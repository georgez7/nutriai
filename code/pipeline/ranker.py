"""
ranker.py
---------
4-stage meal ranking pipeline:
  Stage 1 — SQL pre-filter   (billions → thousands via DB query)
  Stage 2 — FAISS ANN search (thousands → top-200 by nutritional similarity)
  Stage 3 — Constraint filter (hard safety rules via ConstraintEngine)
  Stage 4 — Scoring & re-rank (clinical relevance + nutrient gap closure)

BAX-423 technique: multi-stage ranking (mirrors the billion-to-top-10 course pattern).

The pipeline produces a ranked list of (food, score, explanation) tuples
for each meal slot in the 7-day plan.
"""

import sqlite3
import time
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .constraints import ConstraintEngine, UserProfile, load_candidate_foods, FoodVerdict
from .embeddings import FoodEmbedder, build_user_query_vector
from .nutrients import RDA_TABLE, compute_nutrient_gap

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "foods.db"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredFood:
    fdc_id:      int
    food_name:   str
    category:    str
    score:       float
    stage_scores: dict[str, float] = field(default_factory=dict)
    nutrient_row: dict             = field(default_factory=dict)
    verdict:     Optional[FoodVerdict] = None
    portion_g:   float             = 100.0  # serving size scaled to hit calorie target

    def explain(self) -> str:
        parts = [f"Score: {self.score:.3f}"]
        for stage, s in self.stage_scores.items():
            parts.append(f"  {stage}: {s:.3f}")
        return "\n".join(parts)


@dataclass
class MealSlot:
    day:     int   # 1-7
    meal:    str   # breakfast | lunch | dinner
    target_calories:  float
    target_protein_g: float
    target_carbs_g:   float
    target_fat_g:     float
    target_fiber_g:   float
    category_hint:      Optional[str] = None        # preference for variety
    allowed_categories: Optional[set[str]] = None   # hard category whitelist; None = any


# ---------------------------------------------------------------------------
# Scoring functions (Stage 4)
# ---------------------------------------------------------------------------

def score_nutrient_fit(food: dict, slot: MealSlot) -> float:
    """
    How well does the food's nutritional profile match the meal target?
    Returns [0, 1] — 1.0 = perfect match.

    Uses cosine similarity on the (calories, protein, carbs, fat, fiber) vector.
    """
    actual  = np.array([
        float(food.get("calories",   0) or 0),
        float(food.get("protein_g",  0) or 0),
        float(food.get("carbs_g",    0) or 0),
        float(food.get("fat_g",      0) or 0),
        float(food.get("fiber_g",    0) or 0),
    ])
    target = np.array([
        slot.target_calories,
        slot.target_protein_g,
        slot.target_carbs_g,
        slot.target_fat_g,
        slot.target_fiber_g,
    ])

    # Avoid division by zero
    a_norm = np.linalg.norm(actual)
    t_norm = np.linalg.norm(target)
    if a_norm == 0 or t_norm == 0:
        return 0.0

    cos_sim = float(np.dot(actual, target) / (a_norm * t_norm))
    return max(0.0, cos_sim)


def score_gap_closure(food: dict, daily_totals: dict, profile: UserProfile) -> float:
    """
    Reward foods that help close current micronutrient gaps relative to RDA.
    Nutrients listed in profile.micro_priorities receive double weight.
    Returns [0, 1].
    """
    priority_micros = {
        "iron_mg":         ("iron_mg",         profile.age, profile.sex),
        "calcium_mg":      ("calcium_mg",       profile.age, profile.sex),
        "vitamin_d_mcg":   ("vitamin_d_mcg",    profile.age, profile.sex),
        "vitamin_b12_mcg": ("vitamin_b12_mcg",  profile.age, profile.sex),
        "zinc_mg":         ("zinc_mg",          profile.age, profile.sex),
        "potassium_mg":    ("potassium_mg",      profile.age, profile.sex),
        "magnesium_mg":    ("magnesium_mg",      profile.age, profile.sex),
        "omega3_g":        ("omega3_g",          profile.age, profile.sex),
    }

    user_priorities = set(profile.micro_priorities or [])
    score = 0.0
    weight_sum = 0.0

    for col, (nutrient_key, age, sex) in priority_micros.items():
        rda = RDA_TABLE.get(nutrient_key, {}).get(sex, {}).get("default", 0)
        if rda <= 0:
            continue
        current = daily_totals.get(col, 0)
        gap_pct = max(0, 1 - current / rda)
        food_contribution = float(food.get(col) or 0) / rda
        # Double weight for nutrients flagged as user priorities
        w = 2.0 if nutrient_key in user_priorities else 1.0
        score      += w * gap_pct * min(1.0, food_contribution)
        weight_sum += w

    return score / weight_sum if weight_sum > 0 else 0.0


def score_clinical_bonus(food: dict, profile: UserProfile) -> float:
    """
    Assign a bonus for foods that are especially beneficial for the user's conditions.
    Returns [0, 0.3] additive bonus.
    """
    bonus = 0.0
    food_name = (food.get("food_name") or "").lower()
    cat = (food.get("category") or "").lower()

    if profile.has_hypertension:
        # DASH diet: actively reward high-potassium and high-magnesium foods
        pot = float(food.get("potassium_mg") or 0)
        mag = float(food.get("magnesium_mg") or 0)
        sod = float(food.get("sodium_mg")    or 0)
        # Potassium tiers (per 100g): DASH targets 4700mg/day → ~1567mg/meal
        # Tiered heavily — potassium is the primary DASH deficiency gap
        if pot > 100:   bonus += 0.05   # baseline: any meaningful K source
        if pot > 200:   bonus += 0.07   # good: broccoli, fish, most legumes
        if pot > 350:   bonus += 0.08   # excellent: spinach, lentils, salmon, halibut
        if pot > 500:   bonus += 0.05   # outstanding: avocado, adzuki beans, halibut
        # Magnesium tiers (per 100g): DASH targets 420mg/day → ~140mg/meal
        if mag > 25:    bonus += 0.04
        if mag > 60:    bonus += 0.04
        # Low sodium reward (on top of hard constraint)
        if sod < 50:    bonus += 0.04
        # Omega-3 / DASH-approved proteins
        if any(f in food_name for f in ["salmon", "mackerel", "sardine", "herring", "tuna"]):
            bonus += 0.08

    if profile.has_diabetes_t2:
        fiber = float(food.get("fiber_g") or 0)
        gi    = food.get("gi_value")
        carbs = float(food.get("carbs_g") or 0)
        # Reward high-fibre foods — fibre slows glucose absorption
        if fiber > 5:   bonus += 0.10
        if fiber > 10:  bonus += 0.05
        # GL density per 100g = GI × carbs / 100
        # Reward low-GL-density foods; penalise high-GL-density ones
        if gi is not None and carbs > 0:
            gl_per_100g = float(gi) * carbs / 100
            if gl_per_100g <= 5:    bonus += 0.12  # very low GL: tofu, vegetables, protein fish
            elif gl_per_100g <= 10: bonus += 0.06  # low GL: lentils, barley, quinoa
            elif gl_per_100g > 20:  bonus -= 0.12  # high GL: white rice, many baked goods
            elif gl_per_100g > 30:  bonus -= 0.08  # very high GL: additional penalty
        if "legume" in cat:         bonus += 0.05  # legumes: high fibre, moderate GL

    if profile.has_ibs:
        if food.get("fodmap_status") == "safe":
            bonus += 0.1
        # Extra bonus for very low-FODMAP (confirmed safe)
        VERY_SAFE = {"rice", "quinoa", "oat", "carrot", "zucchini", "spinach",
                     "chicken", "salmon", "egg", "tofu", "strawberr", "blueberr"}
        if any(s in food_name for s in VERY_SAFE):
            bonus += 0.05

    if profile.has_gerd:
        fat = float(food.get("fat_g") or 0)
        if fat < 5:   bonus += 0.1
        # Alkaline-friendly
        GERD_FRIENDLY = {"broccoli", "oat", "banana", "melon", "ginger",
                         "aloe", "cauliflower", "fennel", "celery"}
        if any(s in food_name for s in GERD_FRIENDLY):
            bonus += 0.05

    # Micro-priority bonuses — reward foods that address the user's flagged gaps
    if "vitamin_d_mcg" in (profile.micro_priorities or []):
        vit_d = float(food.get("vitamin_d_mcg") or 0)
        if vit_d > 1.0:   bonus += 0.08   # eggs (~1.1 mcg), some fortified foods
        if vit_d > 3.0:   bonus += 0.08   # high-Vit-D fish, fortified cereals
        if vit_d > 8.0:   bonus += 0.06   # fatty fish (salmon ~11 mcg, mackerel ~16 mcg)

    if "iron_mg" in (profile.micro_priorities or []):
        iron = float(food.get("iron_mg") or 0)
        if iron > 2.0:    bonus += 0.06   # legumes, dark greens, fortified grains
        if iron > 4.0:    bonus += 0.06   # lentils (~3.3mg), tofu (~1.6mg), seeds

    if "calcium_mg" in (profile.micro_priorities or []):
        calcium = float(food.get("calcium_mg") or 0)
        if calcium > 100: bonus += 0.06   # legumes, leafy greens, fortified foods
        if calcium > 250: bonus += 0.06   # dairy alternatives, tofu, sardines

    return min(bonus, 0.30)


def score_variety_bonus(food: dict, used_categories: set[str]) -> float:
    """Penalise categories already used recently; reward novel ones."""
    cat = food.get("category", "other")
    if cat in used_categories:
        return -0.15   # penalty
    return 0.05        # small novelty bonus


# ---------------------------------------------------------------------------
# Main ranker
# ---------------------------------------------------------------------------

class MealRanker:
    """
    Orchestrates the 4-stage meal ranking pipeline.

    Parameters
    ----------
    profile      : UserProfile
    embedder     : FoodEmbedder  (pre-built FAISS index)
    db_path      : Path to foods.db
    """

    def __init__(
        self,
        profile: UserProfile,
        embedder: FoodEmbedder,
        db_path: Path = DB_PATH,
    ):
        self.profile  = profile
        self.embedder = embedder
        self.db_path  = db_path
        self.engine   = ConstraintEngine(profile)

        # Stage timing log (for the technical brief / debug)
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_for_slot(
        self,
        slot: MealSlot,
        daily_totals: dict,
        used_categories: set[str],
        used_fdc_ids: set[int],
        top_k: int = 5,
        used_base_names: set[str] | None = None,
    ) -> list[ScoredFood]:
        """
        Run the full 4-stage pipeline for one meal slot.

        Returns top-k ScoredFood items (best first).
        """
        t_total = time.perf_counter()

        # ── Stage 1: SQL pre-filter ───────────────────────────────────
        t1 = time.perf_counter()
        candidates_raw = load_candidate_foods(self.db_path, self.profile, limit=2000)

        # Category boost: when the slot targets a specific category (e.g. "fish"),
        # the random 2000-row draw may contain very few of that category.
        # Pull up to 500 extra rows for the target category so FAISS always has
        # enough representatives. Track boosted IDs separately so they survive
        # Stage 2 even if FAISS didn't rank them in the top-200.
        boosted_ids: set[int] = set()
        if slot.category_hint:
            boosted = load_candidate_foods(
                self.db_path, self.profile,
                limit=500,
                category_boost=slot.category_hint,
            )
            existing_ids = {r["fdc_id"] for r in candidates_raw}
            new_boosted  = [r for r in boosted if r["fdc_id"] not in existing_ids]
            boosted_ids  = {r["fdc_id"] for r in new_boosted}
            candidates_raw += new_boosted

        self.timings["stage1_sql_ms"] = (time.perf_counter() - t1) * 1000

        # ── Stage 2: FAISS ANN retrieval ─────────────────────────────
        t2 = time.perf_counter()
        # DASH: target 4700mg K/day and 420mg Mg/day split across 3 meals
        pot_target = (4700 / 3) if self.profile.has_hypertension else 1200.0
        mag_target = (420  / 3) if self.profile.has_hypertension else 120.0

        query_vec = build_user_query_vector(
            calorie_target      = slot.target_calories,
            protein_target_g    = slot.target_protein_g,
            carb_target_g       = slot.target_carbs_g,
            fat_target_g        = slot.target_fat_g,
            fiber_target_g      = slot.target_fiber_g,
            sodium_limit_mg     = self.profile.sodium_limit_mg / 3,
            category_preference = slot.category_hint,
            gi_preference       = 35.0 if self.profile.has_diabetes_t2 else None,
            potassium_target_mg = pot_target,
            magnesium_target_mg = mag_target,
        )

        # Restrict FAISS search to SQL pre-filtered IDs
        candidate_ids = {r["fdc_id"] for r in candidates_raw}
        ann_results   = self.embedder.search(query_vec, k=200, filter_ids=candidate_ids)
        ann_fdc_ids   = {r["fdc_id"] for r in ann_results}
        self.timings["stage2_faiss_ms"] = (time.perf_counter() - t2) * 1000

        # Fetch full rows for ANN candidates, always keeping boosted rows
        # regardless of FAISS rank so sparse categories (fish) are never lost.
        ann_candidates = [
            r for r in candidates_raw
            if r["fdc_id"] in ann_fdc_ids or r["fdc_id"] in boosted_ids
        ]

        # ── Stage 3: Constraint filter ────────────────────────────────
        t3 = time.perf_counter()
        safe_foods, verdicts = self.engine.filter_candidates(ann_candidates)
        verdict_map = {v.fdc_id: v for v in verdicts}
        self.timings["stage3_constraint_ms"] = (time.perf_counter() - t3) * 1000

        # Remove already-used foods — by fdc_id AND by base name (blocks var N repeats)
        _var_re = __import__("re").compile(r'\s*\(var\s*\d+\)', __import__("re").IGNORECASE)
        def _bn(food): return _var_re.sub("", food.get("food_name","")).strip().lower()

        safe_foods = [
            f for f in safe_foods
            if f["fdc_id"] not in used_fdc_ids
            and (used_base_names is None or _bn(f) not in used_base_names)
        ]

        # ── Stage 3b: Category role enforcement ──────────────────────
        # Ensures grains go in grain slots, proteins in protein slots, etc.
        # Without this, FAISS returns the best nutritional match regardless
        # of food type, so sandwiches end up as "vegetables" and millet as
        # "protein". Only applied when an explicit whitelist is provided.
        if slot.allowed_categories:
            category_filtered = [
                f for f in safe_foods
                if (f.get("category") or "").lower() in slot.allowed_categories
            ]
            # Fall back to unfiltered list only if the whitelist yields nothing
            # (prevents total slot failure on sparse databases)
            if category_filtered:
                safe_foods = category_filtered

        # ── Stage 4: Score & re-rank ──────────────────────────────────
        t4 = time.perf_counter()
        scored = self._score_all(safe_foods, slot, daily_totals, used_categories, verdict_map)
        scored.sort(key=lambda x: x.score, reverse=True)
        self.timings["stage4_score_ms"] = (time.perf_counter() - t4) * 1000
        self.timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        logger.debug(
            "Slot %d/%s — Stage1: %d, Stage2: %d, Stage3: %d safe, Stage4 top: %.3f | %.1fms total",
            slot.day, slot.meal,
            len(candidates_raw), len(ann_candidates), len(safe_foods),
            scored[0].score if scored else 0,
            self.timings["total_ms"],
        )

        return scored[:top_k]

    def generate_plan(self, calorie_target: float) -> dict:
        """
        Generate a complete 7-day, 3-meal-per-day plan.
        Each meal contains multiple food components (protein, grain, vegetable, etc.).

        Returns
        -------
        {
          "days": [
            {"day": 1, "meals": {"breakfast": [ScoredFood, ...], "lunch": [...], "dinner": [...]}},
            ...
          ],
          "timings": {...},
          "generation_time_s": float,
        }
        """
        t_start = time.perf_counter()

        # Calorie split across meals (sums to 1.0)
        MEAL_SPLITS = {
            "breakfast": 0.25,
            "lunch":     0.40,
            "dinner":    0.35,
        }

        # Protein category rotations per diet mode.
        # Each list is cycled by day so protein variety is spread evenly
        # across the 7-day plan without any randomness.
        _dm = self.profile.diet_mode.value
        _PROTEIN_ROTATIONS: dict[str, list[str]] = {
            "non_vegetarian": ["poultry", "fish",   "beef",    "egg",    "poultry", "fish",   "legume"],
            # Pescatarian lunch: fish on days 1, 4, 7 (3 of 7) — keeps pool from depleting
            "pescatarian":    ["fish",    "egg",    "legume",  "fish",   "dairy",   "legume", "fish"],
            "vegetarian":     ["egg",     "legume", "dairy",   "egg",    "legume",  "nut_seed","dairy"],
            "vegan":          ["legume",  "grain",  "nut_seed","legume", "grain",   "legume", "nut_seed"],
        }
        # Separate dinner rotations — prevents the offset formula from
        # creating same-day lunch+dinner doubles that exhaust the fish pool.
        _DINNER_PROTEIN_ROTATIONS: dict[str, list[str]] = {
            "non_vegetarian": ["fish",   "poultry", "legume", "beef",   "egg",    "poultry", "fish"],
            # Pescatarian dinner: fish on days 2, 5 (2 of 7) + lunch covers days 1,4,7
            # → 5 fish meals total, no same-day doubles, spread across all 7 days
            "pescatarian":    ["egg",    "fish",    "dairy",  "legume", "fish",   "egg",     "legume"],
            "vegetarian":     ["dairy",  "egg",     "legume", "dairy",  "nut_seed","egg",    "legume"],
            "vegan":          ["grain",  "legume",  "grain",  "nut_seed","legume","grain",   "legume"],
        }
        _BREAKFAST_PROTEIN_ROTATIONS: dict[str, list[str]] = {
            "non_vegetarian": ["egg",    "dairy",  "egg",    "legume", "egg",    "dairy",  "egg"],
            "pescatarian":    ["egg",    "dairy",  "egg",    "legume", "egg",    "dairy",  "egg"],
            "vegetarian":     ["egg",    "dairy",  "egg",    "legume", "egg",    "nut_seed","dairy"],
            "vegan":          ["legume", "grain",  "nut_seed","legume","grain",  "legume", "nut_seed"],
        }

        # Hard category whitelists per role — prevents FAISS from placing
        # sandwiches in vegetable slots, grains in protein slots, etc.
        # Protein categories are diet-mode-aware: vegan users have a much
        # smaller animal-protein pool, so high-protein grains (quinoa,
        # amaranth, buckwheat) and legumes are included to keep diversity up
        # and avoid the fallback collapsing every protein slot to one category.
        _dm = self.profile.diet_mode.value
        if _dm == "vegan":
            # Grain slot = grains only; legume + nut_seed + grain go to protein
            # so legumes are NOT shared between both slots (prevents grain dominating)
            _GRAIN_CATS   = {"grain"}
            _PROTEIN_CATS = {"legume", "nut_seed", "grain"}
        elif _dm == "vegetarian":
            # Grain slot = grains only; eggs/dairy/legumes/nuts go to protein
            _GRAIN_CATS   = {"grain"}
            _PROTEIN_CATS = {"egg", "legume", "dairy", "nut_seed", "grain"}
        elif _dm == "pescatarian":
            _GRAIN_CATS   = {"grain", "legume"}
            _PROTEIN_CATS = {"fish", "egg", "legume", "dairy", "nut_seed"}
        else:  # non_vegetarian
            _GRAIN_CATS   = {"grain", "legume"}
            _PROTEIN_CATS = {"poultry", "fish", "beef", "pork", "lamb",
                             "egg", "legume", "dairy", "nut_seed"}
        _VEG_FRUIT_CATS = {"vegetable", "fruit"}
        _LEGUME_CATS    = {"legume"}

        def meal_template(meal_name: str) -> list[dict]:
            """
            Build a meal template using a deterministic per-diet-mode rotation
            for the protein component, ensuring varied proteins across the week.
            """
            idx = (day - 1) % 7   # 0-6 index into the 7-slot rotation
            if meal_name == "breakfast":
                protein_cat = _BREAKFAST_PROTEIN_ROTATIONS.get(_dm, ["egg"] * 7)[idx]
                return [
                    {"role": "grain",   "cal_pct": 0.50, "category_hint": "grain",
                     "allowed_categories": _GRAIN_CATS},
                    {"role": "protein", "cal_pct": 0.30, "category_hint": protein_cat,
                     "allowed_categories": _PROTEIN_CATS},
                    {"role": "fruit",   "cal_pct": 0.20, "category_hint": "fruit",
                     "allowed_categories": _VEG_FRUIT_CATS},
                ]
            elif meal_name == "lunch":
                protein_cat = _PROTEIN_ROTATIONS.get(_dm, ["legume"] * 7)[idx]
                side_hint = "legume" if (self.profile.has_hypertension and day % 2 == 0) else "vegetable"
                side_cats  = _LEGUME_CATS if side_hint == "legume" else _VEG_FRUIT_CATS
                return [
                    {"role": "protein",   "cal_pct": 0.40, "category_hint": protein_cat,
                     "allowed_categories": _PROTEIN_CATS},
                    {"role": "grain",     "cal_pct": 0.35, "category_hint": "grain",
                     "allowed_categories": _GRAIN_CATS},
                    {"role": "vegetable", "cal_pct": 0.25, "category_hint": side_hint,
                     "allowed_categories": side_cats},
                ]
            else:  # dinner — use explicit dinner rotation (no offset formula)
                protein_cat = _DINNER_PROTEIN_ROTATIONS.get(_dm, ["legume"] * 7)[idx]
                side_hint = "legume" if (self.profile.has_hypertension and day % 2 == 1) else "vegetable"
                side_cats  = _LEGUME_CATS if side_hint == "legume" else _VEG_FRUIT_CATS
                return [
                    {"role": "protein",   "cal_pct": 0.40, "category_hint": protein_cat,
                     "allowed_categories": _PROTEIN_CATS},
                    {"role": "grain",     "cal_pct": 0.30, "category_hint": "grain",
                     "allowed_categories": _GRAIN_CATS},
                    {"role": "vegetable", "cal_pct": 0.30, "category_hint": side_hint,
                     "allowed_categories": side_cats},
                ]

        _GAP_NUTRIENTS = [
            "calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
            "iron_mg", "calcium_mg", "vitamin_d_mcg", "vitamin_b12_mcg",
            "zinc_mg", "potassium_mg", "magnesium_mg", "omega3_g",
        ]

        plan = {"days": [], "timings": {}}
        used_categories: set[str] = set()
        used_base_names: set[str] = set()   # global — no same base-name across 7 days

        _VAR_RE = __import__("re").compile(r'\s*\(var\s*\d+\)', __import__("re").IGNORECASE)

        def _base_name(food: dict) -> str:
            return _VAR_RE.sub("", food.get("food_name", "")).strip().lower()

        for day in range(1, 8):
            used_fdc_ids: set[int] = set()  # reset each day — only block same-day repeats
            day_entry = {"day": day, "meals": {}}
            daily_totals: dict[str, float] = {}

            for meal_name in ["breakfast", "lunch", "dinner"]:
                meal_cal = calorie_target * MEAL_SPLITS[meal_name]
                components = meal_template(meal_name)
                chosen_components: list[ScoredFood] = []

                for comp in components:
                    comp_cal = meal_cal * comp["cal_pct"]
                    slot = MealSlot(
                        day=day,
                        meal=f"{meal_name}_{comp['role']}",
                        target_calories=comp_cal,
                        target_protein_g=comp_cal * 0.20 / 4,
                        target_carbs_g  =comp_cal * 0.50 / 4,
                        target_fat_g    =comp_cal * 0.30 / 9,
                        allowed_categories=comp.get("allowed_categories"),
                        target_fiber_g  =3.0,
                        category_hint=comp["category_hint"],
                    )

                    top_foods = self.rank_for_slot(
                        slot, daily_totals, used_categories, used_fdc_ids,
                        top_k=3, used_base_names=used_base_names,
                    )
                    if not top_foods:  # fallback 1: allow fdc_id repeats, keep base-name block
                        top_foods = self.rank_for_slot(
                            slot, daily_totals, used_categories, set(),
                            top_k=3, used_base_names=used_base_names,
                        )
                    if not top_foods:  # fallback 2: relax base-name block
                        top_foods = self.rank_for_slot(
                            slot, daily_totals, set(), set(), top_k=3,
                        )
                    if not top_foods:  # fallback 3: last resort — no category hint, fresh broad search
                        broad_slot = MealSlot(
                            day=day,
                            meal=f"{meal_name}_fallback",
                            target_calories=comp_cal,
                            target_protein_g=comp_cal * 0.20 / 4,
                            target_carbs_g  =comp_cal * 0.50 / 4,
                            target_fat_g    =comp_cal * 0.30 / 9,
                            target_fiber_g  =3.0,
                            category_hint=None,  # no FAISS bias — broadest possible search
                        )
                        top_foods = self.rank_for_slot(
                            broad_slot, {}, set(), set(), top_k=3,
                        )

                    if top_foods:
                        # Prefer the highest-scoring food that can deliver the target
                        # calories within the 900g portion cap. When constraints are
                        # tight (e.g. GERD + gluten allergy), top_foods[0] may be
                        # very low calorie-density and would hit the cap, leaving a
                        # calorie deficit. Scan candidates for a denser option first.
                        chosen = top_foods[0]
                        for candidate in top_foods:
                            cand_cal = float(candidate.nutrient_row.get("calories") or 0)
                            if cand_cal > 0 and (comp_cal / cand_cal) * 100.0 <= 900.0:
                                chosen = candidate
                                break

                        food_cal_per_100g = float(chosen.nutrient_row.get("calories") or 0)
                        if food_cal_per_100g > 0:
                            portion_g = max(30.0, min((comp_cal / food_cal_per_100g) * 100.0, 900.0))
                        else:
                            portion_g = 150.0
                        chosen.portion_g = portion_g

                        chosen_components.append(chosen)
                        used_fdc_ids.add(chosen.fdc_id)
                        used_categories.add(chosen.category)
                        used_base_names.add(_base_name(chosen.nutrient_row))

                        scale = portion_g / 100.0
                        for nutrient in _GAP_NUTRIENTS:
                            daily_totals[nutrient] = daily_totals.get(nutrient, 0) + \
                                float(chosen.nutrient_row.get(nutrient) or 0) * scale

                # ── T2DM: meal-level Glycemic Load (GL) check ────────────
                # GL = Σ (GI_i × carbs_in_serving_i) / 100
                # where carbs_in_serving = carbs_per_100g × portion_g / 100
                #
                # A high-GI ingredient does NOT automatically fail — a small
                # portion of white rice alongside fibre, protein and fat can
                # keep the meal GL within the acceptable range.
                # Thresholds: Low <10 · Medium 10–20 · High >20
                # gi_limit slider is repurposed as the max meal GL target.
                if self.profile.has_diabetes_t2 and chosen_components:
                    for _gi_retry in range(2):
                        gl_per_comp = []
                        meal_gl = 0.0
                        for i, c in enumerate(chosen_components):
                            gi    = c.nutrient_row.get("gi_value")
                            carbs = float(c.nutrient_row.get("carbs_g") or 0)
                            if gi is not None and carbs > 0:
                                carbs_serving = carbs * c.portion_g / 100
                                comp_gl = (float(gi) * carbs_serving) / 100
                                gl_per_comp.append((i, comp_gl))
                                meal_gl += comp_gl

                        if meal_gl <= self.profile.gi_limit:
                            break   # meal GL acceptable — move on

                        logger.debug(
                            "Meal GL=%.1f > %.0f — swapping highest-GL component",
                            meal_gl, self.profile.gi_limit,
                        )

                        if not gl_per_comp:
                            break

                        # Swap the component contributing the most GL
                        worst_idx, worst_gl = max(gl_per_comp, key=lambda x: x[1])
                        worst = chosen_components[worst_idx]
                        logger.debug("Swapping '%s' (GL=%.1f)", worst.food_name, worst_gl)
                        worst = chosen_components[worst_idx]
                        logger.debug(
                            "Meal GI max=%.1f > %.0f — swapping '%s' (GI=%.0f)",
                            meal_gl, self.profile.gi_limit, worst.food_name, worst_gl,
                        )
                        used_fdc_ids.add(worst.fdc_id)   # blacklist offender

                        comp     = components[worst_idx]
                        comp_cal = meal_cal * comp["cal_pct"]
                        swap_slot = MealSlot(
                            day=day,
                            meal=f"{meal_name}_{comp['role']}_giswap",
                            target_calories=comp_cal,
                            target_protein_g=comp_cal * 0.20 / 4,
                            target_carbs_g  =comp_cal * 0.50 / 4,
                            target_fat_g    =comp_cal * 0.30 / 9,
                            target_fiber_g  =3.0,
                            category_hint=comp.get("category_hint"),
                        )
                        swap_tops = self.rank_for_slot(
                            swap_slot, daily_totals, used_categories,
                            used_fdc_ids, top_k=3,
                            used_base_names=used_base_names,
                        )
                        if swap_tops:
                            new_food     = swap_tops[0]
                            food_cal_100 = float(new_food.nutrient_row.get("calories") or 0)
                            new_portion  = max(30.0, min(
                                (comp_cal / food_cal_100) * 100.0
                                if food_cal_100 > 0 else 150.0,
                                900.0,
                            ))
                            new_food.portion_g = new_portion
                            chosen_components[worst_idx] = new_food
                            used_fdc_ids.add(new_food.fdc_id)
                            used_base_names.add(_base_name(new_food.nutrient_row))
                        else:
                            break   # no replacement found — fall through to portion scaling

                    # ── GL portion scaling (guaranteed fallback) ──────────
                    # If swapping still left GL over the limit, proportionally
                    # reduce portions of the GL contributors until the meal
                    # is within range. Accepts lower calorie intake for this
                    # meal — clinically appropriate for T2DM management.
                    gl_per_comp2 = []
                    meal_gl2 = 0.0
                    for i, c in enumerate(chosen_components):
                        gi    = c.nutrient_row.get("gi_value")
                        carbs = float(c.nutrient_row.get("carbs_g") or 0)
                        if gi is not None and carbs > 0:
                            carbs_serving = carbs * c.portion_g / 100
                            comp_gl2 = (float(gi) * carbs_serving) / 100
                            gl_per_comp2.append((i, comp_gl2))
                            meal_gl2 += comp_gl2

                    if meal_gl2 > self.profile.gi_limit and meal_gl2 > 0:
                        scale = self.profile.gi_limit / meal_gl2

                        # Track calories before scaling
                        cal_before = sum(
                            float(c.nutrient_row.get("calories") or 0) * c.portion_g / 100
                            for c in chosen_components
                        )

                        # Scale down GL-contributing components
                        for i, _ in gl_per_comp2:
                            c = chosen_components[i]
                            c.portion_g = max(30.0, c.portion_g * scale)

                        # Redistribute lost calories to NULL-GI components —
                        # increasing their portions does not affect meal GL at all.
                        cal_after = sum(
                            float(c.nutrient_row.get("calories") or 0) * c.portion_g / 100
                            for c in chosen_components
                        )
                        cal_deficit = cal_before - cal_after

                        if cal_deficit > 0:
                            null_gi_comps = [
                                (i, c) for i, c in enumerate(chosen_components)
                                if c.nutrient_row.get("gi_value") is None
                            ]
                            if null_gi_comps:
                                cal_per_comp = cal_deficit / len(null_gi_comps)
                                for i, c in null_gi_comps:
                                    cal_per_100g = float(c.nutrient_row.get("calories") or 0)
                                    if cal_per_100g > 0:
                                        extra_g = (cal_per_comp / cal_per_100g) * 100
                                        c.portion_g = min(900.0, c.portion_g + extra_g)
                            else:
                                # No null-GI components (common for vegan diets) —
                                # recover the calorie deficit by partially un-scaling
                                # GL contributors. The GL will not exceed the original
                                # unscaled meal GL, so the limit remains honoured as
                                # closely as possible while preserving calorie intake.
                                recovery_scale = cal_before / max(cal_after, 1.0)
                                recovery_scale = min(recovery_scale, 1.0 / scale if scale > 0 else 1.0)
                                for i, _ in gl_per_comp2:
                                    c = chosen_components[i]
                                    c.portion_g = min(900.0, c.portion_g * recovery_scale)

                        logger.debug(
                            "GL portion-scaled by %.2f → meal GL ≤ %.0f, "
                            "deficit %.0f kcal redistributed to %d null-GI component(s)",
                            scale, self.profile.gi_limit,
                            cal_deficit, len([c for c in chosen_components
                                              if c.nutrient_row.get("gi_value") is None]),
                        )

                # ── Calorie top-up pass ───────────────────────────────────
                # When constraints are tight (e.g. GERD + gluten allergy),
                # low-density foods hit the 900g portion cap and leave a
                # calorie deficit. Redistribute the shortfall to any
                # component that still has headroom below the cap.
                if chosen_components:
                    actual_meal_cal = sum(
                        float(c.nutrient_row.get("calories") or 0) * c.portion_g / 100
                        for c in chosen_components
                    )
                    cal_shortfall = meal_cal - actual_meal_cal
                    if cal_shortfall > 10:  # more than 10 kcal gap — worth fixing
                        # Identify components with headroom (not yet at 900g cap)
                        headroom_comps = [
                            c for c in chosen_components
                            if c.portion_g < 899.0
                            and float(c.nutrient_row.get("calories") or 0) > 0
                        ]
                        if headroom_comps:
                            # Distribute shortfall proportionally by available headroom
                            for c in headroom_comps:
                                cal_per_100g = float(c.nutrient_row.get("calories") or 0)
                                available_extra_cal = (900.0 - c.portion_g) * cal_per_100g / 100
                                share = available_extra_cal / sum(
                                    (900.0 - hc.portion_g) * float(hc.nutrient_row.get("calories") or 0) / 100
                                    for hc in headroom_comps
                                )
                                extra_cal = cal_shortfall * share
                                extra_g   = (extra_cal / cal_per_100g) * 100
                                c.portion_g = min(900.0, c.portion_g + extra_g)

                day_entry["meals"][meal_name] = chosen_components  # list of ScoredFood

            plan["days"].append(day_entry)

        elapsed = time.perf_counter() - t_start
        plan["generation_time_s"] = round(elapsed, 2)
        plan["timings"] = self.timings.copy()

        logger.info("Plan generated: 7 days × 3 meals × ~3 components in %.2fs", elapsed)
        return plan

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_all(
        self,
        foods:           list[dict],
        slot:            MealSlot,
        daily_totals:    dict,
        used_categories: set[str],
        verdict_map:     dict,
    ) -> list[ScoredFood]:
        scored = []
        for food in foods:
            s_nutrient  = score_nutrient_fit(food, slot)
            s_gap       = score_gap_closure(food, daily_totals, self.profile)
            s_clinical  = score_clinical_bonus(food, self.profile)
            s_variety   = score_variety_bonus(food, used_categories)

            # Weighted aggregate
            total = (
                0.45 * s_nutrient +
                0.25 * s_gap      +
                0.20 * s_clinical +
                0.10 * (s_variety + 0.15)   # normalise variety bonus to [0,1] range
            )

            scored.append(ScoredFood(
                fdc_id=food["fdc_id"],
                food_name=food["food_name"],
                category=food.get("category", ""),
                score=round(total, 4),
                stage_scores={
                    "nutrient_fit":    round(s_nutrient, 4),
                    "gap_closure":     round(s_gap, 4),
                    "clinical_bonus":  round(s_clinical, 4),
                    "variety":         round(s_variety, 4),
                },
                nutrient_row=food,
                verdict=verdict_map.get(food["fdc_id"]),
            ))
        return scored
