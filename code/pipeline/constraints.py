"""
constraints.py
--------------
Clinical constraint engine for NutriAI.

Applies hard safety rules for:
  - IBS / low-FODMAP
  - GERD / acid reflux
  - Type 2 Diabetes (GI limits)
  - Hypertension / DASH diet
  - Allergen exclusions (uses Bloom filter for O(1) lookup)
  - Dietary mode (vegan / vegetarian / pescatarian / non-vegetarian)
  - Religious constraints (no pork, halal, etc.)

Each filter returns (passed: bool, reason: str) for the "Explain" feature.
"""

import sqlite3
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

from .bloom_filter import BloomFilter, build_allergen_filter, build_fodmap_filter

logger = logging.getLogger(__name__)

DB_PATH     = Path(__file__).parent.parent.parent / "data" / "foods.db"
FODMAP_CSV  = Path(__file__).parent.parent.parent / "data" / "fodmap_list.csv"


# ---------------------------------------------------------------------------
# User profile dataclass
# ---------------------------------------------------------------------------

class DietMode(str, Enum):
    VEGAN          = "vegan"
    VEGETARIAN     = "vegetarian"
    PESCATARIAN    = "pescatarian"
    NON_VEGETARIAN = "non_vegetarian"


@dataclass
class UserProfile:
    # Demographics
    name:          str   = "User"
    age:           int   = 30
    sex:           str   = "female"   # male | female | other

    # Calorie target (daily kcal)
    calorie_target: float = 2000.0

    # Dietary mode
    diet_mode: DietMode = DietMode.NON_VEGETARIAN

    # Clinical conditions (any combination)
    has_ibs:          bool = False
    has_gerd:         bool = False
    has_diabetes_t2:  bool = False
    has_hypertension: bool = False

    # Allergens (list of strings, e.g. ["gluten", "dairy", "tree nuts"])
    allergens: list[str] = field(default_factory=list)

    # Cultural / religious
    no_pork:    bool = False
    no_beef:    bool = False
    halal_only: bool = False
    kosher_only: bool = False

    # Micronutrient priorities (for gap flagging)
    micro_priorities: list[str] = field(default_factory=list)

    # Sodium cap (mg/day) — overridden by DASH for HTN
    sodium_limit_mg: float = 2300.0

    # GI limit (for DM2)
    gi_limit: float = 55.0

    def __post_init__(self):
        if self.has_hypertension:
            self.sodium_limit_mg = min(self.sodium_limit_mg, 1500.0)
        if isinstance(self.diet_mode, str):
            self.diet_mode = DietMode(self.diet_mode)


# ---------------------------------------------------------------------------
# Individual constraint result
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    passed: bool
    rule:   str           # short rule name
    reason: str           # human-readable explanation (for Explain feature)
    severity: str = "hard"  # hard | soft


# ---------------------------------------------------------------------------
# Constraint engine
# ---------------------------------------------------------------------------

class ConstraintEngine:
    """
    Evaluates a food item against a user's clinical + dietary constraints.

    Usage
    -----
    engine = ConstraintEngine(profile)
    verdict = engine.evaluate(food_row)   # food_row = dict from SQLite
    if not verdict.passed:
        print(verdict.reasons)
    """

    def __init__(self, profile: UserProfile):
        self.profile = profile

        # Build Bloom filters
        self._allergen_bf: Optional[BloomFilter] = None
        self._fodmap_bf:   Optional[BloomFilter] = None

        if profile.allergens:
            self._allergen_bf = build_allergen_filter(profile.allergens)
            logger.info(
                "Allergen filter built for: %s  (%s)",
                profile.allergens, self._allergen_bf,
            )

        if profile.has_ibs:
            self._fodmap_bf = build_fodmap_filter(str(FODMAP_CSV))
            logger.info("FODMAP filter built: %s", self._fodmap_bf)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, food: dict) -> "FoodVerdict":
        """
        Run all applicable constraints on *food*.

        Parameters
        ----------
        food : dict  — a row from the foods table (must include all columns)

        Returns
        -------
        FoodVerdict  — .passed is True only if ALL hard constraints pass.
        """
        results: list[ConstraintResult] = []

        results += self._check_allergens(food)
        results += self._check_diet_mode(food)
        results += self._check_ibs(food)
        results += self._check_gerd(food)
        results += self._check_diabetes(food)
        results += self._check_hypertension(food)
        results += self._check_religious(food)

        failed_hard = [r for r in results if not r.passed and r.severity == "hard"]
        passed      = len(failed_hard) == 0

        return FoodVerdict(
            fdc_id=food.get("fdc_id", 0),
            food_name=food.get("food_name", ""),
            passed=passed,
            results=results,
        )

    def filter_candidates(self, foods: list[dict]) -> tuple[list[dict], list["FoodVerdict"]]:
        """
        Filter a list of food dicts.
        Returns (safe_foods, all_verdicts).
        safe_foods contains only those that passed all hard constraints.
        """
        safe, verdicts = [], []
        for food in foods:
            verdict = self.evaluate(food)
            verdicts.append(verdict)
            if verdict.passed:
                safe.append(food)
        logger.debug(
            "ConstraintEngine: %d/%d foods passed all hard constraints",
            len(safe), len(foods),
        )
        return safe, verdicts

    # ------------------------------------------------------------------
    # Allergen check (Bloom filter — O(1) lookup)
    # ------------------------------------------------------------------

    def _check_allergens(self, food: dict) -> list[ConstraintResult]:
        if not self._allergen_bf:
            return []

        results = []
        food_name_lower = food.get("food_name", "").lower()
        allergen_flags  = food.get("allergen_flags", "") or ""

        # Check food name against Bloom filter
        name_hit = food_name_lower in self._allergen_bf
        # Check pre-computed allergen flags column
        flag_hits = [
            a for a in self.profile.allergens
            if a.lower() in allergen_flags.lower()
        ]

        if name_hit or flag_hits:
            triggered = flag_hits or ["allergen detected in name"]
            results.append(ConstraintResult(
                passed=False,
                rule="allergen_exclusion",
                reason=(
                    f"{food.get('food_name')} excluded — allergen detected: "
                    f"{', '.join(triggered)}. "
                    f"User allergens: {', '.join(self.profile.allergens)}."
                ),
                severity="hard",
            ))
        else:
            results.append(ConstraintResult(
                passed=True,
                rule="allergen_exclusion",
                reason="No allergens detected.",
            ))

        return results

    # ------------------------------------------------------------------
    # Diet mode check
    # ------------------------------------------------------------------

    def _check_diet_mode(self, food: dict) -> list[ConstraintResult]:
        diet_tags = food.get("diet_tags", "") or ""
        mode      = self.profile.diet_mode.value

        # Use exact tag match (split by comma) to avoid "vegetarian" matching inside "non_vegetarian"
        tags_set = {t.strip() for t in diet_tags.split(",")}
        if mode in tags_set:
            return [ConstraintResult(passed=True, rule="diet_mode", reason=f"Compatible with {mode}.")]

        category = food.get("category", "").lower()
        food_name = food.get("food_name", "").lower()

        # Strict checks
        if mode == DietMode.VEGAN.value:
            is_animal = any(c in category for c in ["dairy", "egg", "poultry", "beef", "pork", "fish", "lamb"])
            if is_animal:
                return [ConstraintResult(
                    passed=False, rule="diet_mode",
                    reason=f"{food.get('food_name')} excluded — contains animal products (user is vegan).",
                    severity="hard",
                )]
        elif mode == DietMode.VEGETARIAN.value:
            is_meat = any(c in category for c in ["poultry", "beef", "pork", "lamb", "fish"])
            if is_meat:
                return [ConstraintResult(
                    passed=False, rule="diet_mode",
                    reason=f"{food.get('food_name')} excluded — meat product (user is vegetarian).",
                    severity="hard",
                )]
        elif mode == DietMode.PESCATARIAN.value:
            is_meat = any(c in category for c in ["poultry", "beef", "pork", "lamb"])
            if is_meat:
                return [ConstraintResult(
                    passed=False, rule="diet_mode",
                    reason=f"{food.get('food_name')} excluded — non-fish meat (user is pescatarian).",
                    severity="hard",
                )]

        return [ConstraintResult(passed=True, rule="diet_mode", reason="Diet mode compatible.")]

    # ------------------------------------------------------------------
    # IBS / low-FODMAP check
    # ------------------------------------------------------------------

    # Monash-verified high-FODMAP trigger words (fallback)
    _IBS_TRIGGERS: set[str] = {
        "garlic", "onion", "leek", "spring onion", "shallot",
        "wheat", "rye", "barley", "spelt",
        "apple", "pear", "mango", "watermelon", "cherry", "apricot", "nectarine",
        "milk", "yogurt", "ice cream", "soft cheese", "ricotta", "condensed milk",
        "lentils", "chickpeas", "kidney beans", "baked beans",
        "cashews", "pistachios",
        "honey", "high fructose corn syrup", "agave",
        "cauliflower", "mushroom", "asparagus", "artichoke",
        "lactose",
    }

    def _check_ibs(self, food: dict) -> list[ConstraintResult]:
        if not self.profile.has_ibs:
            return []

        food_name_lower = food.get("food_name", "").lower()

        # 1. Check Bloom filter (fast path)
        if self._fodmap_bf and food_name_lower in self._fodmap_bf:
            return [ConstraintResult(
                passed=False, rule="ibs_fodmap",
                reason=(
                    f"{food.get('food_name')} excluded — identified as high-FODMAP "
                    f"(Monash University list). Unsafe for IBS-D."
                ),
                severity="hard",
            )]

        # 2. Check stored FODMAP status column
        if food.get("fodmap_status") == "unsafe":
            return [ConstraintResult(
                passed=False, rule="ibs_fodmap",
                reason=f"{food.get('food_name')} excluded — marked high-FODMAP in database.",
                severity="hard",
            )]

        # 3. Keyword fallback
        triggered = [t for t in self._IBS_TRIGGERS if t in food_name_lower]
        if triggered:
            return [ConstraintResult(
                passed=False, rule="ibs_fodmap",
                reason=(
                    f"{food.get('food_name')} excluded — contains high-FODMAP ingredient "
                    f"'{triggered[0]}'. Unsafe for IBS."
                ),
                severity="hard",
            )]

        return [ConstraintResult(passed=True, rule="ibs_fodmap", reason="Low-FODMAP food — safe for IBS.")]

    # ------------------------------------------------------------------
    # GERD / acid reflux check
    # ------------------------------------------------------------------

    _GERD_TRIGGERS: dict[str, str] = {
        "citrus":       "citrus fruit — high acidity triggers GERD",
        "lemon":        "lemon — high acidity triggers GERD",
        "lime":         "lime — high acidity triggers GERD",
        "orange juice": "orange juice — high acidity triggers GERD",
        "grapefruit":   "grapefruit — high acidity triggers GERD",
        "tomato":       "tomato — high acidity triggers GERD",
        "tomato sauce": "tomato sauce — high acidity triggers GERD",
        "coffee":       "coffee — caffeine relaxes LES, worsens GERD",
        "espresso":     "espresso — caffeine worsens GERD",
        "chocolate":    "chocolate — relaxes lower oesophageal sphincter",
        "mint":         "peppermint — relaxes lower oesophageal sphincter",
        "fried":        "fried food — delays gastric emptying, triggers GERD",
        "spicy":        "spicy food — irritates oesophagus",
        "chili":        "chili — spicy, triggers GERD",
        "hot sauce":    "hot sauce — spicy, triggers GERD",
        "alcohol":      "alcohol — triggers GERD",
        "beer":         "beer — carbonation + alcohol trigger GERD",
        "wine":         "wine — alcohol + acidity trigger GERD",
        "soda":         "carbonated drinks — worsen GERD symptoms",
        "energy drink": "energy drink — caffeine + carbonation trigger GERD",
        "fatty":        "high-fat food — delays gastric emptying",
        "butter":       "butter — high fat, may trigger GERD",
        "cream":        "cream — high fat, may trigger GERD",
    }

    def _check_gerd(self, food: dict) -> list[ConstraintResult]:
        if not self.profile.has_gerd:
            return []

        food_name_lower = food.get("food_name", "").lower()
        for trigger, explanation in self._GERD_TRIGGERS.items():
            if trigger in food_name_lower:
                return [ConstraintResult(
                    passed=False, rule="gerd_trigger",
                    reason=f"{food.get('food_name')} excluded — {explanation}.",
                    severity="hard",
                )]

        # Soft flag: high fat
        fat = float(food.get("fat_g") or 0)
        if fat > 20:
            return [ConstraintResult(
                passed=False, rule="gerd_high_fat",
                reason=(
                    f"{food.get('food_name')} excluded — fat content {fat:.1f}g/100g "
                    f"exceeds GERD threshold (>20g). High fat delays gastric emptying."
                ),
                severity="hard",
            )]

        return [ConstraintResult(passed=True, rule="gerd_trigger", reason="No GERD trigger foods detected.")]

    # ------------------------------------------------------------------
    # Type 2 Diabetes check (GI limit)
    # ------------------------------------------------------------------

    def _check_diabetes(self, food: dict) -> list[ConstraintResult]:
        if not self.profile.has_diabetes_t2:
            return []

        gi = food.get("gi_value")
        if gi is None:
            return [ConstraintResult(
                passed=True, rule="diabetes_gl",
                reason=f"{food.get('food_name')} — GI unknown; GL assessed at meal level.",
                severity="soft",
            )]

        gi = float(gi)
        # Only hard-exclude extreme outliers (pure sugars, glucose drinks).
        # Meal-level Glycemic Load is the real constraint enforced in the ranker.
        if gi > 85:
            return [ConstraintResult(
                passed=False, rule="diabetes_gl",
                reason=(
                    f"{food.get('food_name')} excluded — GI={gi:.0f} is an extreme "
                    f"outlier (pure sugar / glucose). Meal GL assessed for all others."
                ),
                severity="hard",
            )]

        return [ConstraintResult(
            passed=True, rule="diabetes_gl",
            reason=(
                f"GI={gi:.0f} — ingredient permitted. "
                f"Meal Glycemic Load assessed after portioning."
            ),
        )]

    # ------------------------------------------------------------------
    # Hypertension / DASH check
    # ------------------------------------------------------------------

    def _check_hypertension(self, food: dict) -> list[ConstraintResult]:
        if not self.profile.has_hypertension:
            return []

        results = []

        # Per-meal sodium limit (daily / 3)
        per_meal_sodium_limit = self.profile.sodium_limit_mg / 3
        sodium = float(food.get("sodium_mg") or 0)
        if sodium > per_meal_sodium_limit:
            results.append(ConstraintResult(
                passed=False, rule="hypertension_sodium",
                reason=(
                    f"{food.get('food_name')} excluded — sodium {sodium:.0f}mg/100g "
                    f"exceeds per-meal limit of {per_meal_sodium_limit:.0f}mg "
                    f"(DASH: ≤{self.profile.sodium_limit_mg:.0f}mg/day)."
                ),
                severity="hard",
            ))

        # Saturated fat soft limit (DASH recommends low sat fat)
        sat_fat = float(food.get("saturated_fat_g") or 0)
        if sat_fat > 10:
            results.append(ConstraintResult(
                passed=False, rule="hypertension_sat_fat",
                reason=(
                    f"{food.get('food_name')} excluded — saturated fat {sat_fat:.1f}g "
                    f"exceeds DASH guideline (>10g/100g)."
                ),
                severity="hard",
            ))

        if not results:
            results.append(ConstraintResult(
                passed=True, rule="hypertension",
                reason="Sodium and saturated fat within DASH diet limits.",
            ))

        return results

    # ------------------------------------------------------------------
    # Religious / cultural constraints
    # ------------------------------------------------------------------

    _PORK_KEYWORDS  = {"pork", "ham", "bacon", "prosciutto", "salami", "lard",
                       "sausage", "chorizo", "pepperoni", "pancetta"}
    _BEEF_KEYWORDS  = {"beef", "steak", "veal", "brisket", "hamburger", "meatball"}
    _ALCOHOL_KW     = {"wine", "beer", "vodka", "whiskey", "alcohol", "liqueur"}

    def _check_religious(self, food: dict) -> list[ConstraintResult]:
        results = []
        food_name_lower = food.get("food_name", "").lower()
        category_lower  = food.get("category",  "").lower()

        if self.profile.no_pork:
            hit = next((k for k in self._PORK_KEYWORDS if k in food_name_lower or k in category_lower), None)
            if hit:
                results.append(ConstraintResult(
                    passed=False, rule="no_pork",
                    reason=f"{food.get('food_name')} excluded — contains pork (user restriction).",
                    severity="hard",
                ))
            else:
                results.append(ConstraintResult(passed=True, rule="no_pork", reason="No pork detected."))

        if self.profile.no_beef:
            hit = next((k for k in self._BEEF_KEYWORDS if k in food_name_lower or k in category_lower), None)
            if hit:
                results.append(ConstraintResult(
                    passed=False, rule="no_beef",
                    reason=f"{food.get('food_name')} excluded — contains beef (user restriction).",
                    severity="hard",
                ))

        return results


# ---------------------------------------------------------------------------
# Verdict container
# ---------------------------------------------------------------------------

@dataclass
class FoodVerdict:
    fdc_id:    int
    food_name: str
    passed:    bool
    results:   list[ConstraintResult]

    @property
    def failure_reasons(self) -> list[str]:
        return [r.reason for r in self.results if not r.passed and r.severity == "hard"]

    @property
    def explain(self) -> str:
        """Single-string explanation for the UI 'Explain' feature."""
        if self.passed:
            return f"✅ {self.food_name} — passed all constraints."
        reasons = self.failure_reasons
        return f"❌ {self.food_name} — excluded.\n" + "\n".join(f"  • {r}" for r in reasons)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"FoodVerdict({status}, {self.food_name!r}, {len(self.failure_reasons)} failures)"


# ---------------------------------------------------------------------------
# Database-level filtering helper
# ---------------------------------------------------------------------------

def load_candidate_foods(
    db_path: Path,
    profile: UserProfile,
    limit: int = 2000,
    category_boost: str | None = None,
) -> list[dict]:
    """
    Fast SQL pre-filter: push simple constraints into the query to reduce
    the candidate pool before the full ConstraintEngine evaluation.

    category_boost : when set, restricts the query to that category so the
                     caller gets a focused pool (e.g. all fish) rather than
                     a random cross-category sample.
    """
    # Global category exclusions — never useful for meal planning
    EXCLUDED_CATEGORIES = ("beverage", "baby", "baby food")

    # Food name fragments that should never appear in a meal plan
    EXCLUDED_NAME_FRAGMENTS = (
        "cornmeal",
        "nutritional powder mix",
        "protein, light, nfs",
        "baby toddler bar",
        "baby toddler",
        "corn grain",
        "meat extender",
    )

    conditions = [
        "calories > 0",
        "category NOT IN ({})".format(", ".join("?" * len(EXCLUDED_CATEGORIES))),
    ] + [
        "LOWER(food_name) NOT LIKE ?" for _ in EXCLUDED_NAME_FRAGMENTS
    ]
    params: list = list(EXCLUDED_CATEGORIES) + [f"%{frag}%" for frag in EXCLUDED_NAME_FRAGMENTS]

    # Diet mode pre-filter (SQL tag check)
    mode = profile.diet_mode.value
    conditions.append("diet_tags LIKE ?")
    params.append(f"%{mode}%")

    # Extra hard exclusion for pescatarian: never allow land meat categories
    # (guards against any tagging inconsistency in the DB)
    if mode == DietMode.PESCATARIAN.value:
        conditions.append("category NOT IN ('poultry','beef','pork','lamb')")

    # No pork
    if profile.no_pork:
        conditions.append("(allergen_flags NOT LIKE '%pork%' AND category NOT LIKE '%pork%')")

    # No beef
    if profile.no_beef:
        conditions.append("(category NOT LIKE '%beef%')")

    # GERD: exclude obviously fried / high-fat
    if profile.has_gerd:
        conditions.append("fat_g <= 20")

    # HTN: per-100g sodium pre-filter (generous — exact check in engine)
    if profile.has_hypertension:
        per_meal_limit = profile.sodium_limit_mg / 3
        conditions.append("sodium_mg <= ?")
        params.append(per_meal_limit * 1.5)  # 1.5x buffer for portion size

    # DM2: pre-filter only removes extreme high-GI foods (pure sugars, glucose drinks).
    # Meal-level Glycemic Load (GL) is the real constraint — applied in the ranker
    # after portions are computed. Individual ingredient GI alone doesn't disqualify
    # a food because a high-GI ingredient in a small portion alongside fibre and
    # protein can result in a perfectly acceptable meal GL.
    if profile.has_diabetes_t2:
        conditions.append("(gi_value IS NULL OR gi_value <= 85)")

    # FODMAP: exclude obviously unsafe (fast path via DB column)
    if profile.has_ibs:
        conditions.append("fodmap_status != 'unsafe'")

    # Category boost: restrict to a specific category so sparse categories
    # (e.g. "fish" in a mostly-plant database) are guaranteed to appear.
    if category_boost:
        conditions.append("category = ?")
        params.append(category_boost.lower())

    where_clause = " AND ".join(conditions)
    sql = f"SELECT * FROM foods WHERE {where_clause} ORDER BY RANDOM() LIMIT {limit}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    logger.info("SQL pre-filter: %d candidates (limit=%d)", len(rows), limit)
    return [dict(r) for r in rows]
