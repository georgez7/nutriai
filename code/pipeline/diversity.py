"""
diversity.py
------------
Diversity engine for NutriAI's 7-day meal plan.

Responsibilities:
  1. DiversityScorer  — measures category/nutrient spread across the plan
  2. MaxMarginSampler — selects meals that maximise pairwise distance from used meals
  3. NoRepeatChecker  — enforces zero meal repeats within the 7-day horizon

BAX-423 technique:
  - Diversity engine satisfies Capability #4
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. No-Repeat Checker
# ---------------------------------------------------------------------------

class NoRepeatChecker:
    """
    Tracks which fdc_ids and (food_name) strings have been used.
    Enforces zero exact repeats within the 7-day plan.
    Also enforces a soft "same-category cooldown" window.
    """

    def __init__(self, cooldown_days: int = 2):
        self._used_ids:   set[int]       = set()
        self._used_names: set[str]       = set()
        # category → list of day numbers when it was last used
        self._cat_history: dict[str, list[int]] = defaultdict(list)
        self.cooldown_days = cooldown_days

    def mark_used(self, fdc_id: int, food_name: str, category: str, day: int):
        self._used_ids.add(fdc_id)
        self._used_names.add(food_name.lower().strip())
        self._cat_history[category].append(day)

    def is_repeat(self, fdc_id: int, food_name: str) -> bool:
        if fdc_id in self._used_ids:
            return True
        return food_name.lower().strip() in self._used_names

    def is_category_on_cooldown(self, category: str, current_day: int) -> bool:
        history = self._cat_history.get(category, [])
        if not history:
            return False
        last_used = max(history)
        return (current_day - last_used) < self.cooldown_days

    def filter(self, foods: list[dict], current_day: int) -> list[dict]:
        """
        Return foods that pass the no-repeat check and are not in category cooldown.
        """
        result = []
        for food in foods:
            fdc_id    = food.get("fdc_id", 0)
            food_name = food.get("food_name", "")
            category  = food.get("category", "other")

            if self.is_repeat(fdc_id, food_name):
                continue
            if self.is_category_on_cooldown(category, current_day):
                continue
            result.append(food)
        return result

    @property
    def used_count(self) -> int:
        return len(self._used_ids)


# ---------------------------------------------------------------------------
# 2. Diversity Scorer
# ---------------------------------------------------------------------------

CATEGORY_GROUPS = {
    "protein":  {"poultry", "beef", "pork", "fish", "legume", "egg", "dairy"},
    "produce":  {"vegetable", "fruit"},
    "grain":    {"grain"},
    "fat":      {"nut_seed", "fat_oil"},
    "other":    {"sauce", "spice", "sweet", "snack", "beverage"},
}


def _group_of(category: str) -> str:
    for group, cats in CATEGORY_GROUPS.items():
        if any(c in category.lower() for c in cats):
            return group
    return "other"


@dataclass
class DiversityReport:
    score:            float          # 0-1 composite diversity score
    unique_categories: int
    unique_groups:    int
    category_counts:  dict[str, int]
    group_counts:     dict[str, int]
    repeat_count:     int
    details:          str            # human-readable breakdown


class DiversityScorer:
    """
    Measures diversity across the planned meals.

    Diversity score = weighted average of:
      - Category diversity (unique categories / total meals)
      - Group balance (how evenly spread across achievable groups for this profile)
      - Repeat penalty (0 repeats = perfect)
    """

    def _get_achievable_groups(self, profile) -> set[str]:
        """
        Return the food groups that are structurally reachable for this profile.

        A vegan + tree-nuts user cannot produce a "fat" group (nut_seed blocked,
        fat_oil not used as a meal component) and their "protein" group is limited
        to legumes. Scoring balance against groups that are impossible to achieve
        structurally penalises the plan for the user's constraints, not for poor
        planning. We instead score only against what the profile can produce.
        """
        dm       = getattr(profile, "diet_mode", None)
        dm_value = dm.value if dm is not None else "non_vegetarian"
        allergens = {a.lower().strip() for a in getattr(profile, "allergens", [])}

        # Start with categories available to all diets
        available_cats: set[str] = {"grain", "vegetable", "fruit"}

        # Add protein-source categories based on diet mode
        if dm_value == "vegan":
            available_cats |= {"legume"}
        elif dm_value == "vegetarian":
            available_cats |= {"legume", "dairy", "egg"}
        elif dm_value == "pescatarian":
            available_cats |= {"legume", "dairy", "egg", "fish"}
        else:  # non_vegetarian
            available_cats |= {"legume", "dairy", "egg",
                               "poultry", "fish", "beef", "pork", "lamb"}

        # Apply allergen exclusions
        if "tree nuts" in allergens:
            available_cats.discard("nut_seed")
        else:
            available_cats.add("nut_seed")

        if "dairy" in allergens:
            available_cats.discard("dairy")
        if "eggs" in allergens:
            available_cats.discard("egg")
        if "fish" in allergens:
            available_cats -= {"fish", "shellfish"}
        if "shellfish" in allergens:
            available_cats.discard("shellfish")

        # Map available categories → achievable groups
        # Exclude "other" entirely — sauce/spice/snack shouldn't appear in plans
        achievable: set[str] = set()
        for group, cats in CATEGORY_GROUPS.items():
            if group == "other":
                continue
            if any(c in available_cats for c in cats):
                achievable.add(group)

        return achievable

    def score_plan(self, plan_foods: list[dict], profile=None) -> DiversityReport:
        """
        Parameters
        ----------
        plan_foods : list of food dicts (one per meal slot, in order)
        profile    : optional UserProfile — when provided, group balance is scored
                     against achievable groups only, so constrained profiles
                     (vegan, multiple allergens) are not penalised for groups that
                     are structurally impossible for them to produce.

        Returns
        -------
        DiversityReport
        """
        if not plan_foods:
            return DiversityReport(0.0, 0, 0, {}, {}, 0, "No foods in plan.")

        n = len(plan_foods)
        cat_counts:   dict[str, int] = defaultdict(int)
        group_counts: dict[str, int] = defaultdict(int)
        name_counts:  dict[str, int] = defaultdict(int)

        for food in plan_foods:
            cat   = food.get("category", "other").lower()
            name  = food.get("food_name", "").lower()
            group = _group_of(cat)
            cat_counts[cat]    += 1
            group_counts[group] += 1
            name_counts[name]  += 1

        unique_cats   = len(cat_counts)
        unique_groups = len(group_counts)
        repeat_count  = sum(v - 1 for v in name_counts.values() if v > 1)

        # ── Category entropy score ────────────────────────────────────
        cat_entropy = self._entropy(list(cat_counts.values()), n)
        max_entropy = math.log2(unique_cats) if unique_cats > 1 else 1
        cat_score   = min(1.0, cat_entropy / max_entropy) if max_entropy > 0 else 1.0

        # ── Group balance score ───────────────────────────────────────
        # If a profile is provided, score only against achievable groups —
        # groups that are structurally impossible (e.g. "fat" for vegan +
        # tree nuts) are excluded from the ideal and the imbalance sum.
        # This prevents constrained profiles from being penalised for
        # constraints they set, not for poor meal variety.
        if profile is not None:
            achievable = self._get_achievable_groups(profile)
        else:
            # Fall back: treat every present group as achievable
            achievable = set(group_counts.keys())

        n_achievable    = max(len(achievable), 1)
        ideal_per_group = n / n_achievable
        imbalance = sum(
            abs(group_counts.get(g, 0) - ideal_per_group)
            for g in achievable
        ) / n
        balance_score = max(0.0, 1.0 - imbalance)

        # ── Repeat score ──────────────────────────────────────────────
        repeat_penalty = min(0.5, repeat_count * 0.1)
        repeat_score   = 1.0 - repeat_penalty

        # ── Composite ────────────────────────────────────────────────
        composite = (0.45 * cat_score + 0.30 * balance_score + 0.25 * repeat_score)

        details_lines = [
            f"Meals analysed: {n}",
            f"Unique categories: {unique_cats} / {n}",
            f"Food group spread: {dict(group_counts)}",
            f"Achievable groups: {sorted(achievable)}",
            f"Repeat meals: {repeat_count}",
            f"Category entropy score: {cat_score:.2f}",
            f"Group balance score: {balance_score:.2f}  (ideal {ideal_per_group:.1f}/group)",
            f"Repeat score: {repeat_score:.2f}",
            f"─── Composite diversity score: {composite:.3f} ───",
        ]

        report = DiversityReport(
            score=round(composite, 3),
            unique_categories=unique_cats,
            unique_groups=unique_groups,
            category_counts=dict(cat_counts),
            group_counts=dict(group_counts),
            repeat_count=repeat_count,
            details="\n".join(details_lines),
        )
        logger.info("Diversity score: %.3f (cats=%d, repeats=%d, achievable_groups=%d)",
                    composite, unique_cats, repeat_count, n_achievable)
        return report

    @staticmethod
    def _entropy(counts: list[int], total: int) -> float:
        """Shannon entropy (bits) of a count distribution."""
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                h -= p * math.log2(p)
        return h


# ---------------------------------------------------------------------------
# 3. Max-Margin Sampler
# ---------------------------------------------------------------------------

class MaxMarginSampler:
    """
    Selects food items that maximise the minimum pairwise distance from
    already-selected items in the nutritional embedding space.

    This ensures the plan is spread across the nutrient space, not just
    category-diverse.
    """

    def __init__(self, feature_cols: Optional[list[str]] = None):
        self.feature_cols = feature_cols or [
            "calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
            "sodium_mg", "calcium_mg", "iron_mg",
        ]
        self._selected_vectors: list[np.ndarray] = []

    def _to_vec(self, food: dict) -> np.ndarray:
        vec = np.array([float(food.get(c) or 0) for c in self.feature_cols])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _min_distance_to_selected(self, vec: np.ndarray) -> float:
        if not self._selected_vectors:
            return float("inf")
        dists = [float(np.linalg.norm(vec - sv)) for sv in self._selected_vectors]
        return min(dists)

    def select(self, candidates: list[dict], n: int = 1) -> list[dict]:
        """
        Select `n` candidates that maximise margin from already-selected items.
        """
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)

        for _ in range(min(n, len(remaining))):
            if not remaining:
                break
            # Score each remaining candidate by min distance to selected set
            vecs   = [self._to_vec(f) for f in remaining]
            scores = [self._min_distance_to_selected(v) for v in vecs]
            best_i = int(np.argmax(scores))
            chosen = remaining.pop(best_i)
            self._selected_vectors.append(vecs[best_i])
            selected.append(chosen)

        return selected

    def mark_selected(self, food: dict):
        """Register an externally selected food."""
        self._selected_vectors.append(self._to_vec(food))

    def reset(self):
        self._selected_vectors = []
