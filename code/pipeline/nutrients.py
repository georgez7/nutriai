"""
nutrients.py
------------
Macro and micronutrient aggregator for NutriAI.

Computes:
  - Per-meal and daily nutrient totals
  - RDA comparison by age and sex (NIH DRI tables)
  - Gap flagging (< 80% RDA threshold)
  - Formatted nutrient summary for the UI
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NIH Dietary Reference Intakes (RDA) — simplified tables
# Sources: NIH ODS, NAP DRI reports
# Keys: nutrient column name → {sex: {age_group: value}} in nutrient's native unit
# ---------------------------------------------------------------------------

# Age group helper — maps age (int) to a string key
def _age_group(age: int) -> str:
    if age < 1:    return "0-0.5"
    if age < 4:    return "1-3"
    if age < 9:    return "4-8"
    if age < 14:   return "9-13"
    if age < 19:   return "14-18"
    if age < 31:   return "19-30"
    if age < 51:   return "31-50"
    if age < 71:   return "51-70"
    return "71+"


# RDA_TABLE[nutrient_col][sex][age_group] = daily RDA value
RDA_TABLE: dict[str, dict[str, dict]] = {
    # ── Macros (per day) ──────────────────────────────────────────────
    "calories": {  # kcal — approximate EER (sedentary)
        "female": {"19-30": 2000, "31-50": 1800, "51-70": 1600, "default": 1800},
        "male":   {"19-30": 2500, "31-50": 2300, "51-70": 2100, "default": 2300},
        "other":  {"default": 2000},
    },
    "protein_g": {  # g/day
        "female": {"14-18": 46, "19-30": 46, "31-50": 46, "51-70": 46, "71+": 46, "default": 46},
        "male":   {"14-18": 52, "19-30": 56, "31-50": 56, "51-70": 56, "71+": 56, "default": 56},
        "other":  {"default": 50},
    },
    "carbs_g": {  # g/day (EAR = 100g, RDA = 130g)
        "female": {"default": 130},
        "male":   {"default": 130},
        "other":  {"default": 130},
    },
    "fat_g": {  # No RDA; AI is 20-35% of calories — store as 65g (2000 kcal × 30% / 9)
        "female": {"default": 65},
        "male":   {"default": 78},
        "other":  {"default": 70},
    },
    "fiber_g": {  # g/day
        "female": {"19-30": 25, "31-50": 25, "51-70": 21, "71+": 21, "default": 25},
        "male":   {"19-30": 38, "31-50": 38, "51-70": 30, "71+": 30, "default": 38},
        "other":  {"default": 30},
    },
    # ── Minerals ──────────────────────────────────────────────────────
    "calcium_mg": {  # mg/day
        "female": {
            "9-13": 1300, "14-18": 1300, "19-30": 1000, "31-50": 1000,
            "51-70": 1200, "71+": 1200, "default": 1000,
        },
        "male": {
            "9-13": 1300, "14-18": 1300, "19-30": 1000, "31-50": 1000,
            "51-70": 1000, "71+": 1200, "default": 1000,
        },
        "other": {"default": 1000},
    },
    "iron_mg": {  # mg/day
        "female": {
            "9-13": 8, "14-18": 15, "19-30": 18, "31-50": 18, "51-70": 8,
            "71+": 8, "default": 18,
        },
        "male": {
            "9-13": 8, "14-18": 11, "19-30": 8, "31-50": 8, "51-70": 8,
            "71+": 8, "default": 8,
        },
        "other": {"default": 13},
    },
    "zinc_mg": {  # mg/day
        "female": {"14-18": 9, "19-30": 8, "31-50": 8, "51-70": 8, "71+": 8, "default": 8},
        "male":   {"14-18": 11, "19-30": 11, "31-50": 11, "51-70": 11, "71+": 11, "default": 11},
        "other":  {"default": 9},
    },
    "potassium_mg": {  # mg/day
        "female": {"19-30": 1700, "31-50": 1700, "51-70": 1700, "71+": 1700, "default": 1700},
        "male":   {"19-30": 1700, "31-50": 1700, "51-70": 1700, "71+": 1700, "default": 1700},
        "other":  {"default": 1700},
    },
    "magnesium_mg": {  # mg/day
        "female": {
            "14-18": 360, "19-30": 310, "31-50": 320, "51-70": 320,
            "71+": 320, "default": 320,
        },
        "male": {
            "14-18": 410, "19-30": 400, "31-50": 420, "51-70": 420,
            "71+": 420, "default": 420,
        },
        "other": {"default": 370},
    },
    "sodium_mg": {  # mg/day — upper limit, not RDA
        "female": {"default": 2300},
        "male":   {"default": 2300},
        "other":  {"default": 2300},
    },
    # ── Vitamins ──────────────────────────────────────────────────────
    "vitamin_c_mg": {  # mg/day
        "female": {"14-18": 65, "19-30": 75, "31-50": 75, "51-70": 75, "71+": 75, "default": 75},
        "male":   {"14-18": 75, "19-30": 90, "31-50": 90, "51-70": 90, "71+": 90, "default": 90},
        "other":  {"default": 82},
    },
    "vitamin_d_mcg": {  # mcg/day (600-800 IU = 15-20 mcg)
        "female": {"14-18": 15, "19-30": 15, "31-50": 15, "51-70": 15, "71+": 20, "default": 15},
        "male":   {"14-18": 15, "19-30": 15, "31-50": 15, "51-70": 15, "71+": 20, "default": 15},
        "other":  {"default": 15},
    },
    "vitamin_b12_mcg": {  # mcg/day
        "female": {"14-18": 2.4, "19-30": 2.4, "31-50": 2.4, "51-70": 2.4, "71+": 2.4, "default": 2.4},
        "male":   {"14-18": 2.4, "19-30": 2.4, "31-50": 2.4, "51-70": 2.4, "71+": 2.4, "default": 2.4},
        "other":  {"default": 2.4},
    },
    "phosphorus_mg": {  # mg/day
        "female": {"14-18": 1250, "19-30": 700, "31-50": 700, "51-70": 700, "71+": 700, "default": 700},
        "male":   {"14-18": 1250, "19-30": 700, "31-50": 700, "51-70": 700, "71+": 700, "default": 700},
        "other":  {"default": 700},
    },
    "omega3_g": {  # g/day — NIH AI: 1.1g (female), 1.6g (male)
        "female": {"14-18": 0.1, "19-30":  0.1, "31-50":  0.1, "51-70":  0.1, "71+":  0.1, "default":  0.1},
        "male":   {"14-18": 0.1, "19-30":  0.1, "31-50":  0.1, "51-70":  0.1, "71+":  0.1, "default":  0.1},
        "other":  {"default":  0.1},
    },
}


def get_rda(nutrient: str, age: int, sex: str) -> float:
    """Look up the RDA for a nutrient given age and sex."""
    sex_key = sex.lower() if sex.lower() in ("female", "male") else "other"
    table = RDA_TABLE.get(nutrient, {}).get(sex_key, {})
    age_key = _age_group(age)
    return float(table.get(age_key) or table.get("default") or 0)


def compute_nutrient_gap(
    daily_total: float,
    nutrient:    str,
    age:         int,
    sex:         str,
) -> tuple[float, bool]:
    """
    Returns (pct_of_rda, is_below_threshold).
    Threshold = 80% of RDA.
    """
    rda = get_rda(nutrient, age, sex)
    if rda <= 0:
        return 100.0, False
    pct = (daily_total / rda) * 100
    below = pct < 80.0
    return round(pct, 1), below


# ---------------------------------------------------------------------------
# Daily totals aggregator
# ---------------------------------------------------------------------------

TRACKED_NUTRIENTS = [
    "calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
    "saturated_fat_g", "sodium_mg",
    "calcium_mg", "iron_mg", "vitamin_c_mg",
    "vitamin_d_mcg", "vitamin_b12_mcg",
    "zinc_mg", "potassium_mg", "magnesium_mg", "phosphorus_mg",
    "omega3_g",
]

NUTRIENT_LABELS = {
    "calories":         "Calories (kcal)",
    "protein_g":        "Protein (g)",
    "carbs_g":          "Carbohydrates (g)",
    "fat_g":            "Fat (g)",
    "fiber_g":          "Fibre (g)",
    "saturated_fat_g":  "Saturated Fat (g)",
    "sodium_mg":        "Sodium (mg)",
    "calcium_mg":       "Calcium (mg)",
    "iron_mg":          "Iron (mg)",
    "vitamin_c_mg":     "Vitamin C (mg)",
    "vitamin_d_mcg":    "Vitamin D (mcg)",
    "vitamin_b12_mcg":  "Vitamin B12 (mcg)",
    "zinc_mg":          "Zinc (mg)",
    "potassium_mg":     "Potassium (mg)",
    "magnesium_mg":     "Magnesium (mg)",
    "phosphorus_mg":    "Phosphorus (mg)",
    "omega3_g":         "Omega-3 Fatty Acids (g)",
}


@dataclass
class MealNutrients:
    """Nutrient totals for a single meal."""
    meal_name: str
    totals:    dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_food_row(cls, meal_name: str, food_row: dict, portion_g: float = 200.0) -> "MealNutrients":
        """
        Compute nutrient totals for a serving.
        USDA data is per 100g; scale to portion_g.
        """
        scale = portion_g / 100.0
        totals = {}
        for nutrient in TRACKED_NUTRIENTS:
            raw = float(food_row.get(nutrient) or 0.0)
            totals[nutrient] = round(raw * scale, 2)
        return cls(meal_name=meal_name, totals=totals)

    def __add__(self, other: "MealNutrients") -> "MealNutrients":
        combined = {k: self.totals.get(k, 0) + other.totals.get(k, 0)
                    for k in set(self.totals) | set(other.totals)}
        return MealNutrients(meal_name="combined", totals=combined)


@dataclass
class DayNutrients:
    """Aggregated nutrients for one day."""
    day:     int
    meals:   dict[str, MealNutrients] = field(default_factory=dict)

    @property
    def daily_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for meal in self.meals.values():
            for k, v in meal.totals.items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def gap_report(self, age: int, sex: str, threshold: float = 80.0) -> list[dict]:
        """
        Returns a list of gap dicts for each tracked nutrient.
        Flags nutrients below `threshold`% of RDA.
        """
        totals = self.daily_totals
        gaps = []
        for nutrient in TRACKED_NUTRIENTS:
            rda = get_rda(nutrient, age, sex)
            if rda <= 0:
                continue
            total  = totals.get(nutrient, 0)
            pct    = round((total / rda) * 100, 1)
            below  = pct < threshold
            gaps.append({
                "nutrient":  nutrient,
                "label":     NUTRIENT_LABELS.get(nutrient, nutrient),
                "total":     round(total, 2),
                "rda":       rda,
                "pct_rda":   pct,
                "below_80":  below,
                "flag":      "⚠️" if below else "✅",
            })
        return gaps


# ---------------------------------------------------------------------------
# Week-level aggregator
# ---------------------------------------------------------------------------

class NutrientAggregator:
    """
    Aggregates nutrients across the full 7-day plan and computes
    per-day and weekly RDA gap reports.
    """

    def __init__(self, age: int, sex: str, calorie_target: float):
        self.age             = age
        self.sex             = sex
        self.calorie_target  = calorie_target
        self.days: list[DayNutrients] = []

    def add_day(self, day: DayNutrients):
        self.days.append(day)

    def weekly_summary(self) -> dict:
        """
        Returns {
          "days": [DayNutrients],
          "weekly_totals": {...},
          "weekly_gap_report": [...],
          "days_below_80": {nutrient: count_of_days},
          "avg_pct_rda": {nutrient: avg_pct},
        }
        """
        weekly_totals: dict[str, float] = {}
        days_below:    dict[str, int]   = {n: 0 for n in TRACKED_NUTRIENTS}
        avg_pct:       dict[str, list]  = {n: [] for n in TRACKED_NUTRIENTS}

        for day in self.days:
            for gap in day.gap_report(self.age, self.sex):
                n = gap["nutrient"]
                weekly_totals[n] = weekly_totals.get(n, 0) + gap["total"]
                avg_pct[n].append(gap["pct_rda"])
                if gap["below_80"]:
                    days_below[n] = days_below.get(n, 0) + 1

        avg_pct_flat = {
            n: round(sum(v) / len(v), 1) if v else 0
            for n, v in avg_pct.items()
        }

        # Weekly gap report (totals vs 7 × daily RDA)
        weekly_gap_report = []
        for nutrient in TRACKED_NUTRIENTS:
            rda   = get_rda(nutrient, self.age, self.sex) * 7
            total = weekly_totals.get(nutrient, 0)
            pct   = round((total / rda * 100), 1) if rda > 0 else 100.0
            weekly_gap_report.append({
                "nutrient":     nutrient,
                "label":        NUTRIENT_LABELS.get(nutrient, nutrient),
                "weekly_total": round(total, 2),
                "weekly_rda":   rda,
                "pct_rda":      pct,
                "days_below_80": days_below.get(nutrient, 0),
                "avg_pct_rda":  avg_pct_flat.get(nutrient, 0),
                "flag": "⚠️" if days_below.get(nutrient, 0) >= 3 else "✅",
            })

        return {
            "days":             self.days,
            "weekly_totals":    weekly_totals,
            "weekly_gap_report": weekly_gap_report,
            "days_below_80":    days_below,
            "avg_pct_rda":      avg_pct_flat,
        }

    @staticmethod
    def format_day_summary(day: DayNutrients, age: int, sex: int) -> str:
        """Pretty-print a one-day nutrient summary."""
        totals = day.daily_totals
        gaps   = day.gap_report(age, sex)
        lines  = [f"=== Day {day.day} Nutrient Summary ==="]

        # Macros
        lines.append("\n📊 Macros:")
        for n in ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
            g = next((x for x in gaps if x["nutrient"] == n), None)
            if g:
                lines.append(f"  {g['label']:25s} {g['total']:7.1f}  ({g['pct_rda']:.0f}% RDA) {g['flag']}")

        # Micros
        lines.append("\n💊 Micronutrients:")
        for n in ["calcium_mg", "iron_mg", "vitamin_d_mcg", "vitamin_b12_mcg", "zinc_mg",
                  "potassium_mg", "magnesium_mg", "omega3_g"]:
            g = next((x for x in gaps if x["nutrient"] == n), None)
            if g:
                lines.append(f"  {g['label']:25s} {g['total']:7.2f}  ({g['pct_rda']:.0f}% RDA) {g['flag']}")

        below = [g["label"] for g in gaps if g["below_80"]]
        if below:
            lines.append(f"\n⚠️  Below 80% RDA: {', '.join(below)}")
        else:
            lines.append("\n✅ All nutrients ≥ 80% RDA")

        return "\n".join(lines)
