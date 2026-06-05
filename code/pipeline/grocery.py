"""
grocery.py
----------
Grocery list generator for the NutriAI 7-day meal plan.

Consolidates all food components across 21 meals, converts gram
weights to purchase units, groups by store section, and prices items
using the Spoonacular Ingredients API (live) with category-average fallback.

Spoonacular flow (no OAuth — just API key):
  GET /food/ingredients/search?query={name}&number=1&apiKey={key}
      → ingredient id
  GET /food/ingredients/{id}/information?amount=100&unit=grams&apiKey={key}
      → estimatedCost: { value: <cents per 100g>, unit: "US Cents" }
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price table — average US supermarket prices ($/kg, 2024 estimates)
# Used as fallback when Spoonacular doesn't find an ingredient.
# ---------------------------------------------------------------------------

PRICE_PER_KG: dict[str, float] = {
    "vegetable": 3.50,
    "fruit":     4.20,
    "grain":     2.80,
    "legume":    3.00,
    "dairy":     5.00,
    "egg":       5.50,
    "poultry":   9.00,
    "fish":     14.00,
    "beef":     13.00,
    "pork":      8.00,
    "lamb":     12.00,
    "nut_seed": 18.00,
    "fat_oil":   5.00,
    "baked":     4.00,
    "beverage":  2.50,
    "sauce":     3.00,
    "snack":     6.00,
    "spice":    20.00,
    "sweet":     5.00,
    "other":     4.00,
}

SECTION_MAP: dict[str, str] = {
    "vegetable": "🥦 Produce",
    "fruit":     "🥦 Produce",
    "grain":     "🌾 Grains & Pantry",
    "legume":    "🌾 Grains & Pantry",
    "baked":     "🌾 Grains & Pantry",
    "fat_oil":   "🌾 Grains & Pantry",
    "sauce":     "🌾 Grains & Pantry",
    "spice":     "🌾 Grains & Pantry",
    "dairy":     "🧀 Dairy & Eggs",
    "egg":       "🧀 Dairy & Eggs",
    "poultry":   "🥩 Meat & Fish",
    "fish":      "🥩 Meat & Fish",
    "beef":      "🥩 Meat & Fish",
    "pork":      "🥩 Meat & Fish",
    "lamb":      "🥩 Meat & Fish",
    "nut_seed":  "🥜 Nuts & Seeds",
    "snack":     "🛒 Other",
    "beverage":  "🛒 Other",
    "sweet":     "🛒 Other",
    "other":     "🛒 Other",
}

SECTION_ORDER = [
    "🥦 Produce",
    "🥩 Meat & Fish",
    "🧀 Dairy & Eggs",
    "🌾 Grains & Pantry",
    "🥜 Nuts & Seeds",
    "🛒 Other",
]

# ---------------------------------------------------------------------------
# Spoonacular API client
# ---------------------------------------------------------------------------

SPOONACULAR_API_KEY = "90ffb3699dc8489f8a622eb7664b829b"
SPOONACULAR_BASE    = "https://api.spoonacular.com"


class SpoonacularPricer:
    """
    Fetches ingredient price estimates from the Spoonacular API.

    Uses two endpoints per ingredient:
      1. /food/ingredients/search  — resolve name → ingredient id
      2. /food/ingredients/{id}/information?amount=100&unit=grams
                                   — get estimatedCost in US cents per 100g

    Results are cached in-memory so each ingredient is only looked up once
    per session (important for the free tier: 150 points/day).
    """

    def __init__(self, api_key: str = SPOONACULAR_API_KEY):
        self.api_key = api_key
        self._cache: dict[str, Optional[float]] = {}   # name → price per 100g (USD)
        self.session = requests.Session()

    def get_price_per_100g(self, food_name: str) -> tuple[Optional[float], str]:
        """
        Look up the price for food_name.

        Returns
        -------
        (price_per_100g_usd, source)
          price_per_100g_usd : USD per 100g, or None if not found
          source             : "spoonacular" | "not_found"
        """
        key = food_name.lower().strip()
        if key in self._cache:
            v = self._cache[key]
            return (v, "spoonacular") if v is not None else (None, "not_found")

        try:
            # Step 1 — search for ingredient id
            r1 = self.session.get(
                f"{SPOONACULAR_BASE}/food/ingredients/search",
                params={"query": food_name, "number": 1, "apiKey": self.api_key},
                timeout=10,
            )
            if not r1.ok:
                logger.warning("Spoonacular search %d for '%s'", r1.status_code, food_name)
                self._cache[key] = None
                return None, "not_found"

            results = r1.json().get("results", [])
            if not results:
                self._cache[key] = None
                return None, "not_found"

            ingredient_id = results[0]["id"]

            # Step 2 — get cost per 100g
            r2 = self.session.get(
                f"{SPOONACULAR_BASE}/food/ingredients/{ingredient_id}/information",
                params={"amount": 100, "unit": "grams", "apiKey": self.api_key},
                timeout=10,
            )
            if not r2.ok:
                logger.warning("Spoonacular info %d for id=%d", r2.status_code, ingredient_id)
                self._cache[key] = None
                return None, "not_found"

            cost_info = r2.json().get("estimatedCost", {})
            cents      = cost_info.get("value", 0)

            if cents and float(cents) > 0:
                price_per_100g = float(cents) / 100   # US cents → USD
                self._cache[key] = price_per_100g
                logger.debug("Spoonacular '%s' → $%.4f/100g", food_name, price_per_100g)
                return price_per_100g, "spoonacular"

        except Exception as e:
            logger.warning("Spoonacular lookup failed for '%s': %s", food_name, e)

        self._cache[key] = None
        return None, "not_found"

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GroceryItem:
    name:          str
    category:      str
    section:       str
    total_g:       float
    purchase_unit: str
    est_cost_usd:  float
    price_source:  str = "estimated"   # "spoonacular" | "estimated"


@dataclass
class GroceryList:
    items:          list[GroceryItem]
    by_section:     dict[str, list[GroceryItem]]
    total_cost_usd: float
    total_items:    int
    notes:          list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r'\s*\(var\s*\d+\)', re.IGNORECASE)


def _clean_name(name: str) -> str:
    """Strip demo-database variation suffixes like '(var 83)'."""
    return _VAR_RE.sub("", name).strip()


def _purchase_unit(total_g: float, category: str) -> str:
    """Convert a gram total to a readable purchase unit."""
    if category == "spice":
        return f"{total_g:.0f}g"
    if category == "egg":
        eggs = max(1, round(total_g / 50))
        return f"{eggs} egg{'s' if eggs != 1 else ''}"
    if total_g < 50:
        return f"{total_g:.0f}g"
    if total_g < 950:
        rounded = round(total_g / 50) * 50
        return f"{rounded:.0f}g"
    return f"{total_g / 1000:.2f}kg"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_grocery_list(
    plan:   list[list[dict]],
    pricer: Optional[SpoonacularPricer] = None,
) -> GroceryList:
    """
    Consolidate all meal components across the 7-day plan into a
    deduplicated grocery list with quantities and cost estimates.

    Parameters
    ----------
    plan   : list of 21 meal-lists (each meal = list of food dicts with portion_g)
    pricer : optional SpoonacularPricer — fetches live prices when provided;
             falls back to PRICE_PER_KG category averages otherwise

    Returns
    -------
    GroceryList
    """
    # Aggregate portions by food name
    aggregated: dict[str, dict] = {}
    for meal in plan:
        for food in meal:
            name     = _clean_name(food.get("food_name", "Unknown"))
            category = (food.get("category") or "other").lower()
            portion  = float(food.get("portion_g") or 100.0)
            if name not in aggregated:
                aggregated[name] = {"category": category, "total_g": 0.0}
            aggregated[name]["total_g"] += portion

    # Price each item
    spoon_hits = 0
    items: list[GroceryItem] = []

    for name, data in aggregated.items():
        cat     = data["category"]
        total_g = data["total_g"]
        section = SECTION_MAP.get(cat, "🛒 Other")
        unit    = _purchase_unit(total_g, cat)
        source  = "estimated"

        cost = None
        if pricer:
            price_per_100g, source = pricer.get_price_per_100g(name)
            if price_per_100g is not None:
                cost = price_per_100g * (total_g / 100)
                spoon_hits += 1

        if cost is None:
            cost   = (total_g / 1000) * PRICE_PER_KG.get(cat, PRICE_PER_KG["other"])
            source = "estimated"

        items.append(GroceryItem(
            name=name,
            category=cat,
            section=section,
            total_g=round(total_g, 1),
            purchase_unit=unit,
            est_cost_usd=round(cost, 2),
            price_source=source,
        ))

    if pricer:
        logger.info("Spoonacular prices: %d/%d items found", spoon_hits, len(items))

    items.sort(key=lambda x: (
        SECTION_ORDER.index(x.section) if x.section in SECTION_ORDER else 99,
        x.name,
    ))

    by_section: dict[str, list[GroceryItem]] = {}
    for item in items:
        by_section.setdefault(item.section, []).append(item)

    total_cost = round(sum(i.est_cost_usd for i in items), 2)

    spoon_count = sum(1 for i in items if i.price_source == "spoonacular")
    est_count   = len(items) - spoon_count

    if pricer and spoon_count > 0:
        price_note = (
            f"Prices: {spoon_count} items from Spoonacular live data, "
            f"{est_count} items from category averages (~est)."
        )
    else:
        price_note = "Prices are average US supermarket estimates (2024) — vary by store and region."

    notes = [
        price_note,
        "Quantities reflect total weekly need. Adjust for your household size.",
        "Spice quantities assume you already have pantry staples — only purchase if needed.",
    ]

    logger.info("Grocery list: %d items, est. $%.2f", len(items), total_cost)

    return GroceryList(
        items=items,
        by_section=by_section,
        total_cost_usd=total_cost,
        total_items=len(items),
        notes=notes,
    )
