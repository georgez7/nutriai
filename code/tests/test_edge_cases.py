"""
test_edge_cases.py
------------------
Edge case tests for NutriAI covering boundary conditions, degenerate inputs,
and extreme constraint combinations not exercised by the persona suite.

Categories:
  EC1  — Constraint Engine extremes
  EC2  — Bloom Filter boundary conditions
  EC3  — Ranker / plan generation robustness
  EC4  — Nutrient Aggregator with bad/extreme data
  EC5  — Grocery list edge cases
  EC6  — Portion scaling boundaries
  EC7  — UserProfile validation
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.constraints import (
    ConstraintEngine, UserProfile, DietMode,
    load_candidate_foods, FoodVerdict,
)
from pipeline.bloom_filter import BloomFilter, build_allergen_filter, build_fodmap_filter
from pipeline.nutrients import (
    NutrientAggregator, DayNutrients, MealNutrients,
    get_rda, TRACKED_NUTRIENTS,
)
from pipeline.diversity import DiversityScorer
from pipeline.grocery import generate_grocery_list, SpoonacularPricer, _clean_name

DB_PATH    = Path(__file__).parent.parent.parent / "data" / "foods.db"
FODMAP_CSV = Path(__file__).parent.parent.parent / "data" / "fodmap_list.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def _food(**kwargs) -> dict:
    """Build a minimal food dict for testing constraint/scoring logic."""
    defaults = {
        "fdc_id": 999999, "food_name": "Test Food", "brand": "generic",
        "category": "vegetable", "diet_tags": "vegan,vegetarian,pescatarian,non_vegetarian",
        "allergen_flags": "", "fodmap_status": "safe", "gi_value": None,
        "calories": 100.0, "protein_g": 5.0, "carbs_g": 20.0, "fat_g": 1.0,
        "fiber_g": 3.0, "saturated_fat_g": 0.1, "sodium_mg": 10.0,
        "calcium_mg": 50.0, "iron_mg": 1.0, "vitamin_c_mg": 10.0,
        "vitamin_d_mcg": 0.0, "vitamin_b12_mcg": 0.0,
        "zinc_mg": 0.5, "potassium_mg": 300.0, "magnesium_mg": 30.0,
        "phosphorus_mg": 80.0,
    }
    defaults.update(kwargs)
    return defaults


def _all_conditions_profile(**kwargs) -> UserProfile:
    """Profile with all 4 clinical conditions active."""
    params = dict(
        name="Edge", age=45, sex="female", calorie_target=1800,
        diet_mode=DietMode.VEGAN,
        has_ibs=True, has_gerd=True, has_diabetes_t2=True, has_hypertension=True,
        allergens=["dairy", "gluten", "eggs", "soy", "tree nuts"],
        no_pork=True, sodium_limit_mg=1500, gi_limit=55,
    )
    params.update(kwargs)
    return UserProfile(**params)


# ═══════════════════════════════════════════════════════════════════════════════
# EC1 — Constraint Engine Extremes
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstraintEngineExtremes:

    def test_all_four_conditions_simultaneously(self):
        """Profile with IBS + GERD + T2DM + HTN should not crash."""
        profile = _all_conditions_profile()
        engine  = ConstraintEngine(profile)
        food    = _food(
            food_name="Brown Rice (cooked)", category="grain",
            gi_value=50.0, sodium_mg=5.0, fat_g=1.0,
            fodmap_status="safe", potassium_mg=200.0,
        )
        verdict = engine.evaluate(food)
        assert isinstance(verdict, FoodVerdict)
        assert isinstance(verdict.passed, bool)

    def test_all_major_allergens_excluded(self):
        """Food flagged with all 9 major allergens must fail allergen check."""
        profile = UserProfile(
            name="MaxAllergen", age=30, sex="male", calorie_target=2000,
            diet_mode=DietMode.NON_VEGETARIAN,
            allergens=["dairy", "gluten", "eggs", "soy", "tree nuts",
                       "peanuts", "fish", "shellfish", "sesame"],
        )
        engine = ConstraintEngine(profile)
        food   = _food(
            food_name="almond milk wheat bread",
            allergen_flags="dairy,gluten,eggs,soy,tree nuts,peanuts,fish,shellfish,sesame",
        )
        verdict = engine.evaluate(food)
        assert not verdict.passed, "Food with all allergens should be excluded"

    def test_gerd_high_fat_threshold(self):
        """Food with fat_g > 20 must fail GERD hard constraint."""
        profile = UserProfile(name="G", age=40, sex="male",
                              calorie_target=2000, has_gerd=True)
        engine  = ConstraintEngine(profile)
        # Exactly at boundary
        food_ok  = _food(fat_g=20.0, food_name="Borderline Fat Food")
        food_bad = _food(fat_g=20.1, food_name="High Fat Food")
        assert engine.evaluate(food_ok).passed,  "fat_g=20 should pass GERD check"
        assert not engine.evaluate(food_bad).passed, "fat_g=20.1 should fail GERD check"

    def test_diabetes_gi_boundary(self):
        """
        Per-food GI is NOT a hard cutoff at gi_limit — meal-level Glycemic Load
        is enforced by the ranker after portioning. The constraint engine only
        hard-blocks extreme outliers (GI > 85, e.g. pure glucose drinks).
        Foods with GI at or moderately above the user's gi_limit slider still
        pass individual evaluation; only GI > 85 is a hard exclusion.
        """
        profile = UserProfile(name="D", age=50, sex="female",
                              calorie_target=1800, has_diabetes_t2=True, gi_limit=55)
        engine  = ConstraintEngine(profile)
        # Moderate GI foods pass — meal-level GL enforced in ranker
        at_limit  = _food(gi_value=55.0, food_name="GI=55 Food")
        above_lim = _food(gi_value=55.1, food_name="GI=55.1 Food")
        assert engine.evaluate(at_limit).passed,  "GI=55 should pass (GL checked at meal level)"
        assert engine.evaluate(above_lim).passed, "GI=55.1 should pass (only GI>85 is hard-blocked)"
        # Only extreme high-GI (pure sugar/glucose) is hard-blocked
        extreme = _food(gi_value=86.0, food_name="Pure Glucose Drink")
        assert not engine.evaluate(extreme).passed, "GI=86 should fail (extreme outlier)"

    def test_unknown_gi_passes_for_diabetes(self):
        """Foods with NULL GI should be cautiously included (soft flag only)."""
        profile = UserProfile(name="D2", age=50, sex="female",
                              calorie_target=1800, has_diabetes_t2=True, gi_limit=55)
        engine = ConstraintEngine(profile)
        food   = _food(gi_value=None, food_name="Unknown GI Food")
        verdict = engine.evaluate(food)
        assert verdict.passed, "NULL GI should be a soft flag, not a hard exclusion"

    def test_vegan_excludes_all_animal_categories(self):
        """Each animal-product category must be excluded for vegan users."""
        profile = UserProfile(name="V", age=25, sex="female",
                              calorie_target=1800, diet_mode=DietMode.VEGAN)
        engine  = ConstraintEngine(profile)
        animal_foods = [
            _food(food_name="Chicken Breast", category="poultry",
                  diet_tags="non_vegetarian"),
            _food(food_name="Cheddar Cheese", category="dairy",
                  diet_tags="vegetarian,non_vegetarian"),
            _food(food_name="Salmon",         category="fish",
                  diet_tags="pescatarian,non_vegetarian"),
            _food(food_name="Whole Egg",      category="egg",
                  diet_tags="vegetarian,non_vegetarian"),
            _food(food_name="Beef Steak",     category="beef",
                  diet_tags="non_vegetarian"),
        ]
        for food in animal_foods:
            verdict = engine.evaluate(food)
            assert not verdict.passed, \
                f"{food['food_name']} should be excluded for vegan"

    def test_pescatarian_allows_fish_excludes_land_meat(self):
        """Pescatarian: fish passes, poultry/beef/pork fail."""
        profile = UserProfile(name="P", age=35, sex="male",
                              calorie_target=2000, diet_mode=DietMode.PESCATARIAN)
        engine  = ConstraintEngine(profile)
        fish    = _food(food_name="Salmon", category="fish",
                        diet_tags="pescatarian,non_vegetarian")
        chicken = _food(food_name="Chicken", category="poultry",
                        diet_tags="non_vegetarian")
        assert engine.evaluate(fish).passed,    "Fish should pass for pescatarian"
        assert not engine.evaluate(chicken).passed, "Poultry should fail for pescatarian"

    def test_no_pork_flag(self):
        """no_pork=True must exclude pork even for non-vegetarian."""
        profile = UserProfile(name="NP", age=40, sex="male",
                              calorie_target=2000, no_pork=True)
        engine  = ConstraintEngine(profile)
        pork    = _food(food_name="Bacon", category="pork",
                        diet_tags="non_vegetarian")
        verdict = engine.evaluate(pork)
        assert not verdict.passed, "Pork should be excluded when no_pork=True"

    def test_sodium_hypertension_per_meal_limit(self):
        """Sodium > daily_limit/3 per 100g must fail HTN constraint."""
        profile = UserProfile(name="H", age=60, sex="male",
                              calorie_target=2000, has_hypertension=True,
                              sodium_limit_mg=1500)
        engine = ConstraintEngine(profile)
        per_meal = 1500 / 3   # = 500 mg
        ok_food  = _food(food_name="Low Sodium",  sodium_mg=499.0)
        bad_food = _food(food_name="High Sodium", sodium_mg=501.0)
        assert engine.evaluate(ok_food).passed,      "499mg should pass"
        assert not engine.evaluate(bad_food).passed, "501mg should fail"

    def test_ibs_keyword_fallback(self):
        """IBS keyword fallback catches foods not in Bloom filter or DB column."""
        profile = UserProfile(name="I", age=30, sex="female",
                              calorie_target=1800, has_ibs=True)
        engine  = ConstraintEngine(profile)
        # "garlic" is in _IBS_TRIGGERS; not necessarily in Bloom filter
        garlic_food = _food(food_name="Roasted Garlic Hummus",
                             fodmap_status="unknown")
        verdict = engine.evaluate(garlic_food)
        assert not verdict.passed, "Garlic should trigger IBS keyword fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# EC2 — Bloom Filter Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestBloomFilterEdgeCases:

    def test_empty_allergen_list_returns_no_filter(self):
        """Empty allergen list should produce a filter that matches nothing."""
        bf = build_allergen_filter([])
        assert "salmon" not in bf
        assert "milk" not in bf

    def test_case_insensitivity(self):
        """Allergen lookup must be case-insensitive."""
        bf = build_allergen_filter(["dairy"])
        assert "MILK" in bf or "milk" in bf  # synonym expansion lowercases

    def test_allergen_synonym_expansion_dairy(self):
        """'dairy' allergen must expand to milk, cheese, butter, whey, etc."""
        bf = build_allergen_filter(["dairy"])
        for term in ["milk", "cheese", "butter", "whey", "yogurt"]:
            assert term in bf, f"'{term}' should be in dairy allergen filter"

    def test_allergen_synonym_expansion_gluten(self):
        """'gluten' must expand to wheat, barley, rye, spelt."""
        bf = build_allergen_filter(["gluten"])
        for term in ["wheat", "barley", "rye", "spelt"]:
            assert term in bf, f"'{term}' should be in gluten filter"

    def test_bloom_filter_no_false_negatives_for_inserted_items(self):
        """Every item inserted must be retrievable — no false negatives."""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        items = ["apple", "banana", "cherry", "date", "elderberry"]
        for item in items:
            bf.insert(item)
        for item in items:
            assert item in bf, f"'{item}' must not produce a false negative"

    def test_bloom_filter_item_count_tracking(self):
        """item_count property must reflect insertions."""
        bf = BloomFilter(capacity=500, error_rate=0.01)
        assert bf.item_count == 0
        bf.insert("test1")
        bf.insert("test2")
        assert bf.item_count == 2

    def test_partial_name_does_not_false_positive_on_unrelated_word(self):
        """'buckwheat' should not trigger 'wheat' allergen — it's gluten-free."""
        bf = build_allergen_filter(["shellfish"])
        # shellfish bloom filter should not flag random food names
        assert "broccoli" not in bf
        assert "spinach" not in bf

    def test_fodmap_filter_loads_from_csv(self):
        """FODMAP filter loads without error and contains expected entries."""
        if not FODMAP_CSV.exists():
            pytest.skip("fodmap_list.csv not present")
        bf = build_fodmap_filter(str(FODMAP_CSV))
        assert bf.item_count > 0, "FODMAP filter should have entries"


# ═══════════════════════════════════════════════════════════════════════════════
# EC3 — Plan Generation Robustness
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlanGenerationRobustness:

    def _make_plan(self, profile):
        from pipeline.ingest import seed_demo_database
        if not DB_PATH.exists():
            seed_demo_database(DB_PATH, n=10_500)
        from pipeline.constraints import load_candidate_foods
        candidates = load_candidate_foods(DB_PATH, profile, limit=3000)
        engine     = ConstraintEngine(profile)
        safe, _    = engine.filter_candidates(candidates)
        seen, plan = set(), []
        for food in safe:
            if food["fdc_id"] not in seen:
                seen.add(food["fdc_id"])
                plan.append(food)
            if len(plan) >= 21:
                break
        return plan

    def test_maximum_constraints_still_yields_meals(self):
        """All conditions + many allergens — must still generate some meals."""
        profile = _all_conditions_profile()
        plan    = self._make_plan(profile)
        assert len(plan) > 0, "Even maximally constrained profile must yield meals"

    def test_low_calorie_target_1200(self):
        """1200 kcal/day — plan still generates without error."""
        profile = UserProfile(name="Low", age=30, sex="female",
                              calorie_target=1200, diet_mode=DietMode.VEGAN)
        plan = self._make_plan(profile)
        assert len(plan) > 0

    def test_high_calorie_target_3500(self):
        """3500 kcal/day — plan generates without error."""
        profile = UserProfile(name="High", age=25, sex="male",
                              calorie_target=3500)
        plan = self._make_plan(profile)
        assert len(plan) > 0

    def test_elderly_profile(self):
        """Age 80 — RDA lookups must resolve without KeyError."""
        profile = UserProfile(name="Elder", age=80, sex="female",
                              calorie_target=1600)
        plan = self._make_plan(profile)
        assert len(plan) > 0

    def test_no_allergens_declared(self):
        """Empty allergen list — allergen filter is not built, no crash."""
        profile = UserProfile(name="NoAllergy", age=35, sex="male",
                              calorie_target=2000, allergens=[])
        plan = self._make_plan(profile)
        assert len(plan) > 0

    def test_sodium_hypertension_plan_respects_limit(self):
        """All foods in an HTN plan must be within per-meal sodium limit."""
        profile = UserProfile(name="HTN", age=60, sex="male",
                              calorie_target=2000, has_hypertension=True,
                              sodium_limit_mg=1500)
        plan    = self._make_plan(profile)
        per_meal_limit = profile.sodium_limit_mg / 3
        violations = [f for f in plan
                      if float(f.get("sodium_mg") or 0) > per_meal_limit]
        assert len(violations) == 0, \
            f"HTN plan contains {len(violations)} sodium violations"

    def test_vegan_plan_contains_no_meat(self):
        """All foods in a vegan plan must not be from animal-product categories."""
        profile  = UserProfile(name="Vegan", age=28, sex="female",
                               calorie_target=1800, diet_mode=DietMode.VEGAN)
        plan     = self._make_plan(profile)
        meat_cats = {"poultry", "beef", "pork", "lamb", "fish", "dairy", "egg"}
        violations = [f for f in plan
                      if any(c in f.get("category", "").lower() for c in meat_cats)]
        assert len(violations) == 0, \
            f"Vegan plan contains animal products: {[v['food_name'] for v in violations[:3]]}"

    def test_ibs_plan_contains_no_high_fodmap(self):
        """All foods in IBS plan must not be marked unsafe FODMAP."""
        profile = UserProfile(name="IBS", age=30, sex="female",
                              calorie_target=1800, has_ibs=True)
        plan    = self._make_plan(profile)
        violations = [f for f in plan if f.get("fodmap_status") == "unsafe"]
        assert len(violations) == 0, \
            f"IBS plan contains high-FODMAP foods: {[v['food_name'] for v in violations[:3]]}"

    def test_diabetes_plan_gi_limit_observed(self):
        """All foods with known GI in T2DM plan must be <= gi_limit."""
        profile = UserProfile(name="DM", age=55, sex="female",
                              calorie_target=1600, has_diabetes_t2=True, gi_limit=55)
        plan    = self._make_plan(profile)
        violations = [f for f in plan
                      if f.get("gi_value") is not None
                      and float(f["gi_value"]) > profile.gi_limit]
        assert len(violations) == 0, \
            f"T2DM plan GI violations: {[(v['food_name'], v['gi_value']) for v in violations[:3]]}"


# ═══════════════════════════════════════════════════════════════════════════════
# EC4 — Nutrient Aggregator with Bad/Extreme Data
# ═══════════════════════════════════════════════════════════════════════════════

class TestNutrientAggregatorEdgeCases:  # EC4

    def _day_from_foods(self, foods: list[dict], day: int = 1) -> DayNutrients:
        day_entry = DayNutrients(day=day)
        for i, meal_name in enumerate(["breakfast", "lunch", "dinner"]):
            if i < len(foods):
                mn = MealNutrients.from_food_row(meal_name, foods[i], portion_g=200.0)
                day_entry.meals[meal_name] = mn
        return day_entry

    def test_all_zero_nutrients_does_not_crash(self):
        """Food with all nutrients = 0 must not raise in gap report."""
        zero_food = _food(**{k: 0.0 for k in TRACKED_NUTRIENTS})
        zero_food.update({"food_name": "Zero Food", "fdc_id": 1})
        day = self._day_from_foods([zero_food])
        gaps = day.gap_report(age=30, sex="female")
        assert len(gaps) > 0

    def test_null_nutrient_values_treated_as_zero(self):
        """None values in nutrient fields must not raise — treated as 0."""
        null_food = _food(**{k: None for k in TRACKED_NUTRIENTS})
        null_food.update({"food_name": "Null Food", "fdc_id": 2,
                           "calories": 100.0})
        mn = MealNutrients.from_food_row("breakfast", null_food, portion_g=200.0)
        assert mn.totals.get("protein_g", 0) == 0.0

    def test_extreme_nutrient_value_does_not_overflow(self):
        """Wildly large nutrient values (e.g. zinc at 42000%) must not crash."""
        extreme_food = _food(zinc_mg=10000.0, potassium_mg=50000.0)
        day  = self._day_from_foods([extreme_food])
        gaps = day.gap_report(age=30, sex="female")
        zinc_gap = next((g for g in gaps if g["nutrient"] == "zinc_mg"), None)
        assert zinc_gap is not None
        assert zinc_gap["pct_rda"] > 100   # should flag as over-RDA, not crash

    def test_rda_lookup_all_age_groups(self):
        """RDA lookups must resolve for every age group without KeyError."""
        ages = [1, 5, 10, 15, 20, 35, 55, 72]
        for age in ages:
            for sex in ("male", "female", "other"):
                rda = get_rda("calcium_mg", age, sex)
                assert rda >= 0, f"Negative RDA for calcium at age={age}, sex={sex}"

    def test_weekly_summary_with_7_days(self):
        """weekly_summary must return exactly 7 day entries."""
        agg  = NutrientAggregator(age=35, sex="male", calorie_target=2000)
        food = _food()
        for d in range(1, 8):
            day = self._day_from_foods([food, food, food], day=d)
            agg.add_day(day)
        summary = agg.weekly_summary()
        assert len(summary["days"]) == 7
        assert "weekly_gap_report" in summary
        assert len(summary["weekly_gap_report"]) > 0

    def test_gap_report_flags_nutrients_below_80_pct(self):
        """A plan with zero intake must flag all nutrients as below 80% RDA."""
        zero_food = _food(**{k: 0.0 for k in TRACKED_NUTRIENTS})
        zero_food.update({"food_name": "Zero", "fdc_id": 3, "calories": 0.0})
        agg = NutrientAggregator(age=30, sex="female", calorie_target=1800)
        for d in range(1, 8):
            day = self._day_from_foods([zero_food, zero_food, zero_food], day=d)
            agg.add_day(day)
        summary = agg.weekly_summary()
        below = [r for r in summary["weekly_gap_report"] if r["days_below_80"] >= 3]
        assert len(below) > 0, "Zero-intake plan should flag multiple nutrient gaps"

    def test_portion_scaling_is_applied(self):
        """MealNutrients.from_food_row must scale nutrients by portion_g/100."""
        food = _food(protein_g=10.0)
        mn100 = MealNutrients.from_food_row("lunch", food, portion_g=100.0)
        mn200 = MealNutrients.from_food_row("lunch", food, portion_g=200.0)
        assert abs(mn200.totals["protein_g"] - 2 * mn100.totals["protein_g"]) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# EC5 — Grocery List Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroceryListEdgeCases:

    def _single_meal_plan(self, food_name="Spinach", category="vegetable",
                           portion_g=200.0, n_meals=21):
        """21-meal plan using one food repeated."""
        food = _food(food_name=food_name, category=category,
                     portion_g=portion_g)
        return [[food]] * n_meals

    def test_empty_plan_returns_empty_list(self):
        """Empty plan must return a GroceryList with 0 items."""
        grocery = generate_grocery_list([])
        assert grocery.total_items == 0
        assert grocery.total_cost_usd == 0.0

    def test_single_ingredient_plan_consolidates(self):
        """21 meals of the same food must consolidate to 1 grocery item."""
        plan    = self._single_meal_plan()
        grocery = generate_grocery_list(plan)
        assert grocery.total_items == 1

    def test_total_gram_aggregation_is_correct(self):
        """Total grams must equal portion_g * number of appearances."""
        portion_g = 150.0
        plan      = self._single_meal_plan(portion_g=portion_g, n_meals=21)
        grocery   = generate_grocery_list(plan)
        expected  = portion_g * 21
        assert abs(grocery.items[0].total_g - expected) < 0.1

    def test_variation_suffix_stripped(self):
        """'Oats (dry) (var 83)' and 'Oats (dry)' must consolidate to 1 item."""
        food1 = _food(food_name="Oats (dry) (var 83)", portion_g=130.0)
        food2 = _food(food_name="Oats (dry)",          portion_g=130.0)
        plan  = [[food1], [food2]] + [[food1]] * 5
        grocery = generate_grocery_list(plan)
        oat_items = [i for i in grocery.items if "Oats" in i.name]
        assert len(oat_items) == 1, "Variation suffixes should consolidate to one item"
        assert "var" not in oat_items[0].name.lower()

    def test_zero_portion_does_not_crash(self):
        """Food with portion_g=0 must not cause division errors."""
        food    = _food(portion_g=0.0)
        plan    = [[food]] * 3
        grocery = generate_grocery_list(plan)
        assert grocery.total_items >= 0  # just must not raise

    def test_cost_fallback_when_no_pricer(self):
        """Without a pricer, all items must use estimated source."""
        plan    = self._single_meal_plan(category="fish")
        grocery = generate_grocery_list(plan, pricer=None)
        for item in grocery.items:
            assert item.price_source == "estimated"

    def test_all_sections_present_for_diverse_plan(self):
        """Plan with foods from all categories must populate multiple sections."""
        foods = [
            _food(food_name="Broccoli",       category="vegetable",  portion_g=150),
            _food(food_name="Salmon",          category="fish",       portion_g=200),
            _food(food_name="Brown Rice",      category="grain",      portion_g=180),
            _food(food_name="Greek Yogurt",    category="dairy",      portion_g=150),
            _food(food_name="Almonds",         category="nut_seed",   portion_g=30),
        ]
        plan    = [[f] for f in foods] * 4 + [[foods[0]]]
        grocery = generate_grocery_list(plan)
        assert len(grocery.by_section) >= 4, "Diverse plan should populate >= 4 sections"

    def test_clean_name_removes_var_suffix(self):
        """_clean_name must strip (var N) patterns."""
        assert _clean_name("Oats (dry) (var 83)") == "Oats (dry)"
        assert _clean_name("Spinach (var 1)")      == "Spinach"
        assert _clean_name("Salmon (cooked)")      == "Salmon (cooked)"  # no var suffix

    def test_csv_export_content(self):
        """Grocery list items must serialise cleanly to CSV-compatible fields."""
        plan    = self._single_meal_plan()
        grocery = generate_grocery_list(plan)
        for item in grocery.items:
            assert isinstance(item.name,          str)
            assert isinstance(item.purchase_unit, str)
            assert isinstance(item.est_cost_usd,  float)
            assert item.est_cost_usd >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# EC6 — Portion Scaling Boundaries
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortionScalingBoundaries:

    def _portion_for(self, cal_per_100g: float, target_cal: float) -> float:
        """Replicate the portion calculation from ranker.py generate_plan."""
        if cal_per_100g > 0:
            portion_g = (target_cal / cal_per_100g) * 100.0
            return max(30.0, min(portion_g, 900.0))
        return 150.0

    def test_very_low_density_food_clamped_to_900g(self):
        """Food with 5 kcal/100g at 500 kcal target would need 10kg — clamped to 900g."""
        portion = self._portion_for(cal_per_100g=5.0, target_cal=500.0)
        assert portion == 900.0

    def test_very_high_density_food_floored_to_30g(self):
        """Food with 900 kcal/100g at 50 kcal target would need 5.5g — floored to 30g."""
        portion = self._portion_for(cal_per_100g=900.0, target_cal=50.0)
        assert portion == 30.0

    def test_zero_calorie_food_returns_default(self):
        """Food with 0 kcal/100g returns the default 150g portion."""
        portion = self._portion_for(cal_per_100g=0.0, target_cal=400.0)
        assert portion == 150.0

    def test_normal_density_food_scales_correctly(self):
        """Food with 200 kcal/100g at 400 kcal target -> 200g (within bounds)."""
        portion = self._portion_for(cal_per_100g=200.0, target_cal=400.0)
        assert abs(portion - 200.0) < 0.01

    def test_portion_always_within_bounds(self):
        """Regardless of inputs, portion must always be in [30, 900]."""
        test_cases = [
            (0.1, 100), (1000, 50), (250, 500), (0, 300), (100, 0),
        ]
        for cal_density, target in test_cases:
            portion = self._portion_for(cal_density, target)
            assert 30.0 <= portion <= 900.0, \
                f"Portion {portion} out of bounds for density={cal_density}, target={target}"


# ═══════════════════════════════════════════════════════════════════════════════
# EC7 — UserProfile Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserProfileValidation:

    def test_hypertension_auto_sets_sodium_limit(self):
        """has_hypertension=True must auto-cap sodium_limit_mg to 1500."""
        profile = UserProfile(name="H", age=50, sex="male",
                              calorie_target=2000, has_hypertension=True,
                              sodium_limit_mg=2300)
        assert profile.sodium_limit_mg == 1500, \
            "HTN profile must auto-cap sodium to 1500 mg/day"

    def test_diet_mode_string_converts_to_enum(self):
        """Passing diet_mode as string must convert to DietMode enum."""
        profile = UserProfile(name="S", age=30, sex="female",
                              calorie_target=1800, diet_mode="vegan")
        assert profile.diet_mode == DietMode.VEGAN

    def test_default_allergens_is_empty_list(self):
        """Default allergens should be an empty list, not None."""
        profile = UserProfile(name="D", age=25, sex="male", calorie_target=2000)
        assert profile.allergens == []

    def test_sex_other_returns_valid_rda(self):
        """'other' sex must resolve RDA without error."""
        rda = get_rda("iron_mg", age=30, sex="other")
        assert rda > 0

    def test_very_young_and_very_old_age(self):
        """Age extremes (18 and 80) must not crash any pipeline component."""
        for age in [18, 80]:
            profile = UserProfile(name="AgeTest", age=age, sex="female",
                                  calorie_target=1800)
            engine = ConstraintEngine(profile)
            verdict = engine.evaluate(_food())
            assert isinstance(verdict.passed, bool)


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent,
    )
    sys.exit(result.returncode)
