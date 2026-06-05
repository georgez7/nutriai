"""
conftest.py — pytest fixtures for NutriAI persona tests.
"""
import sys, os
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.constraints import UserProfile, DietMode
from pipeline.ingest import seed_demo_database

DB_PATH = Path(__file__).parent.parent.parent / "data" / "foods.db"

# ── personas ────────────────────────────────────────────────────────────────

PRIYA = UserProfile(
    name="Priya", age=28, sex="female", calorie_target=1800,
    diet_mode=DietMode.VEGETARIAN, has_ibs=True, allergens=["dairy"],
    micro_priorities=["iron_mg", "calcium_mg", "vitamin_d_mcg"], sodium_limit_mg=2300,
)
RAVI = UserProfile(
    name="Ravi", age=42, sex="male", calorie_target=2200,
    diet_mode=DietMode.NON_VEGETARIAN, has_gerd=True, allergens=["gluten"],
    no_pork=True, micro_priorities=["vitamin_b12_mcg", "zinc_mg", "magnesium_mg"],
    sodium_limit_mg=2300,
)
MEI = UserProfile(
    name="Mei", age=55, sex="female", calorie_target=1600,
    diet_mode=DietMode.VEGAN, has_diabetes_t2=True, allergens=["tree nuts"],
    gi_limit=55, micro_priorities=["vitamin_b12_mcg", "iron_mg", "zinc_mg"],
    sodium_limit_mg=2300,
)
JAMES = UserProfile(
    name="James", age=61, sex="male", calorie_target=2000,
    diet_mode=DietMode.PESCATARIAN, has_hypertension=True, allergens=["soy"],
    sodium_limit_mg=1500, micro_priorities=["potassium_mg", "magnesium_mg"],
)

# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def ensure_db():
    """Seed the demo database once per test session if not present."""
    if not DB_PATH.exists():
        seed_demo_database(str(DB_PATH), n=10_500)

@pytest.fixture(scope="session")
def priya(): return PRIYA

@pytest.fixture(scope="session")
def ravi(): return RAVI

@pytest.fixture(scope="session")
def mei(): return MEI

@pytest.fixture(scope="session")
def james(): return JAMES

@pytest.fixture(scope="session")
def priya_plan(ensure_db, priya):
    from tests.test_personas import _make_synthetic_plan
    return _make_synthetic_plan(priya)

@pytest.fixture(scope="session")
def ravi_plan(ensure_db, ravi):
    from tests.test_personas import _make_synthetic_plan
    return _make_synthetic_plan(ravi)

@pytest.fixture(scope="session")
def mei_plan(ensure_db, mei):
    from tests.test_personas import _make_synthetic_plan
    return _make_synthetic_plan(mei)

@pytest.fixture(scope="session")
def james_plan(ensure_db, james):
    from tests.test_personas import _make_synthetic_plan
    return _make_synthetic_plan(james)
