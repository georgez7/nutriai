"""
test_bloom.py
-------------
Unit tests for the BloomFilter class.

Critical guarantee: ZERO false negatives.
  → Every item that was inserted MUST test True on __contains__.
  → This is tested exhaustively with 10 000 items.

Also tests:
  - False positive rate is within the configured error_rate bound
  - O(1) performance (timing sanity check)
  - build_allergen_filter / build_fodmap_filter factory functions
"""

import time
import math
import random
import string
import sys
import os

# Allow running from project root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.bloom_filter import (
    BloomFilter,
    build_allergen_filter,
    build_fodmap_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_string(length: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + " ", k=length))


def run_test(name: str, fn):
    try:
        fn()
        print(f"  ✅  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  ❌  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  💥  ERROR {name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Core Bloom filter tests
# ---------------------------------------------------------------------------

def test_zero_false_negatives():
    """
    CRITICAL: Every inserted item must test True.
    Tests 10 000 items across multiple capacities and error rates.
    """
    for capacity, error_rate in [(100, 0.01), (1000, 0.05), (10_000, 0.001)]:
        bf = BloomFilter(capacity=capacity, error_rate=error_rate)
        inserted = [random_string() for _ in range(capacity)]

        for item in inserted:
            bf.insert(item)

        for item in inserted:
            assert item in bf, (
                f"FALSE NEGATIVE DETECTED — item '{item}' was inserted but "
                f"not found (capacity={capacity}, error_rate={error_rate}). "
                f"This violates the zero-false-negative guarantee!"
            )


def test_false_positive_rate_bounded():
    """
    FP rate must not exceed 3× the configured error_rate
    (statistical tolerance for small sample sizes).
    """
    capacity   = 5_000
    error_rate = 0.01
    bf = BloomFilter(capacity=capacity, error_rate=error_rate)

    # Insert `capacity` random items
    inserted_set = set()
    while len(inserted_set) < capacity:
        inserted_set.add(random_string(16))
    for item in inserted_set:
        bf.insert(item)

    # Test 10 000 items that were NOT inserted
    n_test = 10_000
    n_fp   = 0
    for _ in range(n_test):
        item = random_string(20)   # very different from inserted (20 chars)
        if item not in inserted_set and item in bf:
            n_fp += 1

    measured_fp = n_fp / n_test
    tolerance   = error_rate * 3   # 3× slack
    assert measured_fp <= tolerance, (
        f"False positive rate {measured_fp:.4f} exceeds tolerance "
        f"{tolerance:.4f} (error_rate={error_rate})"
    )


def test_bulk_insert():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    items = [f"food_{i}" for i in range(500)]
    count = bf.bulk_insert(items)
    assert count == 500, f"Expected 500 inserts, got {count}"
    for item in items:
        assert item in bf


def test_is_safe():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.insert("garlic")
    bf.insert("onion")
    assert not bf.is_safe("garlic"),  "garlic should be unsafe (inserted)"
    assert not bf.is_safe("onion"),   "onion should be unsafe (inserted)"
    assert bf.is_safe("spinach"),     "spinach not inserted — should be safe"


def test_item_count():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    assert bf.item_count == 0
    bf.insert("a")
    bf.insert("b")
    bf.insert("c")
    assert bf.item_count == 3


def test_estimated_fp_rate():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    assert bf.estimated_false_positive_rate == 0.0
    bf.bulk_insert([f"item_{i}" for i in range(500)])
    fp = bf.estimated_false_positive_rate
    assert 0 <= fp < 0.05, f"Estimated FP rate {fp} out of expected range"


def test_case_insensitivity():
    """BloomFilter must normalise to lowercase for allergen safety."""
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.insert("Garlic")
    assert "garlic"  in bf
    assert "GARLIC"  in bf
    assert "Garlic"  in bf
    assert "GaRLiC"  in bf


def test_performance_o1():
    """
    Insert and lookup 100 000 items — must complete in < 10 seconds total.
    Each individual lookup is effectively O(1) regardless of set size.
    """
    n = 100_000
    bf = BloomFilter(capacity=n, error_rate=0.01)
    items = [f"food_item_{i:07d}" for i in range(n)]

    t0 = time.perf_counter()
    for item in items:
        bf.insert(item)
    insert_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for item in items:
        _ = item in bf
    lookup_time = time.perf_counter() - t0

    assert insert_time < 10.0, f"Insertion of {n} items took {insert_time:.1f}s (too slow)"
    assert lookup_time < 5.0,  f"Lookup of  {n} items took {lookup_time:.1f}s (too slow)"
    print(f"         [perf] insert={insert_time:.2f}s, lookup={lookup_time:.2f}s for {n} items")


def test_optimal_sizing():
    """Bit size and hash count should follow the optimal formulas."""
    bf = BloomFilter(capacity=10_000, error_rate=0.01)
    # Theoretical: m = -(n*ln p)/(ln 2)^2 ≈ 95850; k = (m/n)*ln2 ≈ 7
    assert 90_000 < bf.bit_size < 120_000, f"Unexpected bit_size: {bf.bit_size}"
    assert 5 <= bf.hash_count <= 10,        f"Unexpected hash_count: {bf.hash_count}"


def test_edge_cases():
    """Empty string, very long string, whitespace."""
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.insert("")
    bf.insert("  ")
    bf.insert("a" * 1000)

    assert "" in bf
    assert "  " in bf
    assert "a" * 1000 in bf


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

def test_allergen_filter_gluten_synonyms():
    """Allergen filter for 'gluten' must catch all wheat/barley/rye synonyms."""
    bf = build_allergen_filter(["gluten"])
    must_catch = [
        "wheat", "barley", "rye", "spelt", "triticale", "kamut",
        "semolina", "farro", "durum", "bulgur", "couscous", "seitan",
    ]
    for food in must_catch:
        assert food in bf, f"Allergen filter MISSED '{food}' (gluten synonym)"


def test_allergen_filter_dairy():
    bf = build_allergen_filter(["dairy"])
    must_catch = ["milk", "cheese", "butter", "whey", "casein", "lactose", "ghee", "yogurt"]
    for food in must_catch:
        assert food in bf, f"Allergen filter MISSED '{food}' (dairy synonym)"


def test_allergen_filter_tree_nuts():
    bf = build_allergen_filter(["tree nuts"])
    must_catch = ["almonds", "cashews", "walnuts", "pistachios", "pecans",
                  "hazelnuts", "macadamia", "pine nuts"]
    for food in must_catch:
        assert food in bf, f"Allergen filter MISSED '{food}' (tree nut)"


def test_allergen_filter_soy():
    bf = build_allergen_filter(["soy"])
    must_catch = ["soy", "tofu", "edamame", "miso", "tempeh", "tamari",
                  "soy sauce", "soy milk"]
    for food in must_catch:
        assert food in bf, f"Allergen filter MISSED '{food}' (soy)"


def test_allergen_filter_zero_false_negatives_for_declared():
    """
    The absolute critical test: declared allergens must NEVER slip through.
    """
    allergens = ["gluten", "dairy", "tree nuts", "shellfish", "soy", "eggs"]
    bf = build_allergen_filter(allergens)

    # The declared allergens themselves must always be caught
    for allergen in allergens:
        assert allergen.lower() in bf, \
            f"FALSE NEGATIVE: declared allergen '{allergen}' passed through filter!"


def test_fodmap_filter_fallback():
    """FODMAP filter should load from hardcoded defaults if CSV missing."""
    bf = build_fodmap_filter("/nonexistent/path/fodmap.csv")
    high_fodmap = ["garlic", "onion", "apple", "milk", "cauliflower", "asparagus"]
    for food in high_fodmap:
        assert food in bf, f"FODMAP filter MISSED high-FODMAP food: '{food}'"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  NutriAI — Bloom Filter Test Suite")
    print("=" * 60)

    tests = [
        ("Zero false negatives (10K items, 3 configs)",         test_zero_false_negatives),
        ("False positive rate bounded by error_rate",           test_false_positive_rate_bounded),
        ("Bulk insert",                                         test_bulk_insert),
        ("is_safe() helper",                                    test_is_safe),
        ("Item count tracking",                                 test_item_count),
        ("Estimated FP rate",                                   test_estimated_fp_rate),
        ("Case-insensitive normalisation",                      test_case_insensitivity),
        ("Performance O(1) — 100K inserts + lookups",          test_performance_o1),
        ("Optimal bit_size and hash_count",                     test_optimal_sizing),
        ("Edge cases (empty, whitespace, very long)",           test_edge_cases),
        ("Allergen filter — gluten synonyms",                   test_allergen_filter_gluten_synonyms),
        ("Allergen filter — dairy synonyms",                    test_allergen_filter_dairy),
        ("Allergen filter — tree nuts",                         test_allergen_filter_tree_nuts),
        ("Allergen filter — soy",                               test_allergen_filter_soy),
        ("Allergen filter — zero false negatives (declared)",   test_allergen_filter_zero_false_negatives_for_declared),
        ("FODMAP filter — fallback to hardcoded list",          test_fodmap_filter_fallback),
    ]

    results = []
    for name, fn in tests:
        results.append(run_test(name, fn))

    passed = sum(results)
    total  = len(results)
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  🎉  All Bloom filter tests passed!")
        print("  ✅  Zero false-negative guarantee confirmed.")
    else:
        failed = [tests[i][0] for i, r in enumerate(results) if not r]
        print(f"  ❌  Failed: {failed}")
    print("=" * 60 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
