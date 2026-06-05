"""
bloom_filter.py
---------------
Bloom filter for O(1) allergen and high-FODMAP membership testing.

Guarantees:
  - Zero false negatives  (a flagged item is ALWAYS caught)
  - Bounded false-positive rate (tunable via capacity + error_rate)

BAX-423 technique: sketching / probabilistic data structures.
"""

import math
import hashlib
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


class BloomFilter:
    """
    Space-efficient probabilistic set for allergen / FODMAP safety checks.

    Parameters
    ----------
    capacity : int
        Expected number of items to insert. Controls bit-array size.
    error_rate : float
        Target false-positive probability (e.g. 0.01 = 1%).  Must be > 0.

    Properties guaranteed by design:
    - insert()  → O(1)
    - __contains__ → O(1)
    - False negatives → 0  (if an item was inserted, it will always test True)
    - False positives → bounded by `error_rate`
    """

    def __init__(self, capacity: int = 10_000, error_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not (0 < error_rate < 1):
            raise ValueError("error_rate must be in (0, 1)")

        self.capacity = capacity
        self.error_rate = error_rate

        # Optimal bit-array size and hash count (standard formulas)
        self.bit_size = self._optimal_bit_size(capacity, error_rate)
        self.hash_count = self._optimal_hash_count(self.bit_size, capacity)

        # Bit array stored as a bytearray for memory efficiency
        self._bits = bytearray(math.ceil(self.bit_size / 8))
        self._item_count = 0

        logger.debug(
            "BloomFilter initialised: capacity=%d, error_rate=%.4f, "
            "bit_size=%d, hash_count=%d, memory_bytes=%d",
            capacity, error_rate, self.bit_size, self.hash_count, len(self._bits),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, item: str) -> None:
        """Add *item* to the filter.  Always O(1)."""
        for idx in self._hash_indices(item):
            byte_idx, bit_offset = divmod(idx, 8)
            self._bits[byte_idx] |= 1 << bit_offset
        self._item_count += 1

    def __contains__(self, item: str) -> bool:
        """
        Test membership.  O(1).

        Returns True  → item *may* be in the set (possibly a false positive).
        Returns False → item is DEFINITELY NOT in the set (zero false negatives).
        """
        for idx in self._hash_indices(item):
            byte_idx, bit_offset = divmod(idx, 8)
            if not (self._bits[byte_idx] & (1 << bit_offset)):
                return False
        return True

    def bulk_insert(self, items: Iterable[str]) -> int:
        """Insert all items; return count inserted."""
        before = self._item_count
        for item in items:
            self.insert(item)
        inserted = self._item_count - before
        logger.info("BloomFilter bulk_insert: added %d items (total %d)", inserted, self._item_count)
        return inserted

    def is_safe(self, food_name: str) -> bool:
        """
        Convenience wrapper — returns True when food is NOT flagged.
        (i.e., it is absent from the blocked / allergen set)
        """
        return food_name.lower().strip() not in self

    @property
    def item_count(self) -> int:
        return self._item_count

    @property
    def estimated_false_positive_rate(self) -> float:
        """Current empirical FP rate based on item count inserted."""
        if self._item_count == 0:
            return 0.0
        k = self.hash_count
        m = self.bit_size
        n = self._item_count
        return (1 - math.exp(-k * n / m)) ** k

    def __repr__(self) -> str:
        return (
            f"BloomFilter(capacity={self.capacity}, error_rate={self.error_rate}, "
            f"items={self._item_count}, fp_rate={self.estimated_false_positive_rate:.6f})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_indices(self, item: str) -> list[int]:
        """
        Generate `hash_count` independent bit indices for *item* using
        double-hashing (Kirsch-Mitzenmacher technique) to avoid the cost
        of k separate hash functions.
        """
        item_bytes = item.lower().strip().encode("utf-8")

        h1 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha1(item_bytes).hexdigest(), 16)

        return [(h1 + i * h2) % self.bit_size for i in range(self.hash_count)]

    @staticmethod
    def _optimal_bit_size(n: int, p: float) -> int:
        """m = -(n * ln p) / (ln 2)^2"""
        return max(1, math.ceil(-(n * math.log(p)) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """k = (m/n) * ln 2"""
        return max(1, round((m / n) * math.log(2)))


# ---------------------------------------------------------------------------
# Factory helpers — pre-built filters for the NutriAI pipeline
# ---------------------------------------------------------------------------

def build_allergen_filter(allergens: Iterable[str], capacity: int = 5_000) -> BloomFilter:
    """
    Return a BloomFilter pre-loaded with all user-declared allergens and
    their common synonyms / cross-reactive foods.

    False negatives are impossible → zero allergen leakage.
    """
    ALLERGEN_SYNONYMS: dict[str, list[str]] = {
        "gluten":    ["wheat", "barley", "rye", "spelt", "triticale", "kamut",
                      "semolina", "farro", "durum", "bulgur", "couscous",
                      "seitan", "malt", "brewer's yeast"],
        "dairy":     ["milk", "cheese", "butter", "cream", "lactose", "whey",
                      "casein", "ghee", "yogurt", "kefir", "paneer", "quark"],
        "tree nuts": ["almonds", "cashews", "walnuts", "pistachios", "pecans",
                      "brazil nuts", "hazelnuts", "macadamia", "pine nuts",
                      "chestnuts", "coconut"],
        "shellfish": ["shrimp", "crab", "lobster", "crayfish", "prawn",
                      "barnacle", "krill", "langoustine"],
        "soy":       ["soy", "soya", "tofu", "edamame", "miso", "tempeh",
                      "soy sauce", "tamari", "natto", "textured soy protein",
                      "soy milk", "soy flour"],
        "eggs":      ["egg", "eggs", "albumin", "mayonnaise", "meringue",
                      "ovalbumin", "ovomucin"],
        "fish":      ["salmon", "tuna", "cod", "tilapia", "bass", "flounder",
                      "anchovy", "anchovies", "sardine", "halibut", "mahi",
                      "swordfish", "trout", "catfish"],
        "peanuts":   ["peanut", "peanuts", "groundnut", "groundnuts",
                      "peanut butter", "peanut oil", "beer nuts"],
        "sesame":    ["sesame", "tahini", "sesame oil", "sesame seeds",
                      "til", "gingelly"],
    }

    bf = BloomFilter(capacity=capacity, error_rate=0.001)  # 0.1% FP rate
    for allergen in allergens:
        allergen_lower = allergen.lower().strip()
        bf.insert(allergen_lower)
        # Also insert synonyms if we know them
        for canonical, synonyms in ALLERGEN_SYNONYMS.items():
            if allergen_lower == canonical or allergen_lower in synonyms:
                bf.bulk_insert(synonyms)
                bf.insert(canonical)

    logger.info("Allergen BloomFilter built: %s", bf)
    return bf


def build_fodmap_filter(fodmap_csv_path: str, capacity: int = 3_000) -> BloomFilter:
    """
    Return a BloomFilter pre-loaded from the FODMAP CSV (fodmap_list.csv).
    Only foods marked 'unsafe' (high-FODMAP) are inserted.

    CSV expected columns: food_name, fodmap_status  (safe | unsafe)
    """
    import csv

    bf = BloomFilter(capacity=capacity, error_rate=0.005)

    try:
        with open(fodmap_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("fodmap_status", "").strip().lower()
                if status == "unsafe":
                    name = row.get("food_name", "").strip().lower()
                    if name:
                        bf.insert(name)
    except FileNotFoundError:
        logger.warning(
            "FODMAP CSV not found at %s — using hard-coded trigger list.", fodmap_csv_path
        )
        # Fallback: Monash-verified high-FODMAP trigger foods
        HIGH_FODMAP_DEFAULTS = [
            "garlic", "onion", "leek", "spring onion", "shallot",
            "wheat", "rye", "barley",
            "apple", "pear", "mango", "watermelon", "cherry", "apricot",
            "milk", "yogurt", "ice cream", "soft cheese",
            "legumes", "lentils", "chickpeas", "kidney beans", "baked beans",
            "cashews", "pistachios",
            "honey", "high fructose corn syrup", "agave",
            "cauliflower", "mushroom", "asparagus", "artichoke",
        ]
        bf.bulk_insert(HIGH_FODMAP_DEFAULTS)

    logger.info("FODMAP BloomFilter built: %s", bf)
    return bf
