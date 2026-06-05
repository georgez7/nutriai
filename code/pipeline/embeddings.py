"""
embeddings.py
-------------
Embeds food records into a nutritional + clinical vector space and builds
a FAISS HNSW index for sub-5ms approximate nearest-neighbour search.

BAX-423 technique: embeddings + approximate nearest-neighbour search (ANN).

Design
------
- Feature vector: 22 normalised dimensions (macros, micros, GI, category OHE)
- Index type: HNSW (Hierarchical Navigable Small World) — O(log n) search
- Index is serialised to disk so it doesn't need to be rebuilt every run

Usage
-----
embedder = FoodEmbedder(db_path="data/foods.db")
embedder.build()                       # one-time build (~2s for 10K records)
results = embedder.search(query_vec, k=50)  # returns list of fdc_ids
"""

import sqlite3
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH    = Path(__file__).parent.parent.parent / "data" / "foods.db"
INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "faiss_index.pkl"

# Nutrient columns and their per-100g normalisation caps
# (values above cap are clipped, then divided by cap → [0, 1])
NUTRIENT_COLS: list[tuple[str, float]] = [
    ("calories",         900.0),
    ("protein_g",         50.0),
    ("carbs_g",           80.0),
    ("fat_g",             60.0),
    ("fiber_g",           30.0),
    ("saturated_fat_g",   30.0),
    ("sodium_mg",       2400.0),
    ("calcium_mg",       1200.0),
    ("iron_mg",            20.0),
    ("vitamin_c_mg",      200.0),
    ("vitamin_d_mcg",      25.0),
    ("vitamin_b12_mcg",    10.0),
    ("zinc_mg",            15.0),
    ("potassium_mg",     4700.0),
    ("magnesium_mg",      420.0),
    ("phosphorus_mg",    1250.0),
    ("omega3_g",            3.5),  # cap at 3.5g/100g (fatty fish range)
]

# GI normalised to [0,1] on a 0-100 scale
GI_CAP = 100.0

# Food categories for one-hot encoding (5 dims)
CATEGORY_OHE = [
    "vegetable", "fruit", "grain", "legume", "dairy",
    "poultry",   "fish",  "beef",  "nut_seed", "egg",
]

# Total feature dims: len(NUTRIENT_COLS) + 1 (GI) + len(CATEGORY_OHE) = 27
FEATURE_DIM = len(NUTRIENT_COLS) + 1 + len(CATEGORY_OHE)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_vector(row: dict) -> np.ndarray:
    """
    Convert a food record dict into a normalised float32 vector.
    Missing values → 0 (conservative: treat unknown as absent).
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Nutrient dims
    for i, (col, cap) in enumerate(NUTRIENT_COLS):
        val = float(row.get(col) or 0.0)
        vec[i] = min(val, cap) / cap

    # GI dim
    gi = row.get("gi_value")
    vec[len(NUTRIENT_COLS)] = (min(float(gi), GI_CAP) / GI_CAP) if gi else 0.5  # mid if unknown

    # Category OHE
    cat = (row.get("category") or "").lower()
    for j, cat_label in enumerate(CATEGORY_OHE):
        if cat_label in cat:
            vec[len(NUTRIENT_COLS) + 1 + j] = 1.0
            break  # one-hot

    return vec


def build_user_query_vector(
    calorie_target: float,
    protein_target_g: float,
    carb_target_g: float,
    fat_target_g: float,
    fiber_target_g: float,
    sodium_limit_mg: float = 800.0,
    category_preference: Optional[str] = None,
    gi_preference: Optional[float] = None,
    potassium_target_mg: float = 1200.0,
    magnesium_target_mg: float = 120.0,
) -> np.ndarray:
    """
    Build a query vector for a user's per-MEAL nutrient targets so FAISS
    can retrieve the most compatible food candidates.

    Targets should be per-meal amounts (daily / 3 for 3-meal plans).
    Pass higher potassium_target_mg / magnesium_target_mg for HTN profiles
    to steer FAISS toward DASH-friendly high-K/Mg foods.
    """
    row = {
        "calories":        calorie_target,
        "protein_g":       protein_target_g,
        "carbs_g":         carb_target_g,
        "fat_g":           fat_target_g,
        "fiber_g":         fiber_target_g,
        "saturated_fat_g": 0,
        "sodium_mg":       sodium_limit_mg,
        "calcium_mg":      300,
        "iron_mg":         3,
        "vitamin_c_mg":    15,
        "vitamin_d_mcg":   5,
        "vitamin_b12_mcg": 0.8,
        "zinc_mg":         3,
        "potassium_mg":    potassium_target_mg,
        "magnesium_mg":    magnesium_target_mg,
        "phosphorus_mg":   350,
        "gi_value":        gi_preference,
        "category":        category_preference or "",
    }
    return build_feature_vector(row)


# ---------------------------------------------------------------------------
# FAISS index builder & searcher
# ---------------------------------------------------------------------------

class FoodEmbedder:
    """
    Manages the FAISS HNSW index over the food database.

    Attributes
    ----------
    fdc_ids : list[int]  — food IDs in index order (row i → fdc_ids[i])
    index   : faiss.IndexHNSWFlat  — the HNSW index
    """

    def __init__(self, db_path: Path = DB_PATH, index_path: Path = INDEX_PATH):
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.fdc_ids: list[int] = []
        self.food_names: list[str] = []
        self._index = None
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_db_count(self) -> int:
        """Return number of records with calories > 0 in the current database."""
        try:
            conn = sqlite3.connect(self.db_path)
            count = conn.execute("SELECT COUNT(*) FROM foods WHERE calories > 0").fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def build(self, force_rebuild: bool = False) -> int:
        """
        Build (or load from cache) the FAISS HNSW index.
        Automatically rebuilds if the database has changed since the index
        was last saved (e.g. after switching from synthetic to real USDA data).
        Returns the number of indexed items.
        """
        if not force_rebuild and self.index_path.exists():
            self._load()
            # Validate the cached index still matches the current database.
            # If the record count has changed (e.g. real data replaced synthetic),
            # the cached fdc_ids are stale and every FAISS search returns zero hits.
            current_count = self._get_db_count()
            if len(self.fdc_ids) != current_count:
                logger.info(
                    "Database changed (%d → %d records) — rebuilding FAISS index",
                    len(self.fdc_ids), current_count,
                )
                force_rebuild = True
            else:
                return len(self.fdc_ids)

        logger.info("Building FAISS HNSW index from %s", self.db_path)
        t0 = time.perf_counter()

        rows = self._load_db_rows()
        if not rows:
            logger.error("No rows found in %s — run ingest.py first", self.db_path)
            return 0

        vectors = np.stack([build_feature_vector(r) for r in rows]).astype(np.float32)
        self.fdc_ids = [r["fdc_id"] for r in rows]
        self.food_names = [r["food_name"] for r in rows]

        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        self._index = self._build_hnsw(vectors)
        self._built = True
        self._save()

        elapsed = time.perf_counter() - t0
        logger.info(
            "FAISS index built: %d items, dim=%d, elapsed=%.2fs",
            len(rows), FEATURE_DIM, elapsed,
        )
        return len(rows)

    def get_similar_foods(self, fdc_id: int, k: int = 10) -> list[dict]:
        """Return foods similar to a given food (by fdc_id)."""
        if not self._built:
            self.build()
        if fdc_id not in self.fdc_ids:
            return []
        idx = self.fdc_ids.index(fdc_id)

        import faiss
        vec = self._index.reconstruct(idx).reshape(1, -1)
        distances, indices = self._index.search(vec, k + 1)  # +1 to skip self

        results = []
        for dist, i in zip(distances[0], indices[0]):
            if i < 0 or i == idx:
                continue
            results.append({
                "fdc_id":    self.fdc_ids[i],
                "food_name": self.food_names[i],
                "distance":  float(dist),
            })
        return results[:k]

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def index_size(self) -> int:
        return len(self.fdc_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_hnsw(self, vectors: np.ndarray):
        """Build a FAISS HNSW index. Falls back to flat L2 if FAISS unavailable."""
        try:
            import faiss
            # HNSW parameters: M=32 (connections per node), efConstruction=200
            index = faiss.IndexHNSWFlat(FEATURE_DIM, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            index.add(vectors)
            logger.info("Built FAISS HNSW index (M=32, efConstruction=200)")
            return index
        except ImportError:
            logger.warning(
                "faiss-cpu not installed — falling back to brute-force numpy search. "
                "Install with: pip install faiss-cpu"
            )
            # Store vectors for numpy fallback
            self._vectors_fallback = vectors
            return None

    def _load_db_rows(self) -> list[dict]:
        """Load all food records from SQLite."""
        cols = (
            ["fdc_id", "food_name", "category", "gi_value"]
            + [c for c, _ in NUTRIENT_COLS]
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM foods WHERE calories > 0"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _save(self):
        """Serialise index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import faiss
            import tempfile, os
            # Save FAISS index to a temp file then pickle everything
            with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tf:
                faiss.write_index(self._index, tf.name)
                with open(tf.name, "rb") as f:
                    index_bytes = f.read()
            os.unlink(tf.name)
        except (ImportError, Exception):
            index_bytes = None

        payload = {
            "fdc_ids":     self.fdc_ids,
            "food_names":  self.food_names,
            "index_bytes": index_bytes,
            "feature_dim": FEATURE_DIM,
            "vectors_fallback": getattr(self, "_vectors_fallback", None),
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        logger.info("FAISS index saved to %s", self.index_path)

    def _load(self):
        """Load serialised index from disk."""
        logger.info("Loading FAISS index from %s", self.index_path)
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)

        self.fdc_ids = payload["fdc_ids"]
        self.food_names = payload["food_names"]

        index_bytes = payload.get("index_bytes")
        if index_bytes:
            try:
                import faiss, tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tf:
                    tf.write(index_bytes)
                self._index = faiss.read_index(tf.name)
                os.unlink(tf.name)
                logger.info("FAISS HNSW index loaded: %d items", len(self.fdc_ids))
            except ImportError:
                self._index = None
                self._vectors_fallback = payload.get("vectors_fallback")
        else:
            self._index = None
            self._vectors_fallback = payload.get("vectors_fallback")

        self._built = True

    def search(self, query_vector: np.ndarray, k: int = 50,
               filter_ids: Optional[set[int]] = None) -> list[dict]:
        """Search with FAISS HNSW or numpy fallback."""
        if not self._built:
            self.build()

        t0 = time.perf_counter()
        qv = query_vector.astype(np.float32).copy()
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv /= norm

        if self._index is not None:
            results = self._faiss_search(qv, k, filter_ids)
        else:
            results = self._numpy_search(qv, k, filter_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Search: k=%d, results=%d, elapsed=%.2fms", k, len(results), elapsed_ms)
        return results

    def _faiss_search(self, qv: np.ndarray, k: int,
                      filter_ids: Optional[set[int]]) -> list[dict]:
        import faiss
        fetch_k = min(k * 5 if filter_ids else k, len(self.fdc_ids))
        distances, indices = self._index.search(qv.reshape(1, -1), fetch_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            fdc_id = self.fdc_ids[idx]
            if filter_ids and fdc_id not in filter_ids:
                continue
            results.append({"fdc_id": fdc_id, "food_name": self.food_names[idx], "distance": float(dist)})
            if len(results) >= k:
                break
        return results

    def _numpy_search(self, qv: np.ndarray, k: int,
                      filter_ids: Optional[set[int]]) -> list[dict]:
        """Brute-force cosine similarity search (fallback if no FAISS)."""
        if not hasattr(self, "_vectors_fallback") or self._vectors_fallback is None:
            return []
        vecs = self._vectors_fallback
        sims = vecs @ qv
        indices = np.argsort(-sims)
        results = []
        for idx in indices:
            fdc_id = self.fdc_ids[int(idx)]
            if filter_ids and fdc_id not in filter_ids:
                continue
            results.append({
                "fdc_id":    fdc_id,
                "food_name": self.food_names[int(idx)],
                "distance":  float(sims[idx]),
            })
            if len(results) >= k:
                break
        return results
