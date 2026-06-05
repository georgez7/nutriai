# prompts.md — AI Assistance Log
## NutriAI · BAX-423 Final Project · Spring 2026

This file documents how AI assistance (Claude) was used throughout the project.
All code was reviewed, understood, and validated by the student before submission.

---

## Session 1 — Architecture & Data Layer

**Prompt:**
> "I'm building NutriAI for BAX-423. The spec requires a 7-day, 3-meal-per-day diet planner
> with 6 core capabilities (clinical filtering, allergen exclusion, dietary preferences,
> diversity, nutrient analysis, <60s generation). Help me design the pipeline architecture
> using Bloom filters and FAISS."

**AI contribution:** Designed the 4-stage pipeline architecture (SQL pre-filter → FAISS ANN
→ hard filter → weighted re-rank) .

**Student validation:** Reviewed architecture against project spec.

---

## Session 2 — Bloom Filter Implementation

**Prompt:**
> "Implement a BloomFilter class with configurable
> capacity and false-positive rate. Include functions for allergens and FODMAP lists."

**AI contribution:** Implemented `bloom_filter.py` with MD5 + SHA1 double-hashing,
allergen synonym expansion (gluten→wheat/barley/rye, dairy→milk/cheese/etc.),
and FODMAP CSV loader with hardcoded fallback.

**Student validation:** Verified zero false-negative property mathematically.
Confirmed allergen synonyms against clinical nutrition references.

---

## Session 3 — USDA Ingest & Demo Database

**Prompt:**
> "Write an ingestion module that fetches from the USDA FoodData Central API
> and also a seed_demo_database() function with synthetic data
> for offline use. The DB should have 10,500+ records covering all food categories."

**AI contribution:** Implemented `ingest.py` with `USDAClient`, `FoodRecord` dataclass,
`DataIngestor.run()`, and `seed_demo_database()`. Added SQLite schema with indexes.

**Student validation:** Reviewed synthetic nutrient values against USDA reference tables.
Corrected diet_tags for poultry (should be non_vegetarian only, not pescatarian).


---

## Session 4 — Constraint Engine

**Prompt:**
> "Build a ConstraintEngine that applies hard rules for: allergens (via Bloom filter),
> GERD triggers, FODMAP/IBS, GI limit for diabetes, sodium for hypertension, and
> dietary mode (vegan/vegetarian/pescatarian).

**AI contribution:** Implemented `constraints.py` with `UserProfile` dataclass,
`ConstraintEngine.evaluate()`, and per-rule `ConstraintResult` objects.

**Student validation:** Found and fixed critical diet_mode substring bug
("vegetarian" is a substring of "non_vegetarian"). Added pescatarian category
exclusion in SQL pre-filter.

---

## Session 5 — Ranker, Nutrients, Diversity

**Prompt:**
> "Implement the 4-stage ranker, nutrient aggregator with 15-nutrient RDA tracking,
> and diversity scorer."

**AI contribution:** Implemented `ranker.py`, `nutrients.py`, and `diversity.py`.


**Student validation:** Fixed entropy normalization bug (was normalising against total
meals instead of unique categories, causing scores to be artificially low for constrained
profiles like Mei/Priya). Verified RDA values against NIH DRI tables for all 15 nutrients.

---

## Session 6 — Tests

**Prompt:**
> "Write a comprehensive test suite: 16 Bloom filter unit tests and persona integration
> tests for all 4 test personas (Priya/Ravi/Mei/James) covering all 6 capabilities."

**AI contribution:** Implemented `test_bloom.py` and `test_personas.py` with run_all()
standalone runner and pytest-compatible wrapper functions.

**Student validation:** Debugged session fixture caching issue where stale DB data caused false test failures.
Revised C5 test to validate analysis capability (gap detection + flagging) rather than plan optimality

---

## Session 7 — Streamlit App

**Prompt:**
> "Build a Streamlit app with: sidebar profile form, 4 tabs (meal plan, nutrition charts,
> diversity analysis, explain/audit), Plotly charts for daily calories and micronutrient
> radar, and per-meal explain popovers."

**AI contribution:** Implemented `app.py` with sidebar, 4 tabs, Plotly bar/radar/pie charts,
dark-mode CSS, and constraint audit popover.

**Student validation:** Verified all 6 capabilities are demonstrated in the UI.


---