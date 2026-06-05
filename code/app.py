"""
app.py — NutriAI Streamlit UI
==============================
7-day personalised diet planner demonstrating all 6 core capabilities.

Run:
    cd BigDataFinal/code
    streamlit run app.py
"""

import sys
import time
import sqlite3
import logging
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DB_PATH  = DATA_DIR / "foods.db"
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

# ── lazy pipeline imports ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from pipeline.ingest      import seed_demo_database
    from pipeline.constraints import ConstraintEngine, UserProfile, DietMode, load_candidate_foods
    from pipeline.nutrients   import NutrientAggregator, DayNutrients, MealNutrients
    from pipeline.diversity   import DiversityScorer
    from pipeline.bloom_filter import build_allergen_filter, build_fodmap_filter
    from pipeline.grocery     import generate_grocery_list
    return {
        "seed_demo_database":    seed_demo_database,
        "ConstraintEngine":      ConstraintEngine,
        "UserProfile":           UserProfile,
        "DietMode":              DietMode,
        "load_candidate_foods":  load_candidate_foods,
        "NutrientAggregator":   NutrientAggregator,
        "DayNutrients":          DayNutrients,
        "MealNutrients":         MealNutrients,
        "DiversityScorer":       DiversityScorer,
        "build_allergen_filter": build_allergen_filter,
        "build_fodmap_filter":   build_fodmap_filter,
        "generate_grocery_list": generate_grocery_list,
    }

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriAI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .meal-card {
    background: #1e2530;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 4px solid #3b82f6;
  }
  .meal-card h4 { margin: 0 0 4px 0; color: #93c5fd; font-size: 0.95rem; }
  .meal-card .food-name { font-weight: 600; font-size: 1.05rem; color: #f1f5f9; }
  .meal-card .macros { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; margin-right: 4px;
  }
  .badge-green  { background: #166534; color: #bbf7d0; }
  .badge-blue   { background: #1e3a5f; color: #93c5fd; }
  .badge-orange { background: #7c2d12; color: #fed7aa; }
  .badge-red    { background: #7f1d1d; color: #fecaca; }
  .gap-row { padding: 6px 0; border-bottom: 1px solid #1e2530; }
  .explain-box {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 12px; font-size: 0.85rem; color: #94a3b8; margin-top: 8px;
  }
  .metric-card {
    background: #1e2530; border-radius: 8px; padding: 16px;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── DB bootstrap ─────────────────────────────────────────────────────────────
USDA_API_KEY = "gHPUDW2yjsv3rfxcUiBgYnuQSDewuclLWb2HMX9c"


def ensure_database(pipe):
    if not DB_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("⚙️  Fetching food data from USDA FoodData Central API (~3 min)…"):
            from pipeline.ingest import DataIngestor
            ingestor = DataIngestor(api_key=USDA_API_KEY, db_path=DB_PATH)
            n = ingestor.run(target=10_000)
        st.success(f"✅ Database ready — {n:,} USDA foods indexed.")


# ── plan generator ───────────────────────────────────────────────────────────
def generate_plan(pipe, profile):
    from pipeline.ranker import MealRanker
    from pipeline.embeddings import FoodEmbedder

    t0 = time.perf_counter()

    embedder = FoodEmbedder(DB_PATH)
    embedder.build()

    ranker = MealRanker(profile, embedder, DB_PATH)
    result = ranker.generate_plan(profile.calorie_target)

    # Flatten to list of 21 meal-component-lists (each meal = list of food dicts)
    _EMPTY = {
        "fdc_id": -1, "food_name": "No meal available",
        "category": "", "diet_tags": "", "allergen_flags": "",
        "fodmap_status": "", "gi_value": None,
        "calories": 0, "protein_g": 0, "carbs_g": 0,
        "fat_g": 0, "fiber_g": 0, "portion_g": 0,
    }
    plan = []
    for day_entry in result["days"]:
        for meal_name in ["breakfast", "lunch", "dinner"]:
            components = day_entry["meals"].get(meal_name, [])
            meal_foods = []
            for scored in components:
                food_dict = dict(scored.nutrient_row)
                food_dict["portion_g"] = scored.portion_g
                meal_foods.append(food_dict)
            plan.append(meal_foods if meal_foods else [_EMPTY])

    elapsed = time.perf_counter() - t0
    return plan, elapsed


# ── sidebar — user profile form ──────────────────────────────────────────────
def sidebar_profile(pipe):
    DietMode = pipe["DietMode"]

    st.sidebar.markdown("## 🥗 NutriAI")
    st.sidebar.markdown("*Personalised 7-day diet planner*")
    st.sidebar.divider()

    st.sidebar.markdown("### 👤 Your Profile")
    name = st.sidebar.text_input("Name", value="Alex")
    age  = st.sidebar.slider("Age", 18, 80, 35)
    sex  = st.sidebar.radio("Biological sex", ["male", "female"], horizontal=True)
    calorie_target = st.sidebar.slider("Daily calorie target (kcal)", 1200, 3500, 2000, 50)

    st.sidebar.markdown("### 🥦 Diet Mode")
    diet_mode_label = st.sidebar.selectbox(
        "Dietary preference",
        ["Non-Vegetarian", "Pescatarian", "Vegetarian", "Vegan"],
    )
    diet_map = {
        "Non-Vegetarian": DietMode.NON_VEGETARIAN,
        "Pescatarian":    DietMode.PESCATARIAN,
        "Vegetarian":     DietMode.VEGETARIAN,
        "Vegan":          DietMode.VEGAN,
    }
    diet_mode = diet_map[diet_mode_label]

    st.sidebar.markdown("### 🏥 Clinical Conditions")
    has_ibs         = st.sidebar.checkbox("IBS (low-FODMAP)")
    has_gerd        = st.sidebar.checkbox("GERD / Acid reflux")
    has_diabetes_t2 = st.sidebar.checkbox("Type 2 Diabetes")
    has_hypertension = st.sidebar.checkbox("Hypertension (DASH)")

    sodium_limit = 2300
    gi_limit     = 100
    if has_hypertension:
        sodium_limit = st.sidebar.slider("Sodium limit (mg/day)", 1000, 2300, 1500, 100)
    if has_diabetes_t2:
        gi_limit = st.sidebar.slider(
            "Max meal glycemic load (GL)",
            min_value=10, max_value=70, value=55, step=1,
            help="GL = Σ (GI × carbs_in_serving) / 100 across all meal components. "
                 "Low GL < 10 · Medium GL 10–20 · High GL > 20. "
                 "A per-meal target of ~30 aligns with a daily GL < 100 "
                 "(standard low-GL diet for T2DM). "
                 "GL accounts for portion size — a high-GI ingredient in a small "
                 "portion alongside fibre and protein can still yield an acceptable meal GL."
        )

    st.sidebar.markdown("### 🚫 Allergens")
    allergen_options = ["dairy", "gluten", "eggs", "soy", "tree nuts",
                        "peanuts", "fish", "shellfish", "sesame"]
    allergens = st.sidebar.multiselect("Select allergens to exclude", allergen_options)

    st.sidebar.markdown("### 🔴 Religious / Ethical")
    no_pork = st.sidebar.checkbox("No pork / halal-friendly")
    no_beef = st.sidebar.checkbox("No beef / kosher-friendly")

    profile = pipe["UserProfile"](
        name=name, age=age, sex=sex,
        calorie_target=calorie_target,
        diet_mode=diet_mode,
        has_ibs=has_ibs,
        has_gerd=has_gerd,
        has_diabetes_t2=has_diabetes_t2,
        has_hypertension=has_hypertension,
        allergens=allergens,
        no_pork=no_pork,
        no_beef=no_beef,
        sodium_limit_mg=sodium_limit,
        gi_limit=gi_limit,
        micro_priorities=["iron_mg", "calcium_mg", "vitamin_d_mcg",
                          "vitamin_b12_mcg", "zinc_mg", "omega3_g"],
    )

    generate = st.sidebar.button("🚀 Generate My 7-Day Plan", type="primary", use_container_width=True)
    return profile, generate


# ── helpers ──────────────────────────────────────────────────────────────────
def format_macros(food):
    portion = food.get("portion_g", 100.0)
    scale = portion / 100.0
    cal  = (food.get("calories",  0) or 0) * scale
    prot = (food.get("protein_g", 0) or 0) * scale
    carb = (food.get("carbs_g",   0) or 0) * scale
    fat  = (food.get("fat_g",     0) or 0) * scale
    return f"🔥 {cal:.0f} kcal · 🥩 {prot:.1f}g protein · 🌾 {carb:.1f}g carbs · 🫙 {fat:.1f}g fat · ({portion:.0f}g serving)"


def diet_badge(food):
    tags = food.get("diet_tags", "") or ""
    if "vegan" in tags:          return '<span class="badge badge-green">Vegan</span>'
    if "vegetarian" in tags:     return '<span class="badge badge-green">Vegetarian</span>'
    if "pescatarian" in tags:    return '<span class="badge badge-blue">Pescatarian</span>'
    return '<span class="badge badge-orange">Non-Veg</span>'


def gi_badge(food):
    gi = food.get("gi_value")
    if gi is None:  return ""
    if gi <= 55:    return '<span class="badge badge-green">Low GI</span>'
    if gi <= 70:    return '<span class="badge badge-orange">Med GI</span>'
    return '<span class="badge badge-red">High GI</span>'


# ── tabs ─────────────────────────────────────────────────────────────────────
def meal_total_macros(foods: list[dict]) -> dict:
    """Sum scaled macros across all components in a meal."""
    totals = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    for f in foods:
        scale = f.get("portion_g", 100) / 100
        for k in totals:
            totals[k] += (f.get(k, 0) or 0) * scale
    return totals


def render_plan_tab(plan):
    st.markdown("### 📅 Your 7-Day Meal Plan")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_names = ["🌅 Breakfast", "☀️ Lunch", "🌙 Dinner"]
    comp_labels = ["Grain / Base", "Protein", "Vegetable / Fruit"]

    for day_idx in range(7):
        meals = plan[day_idx * 3: day_idx * 3 + 3]
        if not meals:
            continue
        with st.expander(f"**{day_names[day_idx]}**", expanded=(day_idx == 0)):
            cols = st.columns(3)
            for col, meal_label, meal_foods in zip(cols, meal_names, meals):
                with col:
                    totals = meal_total_macros(meal_foods)
                    # Meal header with total macros
                    header = f"""<div class="meal-card">
  <h4>{meal_label}</h4>
  <div class="macros">🔥 {totals['calories']:.0f} kcal total · 🥩 {totals['protein_g']:.1f}g · 🌾 {totals['carbs_g']:.1f}g · 🫙 {totals['fat_g']:.1f}g</div>
</div>"""
                    st.markdown(header, unsafe_allow_html=True)

                    # Each component
                    for i, food in enumerate(meal_foods):
                        role = comp_labels[i] if i < len(comp_labels) else f"Item {i+1}"
                        badges = diet_badge(food) + gi_badge(food)
                        scale = food.get("portion_g", 100) / 100
                        cal = (food.get("calories", 0) or 0) * scale
                        card = f"""<div style="background:#151e2d;border-radius:8px;padding:8px 12px;margin:4px 0;border-left:3px solid #475569">
  <div style="font-size:0.75rem;color:#64748b;margin-bottom:2px">{role}</div>
  <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem">{food.get('food_name','')}</div>
  <div style="font-size:0.78rem;color:#94a3b8">{cal:.0f} kcal · {food.get('portion_g',100):.0f}g serving</div>
  <div style="margin-top:4px">{badges}</div>
</div>"""
                        st.markdown(card, unsafe_allow_html=True)

                    with st.popover("🔍 Explain"):
                        for food in meal_foods:
                            st.markdown(f"**{food.get('food_name','')}**")
                            st.write(f"Category: {food.get('category','')} · "
                                     f"FODMAP: {food.get('fodmap_status','—')} · "
                                     f"GI: {food.get('gi_value') or 'N/A'}")
                            st.write(f"Allergens: {food.get('allergen_flags','none') or 'none'} · "
                                     f"Fibre: {food.get('fiber_g',0):.1f}g · "
                                     f"Sodium: {food.get('sodium_mg',0):.0f}mg")
                            st.divider()


def render_nutrition_tab(plan, profile, pipe):
    NutrientAggregator = pipe["NutrientAggregator"]
    DayNutrients       = pipe["DayNutrients"]
    MealNutrients      = pipe["MealNutrients"]

    st.markdown("### 📊 Nutritional Analysis")

    # Build aggregator — each day_meals[i] is now a list of food dicts
    agg = NutrientAggregator(profile.age, profile.sex, profile.calorie_target)
    for day_num in range(1, 8):
        day_meals = plan[(day_num - 1) * 3: day_num * 3]
        day_entry = DayNutrients(day=day_num)
        for i, meal_name in enumerate(["breakfast", "lunch", "dinner"]):
            if i < len(day_meals):
                # Sum components: start with first, add the rest
                components = day_meals[i]
                if components:
                    mn = MealNutrients.from_food_row(meal_name, components[0],
                                                     portion_g=components[0].get("portion_g", 150.0))
                    for comp in components[1:]:
                        mn = mn + MealNutrients.from_food_row(
                            meal_name, comp, portion_g=comp.get("portion_g", 150.0)
                        )
                    day_entry.meals[meal_name] = mn
        agg.add_day(day_entry)

    summary = agg.weekly_summary()
    gap_report = summary["weekly_gap_report"]

    # ── Macro bar chart (daily calories) ─────────────────────────────────────
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_cals = []
    for day_num in range(1, 8):
        day_meals = plan[(day_num - 1) * 3: day_num * 3]
        total_cal = sum(
            float(f.get("calories", 0) or 0) * f.get("portion_g", 150.0) / 100
            for meal in day_meals for f in meal
        )
        daily_cals.append(total_cal)

    fig_cal = go.Figure(go.Bar(
        x=day_names, y=daily_cals,
        marker_color=["#3b82f6" if c >= profile.calorie_target * 0.85 else "#ef4444"
                      for c in daily_cals],
        text=[f"{c:.0f}" for c in daily_cals], textposition="outside",
    ))
    fig_cal.add_hline(y=profile.calorie_target, line_dash="dash", line_color="#94a3b8",
                       annotation_text=f"Target: {profile.calorie_target} kcal")
    fig_cal.update_layout(
        title="Daily Caloric Intake vs Target",
        xaxis_title="Day", yaxis_title="kcal",
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#f1f5f9", height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # ── Micronutrient gap report ──────────────────────────────────────────────
    st.markdown("#### Micronutrient Gap Report")
    st.caption("Shows average % of RDA achieved per nutrient over the week. "
               "Cells flagged 🚨 are below 80% RDA.")

    gap_df = pd.DataFrame(gap_report)
    if not gap_df.empty:
        gap_df["avg_pct_rda"] = gap_df["avg_pct_rda"].round(1)
        gap_df["status"] = gap_df.apply(
            lambda r: "🚨 Below 80%" if r["days_below_80"] >= 4 else
                      ("⚠️ Borderline"  if r["days_below_80"] >= 2 else "✅ Adequate"),
            axis=1
        )
        gap_df = gap_df.rename(columns={
            "nutrient": "Nutrient",
            "avg_pct_rda": "Avg % RDA",
            "days_below_80": "Days Below 80%",
            "status": "Status",
        })
        st.dataframe(
            gap_df[["Nutrient", "Avg % RDA", "Days Below 80%", "Status"]],
            use_container_width=True, hide_index=True,
        )

        # Radar chart for priority micros
        priority = profile.micro_priorities or []
        radar_rows = [r for r in gap_report if r["nutrient"] in priority]
        if radar_rows:
            labels = [r["nutrient"].replace("_mg","").replace("_mcg","").replace("_"," ").title()
                      for r in radar_rows]
            values = [min(r["avg_pct_rda"], 150) for r in radar_rows]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(59,130,246,0.2)",
                line_color="#3b82f6",
                name="% RDA",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[80] * (len(labels) + 1),
                theta=labels + [labels[0]],
                line_color="#ef4444", line_dash="dash",
                name="80% threshold",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 150], tickcolor="#94a3b8"),
                           bgcolor="#0f172a"),
                paper_bgcolor="#0f172a", font_color="#f1f5f9",
                title="Priority Micronutrient Coverage (% RDA)",
                height=400, showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


def render_diversity_tab(plan, pipe, profile=None):
    DiversityScorer = pipe["DiversityScorer"]
    st.markdown("### 🌈 Diversity Analysis")

    flat_plan = [f for meal in plan for f in meal]
    scorer = DiversityScorer()
    report = scorer.score_plan(flat_plan, profile=profile)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Diversity Score", f"{report.score:.3f}", delta="≥ 0.7 target" if report.score >= 0.7 else "< 0.7 target")
    with col2:
        st.metric("Unique Categories", report.unique_categories)
    with col3:
        st.metric("Unique Food Groups", report.unique_groups)
    with col4:
        st.metric("Repeated Meals", report.repeat_count, delta=("0 ✅" if report.repeat_count == 0 else None),
                  delta_color="normal")

    # Category pie chart
    if report.category_counts:
        fig_pie = px.pie(
            values=list(report.category_counts.values()),
            names=list(report.category_counts.keys()),
            title="Meal Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_layout(
            paper_bgcolor="#0f172a", font_color="#f1f5f9", height=380,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Details
    with st.expander("Diversity score breakdown"):
        st.code(report.details)


def render_explain_tab(plan, profile, pipe):
    st.markdown("### 🔍 Explain Decisions (C1 / C2 / C3 Audit)")
    st.caption("Select a meal component to see why it was included.")

    engine = pipe["ConstraintEngine"](profile)
    day_names_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    meal_labels = ["Breakfast", "Lunch", "Dinner"]
    comp_roles = ["Grain/Base", "Protein", "Side"]

    def _meal_gl(meal_foods: list[dict]) -> tuple[float, list[dict]]:
        """Compute total Glycemic Load for a meal and per-component breakdown."""
        breakdown = []
        total_gl  = 0.0
        for f in meal_foods:
            gi      = f.get("gi_value")
            carbs   = float(f.get("carbs_g") or 0)
            portion = float(f.get("portion_g") or 100)
            if gi is not None and carbs > 0:
                carbs_serving = carbs * portion / 100
                comp_gl = (float(gi) * carbs_serving) / 100
                total_gl += comp_gl
            else:
                comp_gl = None
            breakdown.append({
                "name":    f.get("food_name", ""),
                "gi":      gi,
                "carbs_g": round(carbs * portion / 100, 1),
                "comp_gl": round(comp_gl, 2) if comp_gl is not None else None,
            })
        return round(total_gl, 2), breakdown

    # Build flat list of (label, food_dict, meal_idx) for the selectbox
    options = []
    for d in range(7):
        for m, meal in enumerate(meal_labels):
            idx = d * 3 + m
            if idx < len(plan):
                for c, food in enumerate(plan[idx]):
                    role = comp_roles[c] if c < len(comp_roles) else f"Item {c+1}"
                    label = f"Day {d+1} ({day_names_short[d]}) {meal} — {role}: {food.get('food_name','')}"
                    options.append((label, food, idx))

    if not options:
        st.info("Generate a plan first.")
        return

    labels = [o[0] for o in options]
    selected = st.selectbox("Select a component to explain:", labels)
    if selected:
        sel_idx  = labels.index(selected)
        food     = options[sel_idx][1]
        meal_idx = options[sel_idx][2]
        verdict  = engine.evaluate(food)

        st.markdown(f"#### 🍽️ {food.get('food_name','')}")

        # ── Meal-level GL panel (T2DM only) ──────────────────────────────
        if profile.has_diabetes_t2:
            meal_foods        = plan[meal_idx]
            total_gl, gl_bkdn = _meal_gl(meal_foods)
            gl_limit          = profile.gi_limit  # repurposed as GL limit
            gl_status         = "🟢 Low" if total_gl < 10 else ("🟡 Medium" if total_gl <= 20 else "🔴 High")
            limit_ok          = total_gl <= gl_limit

            st.markdown("**Meal Glycemic Load (GL)**")
            st.caption("GL = Σ (GI × carbs_in_serving) / 100 across all components")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Meal GL",   f"{total_gl:.1f}", delta=gl_status)
            m2.metric("GL Limit",        f"{gl_limit:.0f}")
            m3.metric("Status", "✅ Within limit" if limit_ok else "⚠️ Exceeds limit",
                      delta=None)

            gl_rows = []
            for row in gl_bkdn:
                if row["comp_gl"] is not None:
                    gl_rows.append({
                        "Ingredient":    row["name"],
                        "GI":            row["gi"],
                        "Carbs (g)":     row["carbs_g"],
                        "Component GL":  row["comp_gl"],
                    })
                else:
                    gl_rows.append({
                        "Ingredient":   row["name"],
                        "GI":           "N/A",
                        "Carbs (g)":    row["carbs_g"],
                        "Component GL": "N/A (no GI data)",
                    })
            st.dataframe(gl_rows, use_container_width=True, hide_index=True)
            st.divider()

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Food details**")
            st.json({
                "category":       food.get("category"),
                "diet_tags":      food.get("diet_tags"),
                "allergen_flags": food.get("allergen_flags") or "none",
                "fodmap_status":  food.get("fodmap_status"),
                "gi_value":       food.get("gi_value"),
                "sodium_mg":      food.get("sodium_mg"),
                "calories":       food.get("calories"),
                "portion_g":      food.get("portion_g"),
            })
        with cols[1]:
            st.markdown("**Constraint checks**")
            for result in verdict.results:
                icon = "✅" if result.passed else "❌"
                severity = f" [{result.severity}]" if not result.passed else ""
                st.markdown(f"{icon} **{result.rule}**{severity}  \n{result.reason}")


# ── grocery tab ──────────────────────────────────────────────────────────────
def render_grocery_tab(plan, pipe):
    from pipeline.grocery import SpoonacularPricer
    generate_grocery_list = pipe["generate_grocery_list"]
    st.markdown("### 🛒 Weekly Grocery List")

    # Initialise Spoonacular pricer once per session
    if "spoon_pricer" not in st.session_state:
        st.session_state["spoon_pricer"] = SpoonacularPricer()

    pricer = st.session_state["spoon_pricer"]

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(f"🌿 Live prices via Spoonacular · {pricer.cache_size} ingredients cached this session")
    with col_btn:
        if st.button("🔄 Refresh Prices", type="secondary"):
            st.session_state["spoon_pricer"] = SpoonacularPricer()
            pricer = st.session_state["spoon_pricer"]

    with st.spinner("Fetching ingredient prices from Spoonacular…"):
        grocery = generate_grocery_list(plan, pricer=pricer)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Ingredients", grocery.total_items)
    with col2:
        st.metric("Est. Weekly Cost", f"${grocery.total_cost_usd:.2f}")
    with col3:
        st.metric("Est. Daily Cost", f"${grocery.total_cost_usd / 7:.2f}")

    st.caption("💡 " + grocery.notes[0])
    st.divider()

    # Build CSV for download
    import io, csv as csv_mod
    buf = io.StringIO()
    writer = csv_mod.writer(buf)
    writer.writerow(["Section", "Item", "Quantity", "Est. Cost (USD)"])
    for item in grocery.items:
        writer.writerow([item.section, item.name, item.purchase_unit, f"${item.est_cost_usd:.2f}"])
    writer.writerow([])
    writer.writerow(["", "", "TOTAL", f"${grocery.total_cost_usd:.2f}"])
    st.download_button(
        label="⬇️ Download as CSV",
        data=buf.getvalue(),
        file_name="nutriai_grocery_list.csv",
        mime="text/csv",
    )

    st.markdown("")

    # Items by section
    from pipeline.grocery import SECTION_ORDER
    for section in SECTION_ORDER:
        items = grocery.by_section.get(section, [])
        if not items:
            continue
        section_cost = sum(i.est_cost_usd for i in items)
        with st.expander(f"**{section}** — {len(items)} items · ${section_cost:.2f}", expanded=True):
            h1, h2, h3, h4 = st.columns([3.5, 1.8, 1.4, 1.3])
            h1.markdown("**Item**")
            h2.markdown("**Quantity**")
            h3.markdown("**Price**")
            h4.markdown("**Source**")
            st.divider()
            for item in items:
                c1, c2, c3, c4 = st.columns([3.5, 1.8, 1.4, 1.3])
                c1.write(item.name)
                c2.write(item.purchase_unit)
                c3.write(f"${item.est_cost_usd:.2f}")
                if item.price_source == "spoonacular":
                    c4.markdown('<span class="badge badge-green">🌿 Live</span>', unsafe_allow_html=True)
                else:
                    c4.markdown('<span class="badge badge-orange">~est.</span>', unsafe_allow_html=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    pipe = _load_pipeline()
    ensure_database(pipe)

    profile, generate = sidebar_profile(pipe)

    # Header
    st.title("🥗 NutriAI — Personalised Diet Planner")
    st.caption("BAX-423 Big Data · Final Project · UC Davis GSM · Spring 2026")

    if not generate and "plan" not in st.session_state:
        st.info("👈 Fill in your profile on the left, then click **Generate My 7-Day Plan**.")
        st.markdown("""
        **Capabilities demonstrated:**
        - 🏥 **C1** — Clinical condition filtering: IBS (FODMAP), GERD, Type 2 Diabetes, Hypertension
        - 🚫 **C2** — Allergen exclusion (Bloom filter, zero false negatives guaranteed)
        - 🥦 **C3** — Dietary preference: Vegan / Vegetarian / Pescatarian / Non-Veg
        - 🌈 **C4** — Diversity engine: Shannon entropy scoring, no-repeat guarantees
        - 📊 **C5** — Macro & micronutrient analysis: 15 nutrients tracked, gap flagging
        - ⚡ **C6** — Sub-60-second generation with FAISS ANN + Bloom filter pre-screening

        **BAX-423 techniques used:** Bloom filter (allergen/FODMAP), FAISS HNSW embeddings,
        multi-stage ranking pipeline (SQL → ANN → hard filter → weighted re-rank),
        Thompson Bandit RL for category diversity.
        """)
        return

    if generate:
        with st.spinner("⚡ Generating your personalised 7-day plan…"):
            plan, elapsed = generate_plan(pipe, profile)
        st.session_state["plan"]    = plan
        st.session_state["elapsed"] = elapsed
        st.session_state["profile"] = profile
        st.session_state.pop("diversity_score", None)  # force recompute for new plan

    plan    = st.session_state.get("plan", [])
    elapsed = st.session_state.get("elapsed", 0)
    profile = st.session_state.get("profile", profile)

    if not plan:
        st.warning("⚠️ No meals could be generated with the current constraints. "
                   "Try relaxing some restrictions.")
        return

    # Stats bar
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Meals Generated", len(plan))
    with col2: st.metric("Generation Time", f"{elapsed:.2f}s", delta="< 60s ✅" if elapsed < 60 else "⚠️ slow")
    with col3:
        diversity_score = None
        if "diversity_score" not in st.session_state:
            scorer = pipe["DiversityScorer"]()
            flat = [f for meal in plan for f in meal]
            rep = scorer.score_plan(flat, profile=profile)
            st.session_state["diversity_score"] = rep.score
        st.metric("Diversity Score", f"{st.session_state['diversity_score']:.3f}",
                  delta="≥ 0.7 ✅" if st.session_state["diversity_score"] >= 0.7 else "< 0.7")
    with col4:
        avg_cal = sum(
            float(f.get("calories", 0) or 0) * f.get("portion_g", 150.0) / 100
            for meal in plan for f in meal
        ) / 7
        st.metric("Avg Daily kcal", f"{avg_cal:.0f}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📅 Meal Plan", "📊 Nutrition", "🌈 Diversity", "🔍 Explain", "🛒 Grocery List"]
    )
    with tab1:
        render_plan_tab(plan)
    with tab2:
        render_nutrition_tab(plan, profile, pipe)
    with tab3:
        render_diversity_tab(plan, pipe, profile=profile)
    with tab4:
        render_explain_tab(plan, profile, pipe)
    with tab5:
        render_grocery_tab(plan, pipe)


if __name__ == "__main__":
    main()
