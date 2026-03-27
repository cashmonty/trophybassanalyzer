"""Lake Detail -- deep dive into individual lake statistics and fishing intel."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.config import DATA_DIR, load_lakes
from src.analysis.correlations import compute_feature_importance
from src.dashboard.ui import apply_figure_style, bootstrap_dashboard, lake_label, render_page_header

ctx = bootstrap_dashboard("Lake Detail")
COLORS = ctx.colors

render_page_header(
    "Lake Detail",
    "Per-lake breakdown with local structure notes, catch timing, and feature importance for the selected waters.",
)

# ---------------------------------------------------------------------------
# Data — load directly if not in session_state
# ---------------------------------------------------------------------------
df = ctx.merged_df
lake_configs = ctx.lake_configs

if df is None or (hasattr(df, "empty") and df.empty):
    try:
        df = pd.read_parquet(DATA_DIR / "processed" / "merged.parquet")
        df["datetime"] = pd.to_datetime(df["datetime"])
    except Exception:
        st.info("No data available. Run the pipeline first.")
        st.stop()

if not lake_configs:
    try:
        lake_configs = {lc.key: lc for lc in load_lakes()}
    except Exception:
        pass

TROPHY_WEIGHT = 7.0

# Lake structure/tips knowledge base
LAKE_INTEL = {
    "monroe": {
        "key_areas": "Cutright/Salt Creek arm (pre-spawn staging), Allen's Creek (standing timber), "
                     "Pine Grove (deep points), Fairfax Beach area (spawning flats)",
        "best_pattern": "Pre-spawn: jerkbait on secondary points in creek arms. "
                        "Summer: deep cranking 15-20ft timber. Fall: topwater in creek arms.",
        "local_tip": "Standing timber in 15-25ft is the top big-bass structure. "
                     "South wind pushes bait into the north bank and gets fish moving.",
    },
    "patoka": {
        "key_areas": "Lick Fork arm (clear water, big fish), main lake points, "
                     "standing timber on south end, dam area bluffs",
        "best_pattern": "Clear water = finesse. Drop shot and shaky head in 15-25ft. "
                        "Pre-spawn: suspending jerkbait on rocky points. Night: black jig on bluffs.",
        "local_tip": "Clearest lake in Indiana, so downsize your line and baits. "
                     "Best trophy genetics in the state. Dawn bite is critical.",
    },
    "brookville": {
        "key_areas": "Whitewater arm (largemouth), main lake bluffs (smallmouth), "
                     "Dunlapsville area, standing timber in 30-50ft range",
        "best_pattern": "Deep-water lake: vertical jigging spoons in summer. "
                        "Pre-spawn: crawdad jig on steep bluff transitions. "
                        "Fall: crankbait on long points.",
        "local_tip": "Do not ignore smallmouth. Brookville has state-record potential for both species. "
                     "Deep clear water, bring your electronics.",
    },
    "geist": {
        "key_areas": "Fall Creek inlet (current + bait), Olio Road bridge pilings, "
                     "Saxony Beach riprap, northeast grass beds (summer)",
        "best_pattern": "Power-fish this lake. Spinnerbait and squarebill crankbait. "
                        "Summer: frog on grass mats. Fall: jerkbait on riprap.",
        "local_tip": "Urban lake with tons of docks, and skipping jigs under them is a major edge. "
                     "Tournaments are heavy here so fish pressure is real.",
    },
    "lemon": {
        "key_areas": "Dam area (deep timber), north coves (spawning), "
                     "mid-lake brush piles, laydowns on NW bank",
        "best_pattern": "Stained water = reaction baits. Chatterbait and spinnerbait. "
                        "Pre-spawn: lipless crankbait on points. Summer: Texas rig brush piles.",
        "local_tip": "Smaller lake but productive, with less pressure than Monroe. "
                     "Brush piles hold fish year-round, so milk them carefully.",
    },
    "morse": {
        "key_areas": "Cicero Creek inlet, 236th St bridge area, "
                     "main dam riprap, residential docks on east side",
        "best_pattern": "Shallow power lake. Spinnerbait, swim jig, squarebill. "
                        "Skip docks with a jig. Topwater early and late.",
        "local_tip": "Very shallow, so fish relate to any hard structure they can find. "
                     "Riprap and docks are the two primary patterns.",
    },
}

# ---------------------------------------------------------------------------
# Lake selector
# ---------------------------------------------------------------------------
available_lakes = sorted(df["lake_key"].unique().tolist()) if "lake_key" in df.columns else []

if not available_lakes:
    st.info("No lakes found in the data.")
    st.stop()

selected_lake = st.selectbox(
    "Select Lake",
    available_lakes,
    format_func=lambda key: lake_label(key, lake_configs),
)

lake_df = df[df["lake_key"] == selected_lake].copy()
lake_cfg = lake_configs.get(selected_lake)

if lake_df.empty:
    st.info(f"No data for {selected_lake}.")
    st.stop()

# ---------------------------------------------------------------------------
# Lake info header
# ---------------------------------------------------------------------------
lake_name = lake_cfg.name if lake_cfg else selected_lake.title()
st.subheader(lake_name)

col1, col2, col3, col4, col5 = st.columns(5)

total_catches = int(lake_df["catch_count"].sum()) if "catch_count" in lake_df.columns else 0
total_trophies = int(lake_df["trophy_count"].sum()) if "trophy_count" in lake_df.columns else 0
trophy_rate = (total_trophies / total_catches * 100) if total_catches > 0 else 0
max_fish = lake_df["max_weight"].max() if "max_weight" in lake_df.columns else 0

col1.metric("Total Catches", f"{total_catches:,}")
col2.metric("Trophies (7+)", f"{total_trophies:,}")
col3.metric("Trophy Rate", f"{trophy_rate:.1f}%")
col4.metric("Biggest Fish", f"{max_fish:.2f} lbs" if max_fish > 0 else "N/A")

if lake_cfg:
    depth_str = ""
    if lake_cfg.max_depth_ft:
        depth_str = f"Max depth: {lake_cfg.max_depth_ft}ft"
    if lake_cfg.avg_depth_ft:
        depth_str += f" / Avg: {lake_cfg.avg_depth_ft}ft"
    col5.metric("Depth", depth_str if depth_str else "N/A")

if lake_cfg:
    st.caption(
        f"Location: {lake_cfg.lat:.4f}, {lake_cfg.lon:.4f} | "
        f"USGS Station: {lake_cfg.usgs_station or 'None'} | {lake_cfg.notes}"
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Fishing Intel (pro angler knowledge)
# ---------------------------------------------------------------------------
intel = LAKE_INTEL.get(selected_lake)
if intel:
    st.subheader(f"Fishing Intel: {lake_name}")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"**Key Areas**\n\n{intel['key_areas']}")
    with ic2:
        st.markdown(f"**Best Patterns**\n\n{intel['best_pattern']}")
    with ic3:
        st.markdown(f"**Local Tip**\n\n{intel['local_tip']}")
    st.markdown("---")

# ---------------------------------------------------------------------------
# Folium map
# ---------------------------------------------------------------------------
st.subheader("Lake Location")

if lake_cfg:
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[lake_cfg.lat, lake_cfg.lon], zoom_start=12)
        folium.Marker(
            [lake_cfg.lat, lake_cfg.lon],
            popup=lake_cfg.name,
            tooltip=lake_cfg.name,
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(m)
        st_folium(m, width=700, height=400)
    except ImportError:
        st.info(
            "Install `folium` and `streamlit-folium` for interactive maps: "
            "`pip install folium streamlit-folium`"
        )
        st.map(pd.DataFrame({"lat": [lake_cfg.lat], "lon": [lake_cfg.lon]}))
else:
    st.info("Lake configuration not available for map display.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Historical catch timeline
# ---------------------------------------------------------------------------
st.subheader("Historical Catch Timeline")

if "datetime" in lake_df.columns and "catch_count" in lake_df.columns:
    daily = (
        lake_df.groupby(lake_df["datetime"].dt.date)
        .agg(
            catch_count=("catch_count", "sum"),
            trophy_count=("trophy_count", "sum") if "trophy_count" in lake_df.columns else ("catch_count", "count"),
            max_weight=("max_weight", "max") if "max_weight" in lake_df.columns else ("catch_count", "count"),
        )
        .reset_index()
    )
    daily.columns = ["date", "catch_count", "trophy_count", "max_weight"]
    daily["date"] = pd.to_datetime(daily["date"])

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(
        x=daily["date"], y=daily["catch_count"],
        name="Catches", marker_color=COLORS["water_blue"], opacity=0.6,
    ))
    fig_timeline.add_trace(go.Bar(
        x=daily["date"], y=daily["trophy_count"],
        name="Trophies", marker_color=COLORS["trophy_gold"],
    ))
    fig_timeline.update_layout(
        barmode="overlay",
        title=f"Daily Catches - {lake_name}",
        xaxis_title="Date",
        yaxis_title="Count",
    )
    st.plotly_chart(apply_figure_style(fig_timeline, height=400), use_container_width=True)
else:
    st.info("Timeline data not available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Lake-specific correlation analysis
# ---------------------------------------------------------------------------
st.subheader("Lake-Specific Feature Importance")

try:
    lake_importance = compute_feature_importance(lake_df)
    if not lake_importance.empty:
        fig_imp = px.bar(
            lake_importance.head(10),
            x="abs_correlation",
            y="feature",
            orientation="h",
            color="pearson_r",
            color_continuous_scale=["#e74c3c", "#95a5a6", COLORS["bass_green"]],
            color_continuous_midpoint=0,
            title=f"Top Features for {lake_name}",
            labels={"abs_correlation": "|Correlation|", "feature": "", "pearson_r": "Direction"},
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(apply_figure_style(fig_imp, height=400), use_container_width=True)
    else:
        st.info("Not enough data for lake-specific feature importance.")
except Exception as e:
    st.warning(f"Error computing lake correlations: {e}")

# ---------------------------------------------------------------------------
# Lake-specific monthly and hourly patterns
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Monthly Pattern")
    if "month" in lake_df.columns and "catch_count" in lake_df.columns:
        monthly = (
            lake_df.groupby("month")
            .agg(catches=("catch_count", "sum"), trophies=("trophy_count", "sum"))
            .reset_index()
        )
        month_names = {i: m for i, m in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
        monthly["month_name"] = monthly["month"].map(month_names)

        fig_m = px.bar(
            monthly, x="month_name", y="trophies",
            color_discrete_sequence=[COLORS["trophy_gold"]],
            title="Trophies by Month",
            labels={"month_name": "Month", "trophies": "Trophy Count"},
        )
        st.plotly_chart(apply_figure_style(fig_m, height=350), use_container_width=True)

with col_right:
    st.subheader("Hourly Pattern")
    if "hour" in lake_df.columns and "catch_count" in lake_df.columns:
        hourly = (
            lake_df.groupby("hour")
            .agg(catches=("catch_count", "sum"), trophies=("trophy_count", "sum"))
            .reset_index()
        )
        fig_h = px.bar(
            hourly, x="hour", y="trophies",
            color_discrete_sequence=[COLORS["bass_green"]],
            title="Trophies by Hour",
            labels={"hour": "Hour of Day", "trophies": "Trophy Count"},
        )
        st.plotly_chart(apply_figure_style(fig_h, height=350), use_container_width=True)
