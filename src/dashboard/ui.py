"""Shared Streamlit UI helpers for the trophy bass dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.config import DATA_DIR, LakeConfig, Settings, load_lakes, load_settings

PALETTE = {
    "ink": "#17231C",
    "olive": "#2F4F34",
    "pine": "#244028",
    "moss": "#6F8A48",
    "gold": "#C7903D",
    "sand": "#EFE6D6",
    "cream": "#F8F4EC",
    "water": "#2E5F74",
    "sky": "#A8C7D8",
    "clay": "#A85A3D",
    "fog": "#D8D3C9",
    "white": "#FFFDFC",
}

PLOTLY_TEMPLATE = "trophy_bass"

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

SPECIAL_LAKE_LABELS = {
    "multi": "Multi-Lake Events",
}


@dataclass
class DashboardContext:
    merged_df: pd.DataFrame
    catches_df: pd.DataFrame
    predictions_df: pd.DataFrame | None
    lake_configs: dict[str, LakeConfig]
    settings: Settings
    selected_lakes: list[str]
    colors: dict[str, str]


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the main historical merged and catch datasets."""
    merged = pd.read_parquet(DATA_DIR / "processed" / "merged.parquet").copy()
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])

    catches = pd.read_parquet(DATA_DIR / "processed" / "catches.parquet").copy()
    catches["date"] = pd.to_datetime(catches["date"])
    catches = catches[catches["weight_lbs"].between(0, 20, inclusive="both")]

    return merged, catches


@st.cache_data(show_spinner=False)
def load_dashboard_predictions() -> pd.DataFrame | None:
    """Load the generated 2026 daily predictions when available."""
    path = DATA_DIR / "processed" / "predictions_2026.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def lake_label(lake_key: str, lake_configs: dict[str, LakeConfig]) -> str:
    """Return a presentable label for a lake key."""
    if lake_key in SPECIAL_LAKE_LABELS:
        return SPECIAL_LAKE_LABELS[lake_key]
    if lake_key in lake_configs:
        return lake_configs[lake_key].name
    return str(lake_key).replace("_", " ").title()


def apply_figure_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply the shared plot styling to a Plotly figure."""
    _register_plotly_template()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": '"Avenir Next", "Segoe UI", sans-serif', "color": PALETTE["ink"]},
        margin={"t": 40, "r": 24, "b": 36, "l": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    if height:
        fig.update_layout(height=height)
    return fig


def render_page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    """Render a themed page header."""
    eyebrow_html = (
        f"<div class='tb-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    )
    st.markdown(
        f"""
        <section class="tb-hero">
            {eyebrow_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _register_plotly_template() -> None:
    """Register a shared Plotly template once per session."""
    if PLOTLY_TEMPLATE not in pio.templates:
        pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(
            layout=go.Layout(
                colorway=[
                    PALETTE["olive"],
                    PALETTE["water"],
                    PALETTE["gold"],
                    PALETTE["clay"],
                    PALETTE["moss"],
                    PALETTE["sky"],
                ],
                font={"family": '"Avenir Next", "Segoe UI", sans-serif', "color": PALETTE["ink"]},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title={"font": {"family": "Rockwell, Georgia, serif", "size": 20}},
                xaxis={
                    "showgrid": True,
                    "gridcolor": "rgba(23,35,28,0.08)",
                    "zeroline": False,
                    "linecolor": "rgba(23,35,28,0.10)",
                },
                yaxis={
                    "showgrid": True,
                    "gridcolor": "rgba(23,35,28,0.08)",
                    "zeroline": False,
                    "linecolor": "rgba(23,35,28,0.10)",
                },
            )
        )
    pio.templates.default = PLOTLY_TEMPLATE
    px.defaults.template = PLOTLY_TEMPLATE


def _inject_global_styles() -> None:
    """Inject the dashboard stylesheet."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(199, 144, 61, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(46, 95, 116, 0.12), transparent 28%),
                linear-gradient(180deg, {PALETTE["cream"]} 0%, {PALETTE["sand"]} 54%, #f6efe1 100%);
            color: {PALETTE["ink"]};
        }}
        [data-testid="stAppViewContainer"] > .main {{
            background: transparent;
        }}
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}
        h1, h2, h3 {{
            color: {PALETTE["ink"]};
            font-family: Rockwell, Georgia, serif;
            letter-spacing: 0.02em;
        }}
        p, li, div, label, span {{
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(23,35,28,0.98), rgba(36,64,40,0.94));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        [data-testid="stSidebar"] * {{
            color: #f7f3eb;
        }}
        [data-testid="stSidebar"] .stMultiSelect,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stSelectbox {{
            background: rgba(255,255,255,0.04);
            border-radius: 14px;
            padding: 0.25rem 0.4rem 0.5rem 0.4rem;
        }}
        [data-testid="stMetric"] {{
            background: rgba(255, 253, 252, 0.74);
            border: 1px solid rgba(23, 35, 28, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 50px rgba(23, 35, 28, 0.08);
        }}
        [data-testid="stMetricValue"] {{
            font-family: Rockwell, Georgia, serif;
        }}
        .tb-hero {{
            background:
                linear-gradient(135deg, rgba(23,35,28,0.96) 0%, rgba(46,95,116,0.94) 64%, rgba(199,144,61,0.82) 100%);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 28px;
            padding: 1.7rem 1.8rem 1.5rem 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 26px 60px rgba(23, 35, 28, 0.18);
            position: relative;
            overflow: hidden;
        }}
        .tb-hero::after {{
            content: "";
            position: absolute;
            inset: auto -12% -40% auto;
            width: 240px;
            height: 240px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            filter: blur(4px);
        }}
        .tb-hero h1, .tb-hero p, .tb-hero .tb-eyebrow {{
            position: relative;
            z-index: 2;
        }}
        .tb-hero h1 {{
            color: #fffdf7;
            margin: 0 0 0.4rem 0;
            font-size: 2.3rem;
        }}
        .tb-hero p {{
            color: rgba(255, 250, 242, 0.9);
            margin: 0;
            max-width: 58rem;
            font-size: 1.02rem;
            line-height: 1.5;
        }}
        .tb-eyebrow {{
            display: inline-block;
            margin-bottom: 0.75rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            color: #fff7e6;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .tb-note {{
            background: rgba(255, 253, 252, 0.7);
            border: 1px solid rgba(23, 35, 28, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }}
        .tb-note strong {{
            display: block;
            margin-bottom: 0.35rem;
            font-family: Rockwell, Georgia, serif;
        }}
        .stPlotlyChart,
        [data-testid="stDataFrame"],
        [data-testid="stExpander"] {{
            background: rgba(255, 253, 252, 0.7);
            border: 1px solid rgba(23, 35, 28, 0.08);
            border-radius: 20px;
            padding: 0.6rem;
            box-shadow: 0 18px 40px rgba(23, 35, 28, 0.05);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            background: rgba(255, 253, 252, 0.62);
            padding: 0.4rem 0.85rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PALETTE["olive"]};
            color: #fff9ef;
        }}
        .tb-sidebar-title {{
            font-family: Rockwell, Georgia, serif;
            font-size: 1.05rem;
            margin-bottom: 0.2rem;
        }}
        .tb-sidebar-copy {{
            color: rgba(247,243,235,0.78);
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _filter_catches(
    catches_df: pd.DataFrame, selected_lakes: list[str], include_multi: bool
) -> pd.DataFrame:
    if "lake_key" not in catches_df.columns:
        return catches_df.copy()

    mask = catches_df["lake_key"].isin(selected_lakes)
    if include_multi:
        mask |= catches_df["lake_key"].eq("multi")
    return catches_df[mask].copy()


def _render_sidebar(
    merged_df: pd.DataFrame,
    catches_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    lake_configs: dict[str, LakeConfig],
) -> list[str]:
    available_lakes = sorted(merged_df["lake_key"].dropna().unique().tolist())
    default_lakes = st.session_state.get("selected_lakes", available_lakes)
    default_lakes = [lake for lake in default_lakes if lake in available_lakes] or available_lakes

    st.sidebar.markdown("<div class='tb-sidebar-title'>Pattern Control</div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='tb-sidebar-copy'>Dial the lakes first. Every page below respects the same selection.</div>",
        unsafe_allow_html=True,
    )

    selected_lakes = st.sidebar.multiselect(
        "Lakes",
        options=available_lakes,
        default=default_lakes,
        format_func=lambda key: lake_label(key, lake_configs),
    )
    if not selected_lakes:
        selected_lakes = available_lakes

    st.session_state["selected_lakes"] = selected_lakes

    include_multi = len(selected_lakes) == len(available_lakes)
    filtered_catches = _filter_catches(catches_df, selected_lakes, include_multi)

    trophy_count = int((filtered_catches["weight_lbs"] >= 7).sum()) if not filtered_catches.empty else 0
    st.sidebar.metric("Historical trophies", trophy_count)
    st.sidebar.metric("Selected lakes", len(selected_lakes))

    if predictions_df is not None and not predictions_df.empty:
        future = predictions_df[
            predictions_df["lake_key"].isin(selected_lakes)
            & (predictions_df["date"] >= pd.Timestamp.today().normalize())
        ].copy()
        if not future.empty:
            best = future.nlargest(1, "max_probability").iloc[0]
            st.sidebar.markdown(
                f"""
                <div class="tb-note">
                    <strong>Next strong window</strong>
                    {best['date']:%b %d, %Y}<br>
                    {lake_label(best['lake_key'], lake_configs)}<br>
                    {best['max_probability']:.0%} odds, {best['rating']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    return selected_lakes


def bootstrap_dashboard(page_title: str) -> DashboardContext:
    """Configure the current page and expose filtered dashboard state."""
    st.set_page_config(
        page_title=page_title,
        page_icon=":fish:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _register_plotly_template()
    _inject_global_styles()

    merged_df, catches_df = load_dashboard_data()
    predictions_df = load_dashboard_predictions()
    settings = load_settings()
    lake_configs = {lake.key: lake for lake in load_lakes()}

    selected_lakes = _render_sidebar(merged_df, catches_df, predictions_df, lake_configs)
    include_multi = len(selected_lakes) == merged_df["lake_key"].nunique()

    filtered_merged = merged_df[merged_df["lake_key"].isin(selected_lakes)].copy()
    filtered_catches = _filter_catches(catches_df, selected_lakes, include_multi)
    filtered_predictions = None
    if predictions_df is not None:
        filtered_predictions = predictions_df[predictions_df["lake_key"].isin(selected_lakes)].copy()

    st.session_state["colors"] = {
        "bass_green": PALETTE["olive"],
        "water_blue": PALETTE["water"],
        "trophy_gold": PALETTE["gold"],
        "clay": PALETTE["clay"],
        "moss": PALETTE["moss"],
        "fog": PALETTE["fog"],
    }
    st.session_state["lake_configs"] = lake_configs
    st.session_state["merged_df"] = filtered_merged
    st.session_state["catches_df"] = filtered_catches
    st.session_state["predictions_df"] = filtered_predictions
    st.session_state["settings"] = settings

    return DashboardContext(
        merged_df=filtered_merged,
        catches_df=filtered_catches,
        predictions_df=filtered_predictions,
        lake_configs=lake_configs,
        settings=settings,
        selected_lakes=selected_lakes,
        colors=st.session_state["colors"],
    )
