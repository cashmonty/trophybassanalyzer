"""Shared Streamlit UI helpers for the trophy bass dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.parquet as pq
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

REQUIRED_MERGED_COLUMNS = {"datetime", "lake_key"}
REQUIRED_CATCH_COLUMNS = {"date", "lake_key", "weight_lbs"}
REQUIRED_PREDICTION_COLUMNS = {"date", "lake_key", "max_probability", "rating"}

DASHBOARD_MERGED_COLUMNS = (
    "datetime",
    "date",
    "lake_key",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "moon_illumination",
    "moon_phase_name",
    "solunar_base_score",
    "hour",
    "month",
    "day_of_year",
    "is_weekend",
    "water_temp_estimated",
    "pressure_trend_3h",
    "pressure_trend_6h",
    "front_type",
    "temp_stability_3day",
    "wind_class",
    "temp_change_3day",
    "is_warming_trend",
    "hours_to_front",
    "prefrontal_feed_window",
    "catch_count",
    "max_weight",
    "avg_weight",
    "trophy_count",
    "super_trophy_count",
    "trophy_caught",
    "spawn_phase",
)

DASHBOARD_CATCH_COLUMNS = (
    "date",
    "lake_key",
    "weight_lbs",
    "length_in",
    "angler",
    "place",
    "is_trophy",
    "is_super_trophy",
    "source_type",
    "season_start_year",
)

DASHBOARD_PREDICTION_COLUMNS = (
    "date",
    "lake_key",
    "max_score",
    "mean_score",
    "best_hour",
    "max_probability",
    "mean_probability",
    "moon_phase_name",
    "rating",
)

CATEGORY_COLUMNS = {
    "lake_key",
    "moon_phase_name",
    "front_type",
    "wind_class",
    "spawn_phase",
    "angler",
    "source_type",
    "rating",
}

BOOLEAN_COLUMNS = {"is_weekend", "is_trophy", "is_super_trophy"}


class DashboardDataError(RuntimeError):
    """Raised when dashboard assets are missing or invalid."""


@dataclass
class DashboardContext:
    merged_df: pd.DataFrame
    catches_df: pd.DataFrame
    predictions_df: pd.DataFrame | None
    lake_configs: dict[str, LakeConfig]
    settings: Settings
    selected_lakes: list[str]
    colors: dict[str, str]


def _read_parquet_checked(
    path: Path, label: str, *, columns: tuple[str, ...] | list[str] | None = None
) -> pd.DataFrame:
    """Read a parquet file and convert low-level failures into dashboard-facing errors."""
    if not path.exists():
        raise DashboardDataError(f"Missing required dashboard data file: `{path.as_posix()}`.")
    selected_columns = None
    if columns is not None:
        try:
            available_columns = set(pq.ParquetFile(path).schema.names)
        except Exception as exc:  # pragma: no cover - depends on local parquet engine/runtime
            raise DashboardDataError(
                f"Unable to inspect columns for {label} from `{path.as_posix()}`. {exc}"
            ) from exc
        selected_columns = [column for column in columns if column in available_columns]
    try:
        return pd.read_parquet(path, columns=selected_columns)
    except Exception as exc:  # pragma: no cover - depends on local parquet engine/runtime
        raise DashboardDataError(
            f"Unable to read {label} from `{path.as_posix()}`. {exc}"
        ) from exc


def _require_columns(df: pd.DataFrame, label: str, required: set[str]) -> None:
    """Validate that a dataframe contains the columns the dashboard expects."""
    missing = sorted(required - set(df.columns))
    if missing:
        joined = ", ".join(missing)
        raise DashboardDataError(f"{label} is missing required columns: {joined}.")


def _optimize_dashboard_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink dashboard dataframes so Streamlit sessions stay within memory limits."""
    for column in ("datetime", "date"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])

    for column in CATEGORY_COLUMNS.intersection(df.columns):
        df[column] = df[column].astype("category")

    for column in BOOLEAN_COLUMNS.intersection(df.columns):
        if pd.api.types.is_bool_dtype(df[column]):
            continue
        if df[column].dropna().isin([0, 1, True, False]).all():
            df[column] = df[column].astype("bool")

    float_columns = df.select_dtypes(include=["float64"]).columns
    for column in float_columns:
        df[column] = pd.to_numeric(df[column], downcast="float")

    integer_columns = df.select_dtypes(include=["int64", "int32"]).columns
    for column in integer_columns:
        df[column] = pd.to_numeric(df[column], downcast="integer")

    return df


@st.cache_resource(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the main historical merged and catch datasets."""
    merged = _read_parquet_checked(
        DATA_DIR / "processed" / "merged.parquet",
        "merged history",
        columns=DASHBOARD_MERGED_COLUMNS,
    )
    _require_columns(merged, "Merged history", REQUIRED_MERGED_COLUMNS)
    merged = _optimize_dashboard_frame(merged)
    if "date" not in merged.columns:
        merged["date"] = merged["datetime"].dt.normalize()

    catches = _read_parquet_checked(
        DATA_DIR / "processed" / "catches.parquet",
        "catch ledger",
        columns=DASHBOARD_CATCH_COLUMNS,
    )
    _require_columns(catches, "Catch ledger", REQUIRED_CATCH_COLUMNS)
    catches["date"] = pd.to_datetime(catches["date"])
    catches["weight_lbs"] = pd.to_numeric(catches["weight_lbs"], errors="coerce")
    catches = catches[catches["weight_lbs"].between(0, 20, inclusive="both")]
    catches = _optimize_dashboard_frame(catches)

    return merged, catches


@st.cache_resource(show_spinner=False)
def load_dashboard_predictions() -> pd.DataFrame | None:
    """Load the generated 2026 daily predictions when available."""
    path = DATA_DIR / "processed" / "predictions_2026.parquet"
    if not path.exists():
        return None

    df = _read_parquet_checked(path, "2026 prediction windows", columns=DASHBOARD_PREDICTION_COLUMNS)
    _require_columns(df, "2026 prediction windows", REQUIRED_PREDICTION_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return _optimize_dashboard_frame(df)


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


def render_plotly(fig: go.Figure, *, height: int | None = None) -> None:
    """Render a Plotly figure with the dashboard defaults."""
    styled = apply_figure_style(fig, height=height)
    try:
        st.plotly_chart(styled, width="stretch")
    except TypeError:  # pragma: no cover - compatibility with older Streamlit releases
        st.plotly_chart(styled, use_container_width=True)


def render_dataframe(
    data: pd.DataFrame | pd.io.formats.style.Styler, *, hide_index: bool = False
) -> None:
    """Render a dataframe with the dashboard defaults."""
    try:
        st.dataframe(data, width="stretch", hide_index=hide_index)
    except TypeError:  # pragma: no cover - compatibility with older Streamlit releases
        st.dataframe(data, use_container_width=True)


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

    if not selected_lakes:
        return catches_df.copy()

    if set(selected_lakes) == set(catches_df["lake_key"].dropna().unique()) and include_multi:
        return catches_df

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


def _filter_merged(merged_df: pd.DataFrame, selected_lakes: list[str]) -> pd.DataFrame:
    if "lake_key" not in merged_df.columns or not selected_lakes:
        return merged_df.copy()

    all_lakes = set(merged_df["lake_key"].dropna().unique())
    if set(selected_lakes) == all_lakes:
        return merged_df

    return merged_df[merged_df["lake_key"].isin(selected_lakes)].copy()


def _filter_predictions(
    predictions_df: pd.DataFrame | None, selected_lakes: list[str]
) -> pd.DataFrame | None:
    if predictions_df is None:
        return None
    if "lake_key" not in predictions_df.columns or not selected_lakes:
        return predictions_df.copy()

    all_lakes = set(predictions_df["lake_key"].dropna().unique())
    if set(selected_lakes) == all_lakes:
        return predictions_df

    return predictions_df[predictions_df["lake_key"].isin(selected_lakes)].copy()


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

    try:
        merged_df, catches_df = load_dashboard_data()
        predictions_df = load_dashboard_predictions()
        settings = load_settings()
        lake_configs = {lake.key: lake for lake in load_lakes()}
    except DashboardDataError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # pragma: no cover - defensive UI stop
        st.error(f"Dashboard bootstrap failed: {exc}")
        st.stop()

    selected_lakes = _render_sidebar(merged_df, catches_df, predictions_df, lake_configs)
    include_multi = set(selected_lakes) == set(merged_df["lake_key"].dropna().unique())

    filtered_merged = _filter_merged(merged_df, selected_lakes)
    filtered_catches = _filter_catches(catches_df, selected_lakes, include_multi)
    colors = {
        "bass_green": PALETTE["olive"],
        "water_blue": PALETTE["water"],
        "trophy_gold": PALETTE["gold"],
        "clay": PALETTE["clay"],
        "moss": PALETTE["moss"],
        "fog": PALETTE["fog"],
    }
    filtered_predictions = _filter_predictions(predictions_df, selected_lakes)

    return DashboardContext(
        merged_df=filtered_merged,
        catches_df=filtered_catches,
        predictions_df=filtered_predictions,
        lake_configs=lake_configs,
        settings=settings,
        selected_lakes=selected_lakes,
        colors=colors,
    )
