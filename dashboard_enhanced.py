from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import streamlit as st

# Plotly is optional; the dashboard falls back to Streamlit charts if unavailable.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="Sensor Drift Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------- COLOUR THEME --------------------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F2FBFA 0%, #E8F7F5 55%, #F5FCFB 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #073B3A 0%, #087E78 62%, #18A999 100%);
}
[data-testid="stSidebar"] * { color: #F7FBFF !important; }
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    background: rgba(255,255,255,.08);
    border-radius: 10px; padding: 7px 10px; margin: 3px 0;
}
h1,h2,h3 { color: #073B3A !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#FFFFFF,#E4F5F2);
    border: 1px solid #C8E8E3; border-left: 5px solid #0F9D91;
    border-radius: 14px; padding: 14px;
    box-shadow: 0 5px 15px rgba(7,59,58,.08);
}
div[data-testid="stMetric"] label { color:#386C68 !important; font-weight:650; }
div[data-testid="stMetricValue"] { color:#073B3A !important; }
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(100deg,#087E78,#18A999);
    color:white; border:0; border-radius:10px; font-weight:650;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    color:white; border:0; background:linear-gradient(100deg,#066B66,#0A8F83);
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    background:white; border:1px solid #C8E8E3; border-radius:12px;
}
div[data-testid="stExpander"] {
    background:rgba(255,255,255,.85); border:1px solid #C8E8E3; border-radius:12px;
}
.section-tag {
    display:inline-block; background:#DDF5F0; color:#075E58;
    border:1px solid #B5E3DB; padding:5px 11px; border-radius:30px;
    font-size:.76rem; font-weight:750; letter-spacing:.08em;
}
</style>
""", unsafe_allow_html=True)

CHART_COLORS = ["#087E78", "#18A999", "#0F9D91", "#056F6C",
                "#42B7A5", "#2A8C82", "#65C9B8", "#145E63"]

# ---------------------------- Paths and loading ---------------------------- #
HERE = Path(__file__).resolve().parent

def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (
            (candidate / "gas+sensor+array+drift+dataset" / "Dataset").is_dir()
            or (candidate / "data" / "raw").is_dir()
            or (candidate / "outputs").is_dir()
        ):
            return candidate
    return start

ROOT = find_project_root(HERE)

def find_named_files(filename: str, start: Path):
    """Find likely generated files without searching virtual environments."""
    matches = []
    skip = {".venv", "venv", "site-packages", ".git", "__pycache__"}
    try:
        for p in start.rglob(filename):
            if any(part in skip for part in p.parts):
                continue
            if p.is_file():
                matches.append(p)
    except OSError:
        return []
    return matches


OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"


def read_csv(path: Path, **kwargs):
    try:
        if path.is_file():
            return pd.read_csv(path, **kwargs)
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read {path.name}: {exc}")
    return None


def show_table(filename: str, title: str | None = None, index_col=None):
    frame = read_csv(TABLE_DIR / filename, index_col=index_col)
    if frame is None:
        st.info(f"`{filename}` is not available yet. Run the analysis first.")
        return None
    if title:
        st.subheader(title)
    st.dataframe(frame, use_container_width=True, hide_index=index_col is None)
    return frame


def show_plot(filename: str, caption: str | None = None):
    path = PLOT_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption or filename, use_container_width=True)
    else:
        st.info(f"Plot `{filename}` is not available yet. Run the analysis first.")


def download_csv(path: Path, label: str):
    if path.is_file():
        st.download_button(
            label=f"Download {label}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
        )


with st.sidebar:
    st.header("Dashboard controls")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Data Quality",
            "Descriptive Statistics",
            "Correlation Analysis",
            "Hypothesis Testing",
            "Sensor Drift Ranking",
            "Batch Trends",
            "PCA Visualization",
            "Downloads",
        ],
    )
    st.divider()
    st.caption(f"Project folder: {ROOT}")
    if st.button("Refresh dashboard", use_container_width=True):
        st.rerun()

cleaned_path = OUTPUT_DIR / "cleaned_data.csv"
data = read_csv(cleaned_path)

st.title("Sensor Drift Intelligence")
st.caption("Interactive analysis of the UCI Gas Sensor Array Drift Dataset")

if page == "Overview":
    st.subheader("Project overview")
    if data is None:
        st.warning("The cleaned dataset has not been generated yet.")
        st.code('python ".\\data\\raw\\main.py\\main.py"', language="powershell")
    else:
        feature_count = len([c for c in data.columns if c.startswith("F") and c[1:].isdigit()])
        cards = st.columns(4)
        cards[0].metric("Total measurements", f"{len(data):,}")
        cards[1].metric("Batches", data["batch"].nunique() if "batch" in data else 0)
        cards[2].metric("Sensor features", feature_count)
        cards[3].metric("Gas classes", data["gas"].nunique() if "gas" in data else 0)
        if "batch" in data:
            st.subheader("Measurements by batch")
            st.bar_chart(data.groupby("batch").size().rename("Measurements"))
        if "gas" in data:
            st.subheader("Gas class distribution")
            st.bar_chart(data["gas"].value_counts().rename("Measurements"))
        ranking = show_table("sensor_drift_ranking.csv", "Top sensor drift results")
        if ranking is not None:
            st.dataframe(ranking.head(10), use_container_width=True, hide_index=True)
        show_plot("sensor_drift_ranking.png", "Sensors ranked by standardized drift effect")
elif page == "Data Quality":
    st.subheader("Data quality checks")
    if data is not None:
        cards = st.columns(3)
        cards[0].metric("Rows", f"{len(data):,}")
        cards[1].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
        cards[2].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    for filename, title, index_col in [
        ("missing_values.csv", "Missing-value report", 0),
        ("duplicate_report.csv", "Duplicate-row report", None),
        ("outlier_report.csv", "Outlier report", None),
        ("gas_frequency_by_batch.csv", "Gas frequency by batch", None),
    ]:
        show_table(filename, title, index_col=index_col)
    show_plot("scatter_by_batch.png", "Feature distribution across batches")
elif page == "Descriptive Statistics":
    st.subheader("Descriptive statistics")
    show_table("descriptive_statistics.csv", "Feature summary", index_col=0)
    show_table("excel_dashboard_summary.csv", "Compact dashboard summary")
    show_table("normality_tests.csv", "Normality tests")
    show_plot("F1_boxplot.png", "F1 boxplot")
elif page == "Correlation Analysis":
    st.subheader("Correlation analysis")
    show_plot("correlation_heatmap.png", "Correlation heatmap")
    show_table("correlation_matrix.csv", "Correlation matrix", index_col=0)
elif page == "Hypothesis Testing":
    st.subheader("Hypothesis testing: Batch 1 vs Batch 10")
    show_table("batch1_vs_batch10_tests.csv", "All feature-level tests")
    show_table("significant_drift_results.csv", "Statistically significant results")
elif page == "Sensor Drift Ranking":
    st.subheader("Sensor drift ranking")
    ranking = show_table("sensor_drift_ranking.csv", "Ranking table")
    show_plot("sensor_drift_ranking.png", "Mean absolute standardized effect by sensor")
    if ranking is not None and {"sensor", "mean_absolute_effect"}.issubset(ranking.columns):
        st.bar_chart(ranking.set_index("sensor")[["mean_absolute_effect"]])
elif page == "Batch Trends":
    st.subheader("Batch-to-batch trends")
    show_table("all_batch_comparisons.csv", "All batch comparisons")
    gases = ["ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene"]
    available = [gas for gas in gases if (PLOT_DIR / f"F1_trend_{gas}.png").is_file()]
    if available:
        gas = st.selectbox("Select gas", available)
        show_plot(f"F1_trend_{gas}.png", f"F1 response trend - {gas.title()}")
    else:
        st.info("No F1 trend plots are available yet.")
elif page == "PCA Visualization":
    st.subheader("Principal Component Analysis")
    show_plot("pca_by_batch.png", "PCA projection coloured by batch")
    show_table("pca_explained_variance.csv", "Explained variance")
    show_table("pca_coordinates.csv", "PCA coordinates")
else:
    st.subheader("Download analysis results")
    download_csv(cleaned_path, "cleaned dataset")
    for filename in [
        "descriptive_statistics.csv", "excel_dashboard_summary.csv",
        "correlation_matrix.csv", "batch1_vs_batch10_tests.csv",
        "significant_drift_results.csv", "all_batch_comparisons.csv",
        "sensor_drift_ranking.csv", "pca_coordinates.csv",
        "pca_explained_variance.csv", "missing_values.csv",
        "duplicate_report.csv", "outlier_report.csv",
        "gas_frequency_by_batch.csv", "normality_tests.csv",
    ]:
        download_csv(TABLE_DIR / filename, filename.removesuffix(".csv"))