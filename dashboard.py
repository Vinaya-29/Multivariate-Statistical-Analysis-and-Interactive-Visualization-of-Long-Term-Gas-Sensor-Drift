from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sensor Drift Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Locate the project root. Put this dashboard.py in the main "Data analytics" folder.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(
    (
        p for p in (SCRIPT_DIR, *SCRIPT_DIR.parents)
        if (p / "gas+sensor+array+drift+dataset" / "Dataset").is_dir()
        or (p / "outputs").is_dir()
        or (p / "data" / "raw").is_dir()
    ),
    SCRIPT_DIR,
)
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

st.title("📊 Statistical Analysis of Long-Term Sensor Drift")
st.caption("Interactive dashboard | UCI Gas Sensor Array Drift Dataset")

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
    st.write("**Project folder**")
    st.caption(str(ROOT))
    if st.button("🔄 Refresh dashboard", use_container_width=True):
        st.rerun()

def read_csv(path, **kwargs):
    try:
        if path.is_file():
            return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
    return None

def show_table(filename, title=None, index_col=None):
    df = read_csv(TABLE_DIR / filename, index_col=index_col)
    if df is None:
        st.info(f"`{filename}` is not available yet. Run the analysis first.")
        return None
    if title:
        st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=(index_col is None))
    return df

def show_plot(filename, caption=None):
    path = PLOT_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption or filename, use_container_width=True)
    else:
        st.info(f"Plot `{filename}` is not available yet. Run the analysis first.")

def download_csv(path, label):
    if path.is_file():
        st.download_button(
            label=f"⬇️ Download {label}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
        )

cleaned_path = OUTPUT_DIR / "cleaned_data.csv"
data = read_csv(cleaned_path)

if page == "Overview":
    st.subheader("Project overview")
    if data is not None:
        n_records = len(data)
        n_batches = data["batch"].nunique() if "batch" in data else 0
        n_features = len([c for c in data.columns if c.startswith("F") and c[1:].isdigit()])
        n_gases = data["gas"].nunique() if "gas" in data else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total measurements", f"{n_records:,}")
        c2.metric("Batches", n_batches)
        c3.metric("Sensor features", n_features)
        c4.metric("Gas classes", n_gases)

        if "batch" in data and "gas" in data:
            st.markdown("### Measurements by batch")
            counts = data.groupby("batch").size().rename("Measurements").reset_index()
            st.bar_chart(counts.set_index("batch"))
            st.markdown("### Gas class distribution")
            gas_counts = data["gas"].value_counts().rename_axis("Gas").to_frame("Measurements")
            st.bar_chart(gas_counts)
        st.markdown("### Sensor drift ranking")
        ranking = read_csv(TABLE_DIR / "sensor_drift_ranking.csv")
        if ranking is not None:
            st.dataframe(ranking.head(10), use_container_width=True, hide_index=True)
        show_plot("sensor_drift_ranking.png", "Sensors ranked by standardized drift effect")
    else:
        st.warning("The cleaned dataset has not been generated yet.")
        st.write("Run the analysis script first. The dashboard reads its generated files from the `outputs` folder.")
        st.code('python ".\\data\\raw\\main.py\\main.py"', language="powershell")

elif page == "Data Quality":
    st.subheader("Data quality checks")
    if data is not None:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(data):,}")
        b.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
        c.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    show_table("missing_values.csv", "Missing-value report", index_col=0)
    show_table("duplicate_report.csv", "Duplicate-row report")
    show_table("outlier_report.csv", "Outlier report")
    show_table("gas_frequency_by_batch.csv", "Gas frequency by batch")
    show_plot("scatter_by_batch.png", "Feature distribution across batches")

elif page == "Descriptive Statistics":
    st.subheader("Descriptive statistics")
    show_table("descriptive_statistics.csv", "Feature summary", index_col=0)
    show_table("excel_dashboard_summary.csv", "Compact dashboard summary")
    show_table("normality_tests.csv", "Normality tests")
    st.caption("Feature-level boxplots are saved in outputs/plots. Example:")
    show_plot("F1_boxplot.png", "F1 boxplot")

elif page == "Correlation Analysis":
    st.subheader("Correlation analysis")
    show_plot("correlation_heatmap.png", "Correlation heatmap")
    show_table("correlation_matrix.csv", "Correlation matrix", index_col=0)

elif page == "Hypothesis Testing":
    st.subheader("Hypothesis testing: Batch 1 vs Batch 10")
    show_table("batch1_vs_batch10_tests.csv", "All feature-level tests")
    show_table("significant_drift_results.csv", "Statistically significant results (FDR < 0.05)")

elif page == "Sensor Drift Ranking":
    st.subheader("Sensor drift ranking")
    ranking = show_table("sensor_drift_ranking.csv", "Ranking table")
    show_plot("sensor_drift_ranking.png", "Mean absolute standardized effect by sensor")
    if ranking is not None and "sensor" in ranking.columns and "mean_absolute_effect" in ranking.columns:
        st.markdown("### Compare sensor effects")
        st.bar_chart(ranking.set_index("sensor")[["mean_absolute_effect"]])

elif page == "Batch Trends":
    st.subheader("Batch-to-batch trends")
    show_table("all_batch_comparisons.csv", "All batch comparisons")
    st.markdown("### F1 trend charts by gas")
    gases = ["ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene"]
    available = [g for g in gases if (PLOT_DIR / f"F1_trend_{g}.png").is_file()]
    if available:
        gas_choice = st.selectbox("Select gas", available)
        show_plot(f"F1_trend_{gas_choice}.png", f"F1 response trend — {gas_choice.title()}")
    else:
        st.info("No F1 trend plots are available yet.")

elif page == "PCA Visualization":
    st.subheader("Principal Component Analysis")
    show_plot("pca_by_batch.png", "PCA projection coloured by batch")
    show_table("pca_explained_variance.csv", "Explained variance")
    show_table("pca_coordinates.csv", "PCA coordinates")

elif page == "Downloads":
    st.subheader("Download analysis results")
    st.write("Download generated CSV tables and the cleaned dataset.")
    download_csv(cleaned_path, "cleaned dataset")
    for filename, label in [
        ("descriptive_statistics.csv", "descriptive statistics"),
        ("excel_dashboard_summary.csv", "summary table"),
        ("correlation_matrix.csv", "correlation matrix"),
        ("batch1_vs_batch10_tests.csv", "hypothesis test results"),
        ("significant_drift_results.csv", "significant drift results"),
        ("all_batch_comparisons.csv", "batch comparisons"),
        ("sensor_drift_ranking.csv", "sensor ranking"),
        ("pca_coordinates.csv", "PCA coordinates"),
        ("pca_explained_variance.csv", "PCA explained variance"),
        ("missing_values.csv", "missing values report"),
        ("duplicate_report.csv", "duplicate report"),
        ("outlier_report.csv", "outlier report"),
        ("gas_frequency_by_batch.csv", "gas frequency by batch"),
        ("normality_tests.csv", "normality test results"),
    ]:
        download_csv(TABLE_DIR / filename, label)

st.divider()
st.caption("Dashboard displays results generated by main.py. If a table or chart is missing, run the analysis script successfully and then refresh this page.")
