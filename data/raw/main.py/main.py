

# ==============================================================

# PROJECT: Statistical Analysis of Long-Term Sensor Drift

# Dataset: UCI Gas Sensor Array Drift Dataset

# IDE: Visual Studio Code

# ==============================================================



from pathlib import Path

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.datasets import load_svmlight_file

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy import stats

from statsmodels.stats.multitest import multipletests



warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")



# ==============================================================

# 1. CONFIGURATION

# ==============================================================



SCRIPT_DIR = Path(__file__).resolve().parent

# Find the project root even when this script is nested in data/raw/main.py/.
ROOT = next(
    (
        parent
        for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
        if (parent / "gas+sensor+array+drift+dataset" / "Dataset").is_dir()
        or (parent / "data" / "raw").is_dir()
    ),
    SCRIPT_DIR,
)

DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"


for folder in [PLOT_DIR, TABLE_DIR]:

    folder.mkdir(parents=True, exist_ok=True)



N_FEATURES = 128

FEATURE_COLS = [f"F{i}" for i in range(1, N_FEATURES + 1)]



# Verify these labels against the dataset documentation.

GAS_NAMES = {

    1: "Ethanol",

    2: "Ethylene",

    3: "Ammonia",

    4: "Acetaldehyde",

    5: "Acetone",

    6: "Toluene",

}





# ==============================================================

# 2. FIND AND LOAD DATASET

# ==============================================================



def find_data_directory():
    """Find all ten batch files in common project/dataset locations."""
    expected = [f"batch{i}.dat" for i in range(1, 11)]

    candidates = [
        DATA_DIR,
        ROOT / "gas+sensor+array+drift+dataset" / "Dataset",
        ROOT / "gas+sensor+array+drift+dataset",
        ROOT / "Dataset",
        SCRIPT_DIR / "Dataset",
    ]

    # Also inspect folders beneath the project root, without scanning the
    # virtual environment or generated output directory.
    for parent in (ROOT, *ROOT.parents[:1]):
        dataset_root = parent / "gas+sensor+array+drift+dataset"
        candidates.extend([dataset_root / "Dataset", dataset_root])

    checked = set()
    for folder in candidates:
        folder = folder.resolve()
        if folder in checked:
            continue
        checked.add(folder)
        if all((folder / name).is_file() for name in expected):
            print(f"Dataset found: {folder}")
            return folder

    raise FileNotFoundError(
        "Could not find all 10 batch files (batch1.dat to batch10.dat).\n"
        "Expected them in one of these locations:\n"
        f"  {DATA_DIR}\n"
        f"  {ROOT / 'gas+sensor+array+drift+dataset' / 'Dataset'}\n"
        "Move/extract the Dataset folder into the project root, or copy all "
        "10 batch files into data/raw."
    )



def load_dataset():

    data_dir = find_data_directory()

    frames = []



    print("\nDATASET DIRECTORY:", data_dir)

    print("\nLoading batch files...\n")



    for batch_no in range(1, 11):

        file_path = data_dir / f"batch{batch_no}.dat"



        # The dataset uses one-based feature indices (1 to 128).

        X, y = load_svmlight_file(

            str(file_path),

            n_features=N_FEATURES,

            zero_based=False

        )



        X = X.toarray()



        if X.shape[1] != N_FEATURES:

            raise ValueError(

                f"{file_path.name}: expected {N_FEATURES} features, "

                f"found {X.shape[1]}."

            )



        df = pd.DataFrame(X, columns=FEATURE_COLS)

        df["gas_id"] = y.astype(int)

        df["gas"] = df["gas_id"].map(GAS_NAMES)

        df["batch"] = batch_no



        # Sensor 1 = F1-F8, sensor 2 = F9-F16, etc.

        frames.append(df)



        print(f"Batch {batch_no}: {len(df):,} measurements")



    data = pd.concat(frames, ignore_index=True)



    if data["gas"].isna().any():

        unknown = sorted(

            data.loc[data["gas"].isna(), "gas_id"].unique()

        )

        raise ValueError(

            f"Unrecognized gas labels: {unknown}. "

            "Check GAS_NAMES against the dataset documentation."

        )



    data["batch"] = data["batch"].astype(int)



    print("\nTotal measurements:", f"{len(data):,}")

    print("Feature count:", len(FEATURE_COLS))

    print("Gas classes:", sorted(data["gas"].unique()))



    data.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

    return data





# ==============================================================

# 3. DATA QUALITY CHECKS

# ==============================================================



def quality_analysis(data):

    X = data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)



    # Missing and invalid values

    missing = X.isna().sum().rename("missing_count")

    missing.to_csv(TABLE_DIR / "missing_values.csv")



    # Duplicate measurements and full rows

    duplicates = pd.DataFrame({

        "check": [

            "Duplicate feature vectors",

            "Duplicate complete rows"

        ],

        "count": [

            int(data[FEATURE_COLS].duplicated().sum()),

            int(data.duplicated().sum())

        ]

    })

    duplicates.to_csv(

        TABLE_DIR / "duplicate_report.csv", index=False

    )



    # IQR outlier report

    outlier_records = []



    for col in FEATURE_COLS:

        values = X[col].dropna()



        q1 = values.quantile(0.25)

        q3 = values.quantile(0.75)

        iqr = q3 - q1



        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        count = int(((values < lower) | (values > upper)).sum())



        outlier_records.append({

            "feature": col,

            "lower_bound": lower,

            "upper_bound": upper,

            "outlier_count": count,

            "outlier_percentage": 100 * count / max(len(values), 1)

        })



    pd.DataFrame(outlier_records).to_csv(

        TABLE_DIR / "outlier_report.csv", index=False

    )



    # Sample count by gas and batch

    counts = (

        data.groupby(["batch", "gas"])

        .size()

        .rename("sample_count")

        .reset_index()

    )

    counts.to_csv(

        TABLE_DIR / "gas_frequency_by_batch.csv", index=False

    )



    print("\nData quality checks completed.")

    print("Missing feature values:", int(X.isna().sum().sum()))

    print("Duplicate rows:", int(data.duplicated().sum()))

    print("Outliers reported; they have NOT been automatically deleted.")





# ==============================================================

# 4. DESCRIPTIVE STATISTICS

# ==============================================================



def descriptive_analysis(data):

    # Statistics for every feature, gas and batch

    summary = (

        data.groupby(["batch", "gas"])[FEATURE_COLS]

        .agg(["mean", "median", "std", "count"])

    )

    summary.to_csv(TABLE_DIR / "descriptive_statistics.csv")



    # Compact summary for Excel

    compact = (

        data.groupby(["batch", "gas"])

        .agg(

            sample_count=("gas_id", "size"),

            mean_F1=("F1", "mean"),

            mean_F9=("F9", "mean"),

            mean_F17=("F17", "mean"),

            mean_F25=("F25", "mean")

        )

        .reset_index()

    )

    compact.to_csv(

        TABLE_DIR / "excel_dashboard_summary.csv", index=False

    )



    # Representative feature distributions

    for feature in ["F1", "F9", "F17", "F25"]:

        plt.figure(figsize=(10, 5))

        sns.boxplot(data=data, x="batch", y=feature)

        plt.title(f"{feature}: Response Distribution by Batch")

        plt.xlabel("Batch")

        plt.ylabel("Feature response")

        plt.tight_layout()

        plt.savefig(

            PLOT_DIR / f"{feature}_boxplot.png", dpi=250

        )

        plt.close()



    # Shapiro-Wilk normality test on a sample of F1 values

    # per gas and batch. Large datasets can make this test

    # excessively sensitive to small departures from normality.

    normality = []



    for (batch, gas), group in data.groupby(["batch", "gas"]):

        values = group["F1"].dropna()



        if len(values) < 3:

            continue



        sample = values.sample(

            n=min(len(values), 5000),

            random_state=42

        )



        test = stats.shapiro(sample)



        normality.append({

            "batch": batch,

            "gas": gas,

            "feature": "F1",

            "sample_size": len(sample),

            "statistic": test.statistic,

            "p_value": test.pvalue

        })



    pd.DataFrame(normality).to_csv(

        TABLE_DIR / "normality_tests.csv", index=False

    )



    print("Descriptive statistics completed.")





# ==============================================================

# 5. CORRELATION ANALYSIS

# ==============================================================



def correlation_analysis(data):

    # Sample rows for visualization

    sample = data.sample(

        n=min(len(data), 3000),

        random_state=42

    )



    correlation = sample[FEATURE_COLS].corr()

    correlation.to_csv(TABLE_DIR / "correlation_matrix.csv")



    plt.figure(figsize=(14, 12))

    sns.heatmap(

        correlation,

        cmap="coolwarm",

        center=0,

        xticklabels=False,

        yticklabels=False

    )

    plt.title("Correlation Heatmap: 128 Features")

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "correlation_heatmap.png", dpi=250

    )

    plt.close()



    # Scatter plot

    plt.figure(figsize=(9, 6))

    sns.scatterplot(

        data=sample,

        x="F1",

        y="F9",

        hue="batch",

        palette="viridis",

        alpha=0.65

    )

    plt.title("Feature F1 vs F9 Across Batches")

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "scatter_by_batch.png", dpi=250

    )

    plt.close()



    print("Correlation analysis completed.")





# ==============================================================

# 6. WELCH'S T-TEST AND CONFIDENCE INTERVALS

# ==============================================================



def compare_two_groups(x, y):

    """Return mean difference, Welch test, and 95% CI.



    Difference is defined as mean(y) - mean(x).

    """



    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)



    x = x[np.isfinite(x)]

    y = y[np.isfinite(y)]



    if len(x) < 2 or len(y) < 2:

        return None



    mean_diff = float(np.mean(y) - np.mean(x))

    vx = float(np.var(x, ddof=1))

    vy = float(np.var(y, ddof=1))



    se2 = vx / len(x) + vy / len(y)



    test = stats.ttest_ind(y, x, equal_var=False)



    if se2 > 0:

        dof = se2**2 / (

            (vx / len(x))**2 / (len(x) - 1)

            + (vy / len(y))**2 / (len(y) - 1)

        )



        margin = stats.t.ppf(0.975, dof) * np.sqrt(se2)

    else:

        margin = 0.0



    pooled_sd = np.sqrt((vx + vy) / 2)

    effect = mean_diff / pooled_sd if pooled_sd > 0 else np.nan



    return {

        "n_early": len(x),

        "n_late": len(y),

        "mean_early": float(np.mean(x)),

        "mean_late": float(np.mean(y)),

        "mean_difference": mean_diff,

        "ci_95_lower": mean_diff - margin,

        "ci_95_upper": mean_diff + margin,

        "welch_t": float(test.statistic),

        "p_value": float(test.pvalue),

        "standardized_effect": effect

    }





def hypothesis_testing(data):

    records = []



    for gas in GAS_NAMES.values():

        early = data[

            (data["batch"] == 1) & (data["gas"] == gas)

        ]

        late = data[

            (data["batch"] == 10) & (data["gas"] == gas)

        ]



        for feature in FEATURE_COLS:

            result = compare_two_groups(

                early[feature].dropna(),

                late[feature].dropna()

            )



            if result is None:

                continue



            result.update({

                "gas": gas,

                "feature": feature

            })

            records.append(result)



    results = pd.DataFrame(records)



    if results.empty:

        raise ValueError("No valid groups for hypothesis testing.")



    # Control false discovery rate across all gas-feature tests.

    results["p_adjusted_BH"] = multipletests(

        results["p_value"].fillna(1.0),

        method="fdr_bh"

    )[1]



    results["significant_FDR_0_05"] = (

        results["p_adjusted_BH"] < 0.05

    )



    results.to_csv(

        TABLE_DIR / "batch1_vs_batch10_tests.csv", index=False

    )



    results[results["significant_FDR_0_05"]].to_csv(

        TABLE_DIR / "significant_drift_results.csv",

        index=False

    )



    print("\nHypothesis testing completed.")

    print("Tests performed:", len(results))

    print(

        "Significant results after FDR correction:",

        int(results["significant_FDR_0_05"].sum())

    )



    return results





# ==============================================================

# 7. COMPARE BATCH 1 WITH EVERY LATER BATCH

# ==============================================================



def batch_trend_analysis(data):

    records = []



    for gas in GAS_NAMES.values():

        baseline = data[

            (data["batch"] == 1) & (data["gas"] == gas)

        ]



        for batch in range(2, 11):

            later = data[

                (data["batch"] == batch) & (data["gas"] == gas)

            ]



            for feature in FEATURE_COLS:

                result = compare_two_groups(

                    baseline[feature].dropna(),

                    later[feature].dropna()

                )



                if result is None:

                    continue



                records.append({

                    "gas": gas,

                    "batch": batch,

                    "feature": feature,

                    **result

                })



    trends = pd.DataFrame(records)



    if trends.empty:

        return trends



    trends["p_adjusted_BH"] = multipletests(

        trends["p_value"].fillna(1.0),

        method="fdr_bh"

    )[1]



    trends["significant_FDR_0_05"] = (

        trends["p_adjusted_BH"] < 0.05

    )



    trends.to_csv(

        TABLE_DIR / "all_batch_comparisons.csv", index=False

    )



    # Illustrative trend plot for feature F1

    for gas in GAS_NAMES.values():

        subset = data[data["gas"] == gas]



        if subset.empty:

            continue



        trend = subset.groupby("batch")["F1"].agg(

            mean="mean",

            sem="sem",

            count="count"

        ).reset_index()



        margin = 1.96 * trend["sem"].fillna(0)



        plt.figure(figsize=(9, 5))

        plt.errorbar(

            trend["batch"],

            trend["mean"],

            yerr=margin,

            marker="o",

            capsize=3

        )

        plt.title(f"F1 Response Across Batches: {gas}")

        plt.xlabel("Batch")

        plt.ylabel("Mean F1 response (approx. 95% CI)")

        plt.xticks(range(1, 11))

        plt.tight_layout()

        plt.savefig(

            PLOT_DIR / f"F1_trend_{gas.lower()}.png",

            dpi=250

        )

        plt.close()



    print("Batch trend analysis completed.")

    return trends





# ==============================================================

# 8. RANK THE 16 SENSORS

# ==============================================================



def rank_sensors(test_results):

    results = test_results.copy()



    results["feature_number"] = (

        results["feature"]

        .str.replace("F", "", regex=False)

        .astype(int)

    )



    results["sensor"] = (

        (results["feature_number"] - 1) // 8 + 1

    )



    ranking = (

        results.groupby("sensor")

        .agg(

            mean_absolute_effect=(

                "standardized_effect",

                lambda x: x.abs().mean()

            ),

            median_absolute_effect=(

                "standardized_effect",

                lambda x: x.abs().median()

            ),

            significant_tests=("significant_FDR_0_05", "sum"),

            total_tests=("feature", "count")

        )

        .reset_index()

        .sort_values("mean_absolute_effect", ascending=False)

    )



    ranking["rank"] = range(1, len(ranking) + 1)



    ranking.to_csv(

        TABLE_DIR / "sensor_drift_ranking.csv", index=False

    )



    plt.figure(figsize=(10, 6))

    sns.barplot(

        data=ranking,

        x="sensor",

        y="mean_absolute_effect"

    )

    plt.title("Ranking of Sensors by Standardized Drift Effect")

    plt.xlabel("Sensor number")

    plt.ylabel("Mean absolute standardized effect")

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "sensor_drift_ranking.png", dpi=250

    )

    plt.close()



    return ranking





# ==============================================================

# 9. PRINCIPAL COMPONENT ANALYSIS

# ==============================================================



def perform_pca(data):

    X = data[FEATURE_COLS].replace(

        [np.inf, -np.inf], np.nan

    )



    # Median imputation only as a fallback for missing values.

    X = X.fillna(X.median())



    # Scale across all batches, preserving between-batch shifts.

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)



    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(X_scaled)



    pca_df = pd.DataFrame({

        "PC1": coordinates[:, 0],

        "PC2": coordinates[:, 1],

        "batch": data["batch"].to_numpy(),

        "gas": data["gas"].to_numpy()

    })



    pca_df.to_csv(

        TABLE_DIR / "pca_coordinates.csv", index=False

    )



    variance = pca.explained_variance_ratio_ * 100



    plt.figure(figsize=(10, 7))

    sns.scatterplot(

        data=pca_df,

        x="PC1",

        y="PC2",

        hue="batch",

        palette="viridis",

        alpha=0.5,

        s=18

    )



    plt.title("PCA: Feature Distribution Across Batches")

    plt.xlabel(f"PC1 ({variance[0]:.2f}% variance)")

    plt.ylabel(f"PC2 ({variance[1]:.2f}% variance)")

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "pca_by_batch.png", dpi=250

    )

    plt.close()



    pd.DataFrame({

        "component": ["PC1", "PC2"],

        "variance_explained_percent": variance

    }).to_csv(

        TABLE_DIR / "pca_explained_variance.csv",

        index=False

    )



    print(

        f"PCA variance explained by PC1 + PC2: "

        f"{variance.sum():.2f}%"

    )



    return pca_df





# ==============================================================

# 10. MAIN PROGRAM

# ==============================================================



def main():

    print("=" * 65)

    print("STATISTICAL ANALYSIS OF LONG-TERM SENSOR DRIFT")

    print("=" * 65)



    data = load_dataset()



    quality_analysis(data)

    descriptive_analysis(data)

    correlation_analysis(data)



    tests = hypothesis_testing(data)

    batch_trend_analysis(data)

    ranking = rank_sensors(tests)

    perform_pca(data)



    print("\nTOP 5 SENSORS BY DESCRIPTIVE DRIFT SCORE")

    print(ranking.head(5).to_string(index=False))



    print("\nPROJECT COMPLETED SUCCESSFULLY!")

    print("Cleaned dataset:", OUTPUT_DIR / "cleaned_data.csv")

    print("Statistical tables:", TABLE_DIR)

    print("Plots:", PLOT_DIR)





if __name__ == "__main__":

    main()
