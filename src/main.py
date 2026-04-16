from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_PATH = BASE_DIR / "reports" / "business_insight_report.md"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Generate it first.")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["customer_id"])

    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned


def run_eda(df: pd.DataFrame) -> dict:
    results = {}
    results["total_customers"] = int(df.shape[0])
    results["avg_monthly_spend"] = float(df["monthly_spend"].mean())
    results["avg_satisfaction"] = float(df["satisfaction_score"].mean())
    results["churn_rate"] = float(df["churn"].mean())
    results["high_intent_rate"] = float(df["high_purchase_intent"].mean())
    return results


def create_visualizations(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    churn_by_region = df.groupby("region")["churn"].mean().sort_values(ascending=False)
    sns.barplot(x=churn_by_region.index, y=churn_by_region.values)
    plt.title("Churn Rate by Region")
    plt.ylabel("Churn Rate")
    plt.xlabel("Region")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_by_region.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="monthly_spend",
        y="purchase_frequency",
        hue="churn",
        alpha=0.6,
        palette="Set1",
    )
    plt.title("Monthly Spend vs Purchase Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spend_vs_frequency.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="satisfaction_score", hue="churn", multiple="stack", bins=20)
    plt.title("Satisfaction Score Distribution by Churn")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "satisfaction_vs_churn.png", dpi=150)
    plt.close()

    corr_cols = [
        "tenure_months",
        "annual_income",
        "monthly_spend",
        "purchase_frequency",
        "complaints_last_6m",
        "support_tickets",
        "satisfaction_score",
        "churn",
        "high_purchase_intent",
    ]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close()


def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    features = df[["monthly_spend", "purchase_frequency", "tenure_months", "satisfaction_score"]].copy()
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["segment"] = kmeans.fit_predict(scaled)
    return df


def churn_prediction(df: pd.DataFrame) -> dict:
    target = "churn"
    features = [
        "age",
        "gender",
        "region",
        "acquisition_channel",
        "tenure_months",
        "annual_income",
        "monthly_spend",
        "purchase_frequency",
        "complaints_last_6m",
        "support_tickets",
        "discount_usage_ratio",
        "satisfaction_score",
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1200, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "model": model,
        "classification_report": report,
        "roc_auc": float(auc),
    }


def save_business_report(eda_metrics: dict, segmented_df: pd.DataFrame, model_metrics: dict) -> None:
    segment_summary = (
        segmented_df.groupby("segment")[["monthly_spend", "purchase_frequency", "satisfaction_score", "churn"]]
        .mean()
        .round(2)
    )

    top_churn_region = (
        segmented_df.groupby("region")["churn"].mean().sort_values(ascending=False).index[0]
    )
    high_value_segment = segment_summary["monthly_spend"].idxmax()

    segment_summary_text = segment_summary.to_string()

    content = f"""# Business Insight Report - Customer Behavior Analysis

## 1. Executive Snapshot
- Total customers analyzed: **{eda_metrics["total_customers"]}**
- Average monthly spend: **{eda_metrics["avg_monthly_spend"]:.2f}**
- Average satisfaction score: **{eda_metrics["avg_satisfaction"]:.2f}**
- Churn rate: **{eda_metrics["churn_rate"]:.2%}**
- High purchase-intent rate: **{eda_metrics["high_intent_rate"]:.2%}**

## 2. Key Behavioral Patterns
- Customers with lower satisfaction and more complaints show higher churn tendency.
- Purchase frequency and monthly spend are positively related.
- Region **{top_churn_region}** currently has the highest churn proportion.

## 3. Customer Segmentation Insights
High-value segment (Segment **{high_value_segment}**) shows highest average spend.

Segment averages:

```
{segment_summary_text}
```

## 4. Predictive Analysis (Churn)
- Model used: Logistic Regression
- ROC-AUC: **{model_metrics["roc_auc"]:.3f}**
- Churn precision: **{model_metrics["classification_report"]["1"]["precision"]:.3f}**
- Churn recall: **{model_metrics["classification_report"]["1"]["recall"]:.3f}**

## 5. Business Recommendations
1. Launch retention campaigns for low-satisfaction and high-complaint customers.
2. Run region-specific interventions in high-churn areas (especially **{top_churn_region}**).
3. Build premium bundles for high-value segments and targeted offers for medium-value segments.
4. Use churn-risk scores from the model to prioritize proactive support calls.

## 6. Visual Outputs
Generated in the `output/` folder:
- `churn_by_region.png`
- `spend_vs_frequency.png`
- `satisfaction_vs_churn.png`
- `correlation_matrix.png`
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    print("Loading dataset...")
    df_raw = load_dataset(DATA_PATH)
    print(f"Loaded {df_raw.shape[0]} records and {df_raw.shape[1]} columns.")

    print("Cleaning and preprocessing data...")
    df_clean = clean_data(df_raw)

    print("Running exploratory analysis...")
    eda_metrics = run_eda(df_clean)

    print("Performing customer segmentation...")
    segmented_df = segment_customers(df_clean, n_clusters=4)
    segmented_df.to_csv(OUTPUT_DIR / "customer_data_segmented.csv", index=False)

    print("Training churn prediction model...")
    model_metrics = churn_prediction(segmented_df)

    print("Creating visualizations...")
    create_visualizations(segmented_df)

    print("Writing business report...")
    save_business_report(eda_metrics, segmented_df, model_metrics)

    print("Analysis complete.")
    print(f"Report: {REPORT_PATH}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
