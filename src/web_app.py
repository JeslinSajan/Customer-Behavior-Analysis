from pathlib import Path

import pandas as pd
import streamlit as st

from generate_sample_data import generate_customer_data
from main import (
    DATA_PATH,
    OUTPUT_DIR,
    REPORT_PATH,
    churn_prediction,
    clean_data,
    create_visualizations,
    run_eda,
    save_business_report,
    segment_customers,
)


st.set_page_config(page_title="Customer Behavior Analysis", page_icon="📊", layout="wide")


def run_pipeline() -> dict:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample_df = generate_customer_data(num_customers=3000)
        sample_df.to_csv(DATA_PATH, index=False)

    df_raw = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df_raw)
    eda_metrics = run_eda(df_clean)
    segmented_df = segment_customers(df_clean, n_clusters=4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented_df.to_csv(OUTPUT_DIR / "customer_data_segmented.csv", index=False)

    model_metrics = churn_prediction(segmented_df)
    create_visualizations(segmented_df)
    save_business_report(eda_metrics, segmented_df, model_metrics)

    return {
        "eda_metrics": eda_metrics,
        "segmented_df": segmented_df,
        "model_metrics": model_metrics,
    }


st.title("Industry-Oriented Customer Behavior Analysis")
st.write(
    "Run analysis on customer data and view segmentation, churn prediction, charts, and business insights."
)

st.markdown("### 1) Dataset")
st.caption(f"Current dataset path: `{DATA_PATH}`")
if DATA_PATH.exists():
    st.success("Dataset found.")
else:
    st.warning("Dataset not found. A sample dataset will be generated when you run the analysis.")

if st.button("Run Full Analysis", type="primary"):
    with st.spinner("Running data cleaning, EDA, segmentation, and churn model..."):
        results = run_pipeline()
        st.session_state["results"] = results
    st.success("Analysis completed successfully.")


if "results" in st.session_state:
    results = st.session_state["results"]
    eda = results["eda_metrics"]
    model_metrics = results["model_metrics"]
    segmented_df = results["segmented_df"]

    st.markdown("### 2) Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f'{eda["total_customers"]:,}')
    c2.metric("Avg Monthly Spend", f'{eda["avg_monthly_spend"]:.2f}')
    c3.metric("Avg Satisfaction", f'{eda["avg_satisfaction"]:.2f}')
    c4.metric("Churn Rate", f'{eda["churn_rate"]:.2%}')
    c5.metric("High Intent Rate", f'{eda["high_intent_rate"]:.2%}')

    st.markdown("### 3) Predictive Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f'{model_metrics["roc_auc"]:.3f}')
    m2.metric(
        "Churn Precision",
        f'{model_metrics["classification_report"]["1"]["precision"]:.3f}',
    )
    m3.metric(
        "Churn Recall",
        f'{model_metrics["classification_report"]["1"]["recall"]:.3f}',
    )

    st.markdown("### 4) Segment Preview")
    st.dataframe(segmented_df.head(25), use_container_width=True)

    st.markdown("### 5) Visual Insights")
    image_paths = [
        OUTPUT_DIR / "churn_by_region.png",
        OUTPUT_DIR / "spend_vs_frequency.png",
        OUTPUT_DIR / "satisfaction_vs_churn.png",
        OUTPUT_DIR / "correlation_matrix.png",
    ]

    for image_path in image_paths:
        if image_path.exists():
            st.image(str(image_path), caption=image_path.name, use_container_width=True)

    st.markdown("### 6) Business Insight Report")
    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Report not generated yet.")
else:
    st.info("Click **Run Full Analysis** to generate and display outputs.")
