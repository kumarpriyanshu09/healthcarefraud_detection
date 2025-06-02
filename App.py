import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Healthcare Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    features = pd.read_parquet('processed/Feature Table.parquet')
    return features

@st.cache_data
def load_shap():
    shap_values = np.load('reports/shap_values_full.npy')
    return shap_values

features = load_data()
shap_values = load_shap()

# -------------------- SIDEBAR NAVIGATION -------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "Overview",
        "EDA & Feature Insights",
        "Provider Explorer",
        "Model Explainability",
        "About/Docs"
    ]
)

# -------------------- OVERVIEW TAB ---------------------
if section == "Overview":
    st.title("Healthcare Fraud Detection Dashboard")
    st.markdown("**Welcome to the interactive dashboard.**\nThis app profiles provider fraud using machine learning, SHAP explainability, and real provider-level features.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Providers", f"{features['Provider'].nunique():,}")
    with col2:
        st.metric("Total Claims", f"{features['total_claims'].sum():,}")
    with col3:
        pct_fraud = 100 * features['PotentialFraud'].sum() / len(features)
        st.metric("Fraud Rate (%)", f"{pct_fraud:.1f}%")
    with col4:
        st.metric("Avg. Reimb/Claim", f"${features['avg_reimb'].mean():,.0f}")

    st.divider()
    st.subheader("Quick Facts")
    st.markdown(
        "- **Data covers** inpatient & outpatient claims for 5,410 providers\n"
        "- **Label:** PotentialFraud (binary)\n"
        "- **Model:** XGBoost (AUC 0.95, 88% recall)\n"
        "- **Explainability:** SHAP for both global and per-provider interpretability\n"
    )

    st.image("reports/shap_global_bar_full.png", caption="Top Features Driving Fraud (SHAP Global Bar Plot)", use_column_width=True)

# -------------------- EDA & FEATURE INSIGHTS TAB ---------------------
elif section == "EDA & Feature Insights":
    st.title("Exploratory Data Analysis & Feature Insights")
    st.markdown("### Global Feature Importance\nFeatures with the greatest impact on model fraud predictions (via SHAP):")
    st.image("reports/shap_beeswarm_full.png", caption="SHAP Beeswarm Plot (global feature impacts)", use_column_width=True)

    st.divider()
    st.subheader("Feature Distribution Explorer")
    top_feats = ["total_reimb", "avg_days_between_claims", "total_deductible", "claims_per_bene", "max_reimb", "pct_bene_multiclaim"]
    feat = st.selectbox("Select a feature to visualize:", top_feats)
    st.bar_chart(features[feat])

    st.info("💡 SHAP values confirm that high reimbursements, abnormal claim gaps, and deductible patterns are the strongest fraud indicators in this dataset.")

# -------------------- PROVIDER EXPLORER TAB ----------------------
elif section == "Provider Explorer":
    st.title("Provider Explorer")
    st.markdown("Filter, search, and select providers to view fraud risk scores and detailed features.")

    fraud_filter = st.selectbox("Filter by Fraud Label", ["All", "Fraudulent", "Legitimate"])
    min_claims = st.slider("Minimum Claims", 0, int(features['total_claims'].max()), 0)
    filtered = features.copy()
    if fraud_filter != "All":
        val = 1 if fraud_filter == "Fraudulent" else 0
        filtered = filtered[filtered['PotentialFraud'] == val]
    filtered = filtered[filtered['total_claims'] >= min_claims]
    st.dataframe(filtered[["Provider", "total_claims", "total_reimb", "avg_reimb", "PotentialFraud"]].reset_index(drop=True), use_container_width=True)

    st.divider()
    st.markdown("### Drill Into a Provider")
    provider_ids = filtered["Provider"].unique()
    if len(provider_ids) == 0:
        st.warning("No providers match filter.")
    else:
        sel_id = st.selectbox("Choose Provider ID for details:", provider_ids)
        sel_row = filtered[filtered["Provider"] == sel_id].iloc[0]
        st.json(sel_row.to_dict())

# -------------------- MODEL EXPLAINABILITY TAB ----------------------
elif section == "Model Explainability":
    st.title("Model Explainability – SHAP Analysis")
    st.markdown("Visualize **why** the model flagged a specific provider using SHAP explainability.")

    provider_ids = features["Provider"].unique()
    sel_id = st.selectbox("Choose Provider ID", provider_ids)
    idx = features.index[features["Provider"] == sel_id][0]
    # Show SHAP values for selected provider (waterfall plot pre-generated, e.g. shap_waterfall_full_123.png)
    shap_png_path = f"reports/shap_waterfall_full_{idx}.png"
    if os.path.exists(shap_png_path):
        st.image(shap_png_path, caption=f"SHAP Waterfall for Provider {sel_id}", use_column_width=True)
    else:
        st.warning("SHAP plot not found for this provider. (You can pre-generate them for top flagged providers.)")
    st.write("**Feature values:**")
    st.json(features.iloc[idx][["total_reimb", "avg_days_between_claims", "total_deductible", "claims_per_bene", "PotentialFraud"]].to_dict())

    st.info("You can extend this section to dynamically generate force plots or add peer comparisons as needed.")

# -------------------- ABOUT/DOCS TAB ----------------------
elif section == "About/Docs":
    st.title("About This Project")
    st.markdown("""
    **Healthcare Provider Fraud Detection Dashboard**  
    - Data: Medicare provider claims (inpatient, outpatient, beneficiary, labels)
    - Workflow: Ingestion, cleaning, EDA, feature engineering, modeling, explainability (XGBoost + SHAP)
    - Model: Tuned XGBoost, ROC AUC 0.95, F1 ~0.51, Recall ~0.88
    - Features: ~30 provider-level engineered metrics
    - Explainability: SHAP global and local (per-provider)
    - Docs: See `/reports` for technical docs & visualizations

    _Contact: Your Team | Version 1.0 | Powered by Streamlit_
    """)

# ----- END -----
