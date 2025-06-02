import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide")
st.title("Healthcare Fraud Detection Dashboard")

# ===== EDA Section =====
st.header("Exploratory Data Analysis")
st.image("reports/shap_beeswarm_full.png", caption="SHAP Beeswarm - Full Data")
st.image("reports/shap_global_bar_full.png", caption="SHAP Global Bar - Feature Importance")

# ===== Model Explainability =====
st.header("Model Explainability (SHAP)")
st.image("reports/shap_waterfall_full_123.png", caption="SHAP Waterfall Example")

# ===== Provider Risk Demo =====
st.header("Provider Lookup Demo")
provider_id = st.text_input("Enter Provider ID (demo only)")
if provider_id:
    st.info(f"Demo only: Risk score for provider {provider_id} not available in this version.")

# ===== Downloads =====
st.header("Download Center")
with open("reports/shap_beeswarm_full.png", "rb") as f:
    st.download_button("Download SHAP Beeswarm Plot", f, "shap_beeswarm_full.png")
with open("reports/shap_global_bar_full.png", "rb") as f:
    st.download_button("Download SHAP Bar Plot", f, "shap_global_bar_full.png")
