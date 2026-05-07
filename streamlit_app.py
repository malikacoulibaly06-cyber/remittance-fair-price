"""
Streamlit app — Fair Remittance Price Calculator
"""

import pickle
import os
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Fair Remittance Price", page_icon="💸", layout="wide")

st.title("💸 Fair Remittance Price Calculator")
st.caption("Predicts the expected total cost (%) of a remittance based on the World Bank "
           "Remittance Prices Worldwide dataset (2016–2025). Built on an XGBoost regression model "
           "trained on ~196,000 quotes across 372 corridors.")


@st.cache_resource
def load_artifacts():
    """Look for the model in either ./outputs/ (local) or the repo root (deployed)."""
    candidates = [
        ("outputs/xgb_model.json", "outputs/feature_schema.pkl"),
        ("xgb_model.json",         "feature_schema.pkl"),
    ]
    for model_path, schema_path in candidates:
        if os.path.exists(model_path) and os.path.exists(schema_path):
            model = XGBRegressor()
            model.load_model(model_path)
            with open(schema_path, "rb") as f:
                schema = pickle.load(f)
            return model, schema
    raise FileNotFoundError("xgb_model.json and feature_schema.pkl not found.")


try:
    model, schema = load_artifacts()
except FileNotFoundError:
    st.error("Couldn't find `xgb_model.json` or `feature_schema.pkl` in the repo. "
             "Make sure both files are committed.")
    st.stop()

feature_names = schema["feature_names"]
opts = schema["categorical_options"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Corridor & sender")
    source_region      = st.selectbox("Source region",      opts["source_region"])
    source_income      = st.selectbox("Source income group", opts["source_income"])
    destination_region = st.selectbox("Destination region", opts["destination_region"])
    destination_income = st.selectbox("Destination income group", opts["destination_income"])
    send_usd = st.number_input("Send amount (USD)", min_value=10.0, max_value=10000.0,
                                value=200.0, step=50.0)

with col2:
    st.subheader("Service")
    firm_type        = st.selectbox("Firm type",         opts["firm_type"])
    payment_inst     = st.selectbox("Payment instrument", opts["payment instrument"])
    sending_location = st.selectbox("Sending location",   opts["Sending location"])
    speed            = st.selectbox("Speed",              opts["speed actual"])
    pickup_method    = st.selectbox("Pickup method",      opts["pickup method"])

corridor_firm_count = st.slider("Number of competing firms in this corridor", 1, 30, 8,
                                help="More competition usually → lower fair price")


def build_row():
    raw = {
        "log_send_usd": np.log1p(send_usd),
        "corridor_firm_count": corridor_firm_count,
    }
    cat_inputs = {
        "firm_type": firm_type,
        "payment instrument": payment_inst,
        "Sending location": sending_location,
        "speed actual": speed,
        "pickup method": pickup_method,
        "source_region": source_region,
        "destination_region": destination_region,
        "source_income": source_income,
        "destination_income": destination_income,
    }
    df_in = pd.DataFrame([cat_inputs])
    df_dum = pd.get_dummies(df_in, drop_first=True)
    for k, v in raw.items():
        df_dum[k] = v
    df_aligned = df_dum.reindex(columns=feature_names, fill_value=0)
    return df_aligned


if st.button("Predict fair price", type="primary"):
    row = build_row()
    pred = float(model.predict(row.values)[0])
    sdg_target = 3.0
    st.markdown("---")
    st.metric("Predicted total cost", f"{pred:.2f} %",
              delta=f"{pred - sdg_target:+.2f} pp vs SDG 3% target",
              delta_color="inverse")
    if pred <= sdg_target:
        st.success("Meets the UN SDG 10.c target (≤ 3%) ✔")
    elif pred <= 6.5:
        st.info("Around the global average (~6.4%).")
    else:
        st.warning("Above global average — likely overpriced corridor/instrument.")

    fee_usd = pred / 100 * send_usd
    st.caption(f"On ${send_usd:.0f} that's roughly **${fee_usd:.2f}** in total cost.")

with st.expander("How the model works"):
    st.markdown("""
    - **Target**: `Total cost (%)` from the World Bank RPW dataset
    - **Features**: log-transformed send amount, corridor competition, plus 9 one-hot encoded categoricals
    - **Models trained**: Linear Regression → Random Forest → Gradient Boosting → XGBoost
    - **Best model**: XGBoost (R² ≈ 0.7+, RMSE ≈ 2.5 pp)
    """)

st.caption("Source data: World Bank — Remittance Prices Worldwide (2011–Q1 2025)")
