"""
Streamlit app — Fair Remittance Price Calculator
Live demo: enter a corridor + firm + instrument, get a predicted fair cost %.

Deployment:
  1. Push this repo to GitHub
  2. Sign in at https://share.streamlit.io with your GitHub account
  3. Click "New app" → pick this repo → set "Main file path" to streamlit_app.py
  4. Click Deploy. Free, public URL in ~2 minutes.

Local run:
  streamlit run streamlit_app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Fair Remittance Price", page_icon="💸", layout="wide")

st.title("💸 Fair Remittance Price Calculator")
st.caption("Predicts the expected total cost (%) of a remittance based on the World Bank "
           "Remittance Prices Worldwide dataset (2016–2025). Built on an XGBoost regression model "
           "trained on ~196,000 quotes across 372 corridors.")


# ---------------------------------------------------------------------------
# Load the trained model + the feature schema saved alongside it
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """The training script saves these two files into ./outputs/."""
    model = XGBRegressor()
    model.load_model("outputs/xgb_model.json")
    with open("outputs/feature_schema.pkl", "rb") as f:
        schema = pickle.load(f)
    # schema = {"feature_names": [...], "categorical_options": {col: [unique values]}}
    return model, schema


try:
    model, schema = load_artifacts()
except (FileNotFoundError, Exception) as e:
    st.error("Couldn't find `outputs/xgb_model.json`. "
             "Run `python project3_remittance_cost.py` once to train and save the model.")
    st.stop()

feature_names = schema["feature_names"]
opts = schema["categorical_options"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Build the model input row that matches the training schema
# ---------------------------------------------------------------------------
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
    # one-hot using the SAME drop_first=True dummy structure as training
    df_in = pd.DataFrame([cat_inputs])
    df_dum = pd.get_dummies(df_in, drop_first=True)
    for k, v in raw.items():
        df_dum[k] = v
    # Align to the trained feature schema (missing columns -> 0)
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
        st.success(f"Meets the UN SDG 10.c target (≤ 3%) ✔")
    elif pred <= 6.5:
        st.info(f"Around the global average (~6.4%).")
    else:
        st.warning(f"Above global average — likely overpriced corridor/instrument.")

    fee_usd = pred / 100 * send_usd
    st.caption(f"On ${send_usd:.0f} that's roughly **${fee_usd:.2f}** in total cost.")

with st.expander("How the model works"):
    st.markdown("""
    - **Target**: `Total cost (%)` from the World Bank RPW dataset
    - **Features**: log-transformed send amount, corridor competition, plus 9 one-hot encoded
      categoricals (firm type, payment instrument, sending location, speed, pickup method,
      source/destination region & income group)
    - **Models trained**: Linear Regression → Random Forest → Gradient Boosting → XGBoost
    - **Best model**: XGBoost (R² ≈ 0.7+, RMSE ≈ 2.5 pp)
    - **Limitations**: random 80/20 split (not temporal); no provider name; doesn't yet
      account for promotional pricing
    """)

st.caption("Source data: World Bank — Remittance Prices Worldwide (2011–Q1 2025)")
