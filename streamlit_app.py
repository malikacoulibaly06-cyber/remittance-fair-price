"""
Fair Remittance Price — Comparison Tool
A practical app for senders: enter source country, destination country, and amount;
get a ranked list of every available service with predicted cost in $ and %, plus
guidance on which channel is cheapest.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Fair Remittance Price", page_icon="💸", layout="wide")


# ---------------------------------------------------------------------------
# Load model, feature schema, and corridor lookup
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    candidates = [
        ("outputs/xgb_model.json", "outputs/feature_schema.pkl", "outputs/corridor_lookup.pkl"),
        ("xgb_model.json",          "feature_schema.pkl",          "corridor_lookup.pkl"),
    ]
    for m, s, l in candidates:
        if os.path.exists(m) and os.path.exists(s) and os.path.exists(l):
            model = XGBRegressor()
            model.load_model(m)
            with open(s, "rb") as f:
                schema = pickle.load(f)
            lookup = pd.read_pickle(l)
            return model, schema, lookup
    raise FileNotFoundError("Need xgb_model.json, feature_schema.pkl, and corridor_lookup.pkl.")


try:
    model, schema, lookup = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e) + " Run `python project3_remittance_cost.py` then `python build_lookup.py`.")
    st.stop()

feature_names = schema["feature_names"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💸 Fair Remittance Price — Comparison Tool")
st.caption("Compare every remittance service available for your corridor. "
           "Predicted cost is from an XGBoost model trained on ~196,000 World Bank quotes. "
           "**Use this to pick the cheapest channel before you send.**")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    c1, c2 = st.columns(2)
    with c1:
        source_country = st.selectbox(
            "📤 Sending from",
            sorted(lookup["source_name"].unique()),
            index=sorted(lookup["source_name"].unique()).index("United States")
                  if "United States" in lookup["source_name"].unique() else 0,
        )
    with c2:
        valid_destinations = sorted(
            lookup.loc[lookup["source_name"] == source_country, "destination_name"].unique()
        )
        if not valid_destinations:
            st.warning("No corridors available from that country in the last 8 quarters.")
            st.stop()
        default_dest = "Mexico" if "Mexico" in valid_destinations else valid_destinations[0]
        destination_country = st.selectbox(
            "📥 Sending to",
            valid_destinations,
            index=valid_destinations.index(default_dest),
        )

    send_usd = st.number_input("💵 Send amount (USD)", min_value=10.0, max_value=10000.0,
                                value=200.0, step=50.0)

with right:
    speed_choice = st.radio(
        "⚡ Speed required",
        ["Any", "Same day or faster", "Less than one hour"],
        index=0,
    )
    show_top_n = st.slider("Show top N options", 5, 30, 10)

st.markdown("---")


# ---------------------------------------------------------------------------
# Build the model input matrix from every available service in this corridor
# ---------------------------------------------------------------------------
def predict_for_corridor(src: str, dst: str, amount_usd: float) -> pd.DataFrame:
    sub = lookup[(lookup["source_name"] == src) & (lookup["destination_name"] == dst)].copy()
    if sub.empty:
        return sub

    # Numeric features
    sub["log_send_usd"] = np.log1p(amount_usd)

    cat_cols = ["firm_type", "payment instrument", "Sending location", "speed actual",
                "pickup method", "source_region", "destination_region",
                "source_income", "destination_income"]
    for c in cat_cols:
        sub[c] = sub[c].fillna("UNK").astype(str)

    # One-hot match training schema
    X_cat = pd.get_dummies(sub[cat_cols], drop_first=True)
    X_num = sub[["log_send_usd", "corridor_firm_count"]].astype(float).reset_index(drop=True)
    X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
    X_aligned = X.reindex(columns=feature_names, fill_value=0)

    sub = sub.reset_index(drop=True)
    sub["pred_cost_pct"] = model.predict(X_aligned.values)
    sub["pred_cost_usd"] = sub["pred_cost_pct"] / 100 * amount_usd
    sub["amount_received_usd"] = amount_usd - sub["pred_cost_usd"]
    return sub.sort_values("pred_cost_pct").reset_index(drop=True)


results = predict_for_corridor(source_country, destination_country, send_usd)

# Speed filter
if speed_choice == "Less than one hour":
    results = results[results["speed actual"].str.contains("Less than one hour", na=False)]
elif speed_choice == "Same day or faster":
    results = results[results["speed actual"].isin(
        ["Less than one hour", "Within minutes", "Same day", "Real time"])]

if results.empty:
    st.warning(f"No services match those filters for {source_country} → {destination_country}.")
    st.stop()


# ---------------------------------------------------------------------------
# Smart recommendation banner
# ---------------------------------------------------------------------------
cheapest = results.iloc[0]
fast_options = results[results["speed actual"].str.contains(
    "Less than one hour|Within minutes|Real time", regex=True, na=False)]
cheapest_fast = fast_options.iloc[0] if len(fast_options) else None

st.subheader(f"🏆 Recommendation for {source_country} → {destination_country} on ${send_usd:.0f}")

cols = st.columns(3 if cheapest_fast is not None else 2)

with cols[0]:
    st.success(
        f"**Cheapest overall**\n\n"
        f"**{cheapest['firm']}**  \n"
        f"{cheapest['payment instrument']} → {cheapest['pickup method']}  \n"
        f"⏱️ {cheapest['speed actual']}\n\n"
        f"💰 **{cheapest['pred_cost_pct']:.2f}%**  ≈  **${cheapest['pred_cost_usd']:.2f}**\n\n"
        f"You'd receive ≈ **${cheapest['amount_received_usd']:.2f}**"
    )

if cheapest_fast is not None:
    with cols[1]:
        st.info(
            f"**Cheapest under 1 hour**\n\n"
            f"**{cheapest_fast['firm']}**  \n"
            f"{cheapest_fast['payment instrument']} → {cheapest_fast['pickup method']}\n\n"
            f"💰 **{cheapest_fast['pred_cost_pct']:.2f}%**  ≈  "
            f"**${cheapest_fast['pred_cost_usd']:.2f}**"
        )
    benchmark_col = cols[2]
else:
    benchmark_col = cols[1]

with benchmark_col:
    avg = results["pred_cost_pct"].mean()
    sdg_target = 3.0
    delta_vs_sdg = cheapest["pred_cost_pct"] - sdg_target
    sdg_status = "✅ Beats UN 3% target" if cheapest["pred_cost_pct"] <= 3 \
                 else f"⚠️ {delta_vs_sdg:+.1f} pp vs UN 3% target"
    st.warning(
        f"**Corridor benchmark**\n\n"
        f"Avg of {len(results)} services: **{avg:.2f}%**  \n"
        f"Cheapest savings vs avg: **${(avg - cheapest['pred_cost_pct'])/100*send_usd:.2f}**\n\n"
        f"{sdg_status}"
    )


# ---------------------------------------------------------------------------
# Full ranked comparison table
# ---------------------------------------------------------------------------
st.subheader(f"All {len(results)} services available, ranked cheapest first")

display = results.head(show_top_n).copy()
display.insert(0, "Rank", range(1, len(display) + 1))
display = display.rename(columns={
    "firm":               "Firm",
    "firm_type":          "Type",
    "payment instrument": "Pay with",
    "speed actual":       "Speed",
    "pickup method":      "Pickup as",
    "pred_cost_pct":      "Cost %",
    "pred_cost_usd":      "Cost ($)",
    "amount_received_usd":"Receive ($)",
})[["Rank", "Firm", "Type", "Pay with", "Speed", "Pickup as", "Cost %", "Cost ($)", "Receive ($)"]]


def color_row(row):
    if row["Rank"] == 1:
        return ["background-color: #d4edda"] * len(row)
    if row["Cost %"] <= 3:
        return ["background-color: #e8f5e9"] * len(row)
    if row["Cost %"] >= 8:
        return ["background-color: #ffebee"] * len(row)
    return [""] * len(row)


styled = (display
          .style
          .format({"Cost %": "{:.2f}%", "Cost ($)": "${:.2f}", "Receive ($)": "${:.2f}"})
          .apply(color_row, axis=1))
st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Channel guidance
# ---------------------------------------------------------------------------
with st.expander("💡 Which channel is cheapest? — General guidance"):
    st.markdown("""
- **Internet/online + bank-account pickup** is usually the cheapest combo. Mobile money
  pickup (in markets where it exists) is often cheaper still.
- **Cash-in cash-out** through agents tends to be the most expensive — the convenience
  premium is real.
- **Banks** charge more than dedicated MTOs (Money Transfer Operators) on most corridors.
  Mobile-money providers and digital-only fintechs (Wise, Remitly, WorldRemit) usually win.
- **Speed costs money**, but not always: on busy corridors the cheapest provider is also
  often instant. Use the "Less than one hour" filter to check.
- **Small amounts ($200) are 2–3× more expensive in % terms than $500** — if you can wait
  and consolidate, you save real money.
- **Compare regularly** — provider FX margins change weekly. Cheapest today ≠ cheapest next month.
    """)

with st.expander("📊 How the model works"):
    st.markdown(f"""
- **Data**: World Bank Remittance Prices Worldwide (2011–Q1 2025). This app uses the last
  8 quarters of quotes ({len(lookup):,} unique service combinations across {lookup['corridor'].nunique()}
  corridors and {lookup['firm'].nunique()} firms).
- **Model**: XGBoost regression — R² ≈ 0.7+, MAE ≈ 1.5 percentage points.
- **Features**: log-transformed send amount, corridor competition, firm type, payment
  instrument, sending location, speed, pickup method, source/destination region & income.
- **Limitation**: predictions are *modelled fair price* based on historical patterns, not
  live quotes. Always check the actual provider before sending.
    """)

st.caption("Source: World Bank Remittance Prices Worldwide. Predictions are estimates — verify with the provider.")
