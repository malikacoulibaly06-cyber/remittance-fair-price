"""
Fair Remittance Price — Comparison Tool
A practical app for senders: compare every available service for a corridor,
with predicted cost in USD and percent, and guidance on the cheapest channel.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Fair Remittance Price",
    page_icon="💸",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Load model, schema, and corridor lookup
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
# Sidebar — about, data source, similar tools
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "I'm a student in the United States taking fintech courses, and I wanted "
        "to understand the economics of cross-border payments — specifically why "
        "remittances cost so much and which channels work best for the people "
        "actually sending money home. This tool is the result of that research."
    )

    st.markdown("### Data source")
    st.markdown(
        "Quotes come from the World Bank's "
        "[Remittance Prices Worldwide](https://remittanceprices.worldbank.org/) "
        "database — the same dataset the UN uses to track progress on the SDG 10.c "
        "target (reduce remittance costs to under 3%). It is updated quarterly. "
        "This app uses the eight most recent quarters."
    )

    st.markdown("### Coverage")
    st.markdown(
        f"**{lookup['source_name'].nunique()} sending countries**, "
        f"**{lookup['destination_name'].nunique()} receiving countries**, "
        f"**{lookup['corridor'].nunique()} corridors**, "
        f"**{lookup['firm'].nunique()} firms**.\n\n"
        "The World Bank tracks a representative sample of the largest corridors "
        "by volume — not every country pair in the world. If your corridor isn't "
        "listed, it isn't in the source dataset. I'd like to expand coverage by "
        "merging in provider-published rates and user-reported quotes; that's the "
        "next iteration of the project."
    )

    st.markdown("### Similar tools")
    st.markdown(
        "- **[remittanceprices.worldbank.org](https://remittanceprices.worldbank.org/)** "
        "— the official, independent World Bank portal. The most authoritative source, "
        "but designed as a research tool, not a sender-friendly comparator.\n"
        "- **[monito.com](https://www.monito.com/)** — independent commercial comparison "
        "platform; good UX, broad coverage, monetised through provider referrals (so "
        "ranking can be influenced).\n"
        "- **Provider sites** (Wise, Western Union, Remitly, etc.) all run their own "
        "price-comparison pages, but they are owned by the providers themselves and "
        "naturally favour their own service. Treat their numbers with caution.\n\n"
        "This app sits closer to the World Bank end of the spectrum — model-based, "
        "no referral revenue, no incentive to favour any particular firm."
    )

    st.markdown("---")
    st.caption(
        "Predicted costs are model estimates from an XGBoost regression trained on "
        "~196,000 historical quotes. Always verify with the provider before sending."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Fair Remittance Price")
st.markdown(
    "Compare every remittance service available for your corridor. "
    "Predicted cost is from a regression model trained on World Bank data. "
    "Use it to pick the cheapest channel before you send."
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    c1, c2 = st.columns(2)
    with c1:
        sources = sorted(lookup["source_name"].unique())
        default_src = sources.index("United States") if "United States" in sources else 0
        source_country = st.selectbox("From", sources, index=default_src)
    with c2:
        valid_destinations = sorted(
            lookup.loc[lookup["source_name"] == source_country, "destination_name"].unique()
        )
        if not valid_destinations:
            st.warning("No corridors available from that country in the last 8 quarters.")
            st.stop()
        default_dst = valid_destinations.index("Mexico") if "Mexico" in valid_destinations else 0
        destination_country = st.selectbox("To", valid_destinations, index=default_dst)

    send_usd = st.number_input(
        "Send amount (USD)",
        min_value=10.0, max_value=10000.0, value=200.0, step=50.0,
    )

with right:
    speed_choice = st.radio(
        "Speed required",
        ["Any", "Same day or faster", "Within one hour"],
        index=0,
    )
    show_top_n = st.slider("Show top N services", 5, 30, 10)

st.markdown("")


# ---------------------------------------------------------------------------
# Predict cost for every available service in this corridor
# ---------------------------------------------------------------------------
def predict_for_corridor(src: str, dst: str, amount_usd: float) -> pd.DataFrame:
    sub = lookup[(lookup["source_name"] == src) & (lookup["destination_name"] == dst)].copy()
    if sub.empty:
        return sub

    sub["log_send_usd"] = np.log1p(amount_usd)

    cat_cols = ["firm_type", "payment instrument", "Sending location", "speed actual",
                "pickup method", "source_region", "destination_region",
                "source_income", "destination_income"]
    for c in cat_cols:
        sub[c] = sub[c].fillna("UNK").astype(str)

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

if speed_choice == "Within one hour":
    results = results[results["speed actual"].str.contains("Less than one hour", na=False)]
elif speed_choice == "Same day or faster":
    results = results[results["speed actual"].isin(
        ["Less than one hour", "Within minutes", "Same day", "Real time"])]

if results.empty:
    st.warning(f"No services match those filters for {source_country} → {destination_country}.")
    st.stop()


# ---------------------------------------------------------------------------
# Summary line + benchmark
# ---------------------------------------------------------------------------
cheapest = results.iloc[0]
fast_options = results[results["speed actual"].str.contains(
    "Less than one hour|Within minutes|Real time", regex=True, na=False)]
cheapest_fast = fast_options.iloc[0] if len(fast_options) else None

avg = results["pred_cost_pct"].mean()
sdg_target = 3.0
savings_vs_avg = (avg - cheapest["pred_cost_pct"]) / 100 * send_usd
sdg_status = (
    "below the UN SDG 10.c target of 3%."
    if cheapest["pred_cost_pct"] <= sdg_target
    else f"{cheapest['pred_cost_pct'] - sdg_target:+.1f} percentage points above the UN SDG 10.c target of 3%."
)

st.subheader(f"{source_country} → {destination_country} on ${send_usd:.0f}")

summary_cols = st.columns(3)
with summary_cols[0]:
    st.metric("Cheapest service",
              f"{cheapest['pred_cost_pct']:.2f}%",
              f"${cheapest['pred_cost_usd']:.2f} fee")
    st.caption(f"**{cheapest['firm']}** — {cheapest['payment instrument'].lower()} → "
               f"{cheapest['pickup method'].lower()} ({cheapest['speed actual'].lower()})")

with summary_cols[1]:
    if cheapest_fast is not None:
        st.metric("Cheapest under one hour",
                  f"{cheapest_fast['pred_cost_pct']:.2f}%",
                  f"${cheapest_fast['pred_cost_usd']:.2f} fee")
        st.caption(f"**{cheapest_fast['firm']}** — "
                   f"{cheapest_fast['payment instrument'].lower()} → "
                   f"{cheapest_fast['pickup method'].lower()}")
    else:
        st.metric("Cheapest under one hour", "—")
        st.caption("No services in this corridor settle within an hour.")

with summary_cols[2]:
    st.metric("Corridor average",
              f"{avg:.2f}%",
              f"You save ${savings_vs_avg:.2f} vs. average")
    st.caption(f"Cheapest is {sdg_status}")


# ---------------------------------------------------------------------------
# Full ranked comparison table
# ---------------------------------------------------------------------------
st.markdown(f"##### All {len(results)} services available, ranked cheapest first")

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
        return ["background-color: #e8f3ec"] * len(row)
    if row["Cost %"] <= 3:
        return ["background-color: #f2f8f4"] * len(row)
    if row["Cost %"] >= 8:
        return ["background-color: #fbecea"] * len(row)
    return [""] * len(row)


styled = (display
          .style
          .format({"Cost %": "{:.2f}%", "Cost ($)": "${:.2f}", "Receive ($)": "${:.2f}"})
          .apply(color_row, axis=1))
st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Channel guidance
# ---------------------------------------------------------------------------
with st.expander("Notes on channel costs"):
    st.markdown(
        "- Online + bank-account or mobile-wallet pickup is usually the cheapest "
        "combination on most corridors.\n"
        "- Cash-in / cash-out through agents is consistently the most expensive — "
        "the convenience premium is real.\n"
        "- Banks generally charge more than dedicated money-transfer operators "
        "(MTOs). Mobile-money providers and digital-only fintechs (Wise, Remitly, "
        "WorldRemit) tend to win on price.\n"
        "- Speed and price aren't always a trade-off — on competitive corridors "
        "the cheapest provider is also instant. Use the speed filter to check.\n"
        "- Cost % is regressive on amount: $200 transfers cost roughly 2–3× more "
        "as a percentage than $500. If you can consolidate, you save real money.\n"
        "- Provider FX margins move weekly. Cheapest today isn't necessarily "
        "cheapest next month — re-check before each send."
    )

with st.expander("How the model works"):
    st.markdown(
        f"- **Data**: World Bank Remittance Prices Worldwide, last 8 quarters.\n"
        f"- **Coverage**: {len(lookup):,} unique service combinations across "
        f"{lookup['corridor'].nunique()} corridors and {lookup['firm'].nunique()} firms.\n"
        f"- **Model**: XGBoost regression — R² ≈ 0.7+, MAE ≈ 1.5 percentage points "
        f"on the held-out test set.\n"
        f"- **Features**: log-transformed send amount, corridor competition, firm type, "
        f"payment instrument, sending location, speed, pickup method, and "
        f"source/destination region & income group.\n"
        f"- **Limitation**: predictions are *modelled fair price* based on historical "
        f"patterns, not live quotes. Always verify with the actual provider."
    )
