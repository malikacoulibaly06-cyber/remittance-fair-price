"""
build_lookup.py
---------------
Builds a deduplicated lookup table of every (source country, destination country,
firm, payment instrument, sending location, pickup method, speed) combination
that has appeared in the World Bank RPW dataset over the last 8 quarters.

The Streamlit comparison app uses this lookup to show users every available
service for their chosen corridor.

Run once after `project3_remittance_cost.py`:
    python build_lookup.py

Output:
    corridor_lookup.pkl  (~0.5 MB, commit alongside xgb_model.json + feature_schema.pkl)
"""

import os
import pandas as pd

DATA_PATH = "rpw_dataset_2011_2025_q1.xlsx"
SHEET_NAME = "Dataset (from Q2 2016)"
OUTFILE   = "corridor_lookup.pkl"
RECENT_QUARTERS = 8


def main():
    print(f"[load] reading {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df.dropna(axis=1, how="all")

    # Use only the most recent quarters so we don't show defunct firms
    periods = sorted([p for p in df["period"].dropna().unique()])
    recent = periods[-RECENT_QUARTERS:]
    df_r = df[df["period"].isin(recent)].copy()
    print(f"[filter] keeping last {RECENT_QUARTERS} quarters: {recent[0]} .. {recent[-1]}  "
          f"({len(df_r):,} rows)")

    # Most-recent firm count per corridor (drives the corridor_firm_count feature)
    latest = periods[-1]
    firm_count = (df[df["period"] == latest]
                  .groupby("corridor")["firm"]
                  .nunique()
                  .rename("corridor_firm_count")
                  .reset_index())

    # Dedup keeping the newest quote per (corridor, firm, channel)
    df_r = df_r.sort_values("period", ascending=False)
    key = ["corridor", "firm", "payment instrument", "Sending location", "pickup method"]
    lookup = df_r.drop_duplicates(subset=key, keep="first").copy()

    lookup = lookup.merge(firm_count, on="corridor", how="left")
    lookup["corridor_firm_count"] = lookup["corridor_firm_count"].fillna(5).astype(int)

    keep = ["source_code", "source_name", "source_region", "source_income",
            "destination_code", "destination_name", "destination_region", "destination_income",
            "firm", "firm_type", "payment instrument", "Sending location",
            "speed actual", "pickup method",
            "corridor", "corridor_firm_count"]
    lookup = (lookup[keep]
              .dropna(subset=["source_name", "destination_name", "firm"])
              .reset_index(drop=True))

    lookup.to_pickle(OUTFILE)
    size_mb = os.path.getsize(OUTFILE) / 1024 / 1024
    print(f"[save] {OUTFILE}  ({len(lookup):,} rows, {size_mb:.2f} MB)")
    print(f"       countries: {lookup['source_name'].nunique()} source / "
          f"{lookup['destination_name'].nunique()} destination")
    print(f"       corridors: {lookup['corridor'].nunique()}, firms: {lookup['firm'].nunique()}")


if __name__ == "__main__":
    main()
