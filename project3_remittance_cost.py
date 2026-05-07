"""
============================================================================
PROJECT 3 — Predicting the All-In Cost of a Remittance
A regression model with a fairness/equity audit
Capstone format mirrors the Credit Card Churn deck (Logit / RF / AdaBoost / XGBoost)
============================================================================

Business question:
    What is the expected total cost % for a remittance with a given combination
    of corridor, firm type, payment instrument, send amount, and pickup method —
    and which features drive cost the most?

Dataset:
    World Bank Remittance Prices Worldwide (RPW) — sheet "Dataset (from Q2 2016)"
    ~198,000 quotes, 42 columns, covers 2016 Q2 through 2025 Q1.

Pipeline (mirrors the churn project):
    SECTION 01 · FRAMING  → defined in the docstring above
    SECTION 02 · DATA     → load, clean target, drop low-volume corridors
    SECTION 03 · METHODOLOGY → feature engineering & encoding
    SECTION 04 · MODELS      → 4 models: Linear, RF, GB, XGBoost
    SECTION 05 · RESULTS     → metrics table
    SECTION 06 · INTERPRETABILITY → feature importance plots
    SECTION 07 · INSIGHTS    → fairness audit + key takeaways
    SECTION 08 · ACTION      → "Fair Price Calculator" prediction helper

How to run:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
    python project3_remittance_cost.py

Inputs:
    rpw_dataset_2011_2025_q1.xlsx   (place next to this script, or edit DATA_PATH)

Outputs (all written to ./outputs/):
    model_comparison.csv            — RMSE / MAE / R² per model
    feature_importance_rf.png       — top features from Random Forest
    feature_importance_xgb.png      — top features from XGBoost
    fairness_residuals_region.png   — residual distribution by destination region
    predictions_test.csv            — predicted vs actual on the held-out test set
    fair_price_examples.csv         — predicted "fair" cost for sample corridors
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "rpw_dataset_2011_2025_q1.xlsx"
SHEET_NAME = "Dataset (from Q2 2016)"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Total cost (%) of transaction"

CATEGORICAL_FEATURES = [
    "firm_type",
    "payment instrument",
    "Sending location",
    "speed actual",
    "pickup method",
    "source_region",
    "destination_region",
    "source_income",
    "destination_income",
]

NUMERIC_FEATURES = [
    "log_send_usd",
    "corridor_firm_count",  # engineered: number of competing firms in the corridor-quarter
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# SECTION 02 · DATA — load and clean
# ===========================================================================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the post-Q2-2016 sheet from the RPW workbook."""
    print(f"[load] reading {path} (this takes ~1 minute)...")
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    df = df.dropna(axis=1, how="all")
    print(f"[load] raw shape: {df.shape}")
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, negative cost (FX-margin artifacts), and >50% extreme outliers."""
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] >= 0) & (df[TARGET] <= 50)].copy()
    print(f"[clean] target rows: {before:,} -> {len(df):,} "
          f"(mean={df[TARGET].mean():.2f}%, median={df[TARGET].median():.2f}%)")
    return df


def drop_low_volume_corridors(df: pd.DataFrame, min_quotes: int = 20) -> pd.DataFrame:
    """Drop corridors with fewer than `min_quotes` total quotes (cold-start)."""
    counts = df["corridor"].value_counts()
    keep = counts[counts >= min_quotes].index
    out = df[df["corridor"].isin(keep)].copy()
    print(f"[clean] corridors: {df['corridor'].nunique()} -> {out['corridor'].nunique()}")
    return out


# ===========================================================================
# SECTION 03 · METHODOLOGY — feature engineering
# ===========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric and categorical features used by all four models."""
    # log-transform the send amount (cost % is regressive on amount)
    df["log_send_usd"] = np.log1p(
        pd.to_numeric(df["Surveyed send amount (USD)"], errors="coerce").fillna(200)
    )

    # corridor competition: how many distinct firms quote that corridor in that period
    firm_count = (
        df.groupby(["corridor", "period"])["firm"]
        .nunique()
        .rename("corridor_firm_count")
        .reset_index()
    )
    df = df.merge(firm_count, on=["corridor", "period"], how="left")

    # fill categorical NaNs with a sentinel string
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].fillna("UNK").astype(str)

    return df


def build_design_matrix(df: pd.DataFrame):
    """One-hot encode categoricals, concat with numeric features, return X, y, feature_names."""
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)
    X_num = df[NUMERIC_FEATURES].astype(float)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = df[TARGET].values
    print(f"[features] X shape: {X.shape}  ({X_num.shape[1]} numeric + "
          f"{X_cat.shape[1]} one-hot dummies)")
    return X, y, list(X.columns)


# ===========================================================================
# SECTION 04 · MODELS — train all four
# ===========================================================================
def evaluate(name, model, X_train, X_test, y_train, y_test):
    """Fit, predict, and return a metrics dict."""
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    metrics = {
        "model": name,
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_tr)),
        "rmse_test":  np.sqrt(mean_squared_error(y_test,  y_pred_te)),
        "mae_test":   mean_absolute_error(y_test, y_pred_te),
        "r2_train":   r2_score(y_train, y_pred_tr),
        "r2_test":    r2_score(y_test,  y_pred_te),
    }
    print(f"  [{name:<20}] RMSE_test={metrics['rmse_test']:.3f}  "
          f"MAE_test={metrics['mae_test']:.3f}  "
          f"R²_test={metrics['r2_test']:.3f}  "
          f"(train R²={metrics['r2_train']:.3f})")
    return model, metrics, y_pred_te


def train_all_models(X_train, X_test, y_train, y_test):
    print("\n[models] training four regressors...")

    # MODEL 01 · BASELINE — Linear regression
    lin_model, lin_m, lin_pred = evaluate(
        "LinearRegression",
        LinearRegression(),
        X_train, X_test, y_train, y_test,
    )

    # MODEL 02 · ENSEMBLE — Random Forest
    # Tuned with a small GridSearchCV; comment out the grid to speed up.
    rf_grid = {
        "n_estimators": [200],
        "max_depth": [None, 20],
        "min_samples_leaf": [2],
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    print("  [RF] running 5-fold GridSearchCV on a small grid...")
    rf_search = GridSearchCV(rf, rf_grid, scoring="r2", cv=5, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    print(f"  [RF] best params: {rf_search.best_params_}")
    rf_model, rf_m, rf_pred = evaluate(
        "RandomForest", rf_best, X_train, X_test, y_train, y_test,
    )

    # MODEL 03 · BOOSTED — Gradient Boosting (sklearn — analog of AdaBoost in churn deck)
    gb_model, gb_m, gb_pred = evaluate(
        "GradientBoosting",
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        X_train, X_test, y_train, y_test,
    )

    # MODEL 04 · BOOSTED — XGBoost (the recommended model in the churn deck)
    xgb_model, xgb_m, xgb_pred = evaluate(
        "XGBoost",
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        X_train, X_test, y_train, y_test,
    )

    metrics_df = pd.DataFrame([lin_m, rf_m, gb_m, xgb_m])
    return {
        "LinearRegression": lin_model,
        "RandomForest":     rf_model,
        "GradientBoosting": gb_model,
        "XGBoost":          xgb_model,
    }, metrics_df, {
        "LinearRegression": lin_pred,
        "RandomForest":     rf_pred,
        "GradientBoosting": gb_pred,
        "XGBoost":          xgb_pred,
    }


# ===========================================================================
# SECTION 06 · INTERPRETABILITY — feature importance plots
# ===========================================================================
def plot_feature_importance(model, feature_names, title: str, outfile: str, top_k: int = 20):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, color="steelblue")
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"  [plot] saved {outfile}")


# ===========================================================================
# SECTION 07 · INSIGHTS — fairness audit on residuals by destination region
# ===========================================================================
def fairness_audit(df_test, y_test, y_pred_xgb, outfile: str) -> pd.DataFrame:
    df_test = df_test.copy()
    df_test["actual"] = y_test
    df_test["pred"] = y_pred_xgb
    df_test["residual"] = y_test - y_pred_xgb

    summary = (df_test.groupby("destination_region")["residual"]
               .agg(["mean", "median", "std", "count"])
               .round(3)
               .sort_values("mean", ascending=False))
    print("\n[fairness] residual (actual - predicted) by destination_region:")
    print(summary)

    plt.figure(figsize=(9, 5))
    order = summary.index.tolist()
    sns.boxplot(
        data=df_test[df_test["destination_region"].isin(order)],
        x="destination_region", y="residual", order=order,
        showfliers=False, color="steelblue",
    )
    plt.axhline(0, ls="--", c="red", lw=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Residual (actual - predicted) %")
    plt.title("Fairness audit: cost prediction residuals by destination region\n"
              "(positive = corridor pays more than the model predicts)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"  [plot] saved {outfile}")
    return summary


# ===========================================================================
# SECTION 08 · ACTION — Fair Price Calculator helper
# ===========================================================================
def predict_fair_price(model, feature_names, sample_rows: pd.DataFrame) -> pd.DataFrame:
    """Score new corridor-firm-instrument combinations.

    `sample_rows` must already be passed through engineer_features() and
    one-hot-encoded with the same dummy columns as training (use .reindex).
    """
    aligned = sample_rows.reindex(columns=feature_names, fill_value=0)
    preds = model.predict(aligned.values)
    out = sample_rows.copy()
    out["predicted_total_cost_pct"] = preds.round(2)
    return out


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # ---- Section 02 — Data
    df = load_data(DATA_PATH)
    df = clean_target(df)
    df = drop_low_volume_corridors(df, min_quotes=20)

    # ---- Section 03 — Methodology
    df = engineer_features(df)
    X, y, feature_names = build_design_matrix(df)

    # 80/20 random split (a temporal split — train ≤2022, test 2023-2025 — is a good extension)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[split] train={len(X_train):,}  test={len(X_test):,}")

    # ---- Section 04 — Models
    models, metrics_df, preds = train_all_models(X_train, X_test, y_train, y_test)

    # ---- Section 05 — Results
    metrics_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[save] {metrics_path}")
    print("\n=== MODEL COMPARISON ===")
    print(metrics_df.to_string(index=False))

    # ---- Section 06 — Feature importance (RF + XGB)
    plot_feature_importance(
        models["RandomForest"], feature_names,
        "Random Forest — Top 20 Features",
        os.path.join(OUTPUT_DIR, "feature_importance_rf.png"),
    )
    plot_feature_importance(
        models["XGBoost"], feature_names,
        "XGBoost — Top 20 Features",
        os.path.join(OUTPUT_DIR, "feature_importance_xgb.png"),
    )

    # ---- Section 07 — Fairness audit on best model (XGBoost)
    summary = fairness_audit(
        df_test, y_test, preds["XGBoost"],
        outfile=os.path.join(OUTPUT_DIR, "fairness_residuals_region.png"),
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "fairness_residuals_region.csv"))

    # save predictions
    out_pred = df_test[["corridor", "firm", "firm_type", "payment instrument",
                        "speed actual", "Surveyed send amount (USD)",
                        "destination_region"]].copy()
    out_pred["actual_cost_pct"] = y_test
    out_pred["pred_cost_pct"]   = preds["XGBoost"].round(3)
    out_pred["residual"]        = (out_pred["actual_cost_pct"] - out_pred["pred_cost_pct"]).round(3)
    out_pred.to_csv(os.path.join(OUTPUT_DIR, "predictions_test.csv"), index=False)
    print(f"[save] predictions_test.csv ({len(out_pred):,} rows)")

    # ---- Section 08 — Example "fair price" lookups for a few well-known corridors
    examples = (df_test
                .groupby("corridor")
                .first()
                .reset_index()
                .head(15)[["corridor", "firm_type", "payment instrument", "pickup method",
                           "speed actual", "Surveyed send amount (USD)"]])
    examples_X = X_test.iloc[:15].copy() if hasattr(X_test, "iloc") else pd.DataFrame(X_test[:15], columns=feature_names)
    examples["predicted_fair_cost_pct"] = preds["XGBoost"][:15].round(2)
    examples.to_csv(os.path.join(OUTPUT_DIR, "fair_price_examples.csv"), index=False)
    print(f"[save] fair_price_examples.csv (15 corridors)")

    # ---- Persist the XGBoost model + feature schema for the Streamlit demo ----
    with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(models["XGBoost"], f)
    cat_options = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    schema = {"feature_names": feature_names, "categorical_options": cat_options}
    with open(os.path.join(OUTPUT_DIR, "feature_schema.pkl"), "wb") as f:
        pickle.dump(schema, f)
    print(f"[save] xgb_model.pkl + feature_schema.pkl  (used by streamlit_app.py)")

    print("\n=== DONE ===  outputs/ folder contains all artifacts.")


if __name__ == "__main__":
    main()
