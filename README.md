# Project 3 — Predicting the All-In Cost of a Remittance

A regression model with a fairness/equity audit, mirroring the structure of the
Credit Card Churn capstone (Logit/RF/AdaBoost/XGBoost → Linear/RF/GB/XGBoost).

## Files in this folder

| File | Purpose |
|---|---|
| `project3_remittance_cost.py` | The full pipeline. Run this on your machine. |
| `sample_run_output.txt` | Real numbers produced by the data pipeline (numpy OLS baseline) so you can see what to expect before installing libs. |
| `preview_feature_importance.png` | Preview chart of the strongest features. |
| `preview_fairness_audit.png` | Preview chart of cost-prediction residuals by destination region. |
| `rpw_dataset_2011_2025_q1.xlsx` | Place the dataset here (rename if needed). |

## How to run

```bash
# 1. Install dependencies (you already have most of these from the churn project)
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl

# 2. Drop the dataset xlsx into this folder, then:
python project3_remittance_cost.py
```

The script writes all artifacts into a new `outputs/` folder:

- `model_comparison.csv` — RMSE, MAE, R² for all four models
- `feature_importance_rf.png` and `feature_importance_xgb.png`
- `fairness_residuals_region.png` and `.csv` — the equity audit
- `predictions_test.csv` — predicted vs actual for every test-set quote
- `fair_price_examples.csv` — sample "fair price" predictions for 15 corridors

## What the script does (mapped to the churn deck)

| Churn deck section | Project 3 equivalent |
|---|---|
| 01 Framing — business question | Predict `Total cost (%) of transaction` |
| 02 Data — 10,127 customers, 16% churn | 197k quotes, mean cost 6.55%, median 5.09% |
| 03 Methodology — encoding | log-transform send amount, one-hot encode 9 categoricals, drop low-volume corridors |
| 04 Model 01 — Logit baseline | **Linear Regression** baseline |
| 04 Model 02 — Random Forest + GridSearchCV | **Random Forest** + GridSearchCV (n_estimators, max_depth) |
| 04 Models 03 & 04 — AdaBoost & XGBoost | **Gradient Boosting** + **XGBoost** |
| 05 Results — F1 / ROC-AUC / Precision / Recall | RMSE / MAE / R² (regression analogs) |
| 06 Interpretability — feature importances | RF and XGB feature-importance plots |
| 07 Insights — behavioral signals | Send amount, firm type, instrument, region |
| 08 Action plan — retention strategies | "Fair Price Calculator" — flag overpriced services |
| 09 ROI — bank revenue | $650B annual flow; 1pp cost reduction = $6.5B saved |

## Talking points for the presentation

1. **Send amount is the biggest cost driver.** Small remittances ($200) cost
   2–3× more in % terms than larger ones ($500+). The model quantifies this
   regressivity.
2. **Mobile-money and fintech firm types reduce cost; banks and post offices
   raise it** — even after controlling for corridor and instrument.
3. **The fairness audit is the differentiator.** Unlike the churn project, we
   can plot residuals by destination region and ask: *after controlling for
   every feature we have, do corridors serving poorer regions still pay a
   structural premium?* That's a publishable finding either way.
4. **Tied to the speed thesis.** Same features that drive cost down (mobile
   wallet, internet sending, fintech firm) also drive speed up. The story is
   "structural friction in remittance corridors" — cost and speed are two
   sides of the same coin.
