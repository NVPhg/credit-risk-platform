"""Traditional WoE / IV application scorecard (third benchmark).

- Optimal binning with monotonic WoE trends (OptBinning, MIP solver).
- Variable selection by Information Value (IV >= 0.02).
- Logistic regression on WoE, scaled to points: 600 at 20:1 good:bad odds, 20 PDO,
  the same scale as the dashboard.
- Trained on the identical training sample; scored on the untouched holdout.

Run after train_model.py / compare_models.py:
    python build_scorecard.py
"""
from __future__ import annotations

import json
from getpass import getuser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

import risk_metrics as rm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SC = OUT / "scorecard"
SC.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"postgresql+psycopg2://{getuser()}@/home_credit_risk?host=/tmp")
df = pd.read_sql_query("SELECT * FROM risk_analytics.customer_feature_mart_v2", engine)
engine.dispose()

X = df.drop(columns=["applicant_id", "target", "age_years"], errors="ignore")
y = df["target"].astype(int)
categorical = X.select_dtypes(exclude=np.number).columns.tolist()
for col in categorical:
    X[col] = X[col].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

saved = pd.read_csv(OUT / "model_predictions.csv")
assert np.array_equal(df.loc[X_test.index, "applicant_id"].to_numpy(), saved["applicant_id"].to_numpy())

variables = X.columns.tolist()
binning_fit_params = {
    v: {"monotonic_trend": "auto_asc_desc", "min_prebin_size": 0.02, "max_n_bins": 8}
    for v in variables if v not in categorical
}
binning = BinningProcess(
    variable_names=variables,
    categorical_variables=categorical,
    selection_criteria={"iv": {"min": 0.02, "max": 1.0}},
    binning_fit_params=binning_fit_params,
    min_bin_size=0.02,
    n_jobs=4,
)
scorecard = Scorecard(
    binning_process=binning,
    estimator=LogisticRegression(max_iter=2000, C=1.0),
    scaling_method="pdo_odds",
    scaling_method_params={"pdo": rm.PDO, "odds": rm.BASE_ODDS, "scorecard_points": rm.BASE_SCORE},
    intercept_based=False,
    reverse_scorecard=False,
    rounding=True,
)
scorecard.fit(X_train, y_train)

p_test = scorecard.predict_proba(X_test)[:, 1]
p_train = scorecard.predict_proba(X_train)[:, 1]
points_test = scorecard.score(X_test)
fpr, tpr, _ = roc_curve(y_test, p_test)

# IV table
summary = scorecard.binning_process_.summary()[["name", "dtype", "status", "selected", "n_bins", "iv", "js", "gini"]]
summary = summary.sort_values("iv", ascending=False)
summary.to_csv(SC / "iv_table.csv", index=False)

# Points table (attribute → points)
table = scorecard.table(style="detailed")
points = table[["Variable", "Bin", "Count (%)", "Event rate", "WoE", "IV", "Points"]].copy()
points = points[~points["Bin"].astype(str).isin(["Special", "Missing"]) | (points["Count (%)"] > 0)]
points.to_csv(SC / "points_table.csv", index=False)

metrics = {
    "model": "WoE scorecard",
    "roc_auc": float(roc_auc_score(y_test, p_test)),
    "pr_auc": float(average_precision_score(y_test, p_test)),
    "ks": float(np.max(tpr - fpr)),
    "brier_score": float(brier_score_loss(y_test, p_test)),
    "train_roc_auc": float(roc_auc_score(y_train, p_train)),
    "n_variables": int(summary["selected"].sum()),
    "points_range": [float(points_test.min()), float(points_test.max())],
}
metrics["auc_gap"] = metrics["train_roc_auc"] - metrics["roc_auc"]
(OUT / "scorecard_metrics.json").write_text(json.dumps(metrics, indent=2))
joblib.dump(scorecard, SC / "scorecard.joblib")

saved["scorecard_score"] = p_test
saved["scorecard_points"] = points_test
saved.to_csv(OUT / "model_predictions.csv", index=False)

gap = pd.read_csv(OUT / "generalization_gap.csv")
gap = gap[gap["Model"] != "WoE Scorecard"]
gap = pd.concat([gap, pd.DataFrame([{
    "Model": "WoE Scorecard", "Train ROC-AUC": metrics["train_roc_auc"],
    "Holdout ROC-AUC": metrics["roc_auc"], "AUC Gap": metrics["auc_gap"],
    "Train PR-AUC": float(average_precision_score(y_train, p_train)), "Holdout PR-AUC": metrics["pr_auc"],
    "Train Brier": float(brier_score_loss(y_train, p_train)), "Holdout Brier": metrics["brier_score"],
}])], ignore_index=True)
gap.to_csv(OUT / "generalization_gap.csv", index=False)

print(json.dumps(metrics, indent=2))
print(summary.head(12).to_string(index=False))
