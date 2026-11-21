
import json
from pathlib import Path
from getpass import getuser
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. Load the same PostgreSQL feature mart
user = getuser()
engine = create_engine(
    f"postgresql+psycopg2://{user}@/home_credit_risk?host=/tmp"
)
df = pd.read_sql_query(
    "SELECT * FROM risk_analytics.customer_feature_mart_v2",
    engine
)
engine.dispose()

assert len(df) == 307511
assert df["applicant_id"].is_unique
assert df["target"].notna().all()

# 2. Match the original Logistic Regression feature set
X = df.drop(
    columns=["applicant_id", "target", "age_years"],
    errors="ignore"
).copy()
y = df["target"].astype(int)

categorical_cols = X.select_dtypes(
    exclude=np.number
).columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")

# LightGBM natively handles missing values
# and pandas categorical features.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 3. Verify that both models use identical holdout IDs
baseline_file = OUT / "holdout_predictions.csv"
baseline = pd.read_csv(baseline_file)
test_ids = df.loc[X_test.index, "applicant_id"].to_numpy()

if not np.array_equal(
    baseline["applicant_id"].to_numpy(), test_ids
):
    raise ValueError("Holdout IDs do not match the baseline!")

if not np.array_equal(
    baseline["target"].to_numpy(), y_test.to_numpy()
):
    raise ValueError("Holdout labels do not match!")

print("Holdout verified:", len(test_ids), "applicants")

# 4. Train LightGBM - fixed initial configuration
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    colsample_bytree=0.85,
    subsample=0.85,
    subsample_freq=1,
    reg_lambda=1.0,
    importance_type="gain",
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

print("Training LightGBM...")
model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_cols
)

prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, prob)

metrics = {
    "model": "LightGBM",
    "validation_method": "Stratified random holdout",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "pr_auc": float(average_precision_score(y_test, prob)),
    "ks": float(np.max(tpr - fpr)),
    "brier_score": float(brier_score_loss(y_test, prob))
}

with open(OUT / "lightgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, OUT / "lightgbm_model.joblib")

# 5. Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUT / "feature_importance.csv", index=False
)

# 6. Model comparison
with open(OUT / "metrics.json") as f:
    base_metrics = json.load(f)

comparison = pd.DataFrame({
    "Metric": ["ROC-AUC", "PR-AUC", "KS", "Brier Score"],
    "Logistic Regression": [
        base_metrics["roc_auc"],
        base_metrics["pr_auc"],
        base_metrics["ks"],
        base_metrics["brier_score"]
    ],
    "LightGBM": [
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["ks"],
        metrics["brier_score"]
    ]
})
comparison["Difference"] = (
    comparison["LightGBM"] -
    comparison["Logistic Regression"]
)
comparison.to_csv(OUT / "model_comparison.csv", index=False)

# 7. Save combined ROC chart
base_prob = baseline["predicted_risk"].to_numpy()
fpr_base, tpr_base, _ = roc_curve(y_test, base_prob)

plt.figure(figsize=(8, 5))
plt.plot(
    fpr_base, tpr_base,
    label=f"Logistic AUC {base_metrics['roc_auc']:.3f}"
)
plt.plot(
    fpr, tpr,
    label=f"LightGBM AUC {metrics['roc_auc']:.3f}"
)
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "model_comparison_roc.png", dpi=180)
plt.close()

# 8. Precision-Recall chart
plt.figure(figsize=(8, 5))
for name, scores in [
    ("Logistic Regression", base_prob),
    ("LightGBM", prob)
]:
    precision, recall, _ = precision_recall_curve(
        y_test, scores
    )
    plt.plot(recall, precision, label=name)
plt.axhline(
    y_test.mean(),
    linestyle="--",
    label="Prevalence baseline"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "model_comparison_pr.png", dpi=180)
plt.close()

# 9. Calibration comparison
plt.figure(figsize=(8, 5))
for name, scores in [
    ("Logistic Regression", base_prob),
    ("LightGBM", prob)
]:
    observed, predicted = calibration_curve(
        y_test,
        scores,
        n_bins=10,
        strategy="quantile"
    )
    plt.plot(predicted, observed, "o-", label=name)
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed TARGET Rate")
plt.title("Calibration Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUT / "model_comparison_calibration.png", dpi=180
)
plt.close()

# 10. Save comparable predictions
pd.DataFrame({
    "applicant_id": test_ids,
    "target": y_test.to_numpy(),
    "logistic_score": base_prob,
    "lightgbm_score": prob
}).to_csv(OUT / "model_predictions.csv", index=False)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False, float_format="%.4f"))
print("\nSaved outputs successfully.")
