from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve
)

OUT = Path(__file__).resolve().parent / "outputs"
PREDICTIONS = OUT / "model_predictions.csv"
N_BOOTSTRAP = 500
SEED = 42

if not PREDICTIONS.exists():
    raise FileNotFoundError(
        "Run compare_models.py before validation."
    )

df = pd.read_csv(PREDICTIONS)

required = [
    "applicant_id",
    "target",
    "logistic_score",
    "lightgbm_score"
]
if not set(required).issubset(df.columns):
    raise ValueError("Missing required prediction columns")

if df["applicant_id"].duplicated().any():
    raise ValueError("Duplicate applicant IDs found")

if df[required].isna().any().any():
    raise ValueError("Missing values in predictions")

y = df["target"].to_numpy(dtype=int)
logistic = df["logistic_score"].to_numpy(dtype=float)
lightgbm = df["lightgbm_score"].to_numpy(dtype=float)

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Target must contain both binary classes")

for scores in [logistic, lightgbm]:
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction detected")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction outside [0, 1]")

METRICS = [
    "ROC-AUC",
    "Average Precision",
    "KS",
    "Brier Score"
]

def calculate_metrics(actual, scores):
    fpr, tpr, _ = roc_curve(actual, scores)
    return np.array([
        roc_auc_score(actual, scores),
        average_precision_score(actual, scores),
        np.max(tpr - fpr),
        brier_score_loss(actual, scores)
    ])

baseline = calculate_metrics(y, logistic)
challenger = calculate_metrics(y, lightgbm)

# Paired stratified bootstrap:
# resample the same applicant indices for both models,
# keeping the number of positives and negatives fixed.
rng = np.random.default_rng(SEED)
positive_idx = np.flatnonzero(y == 1)
negative_idx = np.flatnonzero(y == 0)

boot_base = np.empty((N_BOOTSTRAP, len(METRICS)))
boot_lgb = np.empty((N_BOOTSTRAP, len(METRICS)))

print("Running paired stratified bootstrap...")

for i in range(N_BOOTSTRAP):
    sampled_pos = rng.choice(
        positive_idx, size=len(positive_idx), replace=True
    )
    sampled_neg = rng.choice(
        negative_idx, size=len(negative_idx), replace=True
    )
    idx = np.concatenate([sampled_pos, sampled_neg])
    sampled_y = y[idx]
    boot_base[i] = calculate_metrics(
        sampled_y, logistic[idx]
    )
    boot_lgb[i] = calculate_metrics(
        sampled_y, lightgbm[idx]
    )
    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1}/{N_BOOTSTRAP}")

rows = []

for j, metric in enumerate(METRICS):
    diff = challenger[j] - baseline[j]
    boot_diff = boot_lgb[:, j] - boot_base[:, j]

    base_ci = np.quantile(boot_base[:, j], [0.025, 0.975])
    lgb_ci = np.quantile(boot_lgb[:, j], [0.025, 0.975])
    diff_ci = np.quantile(boot_diff, [0.025, 0.975])

    if metric == "Brier Score":
        evidence = (
            "LightGBM lower"
            if diff_ci[1] < 0 else "Inconclusive"
        )
    else:
        evidence = (
            "LightGBM higher"
            if diff_ci[0] > 0 else "Inconclusive"
        )

    rows.append({
        "Metric": metric,
        "Logistic": baseline[j],
        "Logistic CI Low": base_ci[0],
        "Logistic CI High": base_ci[1],
        "LightGBM": challenger[j],
        "LightGBM CI Low": lgb_ci[0],
        "LightGBM CI High": lgb_ci[1],
        "Difference": diff,
        "Difference CI Low": diff_ci[0],
        "Difference CI High": diff_ci[1],
        "Evidence": evidence
    })

results = pd.DataFrame(rows)
results.to_csv(
    OUT / "bootstrap_validation.csv", index=False
)

# Top-decile concentration on the same holdout
def top_decile_summary(scores):
    n = int(np.ceil(0.10 * len(scores)))
    top_indices = np.argsort(scores, kind="stable")[-n:]
    top_target_rate = y[top_indices].mean()
    overall_target_rate = y.mean()
    return {
        "target_rate": top_target_rate,
        "lift": top_target_rate / overall_target_rate,
        "capture": y[top_indices].sum() / y.sum()
    }

log_top = top_decile_summary(logistic)
lgb_top = top_decile_summary(lightgbm)

# Generate GitHub-friendly Markdown validation report
lines = [
    "# Credit Risk Model Validation Report",
    "",
    "## 1. Executive Summary",
    "",
    f"- Holdout sample: {len(df):,} applicants",
    f"- Observed TARGET=1 rate: {y.mean():.2%}",
    "- Baseline: Logistic Regression",
    "- Challenger: LightGBM",
    "- Validation: stratified random holdout (not out-of-time)",
    "",
    "## 2. Bootstrap Performance Comparison",
    "",
    f"Paired stratified bootstrap with {N_BOOTSTRAP} resamples, "
    "95% percentile confidence intervals.",
    "",
    "| Metric | Logistic | LightGBM | Difference | 95% CI for difference |",
    "|---|---:|---:|---:|---:|"
]

for _, row in results.iterrows():
    lines.append(
        f"| {row['Metric']} "
        f"| {row['Logistic']:.4f} "
        f"| {row['LightGBM']:.4f} "
        f"| {row['Difference']:+.4f} "
        f"| [{row['Difference CI Low']:+.4f}, "
        f"{row['Difference CI High']:+.4f}] |"
    )

lines.extend([
    "",
    "Positive differences favor LightGBM except for Brier Score, "
    "where a negative difference favors LightGBM.",
    "",
    "## 3. Risk Concentration",
    "",
    "| Indicator | Logistic | LightGBM |",
    "|---|---:|---:|",
    f"| Top 10% TARGET rate | {log_top['target_rate']:.2%} "
    f"| {lgb_top['target_rate']:.2%} |",
    f"| Top 10% lift | {log_top['lift']:.2f}x "
    f"| {lgb_top['lift']:.2f}x |",
    f"| Positive-case capture | {log_top['capture']:.2%} "
    f"| {lgb_top['capture']:.2%} |",
    "",
    "## 4. Validation Limitations",
    "",
    "- The holdout is random, not chronological.",
    "- The same holdout has been examined during model comparison; "
    "subsequent model selection can introduce selection bias.",
    "- Historical feature availability at application time "
    "has not been independently established.",
    "- In particular, installment-derived aggregates require "
    "a point-in-time leakage audit.",
    "- Bootstrap intervals quantify sampling uncertainty "
    "under the chosen resampling scheme. They do not resolve "
    "selection bias, data leakage, or distribution shift.",
    "- Calibration, fairness, stability, independent validation "
    "and external validation are not yet complete.",
    "- Home Credit TARGET is a payment-difficulty indicator, "
    "not automatically a regulatory probability of default.",
    "",
    "## 5. Validation Decision",
    "",
    "LightGBM is a promising challenger based on observed "
    "holdout performance. Approval for production use is "
    "not recommended without further validation.",
    "",
    "## 6. Reproducibility",
    "",
    f"- Bootstrap seed: {SEED}",
    f"- Bootstrap replications: {N_BOOTSTRAP}",
    "- Input: outputs/model_predictions.csv",
    "- Detailed results: outputs/bootstrap_validation.csv",
    "",
    "Research and portfolio demonstration only."
])

report = "\n".join(lines) + "\n"
(OUT / "model_validation_report.md").write_text(
    report, encoding="utf-8"
)

print("\nBOOTSTRAP VALIDATION RESULTS")
print(
    results[
        [
            "Metric",
            "Difference",
            "Difference CI Low",
            "Difference CI High",
            "Evidence"
        ]
    ].to_string(index=False, float_format="%.4f")
)
print("\nValidation report saved to outputs/")
