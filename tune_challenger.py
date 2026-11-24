"""Remediation of finding F-01 (overfitting): LightGBM challenger v1.1.

- Hyper-parameters tuned with Optuna on a validation fold carved out of the TRAINING
  sample, so the 61,503-applicant holdout stays untouched until the final evaluation
  (also addresses F-05, holdout reuse).
- Early stopping on the validation fold.
- Monotonic constraints on risk drivers with an unambiguous business direction.
- Objective rewards validation AUC and penalises a train-validation gap above 0.02.

Run after compare_models.py and validate_models.py:
    python tune_challenger.py
"""
from __future__ import annotations

import json
from getpass import getuser
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

import risk_metrics as rm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DASH = OUT / "dashboard"
N_TRIALS = 40
SEED = 42

# Expected direction of risk: +1 = higher value, higher PD; -1 = higher value, lower PD.
MONOTONE = {
    "employment_years": -1,
    "due_date_shortfall_rate": +1,
    "due_date_shortfall_count": +1,
    "observed_underpaid_count": +1,
    "refused_count": +1,
    "approval_rate": -1,
}

# 1. Data and the identical holdout split
engine = create_engine(f"postgresql+psycopg2://{getuser()}@/home_credit_risk?host=/tmp")
df = pd.read_sql_query("SELECT * FROM risk_analytics.customer_feature_mart_v2", engine)
engine.dispose()
X = df.drop(columns=["applicant_id", "target", "age_years"], errors="ignore")
y = df["target"].astype(int)
categorical = X.select_dtypes(exclude=np.number).columns.tolist()
for col in categorical:
    X[col] = X[col].astype("category")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

saved = pd.read_csv(OUT / "model_predictions.csv")
assert np.array_equal(df.loc[X_test.index, "applicant_id"].to_numpy(), saved["applicant_id"].to_numpy())

# Validation fold inside the training sample
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=7
)
features = X.columns.tolist()
monotone = [MONOTONE.get(f, 0) for f in features]
fit_set = lgb.Dataset(X_fit, y_fit, categorical_feature=categorical, free_raw_data=False)
val_set = lgb.Dataset(X_val, y_val, reference=fit_set, free_raw_data=False)

BASE = {
    "objective": "binary", "metric": "auc", "verbosity": -1, "seed": SEED,
    "num_threads": 4, "monotone_constraints": monotone,
    "monotone_constraints_method": "advanced", "bagging_freq": 1,
}


def objective(trial):
    params = {
        **BASE,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 48),
        "min_child_samples": trial.suggest_int("min_child_samples", 200, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
        "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 50.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "cat_smooth": trial.suggest_float("cat_smooth", 10, 100),
    }
    booster = lgb.train(
        params, fit_set, num_boost_round=2000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    val_auc = roc_auc_score(y_val, booster.predict(X_val, num_iteration=booster.best_iteration))
    fit_auc = roc_auc_score(y_fit, booster.predict(X_fit, num_iteration=booster.best_iteration))
    gap = fit_auc - val_auc
    trial.set_user_attr("best_iteration", booster.best_iteration)
    trial.set_user_attr("val_auc", val_auc)
    trial.set_user_attr("fit_auc", fit_auc)
    trial.set_user_attr("gap", gap)
    return val_auc - 0.5 * max(0.0, gap - 0.02)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

trials = study.trials_dataframe()
trials.to_csv(OUT / "tuning_trials.csv", index=False)
best = study.best_trial
print(f"Best trial {best.number}: val AUC {best.user_attrs['val_auc']:.4f}, "
      f"gap {best.user_attrs['gap']:.4f}, iterations {best.user_attrs['best_iteration']}")

# 2. Refit on the full training sample with the selected number of trees
final_params = {**BASE, **best.params}
n_trees = int(best.user_attrs["best_iteration"])
final = lgb.train(final_params, lgb.Dataset(X_train, y_train, categorical_feature=categorical),
                  num_boost_round=n_trees)
final.save_model(DASH / "lightgbm_v11_model.txt")

p_train = final.predict(X_train)
p_test = final.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, p_test)
metrics = {
    "model": "LightGBM v1.1",
    "validation_method": "Stratified random holdout (untouched during tuning)",
    "roc_auc": float(roc_auc_score(y_test, p_test)),
    "pr_auc": float(average_precision_score(y_test, p_test)),
    "ks": float(np.max(tpr - fpr)),
    "brier_score": float(brier_score_loss(y_test, p_test)),
    "train_roc_auc": float(roc_auc_score(y_train, p_train)),
    "n_trees": n_trees,
    "tuning": {"trials": N_TRIALS, "best_trial": best.number,
               "validation_auc": best.user_attrs["val_auc"], "validation_gap": best.user_attrs["gap"]},
    "params": {k: (float(v) if isinstance(v, float) else v) for k, v in best.params.items()},
    "monotone_constraints": MONOTONE,
}
metrics["auc_gap"] = metrics["train_roc_auc"] - metrics["roc_auc"]
(OUT / "lightgbm_v11_metrics.json").write_text(json.dumps(metrics, indent=2))

# 3. Predictions, generalisation gap and bootstrap evidence
saved["lightgbm_v11_score"] = p_test
saved.to_csv(OUT / "model_predictions.csv", index=False)

gap = pd.read_csv(OUT / "generalization_gap.csv")
gap = gap[gap["Model"] != "LightGBM v1.1"]
gap = pd.concat([gap, pd.DataFrame([{
    "Model": "LightGBM v1.1",
    "Train ROC-AUC": metrics["train_roc_auc"], "Holdout ROC-AUC": metrics["roc_auc"],
    "AUC Gap": metrics["auc_gap"],
    "Train PR-AUC": float(average_precision_score(y_train, p_train)),
    "Holdout PR-AUC": metrics["pr_auc"],
    "Train Brier": float(brier_score_loss(y_train, p_train)),
    "Holdout Brier": metrics["brier_score"],
}])], ignore_index=True)
gap.to_csv(OUT / "generalization_gap.csv", index=False)

y_h = saved["target"].to_numpy()
rm.paired_bootstrap(y_h, saved["logistic_score"], p_test).to_csv(OUT / "bootstrap_v11_vs_logistic.csv", index=False)
rm.paired_bootstrap(y_h, saved["lightgbm_score"], p_test).to_csv(OUT / "bootstrap_v11_vs_v10.csv", index=False)

print(json.dumps({k: metrics[k] for k in ["roc_auc", "ks", "pr_auc", "brier_score", "train_roc_auc", "auc_gap", "n_trees"]}, indent=2))
