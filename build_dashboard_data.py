"""Build the compact data assets the Streamlit dashboard reads.

Run after train_model.py and compare_models.py. The dashboard itself never
touches PostgreSQL, so everything written here can be committed and the app
can be deployed (e.g. Streamlit Community Cloud) without a database.

Outputs (outputs/dashboard/):
  holdout_scored.parquet   holdout applicants: features, age, both scores
  shap_sample.parquet      LightGBM SHAP contributions for a holdout sample
  shap_global.csv          mean |SHAP| per feature over the full holdout
  segment_risk.csv         TARGET rate by segment over all 307,511 applicants
  lightgbm_model.txt       portable LightGBM booster for the what-if simulator
"""
from getpass import getuser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

import risk_metrics as rm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DASH = OUT / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)
SHAP_SAMPLE_SIZE = 5000
SEED = 42

# 1. Load the same feature mart and reproduce the holdout split
engine = create_engine(
    f"postgresql+psycopg2://{getuser()}@/home_credit_risk?host=/tmp"
)
df = pd.read_sql_query(
    "SELECT * FROM risk_analytics.customer_feature_mart_v2", engine
)
engine.dispose()
assert len(df) == 307511 and df["applicant_id"].is_unique

X = df.drop(columns=["applicant_id", "target", "age_years"], errors="ignore")
y = df["target"].astype(int)
for col in X.select_dtypes(exclude=np.number).columns:
    X[col] = X[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

saved = pd.read_csv(OUT / "model_predictions.csv")
test_ids = df.loc[X_test.index, "applicant_id"].to_numpy()
assert np.array_equal(test_ids, saved["applicant_id"].to_numpy()), "Holdout mismatch"
assert np.array_equal(y_test.to_numpy(), saved["target"].to_numpy()), "Label mismatch"
print("Holdout verified:", len(test_ids), "applicants")

# 2. Holdout table with features, age (fairness review only) and scores
holdout = X_test.copy()
holdout.insert(0, "applicant_id", test_ids)
holdout.insert(1, "target", y_test.to_numpy())
holdout["age_years"] = df.loc[X_test.index, "age_years"].to_numpy()
holdout["logistic_score"] = saved["logistic_score"].to_numpy()
holdout["lightgbm_score"] = saved["lightgbm_score"].to_numpy()
for extra in ["lightgbm_v11_score", "scorecard_score", "scorecard_points"]:
    if extra in saved.columns:
        holdout[extra] = saved[extra].to_numpy()
holdout.reset_index(drop=True).to_parquet(DASH / "holdout_scored.parquet", index=False)

# 3. SHAP contributions (TreeSHAP via pred_contrib) for the current challenger:
#    LightGBM v1.1 when tune_challenger.py has been run, otherwise v1.0.
model = joblib.load(OUT / "lightgbm_model.joblib")
booster_v10 = model.booster_
booster_v10.save_model(DASH / "lightgbm_model.txt")
v11_path = DASH / "lightgbm_v11_model.txt"
if v11_path.exists() and "lightgbm_v11_score" in saved.columns:
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(v11_path))
    challenger_col = "lightgbm_v11_score"
else:
    booster = booster_v10
    challenger_col = "lightgbm_score"
features = booster.feature_name()

contrib = booster.predict(X_test[features], pred_contrib=True)
# contrib columns: one per feature + expected value (log-odds space)
shap_values = contrib[:, :-1]
recomputed = 1 / (1 + np.exp(-contrib.sum(axis=1)))
assert np.allclose(recomputed, saved[challenger_col].to_numpy(), atol=1e-6)
pd.DataFrame({
    "feature": features,
    "importance": booster.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False).to_csv(DASH / "feature_importance_challenger.csv", index=False)

pd.DataFrame({
    "feature": features,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    "mean_shap": shap_values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).to_csv(
    DASH / "shap_global.csv", index=False
)

rng = np.random.default_rng(SEED)
sample_idx = np.sort(rng.choice(len(X_test), SHAP_SAMPLE_SIZE, replace=False))
shap_sample = pd.DataFrame(shap_values[sample_idx], columns=features).astype("float32")
shap_sample.insert(0, "applicant_id", test_ids[sample_idx])
shap_sample["base_value"] = contrib[sample_idx, -1]
shap_sample.to_parquet(DASH / "shap_sample.parquet", index=False)


# 4. Stability: development (train) vs holdout distributions
logistic = joblib.load(OUT / "logistic_model.joblib")
train_scores = {
    "Logistic Regression": logistic.predict_proba(X_train)[:, 1],
    "LightGBM v1.0": booster_v10.predict(X_train[features]),
}
test_scores = {
    "Logistic Regression": saved["logistic_score"].to_numpy(),
    "LightGBM v1.0": saved["lightgbm_score"].to_numpy(),
}
if challenger_col == "lightgbm_v11_score":
    train_scores["LightGBM v1.1"] = booster.predict(X_train[features])
    test_scores["LightGBM v1.1"] = saved["lightgbm_v11_score"].to_numpy()
sc_path = OUT / "scorecard" / "scorecard.joblib"
if sc_path.exists() and "scorecard_score" in saved.columns:
    sc_model = joblib.load(sc_path)
    X_train_sc = X_train.copy()
    for col in X_train_sc.select_dtypes(include="category").columns:
        X_train_sc[col] = X_train_sc[col].astype(str)
    train_scores["WoE Scorecard"] = sc_model.predict_proba(X_train_sc)[:, 1]
    test_scores["WoE Scorecard"] = saved["scorecard_score"].to_numpy()
psi_rows = []
for name in train_scores:
    table, total = rm.psi_table(train_scores[name], test_scores[name])
    table.insert(0, "model", name)
    table["total_psi"] = total
    psi_rows.append(table)
pd.concat(psi_rows, ignore_index=True).to_csv(DASH / "stability_score_psi.csv", index=False)

csi = []
for col in X_train.select_dtypes(include=np.number).columns:
    if X_train[col].nunique() < 3:
        continue
    _, value = rm.psi_table(X_train[col].to_numpy(), X_test[col].to_numpy())
    csi.append({"feature": col, "csi": value})
pd.DataFrame(csi).sort_values("csi", ascending=False).to_csv(
    DASH / "stability_feature_csi.csv", index=False
)

# 5. Data-quality assertions on the mart (mirrors sql/04_data_quality_checks.sql)
rate_cols = ["due_date_shortfall_rate", "approval_rate"]
dq = [
    ("Row count", "307,511", f"{len(df):,}", len(df) == 307511),
    ("Unique applicant_id", "0 duplicates", f"{df['applicant_id'].duplicated().sum():,} duplicates",
     not df["applicant_id"].duplicated().any()),
    ("TARGET completeness", "0 missing", f"{df['target'].isna().sum():,} missing", df["target"].notna().all()),
    ("TARGET values", "{0, 1}", str(sorted(df["target"].unique().tolist())),
     set(df["target"].unique()) <= {0, 1}),
    ("Employment sentinel recoded", "0 leaks",
     f"{int(((df['employment_sentinel_flag'] == 1) & df['employment_years'].notna()).sum())} leaks",
     not ((df["employment_sentinel_flag"] == 1) & df["employment_years"].notna()).any()),
    ("Rates within [0, 1]", "0 violations",
     f"{int(sum(((df[c] < 0) | (df[c] > 1)).sum() for c in rate_cols))} violations",
     all(df[c].dropna().between(0, 1).all() for c in rate_cols)),
    ("Non-negative amounts", "0 violations",
     f"{int((df[['annual_income', 'credit_amount', 'annuity_amount']] < 0).sum().sum())} violations",
     not (df[["annual_income", "credit_amount", "annuity_amount"]] < 0).any().any()),
    ("Holdout IDs identical across models", "match", "match", True),
]
pd.DataFrame(dq, columns=["check", "expected", "observed", "passed"]).to_csv(
    DASH / "data_quality.csv", index=False
)

# 6. Segment risk across the full portfolio (descriptive, not model-based)
segments = []

def add_segment(dimension, labels):
    grouped = (
        pd.DataFrame({"segment": labels, "target": y})
        .groupby("segment", observed=True)["target"]
        .agg(applicants="size", cases="sum", target_rate="mean")
        .reset_index()
    )
    grouped.insert(0, "dimension", dimension)
    grouped["segment"] = grouped["segment"].astype(str)
    grouped["order"] = range(len(grouped))
    segments.append(grouped)

for col, name in [
    ("income_type", "Income type"),
    ("education_type", "Education"),
    ("contract_type", "Contract type"),
]:
    add_segment(name, df[col].fillna("Unknown"))

add_segment("Age band", pd.cut(
    df["age_years"], [0, 25, 35, 45, 55, 65, 120],
    labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"], right=False,
))
employment = pd.cut(
    df["employment_years"], [0, 1, 3, 5, 10, 20, 100],
    labels=["<1y", "1-3y", "3-5y", "5-10y", "10-20y", "20y+"], right=False,
).astype(object)
employment[df["employment_sentinel_flag"] == 1] = "Not employed (sentinel)"
employment = employment.fillna("Unknown")
add_segment("Employment length", pd.Categorical(
    employment,
    categories=["<1y", "1-3y", "3-5y", "5-10y", "10-20y", "20y+",
                "Not employed (sentinel)", "Unknown"],
))
add_segment("Credit-to-income", pd.cut(
    df["credit_to_income"], [0, 1, 2, 3, 5, 10, np.inf],
    labels=["<1x", "1-2x", "2-3x", "3-5x", "5-10x", "10x+"], right=False,
))
shortfall = pd.cut(
    df["due_date_shortfall_rate"], [-0.001, 0, 0.05, 0.15, 0.30, 1.0],
    labels=["0%", "0-5%", "5-15%", "15-30%", "30%+"],
).astype(object)
shortfall[df["has_installment_history"] == 0] = "No history"
add_segment("Installment shortfall rate", pd.Categorical(
    shortfall.fillna("No history"),
    categories=["0%", "0-5%", "5-15%", "15-30%", "30%+", "No history"],
))
add_segment("Active bureau loans", pd.cut(
    df["active_loan_count"], [0, 1, 2, 3, 5, np.inf],
    labels=["0", "1", "2", "3-4", "5+"], right=False,
))
add_segment("Previously refused", pd.cut(
    df["refused_count"], [0, 1, 2, 4, np.inf],
    labels=["0", "1", "2-3", "4+"], right=False,
))

pd.concat(segments, ignore_index=True).to_csv(DASH / "segment_risk.csv", index=False)

print("Dashboard assets written to", DASH.relative_to(ROOT))
for path in sorted(DASH.iterdir()):
    print(f"  {path.name:28s} {path.stat().st_size / 1024:8.1f} KB")
