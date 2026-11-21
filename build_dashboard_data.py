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

_, X_test, _, y_test = train_test_split(
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
holdout.reset_index(drop=True).to_parquet(DASH / "holdout_scored.parquet", index=False)

# 3. SHAP contributions (LightGBM TreeSHAP via pred_contrib)
model = joblib.load(OUT / "lightgbm_model.joblib")
booster = model.booster_
features = booster.feature_name()

contrib = booster.predict(X_test[features], pred_contrib=True)
# contrib columns: one per feature + expected value (log-odds space)
shap_values = contrib[:, :-1]
recomputed = 1 / (1 + np.exp(-contrib.sum(axis=1)))
assert np.allclose(recomputed, saved["lightgbm_score"].to_numpy(), atol=1e-6)

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

booster.save_model(DASH / "lightgbm_model.txt")

# 4. Segment risk across the full portfolio (descriptive, not model-based)
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
