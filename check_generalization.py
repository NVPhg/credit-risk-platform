
from pathlib import Path
from getpass import getuser
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT = Path(__file__).resolve().parent / "outputs"

engine = create_engine(
    f"postgresql+psycopg2://{getuser()}@/home_credit_risk?host=/tmp"
)
df = pd.read_sql_query(
    "SELECT * FROM risk_analytics.customer_feature_mart_v2",
    engine
)
engine.dispose()

X = df.drop(columns=["applicant_id", "target", "age_years"], errors="ignore")
y = df["target"].astype(int)

# Match the categorical representation used during LightGBM training
for col in X.select_dtypes(exclude=np.number).columns:
    X[col] = X[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

saved = pd.read_csv(OUT / "model_predictions.csv")
assert np.array_equal(
    df.loc[X_test.index, "applicant_id"].to_numpy(),
    saved["applicant_id"].to_numpy()
), "Holdout mismatch"

models = {
    "Logistic Regression": joblib.load(OUT / "logistic_model.joblib"),
    "LightGBM": joblib.load(OUT / "lightgbm_model.joblib")
}

rows = []
for name, model in models.items():
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)
    rows.append({
        "Model": name,
        "Train ROC-AUC": train_auc,
        "Holdout ROC-AUC": test_auc,
        "AUC Gap": train_auc - test_auc,
        "Train PR-AUC": average_precision_score(y_train, train_prob),
        "Holdout PR-AUC": average_precision_score(y_test, test_prob),
        "Train Brier": brier_score_loss(y_train, train_prob),
        "Holdout Brier": brier_score_loss(y_test, test_prob)
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "generalization_gap.csv", index=False)

print("\nTRAIN VS HOLDOUT VALIDATION")
print(results.to_string(index=False, float_format="%.4f"))
