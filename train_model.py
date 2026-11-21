
import json
from pathlib import Path
from getpass import getuser
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve
)
from sklearn.calibration import calibration_curve

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# 1. Connect to Postgres.app via local Unix socket
username = getuser()
url = f"postgresql+psycopg2://{username}@/home_credit_risk?host=/tmp"
engine = create_engine(url)

# 2. Extract the validated feature mart
print("Loading customer feature mart...")
df = pd.read_sql_query(
    "SELECT * FROM risk_analytics.customer_feature_mart_v2",
    engine
)
engine.dispose()

assert len(df) == 307511, "Unexpected row count"
assert df["applicant_id"].is_unique, "Duplicate applicants found"
assert df["target"].notna().all(), "Missing TARGET detected"

print("Data shape:", df.shape)
print("TARGET rate:", round(df["target"].mean(), 4))

# 3. Prepare features
# Age excluded from this initial predictive baseline
X = df.drop(
    columns=["applicant_id", "target", "age_years"],
    errors="ignore"
)
y = df["target"].astype(int)

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 4. Stratified holdout split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 5. Train the Logistic Regression baseline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])
print("Training Logistic Regression...")
model.fit(X_train, y_train)

# 6. Holdout predictions and validation metrics
probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

metrics = {
    "model": "Logistic Regression",
    "validation_method": "Stratified random holdout",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "target_rate_test": float(y_test.mean()),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "pr_auc": float(average_precision_score(
        y_test, probabilities
    )),
    "ks": float(np.max(tpr - fpr)),
    "brier_score": float(brier_score_loss(
        y_test, probabilities
    ))
}

print("\nMODEL VALIDATION RESULTS")
for key, value in metrics.items():
    print(f"{key}: {value}")

# 7. Save reproducible model and metrics
joblib.dump(model, OUTPUT / "logistic_model.joblib")
with open(OUTPUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

predictions = pd.DataFrame({
    "applicant_id": df.loc[X_test.index, "applicant_id"].to_numpy(),
    "target": y_test.to_numpy(),
    "predicted_risk": probabilities
})
predictions.to_csv(OUTPUT / "holdout_predictions.csv", index=False)

# 8. ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], "--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Credit Risk Model - ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "roc_curve.png", dpi=180)
plt.close()

# 9. Calibration Curve
observed, predicted = calibration_curve(
    y_test, probabilities, n_bins=10, strategy="quantile"
)
plt.figure(figsize=(7, 5))
plt.plot(predicted, observed, "o-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed TARGET Rate")
plt.title("Credit Risk Model - Calibration")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "calibration_curve.png", dpi=180)
plt.close()

print("\nFinished. Results saved in outputs/")
