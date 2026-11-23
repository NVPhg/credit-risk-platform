"""Credit-risk metrics, validation tests and RAG rules shared by the app and the report.

Everything here is pure NumPy / pandas / SciPy so it can be unit-tested and reused
outside Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- scaling
# 600 points at good:bad odds of 20:1; every 20 points doubles the odds.
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 20
FACTOR = PDO / np.log(2)
GRADES = [("A", 630), ("B", 610), ("C", 590), ("D", 570), ("E", 550), ("F", -np.inf)]
GRADE_ORDER = [g for g, _ in GRADES]


def pd_to_score(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    score = BASE_SCORE + FACTOR * np.log(((1 - p) / p) / BASE_ODDS)
    return float(score) if score.ndim == 0 else score


def score_to_pd(score):
    odds = BASE_ODDS * np.exp((np.asarray(score, dtype=float) - BASE_SCORE) / FACTOR)
    return 1 / (1 + odds)


def score_to_grade(score):
    score = np.atleast_1d(score)
    out = np.full(score.shape, "F", dtype=object)
    for grade, lower in reversed(GRADES):
        out[score >= lower] = grade
    return out


def grade_range(grade):
    bounds = dict(GRADES)
    i = GRADE_ORDER.index(grade)
    lower = bounds[grade]
    if i == 0:
        return f"≥ {lower:.0f}"
    upper = bounds[GRADE_ORDER[i - 1]]
    if np.isinf(lower):
        return f"< {upper:.0f}"
    return f"{lower:.0f} – {upper - 1:.0f}"


# --------------------------------------------------------------------------- discrimination
def ks_curve(y, p, points=400):
    """Cumulative bad and good distributions ordered from riskiest to safest."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="stable")
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    pop = np.arange(1, len(ys) + 1) / len(ys)
    gap = cum_bad - cum_good
    k = int(np.argmax(gap))
    idx = np.unique(np.r_[np.linspace(0, len(ys) - 1, points).astype(int), k])
    curve = pd.DataFrame({
        "population": pop[idx], "cum_bad": cum_bad[idx], "cum_good": cum_good[idx],
    })
    return curve, float(gap[k]), float(pop[k]), float(np.asarray(p)[order][k])


def lorenz_curve(y, p, points=400):
    """Cumulative share of bads captured vs share of population (riskiest first)."""
    curve, _, _, _ = ks_curve(y, p, points)
    out = pd.concat([pd.DataFrame({"population": [0.0], "cum_bad": [0.0]}),
                     curve[["population", "cum_bad"]]], ignore_index=True)
    auc = np.trapezoid(out["cum_bad"], out["population"])
    bad_rate = float(np.mean(y))
    gini = (auc - 0.5) / (0.5 * (1 - bad_rate))  # accuracy ratio
    return out, float(gini)


def gains_table(y, p, n_bins=10):
    """Decile gains/lift table; decile 1 = riskiest 10%."""
    y = np.asarray(y)
    p = np.asarray(p)
    rank = pd.Series(-p).rank(method="first")
    decile = pd.qcut(rank, n_bins, labels=range(1, n_bins + 1)).astype(int).to_numpy()
    df = pd.DataFrame({"decile": decile, "y": y, "p": p})
    t = df.groupby("decile").agg(
        applicants=("y", "size"), bads=("y", "sum"),
        min_pd=("p", "min"), max_pd=("p", "max"), avg_pd=("p", "mean"),
    ).reset_index()
    t["bad_rate"] = t["bads"] / t["applicants"]
    t["cum_bads_captured"] = t["bads"].cumsum() / t["bads"].sum()
    t["cum_population"] = t["applicants"].cumsum() / t["applicants"].sum()
    t["lift"] = t["bad_rate"] / y.mean()
    t["cum_lift"] = t["cum_bads_captured"] / t["cum_population"]
    return t


# --------------------------------------------------------------------------- calibration
def hosmer_lemeshow(y, p, groups=10):
    y = np.asarray(y)
    p = np.asarray(p)
    bins = pd.qcut(pd.Series(p).rank(method="first"), groups, labels=False).to_numpy()
    df = pd.DataFrame({"g": bins, "y": y, "p": p}).groupby("g").agg(
        n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    stat = (((df["obs"] - df["exp"]) ** 2) / (df["exp"] * (1 - df["exp"] / df["n"]))).sum()
    pvalue = 1 - stats.chi2.cdf(stat, groups - 2)
    return float(stat), float(pvalue)


def binomial_grade_test(grades, y, p, alpha=0.05):
    """Two-sided binomial test of observed defaults vs mean predicted PD, per grade."""
    df = pd.DataFrame({"grade": grades, "y": np.asarray(y), "p": np.asarray(p)})
    rows = []
    for grade in GRADE_ORDER:
        g = df[df["grade"] == grade]
        if g.empty:
            continue
        n, k, pd_hat = len(g), int(g["y"].sum()), float(g["p"].mean())
        res = stats.binomtest(k, n, pd_hat)
        rows.append({
            "grade": grade, "applicants": n, "observed_bads": k,
            "observed_rate": k / n, "predicted_pd": pd_hat,
            "expected_bads": n * pd_hat, "p_value": res.pvalue,
            "result": "Pass" if res.pvalue >= alpha else "Fail",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- stability
def psi_table(expected, actual, bins=10):
    """Population Stability Index with bins set on the expected (development) sample."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.Series(pd.cut(expected, edges, include_lowest=True)).value_counts(sort=False, normalize=True)
    a = pd.Series(pd.cut(actual, edges, include_lowest=True)).value_counts(sort=False, normalize=True)
    t = pd.DataFrame({"expected_pct": e.to_numpy(), "actual_pct": a.to_numpy()},
                     index=[str(i) for i in e.index])
    eps = 1e-6
    t["psi"] = (t["actual_pct"] - t["expected_pct"]) * np.log(
        (t["actual_pct"] + eps) / (t["expected_pct"] + eps))
    return t.reset_index(names="bin"), float(t["psi"].sum())


# --------------------------------------------------------------------------- RAG
RAG_COLORS = {"Green": "#2BB673", "Amber": "#FFB36B", "Red": "#FF6B7A", "Not assessed": "#8A97A8"}

RAG_THRESHOLDS = [
    ("Discrimination (Gini)", "Green ≥ 0.40 · Amber 0.30–0.40 · Red < 0.30"),
    ("Separation (KS)", "Green ≥ 0.30 · Amber 0.20–0.30 · Red < 0.20"),
    ("Benchmark significance", "Green if the 95% bootstrap CI of ΔAUC excludes 0"),
    ("Overfitting (train − holdout AUC)", "Green < 0.02 · Amber 0.02–0.05 · Red > 0.05"),
    ("Calibration (grade binomial tests)", "Green: all grades pass · Amber: 1 fails · Red: 2+ fail"),
    ("Rank ordering", "Green if observed rate is strictly monotonic across grades"),
    ("Stability (score PSI)", "Green < 0.10 · Amber 0.10–0.25 · Red > 0.25"),
    ("Fairness (age impact ratio)", "Green ≥ 0.80 · Amber 0.60–0.80 · Red < 0.60"),
    ("Data quality", "Green if all mart assertions pass"),
]


def band(value, green, amber, higher_is_better=True):
    if higher_is_better:
        return "Green" if value >= green else "Amber" if value >= amber else "Red"
    return "Green" if value < green else "Amber" if value <= amber else "Red"


def overall_rating(statuses):
    statuses = list(statuses)
    if "Red" in statuses or "Not assessed" in statuses:
        return "Not approved for production"
    if "Amber" in statuses:
        return "Approved with conditions"
    return "Approved"


# --------------------------------------------------------------------------- scorecard
AGE_BANDS = ([0, 25, 35, 45, 55, 65, 120], ["<25", "25-34", "35-44", "45-54", "55-64", "65+"])


def age_impact(pred, score_col, approval_rate=0.80):
    """Approval rate per age band when approving the lowest-PD share of applicants."""
    cutoff = np.quantile(pred[score_col], approval_rate)
    band = pd.cut(pred["age_years"], AGE_BANDS[0], labels=AGE_BANDS[1], right=False)
    t = (pd.DataFrame({"band": band, "approved": pred[score_col] <= cutoff, "y": pred["target"]})
         .groupby("band", observed=True)
         .agg(applicants=("approved", "size"), approval_rate=("approved", "mean"),
              observed_rate=("y", "mean"))
         .reset_index())
    t["impact_ratio"] = t["approval_rate"] / t["approval_rate"].max()
    return t


def validation_scorecard(pred, boot, gap, psi, dq, model="LightGBM"):
    """RAG assessment of one model. Returns (table, overall rating)."""
    col = {"LightGBM": "lightgbm_score", "Logistic Regression": "logistic_score"}[model]
    y, p = pred["target"].to_numpy(), pred[col].to_numpy()
    _, gini = lorenz_curve(y, p)
    _, ks, _, _ = ks_curve(y, p)
    grades = score_to_grade(pd_to_score(p))
    binom = binomial_grade_test(grades, y, p)
    fails = int((binom["result"] == "Fail").sum())
    rates = binom["observed_rate"].to_numpy()
    monotonic = bool(np.all(np.diff(rates) > 0))
    model_gap = float(gap.set_index("Model").loc[model, "AUC Gap"])
    model_psi = float(psi[psi["model"] == model]["total_psi"].iloc[0])
    impact = age_impact(pred, col)
    min_ir = float(impact["impact_ratio"].min())
    worst_band = str(impact.loc[impact["impact_ratio"].idxmin(), "band"])
    auc = boot.set_index("Metric").loc["ROC-AUC"]
    _, hl_p = hosmer_lemeshow(y, p)

    rows = [
        ("Discrimination (Gini)", f"{gini:.3f}", band(gini, 0.40, 0.30),
         "Gini = 2·AUC − 1 on the 61,503-applicant holdout."),
        ("Separation (KS)", f"{ks:.3f}", band(ks, 0.30, 0.20),
         "Maximum distance between cumulative bad and good distributions."),
        ("Benchmark significance",
         (f"ΔAUC {auc['Difference']:+.3f} [{auc['Difference CI Low']:+.3f}, {auc['Difference CI High']:+.3f}]"
          if model == "LightGBM" else "This model is the benchmark"),
         ("Green" if auc["Difference CI Low"] > 0 else "Amber") if model == "LightGBM" else "Not assessed",
         "Paired stratified bootstrap, 500 resamples, challenger vs logistic benchmark."),
        ("Overfitting (train − holdout AUC)", f"{model_gap:.3f}",
         band(model_gap, 0.02, 0.05, higher_is_better=False),
         "Gap between in-sample and holdout ROC-AUC."),
        ("Calibration (grade binomial tests)", f"{len(binom) - fails}/{len(binom)} grades pass · HL p = {hl_p:.3f}",
         "Green" if fails == 0 else "Amber" if fails == 1 else "Red",
         "Two-sided binomial test of observed vs predicted bads per grade (α = 5%)."),
        ("Rank ordering", "Monotonic" if monotonic else "Not monotonic",
         "Green" if monotonic else "Red", "Observed bad rate across grades A → F."),
        ("Stability (score PSI)", f"{model_psi:.4f}", band(model_psi, 0.10, 0.25, higher_is_better=False),
         "Train vs holdout. Random split, so a low PSI is expected; out-of-time PSI is still required."),
        ("Fairness (age impact ratio)", f"{min_ir:.2f} ({worst_band})", band(min_ir, 0.80, 0.60),
         "Lowest approval-rate ratio vs the best-treated age band at 80% approval."),
        ("Data quality", f"{int(dq['passed'].sum())}/{len(dq)} checks pass",
         "Green" if dq["passed"].all() else "Red", "Grain, label, sentinel, range and holdout-integrity assertions."),
        ("Out-of-time performance", "—", "Not assessed", "Dataset has no application dates."),
        ("Point-in-time feature audit", "—", "Not assessed",
         "Behavioural aggregates may include post-application records."),
    ]
    table = pd.DataFrame(rows, columns=["test", "result", "status", "note"])
    return table, overall_rating(table["status"])
