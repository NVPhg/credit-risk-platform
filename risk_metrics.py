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
RAG_COLORS = {"Green": "#1F9E5A", "Amber": "#D9822B", "Red": "#D64550", "Not assessed": "#7A8699"}

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
    """Any Red blocks approval; Amber or unassessed tests become approval conditions."""
    statuses = list(statuses)
    if "Red" in statuses:
        return "Not approved for production"
    if "Amber" in statuses or "Not assessed" in statuses:
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


def validation_scorecard(pred, boot, gap, psi, dq, model="LightGBM v1.1", col=None,
                         gap_name=None, psi_name=None):
    """RAG assessment of one model. Returns (table, overall rating).

    boot: paired bootstrap of this model vs the logistic benchmark (None for the benchmark
    itself or when not run). gap_name / psi_name: the model's row label in those files.
    """
    col = col or {"LightGBM v1.1": "lightgbm_v11_score", "LightGBM v1.0": "lightgbm_score",
                  "LightGBM": "lightgbm_score", "WoE Scorecard": "scorecard_score",
                  "Logistic Regression": "logistic_score"}[model]
    gap_name = gap_name or model
    psi_name = psi_name or model
    y, p = pred["target"].to_numpy(), pred[col].to_numpy()
    _, gini = lorenz_curve(y, p)
    _, ks, _, _ = ks_curve(y, p)
    grades = score_to_grade(pd_to_score(p))
    binom = binomial_grade_test(grades, y, p)
    fails = int((binom["result"] == "Fail").sum())
    rates = binom["observed_rate"].to_numpy()
    monotonic = bool(np.all(np.diff(rates) > 0))
    model_gap = float(gap.set_index("Model").loc[gap_name, "AUC Gap"])
    psi_rows = psi[psi["model"] == psi_name]
    model_psi = float(psi_rows["total_psi"].iloc[0]) if len(psi_rows) else np.nan
    impact = age_impact(pred, col)
    min_ir = float(impact["impact_ratio"].min())
    worst_band = str(impact.loc[impact["impact_ratio"].idxmin(), "band"])
    _, hl_p = hosmer_lemeshow(y, p)

    if boot is not None:
        auc = boot.set_index("Metric").loc["ROC-AUC"]
        bench = (f"ΔAUC {auc['Difference']:+.3f} [{auc['Difference CI Low']:+.3f}, "
                 f"{auc['Difference CI High']:+.3f}]",
                 "Green" if auc["Difference CI Low"] > 0 else "Red" if auc["Difference CI High"] < 0 else "Amber")
    elif model == "Logistic Regression":
        bench = ("This model is the benchmark", "Not assessed")
    else:
        bench = ("Bootstrap not run for this model", "Not assessed")

    rows = [
        ("Discrimination (Gini)", f"{gini:.3f}", band(gini, 0.40, 0.30),
         "Gini = 2·AUC − 1 on the 61,503-applicant holdout."),
        ("Separation (KS)", f"{ks:.3f}", band(ks, 0.30, 0.20),
         "Maximum distance between cumulative bad and good distributions."),
        ("Benchmark significance", bench[0], bench[1],
         "Paired stratified bootstrap, 500 resamples, vs the logistic benchmark."),
        ("Overfitting (train − holdout AUC)", f"{model_gap:.3f}",
         band(model_gap, 0.02, 0.05, higher_is_better=False),
         "Gap between in-sample and holdout ROC-AUC."),
        ("Calibration (grade binomial tests)", f"{len(binom) - fails}/{len(binom)} grades pass · HL p = {hl_p:.3f}",
         "Green" if fails == 0 else "Amber" if fails == 1 else "Red",
         "Two-sided binomial test of observed vs predicted bads per grade (α = 5%)."),
        ("Rank ordering", "Monotonic" if monotonic else "Not monotonic",
         "Green" if monotonic else "Red", "Observed bad rate across grades A → F."),
        ("Stability (score PSI)", f"{model_psi:.4f}" if not np.isnan(model_psi) else "—",
         band(model_psi, 0.10, 0.25, higher_is_better=False) if not np.isnan(model_psi) else "Not assessed",
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


# --------------------------------------------------------------------------- bootstrap
BOOT_METRICS = ["ROC-AUC", "Average Precision", "KS", "Brier Score"]


def _boot_metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    return np.array([roc_auc_score(y, p), average_precision_score(y, p),
                     np.max(tpr - fpr), brier_score_loss(y, p)])


def paired_bootstrap(y, base, challenger, n_boot=500, seed=42):
    """Paired stratified bootstrap of challenger − base (same format as validate_models.py)."""
    y = np.asarray(y, dtype=int)
    base, challenger = np.asarray(base), np.asarray(challenger)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    b0, c0 = _boot_metrics(y, base), _boot_metrics(y, challenger)
    bb = np.empty((n_boot, 4))
    cc = np.empty((n_boot, 4))
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bb[i], cc[i] = _boot_metrics(y[idx], base[idx]), _boot_metrics(y[idx], challenger[idx])
    rows = []
    for j, metric in enumerate(BOOT_METRICS):
        d = cc[:, j] - bb[:, j]
        lo, hi = np.quantile(d, [0.025, 0.975])
        if metric == "Brier Score":
            evidence = "Challenger lower" if hi < 0 else "Challenger higher" if lo > 0 else "Inconclusive"
        else:
            evidence = "Challenger higher" if lo > 0 else "Challenger lower" if hi < 0 else "Inconclusive"
        rows.append({
            "Metric": metric, "Base": b0[j], "Challenger": c0[j],
            "Difference": c0[j] - b0[j], "Difference CI Low": lo, "Difference CI High": hi,
            "Evidence": evidence,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Basel IRB, EL, pricing
PD_FLOOR = 0.0005  # Basel III floor for retail PDs (5 bp)


def irb_retail_k(pd_, lgd):
    """Basel IRB capital requirement K (share of EAD) for 'other retail' exposures.

    R = 0.03·w + 0.16·(1 − w), w = (1 − e^(−35·PD)) / (1 − e^(−35))
    K = LGD · N[(G(PD) + √R · G(0.999)) / √(1 − R)] − PD · LGD   (no maturity adjustment for retail)
    """
    pd_ = np.clip(np.asarray(pd_, dtype=float), PD_FLOOR, 0.9999)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    r = 0.03 * w + 0.16 * (1 - w)
    cond = stats.norm.cdf((stats.norm.ppf(pd_) + np.sqrt(r) * stats.norm.ppf(0.999)) / np.sqrt(1 - r))
    return np.maximum(lgd * cond - pd_ * lgd, 0.0)


def stress_pd(pd_, odds_multiplier):
    """Scale the odds of default: a uniform logit shift preserving rank order."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    odds = pd_ / (1 - pd_) * odds_multiplier
    return odds / (1 + odds)


def grade_economics(pd_, ead, grades, lgd=0.45, funding=0.06, opex=0.03, hurdle=0.15):
    """Per-grade expected loss, IRB capital and minimum risk-based price (annual rate)."""
    df = pd.DataFrame({"grade": grades, "pd": np.asarray(pd_), "ead": np.asarray(ead, dtype=float)})
    df["k"] = irb_retail_k(df["pd"], lgd)
    df["el"] = df["pd"] * lgd * df["ead"]
    df["capital"] = df["k"] * df["ead"]
    t = df.groupby("grade").agg(
        applicants=("pd", "size"), avg_pd=("pd", "mean"), ead=("ead", "sum"),
        el=("el", "sum"), capital=("capital", "sum"),
    ).reindex([g for g in GRADE_ORDER if g in set(grades)]).reset_index()
    t["el_rate"] = t["el"] / t["ead"]
    t["k_rate"] = t["capital"] / t["ead"]
    t["rwa"] = t["capital"] * 12.5
    t["min_rate"] = funding + opex + t["el_rate"] + t["k_rate"] * hurdle
    return t


STRESS_SCENARIOS = {
    "Baseline": {"odds": 1.0, "lgd_add": 0.00},
    "Adverse": {"odds": 1.5, "lgd_add": 0.05},
    "Severely adverse": {"odds": 2.25, "lgd_add": 0.10},
}


def stress_test(pd_, ead, lgd=0.45, cutoff_score=None, scenarios=STRESS_SCENARIOS):
    """Portfolio EL, capital and approval under PD/LGD stress, cut-off score held fixed."""
    ead = np.asarray(ead, dtype=float)
    rows = []
    for name, s in scenarios.items():
        p = stress_pd(pd_, s["odds"])
        l = min(lgd + s["lgd_add"], 1.0)
        scores = pd_to_score(p)
        grades = score_to_grade(scores)
        approved = scores >= cutoff_score if cutoff_score is not None else np.ones(len(p), bool)
        k = irb_retail_k(p, l)
        rows.append({
            "scenario": name, "odds_multiplier": s["odds"], "lgd": l,
            "avg_pd": float(p.mean()),
            "expected_loss": float((p * l * ead).sum()),
            "el_rate": float((p * l * ead).sum() / ead.sum()),
            "capital": float((k * ead).sum()),
            "rwa": float((k * ead).sum() * 12.5),
            "approval_rate": float(approved.mean()),
            "approved_expected_bad_rate": float(p[approved].mean()) if approved.any() else np.nan,
            "share_grade_ef": float(np.isin(grades, ["E", "F"]).mean()),
        })
    return pd.DataFrame(rows)
