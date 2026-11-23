"""Generate a bank-style Model Validation Report (HTML + PDF) from the project outputs.

Run after build_dashboard_data.py:
    python build_validation_report.py

Writes docs/Model_Validation_Report.html and, when Google Chrome / Chromium is available,
docs/Model_Validation_Report.pdf (rendered headless).
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

import risk_metrics as rm  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DASH = OUT / "dashboard"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- data
pred = pd.read_parquet(DASH / "holdout_scored.parquet")
boot = pd.read_csv(OUT / "bootstrap_validation.csv")
gap = pd.read_csv(OUT / "generalization_gap.csv")
psi = pd.read_csv(DASH / "stability_score_psi.csv")
csi = pd.read_csv(DASH / "stability_feature_csi.csv")
dq = pd.read_csv(DASH / "data_quality.csv")
shap = pd.read_csv(DASH / "shap_global.csv")
lgb_metrics = json.loads((OUT / "lightgbm_metrics.json").read_text())
lr_metrics = json.loads((OUT / "metrics.json").read_text())

y = pred["target"].to_numpy()
p_lgb = pred["lightgbm_score"].to_numpy()
p_lr = pred["logistic_score"].to_numpy()
rate = y.mean()

scorecard, rating = rm.validation_scorecard(pred, boot, gap, psi, dq, "LightGBM")
lr_scorecard, _ = rm.validation_scorecard(pred, boot, gap, psi, dq, "Logistic Regression")
grades = rm.score_to_grade(rm.pd_to_score(p_lgb))
binom = rm.binomial_grade_test(grades, y, p_lgb)
hl_stat, hl_p = rm.hosmer_lemeshow(y, p_lgb)
lr_hl_stat, lr_hl_p = rm.hosmer_lemeshow(y, p_lr)
_, ks_lgb, ks_pop, _ = rm.ks_curve(y, p_lgb)
_, gini_lgb = rm.lorenz_curve(y, p_lgb)
_, gini_lr = rm.lorenz_curve(y, p_lr)
gains = rm.gains_table(y, p_lgb)
impact = rm.age_impact(pred, "lightgbm_score")
g = gap.set_index("Model")
b = boot.set_index("Metric")

# ----------------------------------------------------------------------------- charts
INK, MUTED, GRID = "#1d2430", "#6b7686", "#e3e8ef"
TEAL, BLUE, RED, ORANGE = "#1F9E8C", "#3B7DD8", "#D64550", "#E8933A"
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200,
})


def fig_to_img(fig, width="100%"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width}">'


def chart_roc_ks():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for name, p, c in [("Logistic Regression", p_lr, BLUE), ("LightGBM", p_lgb, TEAL)]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=name)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    ax = axes[1]
    curve, ks, at, _ = rm.ks_curve(y, p_lgb)
    ax.plot(curve["population"] * 100, curve["cum_bad"] * 100, color=RED, lw=2, label="Cumulative bads")
    ax.plot(curve["population"] * 100, curve["cum_good"] * 100, color=TEAL, lw=2, label="Cumulative goods")
    k = curve.iloc[(curve["population"] - at).abs().argmin()]
    ax.vlines(k["population"] * 100, k["cum_good"] * 100, k["cum_bad"] * 100, color=ORANGE, lw=2, ls=":")
    ax.annotate(f"KS = {ks:.3f}", (k["population"] * 100 + 2, (k["cum_bad"] + k["cum_good"]) * 50),
                color=ORANGE, fontweight="bold")
    ax.set(xlabel="Applicants, riskiest first (%)", ylabel="Cumulative share (%)", title="KS: LightGBM")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_lorenz_dist():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for name, p, c, gn in [("Logistic Regression", p_lr, BLUE, gini_lr), ("LightGBM", p_lgb, TEAL, gini_lgb)]:
        curve, _ = rm.lorenz_curve(y, p)
        ax.plot(curve["population"] * 100, curve["cum_bad"] * 100, color=c, lw=2, label=f"{name} (Gini {gn:.3f})")
    ax.plot([0, 100], [0, 100], "--", color=MUTED, lw=1)
    ax.set(xlabel="Applicants declined, riskiest first (%)", ylabel="Bads captured (%)", title="Lorenz curve")
    ax.legend(loc="lower right")
    ax = axes[1]
    scores = rm.pd_to_score(p_lgb)
    bins = np.arange(480, 681, 5)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.55, color=TEAL, label="Good")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.55, color=RED, label="Bad")
    for grade, lower in rm.GRADES[:-1]:
        ax.axvline(lower, color=MUTED, lw=0.8, ls=":")
        ax.text(lower + 1, ax.get_ylim()[1] * 0.95, grade, color=MUTED, fontsize=8)
    ax.set(xlabel="Credit score", ylabel="Density", title="Score distribution by outcome")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    ax = axes[0]
    for name, p, c in [("Logistic Regression", p_lr, BLUE), ("LightGBM", p_lgb, TEAL)]:
        obs, exp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(exp, obs, "o-", color=c, lw=2, ms=4, label=name)
    ax.plot([0, 0.3], [0, 0.3], "--", color=MUTED, lw=1, label="Perfect")
    ax.set(xlim=(0, 0.3), ylim=(0, 0.3), xlabel="Mean predicted PD", ylabel="Observed rate",
           title="Reliability (deciles)")
    ax.legend(loc="upper left")
    ax = axes[1]
    x = np.arange(len(binom))
    ax.bar(x - 0.2, binom["observed_rate"] * 100, 0.4, color=TEAL, label="Observed")
    ax.bar(x + 0.2, binom["predicted_pd"] * 100, 0.4, color=MUTED, label="Predicted")
    ax.set_xticks(x, binom["grade"])
    ax.set(ylabel="Rate (%)", title="Observed vs predicted by grade: LightGBM")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_bootstrap_gap():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    labels = []
    for i, (_, r) in enumerate(boot.iterrows()):
        lo, hi, d = r["Difference CI Low"], r["Difference CI High"], r["Difference"]
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=TEAL, capsize=4, lw=2)
        labels.append(r["Metric"] + (" (lower is better)" if r["Metric"] == "Brier Score" else ""))
    ax.axvline(0, color=MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="LightGBM − Logistic (95% CI)", title="Paired bootstrap, 500 resamples")
    ax = axes[1]
    x = np.arange(2)
    ax.bar(x - 0.2, gap["Train ROC-AUC"], 0.4, color=MUTED, label="Train")
    ax.bar(x + 0.2, gap["Holdout ROC-AUC"], 0.4, color=TEAL, label="Holdout")
    for i, (_, r) in enumerate(gap.iterrows()):
        top = max(r["Train ROC-AUC"], r["Holdout ROC-AUC"])
        ax.text(i, top + 0.006, f"gap {r['AUC Gap']:+.3f}", ha="center", fontsize=8,
                color=RED if r["AUC Gap"] > 0.05 else INK)
    ax.set_xticks(x, ["Logistic", "LightGBM"])
    ax.set(ylim=(0.6, 0.8), ylabel="ROC-AUC", title="Generalisation")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_psi_shap():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    t = psi[psi["model"] == "LightGBM"]
    x = np.arange(1, len(t) + 1)
    ax.bar(x - 0.2, t["expected_pct"] * 100, 0.4, color=MUTED, label="Development")
    ax.bar(x + 0.2, t["actual_pct"] * 100, 0.4, color=TEAL, label="Holdout")
    ax.set_xticks(x)
    ax.set(xlabel="Score decile (development)", ylabel="Share (%)",
           title=f"Score stability: PSI = {t['total_psi'].iloc[0]:.4f}")
    ax.set_ylim(0, 13.5)
    ax.legend(loc="upper center", ncol=2)
    ax = axes[1]
    top = shap.head(10).iloc[::-1]
    ax.barh(top["feature"].str.replace("_", " "), top["mean_abs_shap"], color=TEAL)
    ax.set(xlabel="Mean |SHAP| (log-odds)", title="Top drivers: TreeSHAP")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig_to_img(fig)


def chart_fairness():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    colors = [RED if r < 0.8 else TEAL for r in impact["impact_ratio"]]
    ax.bar(impact["band"].astype(str), impact["approval_rate"] * 100, color=colors)
    ax2 = ax.twinx()
    ax2.plot(impact["band"].astype(str), impact["observed_rate"] * 100, "o-", color=INK, lw=1.5)
    ax2.set_ylabel("Observed TARGET rate (%)", color=INK)
    ax2.grid(False)
    for i, r in enumerate(impact.itertuples()):
        ax.text(i, r.approval_rate * 100 - 9, f"IR {r.impact_ratio:.2f}", ha="center", fontsize=8,
                color="white", fontweight="bold")
    ax.set(ylim=(0, 110), ylabel="Approval rate (%)", xlabel="Age band",
           title="Approval rate by age band at 80% overall approval (bars) vs observed risk (line)")
    fig.tight_layout()
    return fig_to_img(fig)


# ----------------------------------------------------------------------------- tables
def html_table(df, formats=None, cls=""):
    formats = formats or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, row in df.astype(object).iterrows():
        cells = ""
        for c in df.columns:
            v = row[c]
            f = formats.get(c)
            text = f(v) if callable(f) else (f.format(v) if f else v)
            cells += f"<td>{text}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def pill(status):
    return f'<span class="pill {status.replace(" ", "-").lower()}">{status}</span>'


FINDINGS = [
    ("F-01", "High", "Overfitting",
     f"LightGBM train ROC-AUC {g.loc['LightGBM', 'Train ROC-AUC']:.3f} vs holdout "
     f"{g.loc['LightGBM', 'Holdout ROC-AUC']:.3f} (gap {g.loc['LightGBM', 'AUC Gap']:.3f}); "
     "the benchmark shows no gap.",
     "Retrain with early stopping and stronger regularisation tuned on a separate validation fold; re-validate."),
    ("F-02", "High", "No out-of-time validation",
     "The dataset has no application dates, so temporal drift and out-of-time performance cannot be measured.",
     "Obtain dated applications and run an out-of-time test plus monthly PSI before approval."),
    ("F-03", "High", "Point-in-time leakage risk",
     "Installment, bureau and previous-application aggregates use full history; availability at the application date is not evidenced.",
     "Restrict aggregates to records dated before each application and re-measure performance."),
    ("F-04", "Medium", "Age-band disparity",
     f"At 80% approval, applicants under 25 are approved at {impact['approval_rate'].iloc[0]:.0%} vs "
     f"{impact['approval_rate'].max():.0%} for the best-treated band (impact ratio {impact['impact_ratio'].min():.2f}).",
     "Document the business justification, test proxy features (e.g. employment length) and monitor outcomes by age."),
    ("F-05", "Medium", "Holdout reuse",
     "The same holdout was used to compare models; further tuning against it would bias results.",
     "Reserve an untouched test sample for the final approval decision."),
    ("F-06", "Low", "Label definition",
     "TARGET flags payment difficulty and is not a Basel / IFRS 9 default definition.",
     "Map to a 90+ DPD definition before any regulatory or provisioning use."),
    ("F-07", "Low", "Illustrative economics",
     "Cut-off profitability uses assumed margin (10%) and loss (50%) rates.",
     "Replace with product-level margin, LGD and funding cost assumptions."),
]

def sev_pill(sev):
    return f'<span class="sev {sev.lower()}">{sev}</span>'


findings_rows = "".join(
    f"<tr><td class='id'>{fid}</td><td>{sev_pill(sev)}</td><td><b>{title}</b><br>{desc}</td><td>{rec}</td></tr>"
    for fid, sev, title, desc, rec in FINDINGS
)
findings_table = (
    "<table class='findings'><thead><tr><th>ID</th><th>Severity</th><th>Finding</th>"
    f"<th>Recommendation</th></tr></thead><tbody>{findings_rows}</tbody></table>"
)

scorecard_table = html_table(
    scorecard.assign(status=scorecard["status"].map(pill)).rename(columns={
        "test": "Test", "result": "Result", "status": "Status", "note": "Basis"}),
    cls="rag",
)
perf = pd.DataFrame([
    ("ROC-AUC", lr_metrics["roc_auc"], lgb_metrics["roc_auc"]),
    ("Gini (accuracy ratio)", gini_lr, gini_lgb),
    ("KS", lr_metrics["ks"], lgb_metrics["ks"]),
    ("PR-AUC", lr_metrics["pr_auc"], lgb_metrics["pr_auc"]),
    ("Brier score (↓)", lr_metrics["brier_score"], lgb_metrics["brier_score"]),
], columns=["Metric", "Logistic Regression", "LightGBM"])
perf["Difference"] = perf["LightGBM"] - perf["Logistic Regression"]
perf_table = html_table(perf, {"Logistic Regression": "{:.4f}", "LightGBM": "{:.4f}", "Difference": "{:+.4f}"}, "num")

boot_table = html_table(
    boot[["Metric", "Difference", "Difference CI Low", "Difference CI High", "Evidence"]].rename(
        columns={"Difference CI Low": "95% CI low", "Difference CI High": "95% CI high"}),
    {"Difference": "{:+.4f}", "95% CI low": "{:+.4f}", "95% CI high": "{:+.4f}"}, "num",
)
gains_table = html_table(
    gains[["decile", "applicants", "bads", "bad_rate", "cum_bads_captured", "lift", "cum_lift"]].rename(columns={
        "decile": "Decile", "applicants": "Applicants", "bads": "Bads", "bad_rate": "Bad rate",
        "cum_bads_captured": "Cum. bads captured", "lift": "Lift", "cum_lift": "Cum. lift"}),
    {"Applicants": "{:,}", "Bads": "{:,}", "Bad rate": "{:.2%}", "Cum. bads captured": "{:.1%}",
     "Lift": "{:.2f}x", "Cum. lift": "{:.2f}x"}, "num",
)
binom_table = html_table(
    binom[["grade", "applicants", "observed_bads", "expected_bads", "observed_rate", "predicted_pd", "p_value", "result"]]
    .rename(columns={"grade": "Grade", "applicants": "N", "observed_bads": "Observed bads",
                     "expected_bads": "Expected bads", "observed_rate": "Observed", "predicted_pd": "Predicted",
                     "p_value": "p-value", "result": "Result"}),
    {"N": "{:,}", "Observed bads": "{:,}", "Expected bads": "{:,.0f}", "Observed": "{:.2%}",
     "Predicted": "{:.2%}", "p-value": "{:.3f}"}, "num",
)
dq_table = html_table(
    dq.assign(passed=dq["passed"].map({True: "Pass", False: "Fail"})).rename(columns={
        "check": "Assertion", "expected": "Expected", "observed": "Observed", "passed": "Result"}),
)
csi_table = html_table(
    csi.head(6).rename(columns={"feature": "Feature", "csi": "CSI"}), {"CSI": "{:.5f}"}, "num",
)
impact_table = html_table(
    impact.rename(columns={"band": "Age band", "applicants": "Applicants", "approval_rate": "Approval rate",
                           "observed_rate": "TARGET rate", "impact_ratio": "Impact ratio"}),
    {"Applicants": "{:,}", "Approval rate": "{:.1%}", "TARGET rate": "{:.2%}", "Impact ratio": "{:.2f}"}, "num",
)
thresholds_table = html_table(pd.DataFrame(rm.RAG_THRESHOLDS, columns=["Test", "Thresholds"]))
counts = scorecard["status"].value_counts()
today = date.today().strftime("%d %B %Y")
rating_cls = "red" if rating.startswith("Not") else "amber" if "conditions" in rating else "green"

# ----------------------------------------------------------------------------- HTML
HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Model Validation Report: Credit Risk Intelligence PD Model</title>
<style>
@page {{ size: A4; margin: 18mm 16mm 18mm 16mm;
  @bottom-left {{ content: "Credit Risk Intelligence · Model Validation Report"; font: 8pt Helvetica; color: #8a94a3; }}
  @bottom-right {{ content: "Page " counter(page) " of " counter(pages); font: 8pt Helvetica; color: #8a94a3; }} }}
@page :first {{ margin: 0; @bottom-left {{ content: none; }} @bottom-right {{ content: none; }} }}
* {{ box-sizing: border-box; }}
body {{ font: 9.6pt/1.45 Helvetica, Arial, sans-serif; color: {INK}; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
h1, h2, h3 {{ color: #0E1929; margin: 0; }}
h2 {{ font-size: 15pt; margin: 0 0 8px; padding-bottom: 5px; border-bottom: 2px solid #0E1929; }}
h2 .n {{ color: {TEAL}; margin-right: 6px; }}
h3 {{ font-size: 11pt; margin: 14px 0 6px; }}
p {{ margin: 0 0 8px; }}
section {{ break-before: page; }}
section.flow {{ break-before: auto; margin-top: 18px; }}
table {{ width: 100%; border-collapse: collapse; margin: 6px 0 10px; font-size: 8.8pt; break-inside: avoid; }}
th {{ text-align: left; background: #eef2f6; color: #3b4556; font-weight: 700; padding: 6px 8px;
  border-bottom: 1px solid #cfd7e2; }}
td {{ padding: 5px 8px; border-bottom: 1px solid #e3e8ef; vertical-align: top; }}
table.num td:not(:first-child), table.num th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
.pill, .sev {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 7.8pt; font-weight: 700; white-space: nowrap; }}
.pill.green {{ background: #dff5ea; color: #157a4b; }}
.pill.amber {{ background: #fdeedd; color: #a55a10; }}
.pill.red {{ background: #fbe2e4; color: #a62532; }}
.pill.not-assessed {{ background: #eceff3; color: #5d6878; }}
.sev.high {{ background: #a62532; color: white; }}
.sev.medium {{ background: #e8933a; color: white; }}
.sev.low {{ background: #8a97a8; color: white; }}
td.id {{ font-weight: 700; white-space: nowrap; }}
table.findings td:nth-child(3) {{ width: 46%; }}
.cover {{ height: 297mm; background: #0E1929; color: white; padding: 34mm 22mm 20mm; position: relative; }}
.cover .eyebrow {{ color: #31C6B0; font-size: 10pt; letter-spacing: 3px; font-weight: 700; }}
.cover h1 {{ color: white; font-size: 32pt; line-height: 1.1; margin: 14px 0 10px; }}
.cover .sub {{ color: #A8B7CA; font-size: 13pt; max-width: 150mm; }}
.cover .rating {{ margin-top: 26mm; display: inline-block; border: 1.5px solid #FF6B7A; color: #FF6B7A;
  padding: 10px 16px; border-radius: 10px; font-size: 14pt; font-weight: 800; }}
.cover table {{ position: absolute; left: 22mm; right: 22mm; bottom: 24mm; width: auto; color: #DCE6F3; font-size: 9.5pt; }}
.cover td {{ border-bottom: 1px solid #253B54; padding: 7px 0; }}
.cover td:first-child {{ color: #A8B7CA; width: 45mm; }}
.box {{ background: #f4f7fa; border-left: 4px solid {TEAL}; padding: 10px 14px; border-radius: 6px; margin: 8px 0 12px; }}
.box.red {{ border-left-color: {RED}; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 12px; }}
.kpi {{ border: 1px solid #dfe5ec; border-radius: 8px; padding: 8px 10px; }}
.kpi .l {{ color: {MUTED}; font-size: 8pt; text-transform: uppercase; letter-spacing: .5px; }}
.kpi .v {{ font-size: 16pt; font-weight: 800; color: #0E1929; }}
.kpi .d {{ font-size: 8pt; color: {TEAL}; }}
ul {{ margin: 4px 0 8px; padding-left: 16px; }} li {{ margin: 2px 0; }}
.toc td {{ border: none; padding: 3px 0; font-size: 10pt; }}
.toc td:last-child {{ text-align: right; color: {MUTED}; }}
.small {{ font-size: 8.4pt; color: {MUTED}; }}
</style></head><body>

<div class="cover">
  <div class="eyebrow">MODEL RISK MANAGEMENT · INDEPENDENT VALIDATION</div>
  <h1>Model Validation Report<br>Application PD Model</h1>
  <div class="sub">LightGBM challenger vs Logistic Regression benchmark · Home Credit consumer-loan applications</div>
  <div class="rating">Overall rating: {rating}</div>
  <table>
    <tr><td>Model</td><td>Credit Risk Intelligence: application PD (LightGBM v1.0)</td></tr>
    <tr><td>Benchmark</td><td>Logistic Regression v1.0</td></tr>
    <tr><td>Use</td><td>Application scoring, risk grading and approval cut-off (illustrative)</td></tr>
    <tr><td>Validation sample</td><td>{len(pred):,} applicants, stratified random holdout, TARGET rate {rate:.2%}</td></tr>
    <tr><td>Report date</td><td>{today}</td></tr>
    <tr><td>Classification</td><td>Portfolio project · public data · not a production model</td></tr>
  </table>
</div>

<section>
<h2><span class="n">1</span>Executive summary</h2>
<div class="box red"><b>Validation decision: {rating}.</b> LightGBM is a materially and statistically better
discriminator than the benchmark and is well calibrated, but one test is rated Red (overfitting) and two
required tests could not be performed (out-of-time, point-in-time audit). The model should be remediated and
re-validated before any production use.</div>
<div class="kpis">
  <div class="kpi"><div class="l">Gini</div><div class="v">{gini_lgb:.3f}</div><div class="d">{gini_lgb - gini_lr:+.3f} vs benchmark</div></div>
  <div class="kpi"><div class="l">KS</div><div class="v">{ks_lgb:.3f}</div><div class="d">{lgb_metrics['ks'] - lr_metrics['ks']:+.3f} vs benchmark</div></div>
  <div class="kpi"><div class="l">Top-decile lift</div><div class="v">{gains['lift'].iloc[0]:.2f}x</div><div class="d">{gains['cum_bads_captured'].iloc[0]:.1%} of bads captured</div></div>
  <div class="kpi"><div class="l">Train–holdout AUC gap</div><div class="v" style="color:{RED}">{g.loc['LightGBM', 'AUC Gap']:.3f}</div><div class="d" style="color:{RED}">benchmark {g.loc['Logistic Regression', 'AUC Gap']:+.3f}</div></div>
</div>
<h3>Validation scorecard: LightGBM</h3>
<p class="small">{counts.get('Green', 0)} Green · {counts.get('Amber', 0)} Amber · {counts.get('Red', 0)} Red · {counts.get('Not assessed', 0)} Not assessed. Thresholds in Appendix A.</p>
{scorecard_table}
<h3 style="break-before: page">Key findings</h3>
{findings_table}
</section>

<section>
<h2><span class="n">2</span>Scope, model and data</h2>
<h3>2.1 Purpose and scope</h3>
<p>The model estimates the probability that a consumer-loan applicant experiences payment difficulty
(Home Credit <i>TARGET</i>). Scores feed an A–F risk grade and an approval cut-off. This review covers
conceptual soundness, data quality, discrimination, calibration, stability, benchmarking, explainability and
fairness. Implementation and production controls are out of scope.</p>
<h3>2.2 Model design</h3>
<ul>
<li><b>Challenger:</b> LightGBM, 350 trees, learning rate 0.05, 31 leaves, min. 100 samples per leaf, 85% row/column subsampling, L2 = 1.0, native categorical handling.</li>
<li><b>Benchmark:</b> Logistic Regression with median imputation, missing indicators, scaling and one-hot encoding.</li>
<li><b>Features:</b> 25 application, bureau, previous-application and installment-behaviour features. Age is excluded from both models and used only for the fairness review.</li>
<li><b>Score scaling:</b> {rm.BASE_SCORE} points at {rm.BASE_ODDS}:1 good:bad odds, {rm.PDO} points to double the odds. Grades: A ≥ 630, B 610–629, C 590–609, D 570–589, E 550–569, F &lt; 550.</li>
<li><b>Sample design:</b> stratified 80/20 random split (seed 42); both models are scored on identical, programmatically verified holdout IDs.</li>
</ul>
<h3>2.3 Data and data quality</h3>
<p>Features come from a PostgreSQL mart (<code>risk_analytics.customer_feature_mart_v2</code>) built from 7 source
tables (58M+ rows), one row per applicant. All data-quality assertions pass.</p>
{dq_table}
</section>

<section>
<h2><span class="n">3</span>Discrimination and benchmarking</h2>
<p>LightGBM outperforms the benchmark on every metric. The 95% paired-bootstrap interval for each difference excludes zero.</p>
{perf_table}
{chart_roc_ks()}
{chart_lorenz_dist()}
<h3>3.1 Statistical significance</h3>
{boot_table}
<h3>3.2 Gains table: LightGBM</h3>
{gains_table}
</section>

<section>
<h2><span class="n">4</span>Calibration</h2>
<p>Predicted PDs track observed rates closely. All six grades pass a two-sided binomial test at the 5% level, and
the Hosmer–Lemeshow test does not reject (χ² = {hl_stat:.1f}, p = {hl_p:.3f}). For comparison, the benchmark fails
the binomial test in one grade and is rejected by Hosmer–Lemeshow (p = {lr_hl_p:.3f}).</p>
{chart_calibration()}
{binom_table}
<p>Observed rates rise strictly from grade A ({binom['observed_rate'].iloc[0]:.1%}) to F
({binom['observed_rate'].iloc[-1]:.1%}), a {binom['observed_rate'].iloc[-1] / binom['observed_rate'].iloc[0]:.0f}x spread: rank ordering is preserved.</p>
</section>

<section>
<h2><span class="n">5</span>Robustness: overfitting and stability</h2>
<p>LightGBM's in-sample AUC exceeds its holdout AUC by <b>{g.loc['LightGBM', 'AUC Gap']:.3f}</b>, above the 0.05
Red threshold, while the benchmark shows no gap. The challenger still wins on holdout, but the reported lift is
not guaranteed to persist. This is finding F-01.</p>
{chart_bootstrap_gap()}
<p>Score and characteristic stability between development and holdout samples are negligible (PSI well below
0.10). This is expected under a random split and is <b>not</b> evidence of stability over time (finding F-02).</p>
{chart_psi_shap()}
{csi_table}
</section>

<section>
<h2><span class="n">6</span>Explainability and fairness</h2>
<p>TreeSHAP contributions reproduce every prediction exactly (log-odds sum asserted in the pipeline). The main
drivers are employment length, active bureau loans, due-date shortfall rate, education and bureau loan count.
All are directionally intuitive for consumer credit. Individual reason codes are available in the dashboard.</p>
<h3>6.1 Age-band review</h3>
<p>Age is not a model input, yet approval rates at an 80% overall approval rate fall with age band. The gap
mirrors genuinely higher observed rates among younger applicants, but the impact ratio for the under-25 band
({impact['impact_ratio'].min():.2f}) is below the four-fifths benchmark (finding F-04).</p>
{chart_fairness()}
{impact_table}
</section>

<section>
<h2><span class="n">7</span>Limitations and conditions for approval</h2>
<ul>
<li>Random rather than chronological holdout; no out-of-time evidence.</li>
<li>Holdout was reused for model comparison; residual selection bias.</li>
<li>Behavioural aggregates not audited for point-in-time availability.</li>
<li>TARGET is a payment-difficulty flag, not a regulatory default definition.</li>
<li>Fairness review limited to age; other protected attributes not assessed.</li>
<li>Economics used for cut-off analysis are illustrative.</li>
</ul>
<h3>Conditions to move to "Approved with conditions"</h3>
<ol>
<li>Close F-01: train–holdout AUC gap below 0.05, re-validated on an untouched sample.</li>
<li>Close F-02 and F-03: out-of-time test and point-in-time feature audit completed with no material degradation.</li>
<li>Document a business justification and monitoring plan for F-04.</li>
</ol>

<section class="flow">
<h2><span class="n">A</span>Appendix: RAG thresholds and reproducibility</h2>
{thresholds_table}
<p class="small">Thresholds are illustrative and in line with common practice for retail application scorecards.
Reproduce with: <code>python build_dashboard_data.py &amp;&amp; python build_validation_report.py</code>.
Source code: github.com/NVPhg/credit-risk-platform. Data: Home Credit Default Risk (Kaggle).</p>
</section>
</section>
</body></html>
"""

html_path = DOCS / "Model_Validation_Report.html"
html_path.write_text(HTML, encoding="utf-8")
print("Wrote", html_path.relative_to(ROOT))

chrome = next((c for c in [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
] if c and Path(c).exists()), None)
if chrome:
    pdf_path = DOCS / "Model_Validation_Report.pdf"
    subprocess.run([
        chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", html_path.as_uri(),
    ], check=True, capture_output=True)
    print("Wrote", pdf_path.relative_to(ROOT))
else:
    print("Chrome not found: open the HTML file and print it to PDF.")
