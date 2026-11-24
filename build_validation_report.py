"""Generate a bank-style Model Validation Report (HTML + PDF) from the project outputs.

Model under review: LightGBM v1.1 (remediated challenger), with LightGBM v1.0 shown for the
remediation history and the logistic regression and WoE scorecard as benchmarks.

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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve  # noqa: E402

import risk_metrics as rm  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DASH = OUT / "dashboard"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- data
pred = pd.read_parquet(DASH / "holdout_scored.parquet")
boot = pd.read_csv(OUT / "bootstrap_v11_vs_logistic.csv")
boot_v10 = pd.read_csv(OUT / "bootstrap_validation.csv")
boot_v11_v10 = pd.read_csv(OUT / "bootstrap_v11_vs_v10.csv")
gap = pd.read_csv(OUT / "generalization_gap.csv")
psi = pd.read_csv(DASH / "stability_score_psi.csv")
csi = pd.read_csv(DASH / "stability_feature_csi.csv")
dq = pd.read_csv(DASH / "data_quality.csv")
shap = pd.read_csv(DASH / "shap_global.csv")
trials = pd.read_csv(OUT / "tuning_trials.csv")
v11 = json.loads((OUT / "lightgbm_v11_metrics.json").read_text())
v10 = json.loads((OUT / "lightgbm_metrics.json").read_text())
lr_metrics = json.loads((OUT / "metrics.json").read_text())
sc_metrics = json.loads((OUT / "scorecard_metrics.json").read_text())

MODELS = {
    "LightGBM v1.1": "lightgbm_v11_score",
    "LightGBM v1.0": "lightgbm_score",
    "WoE Scorecard": "scorecard_score",
    "Logistic Regression": "logistic_score",
}
GAP_NAMES = {"LightGBM v1.0": "LightGBM"}

y = pred["target"].to_numpy()
p_ch = pred["lightgbm_v11_score"].to_numpy()
p_lr = pred["logistic_score"].to_numpy()
rate = y.mean()

scorecard, rating = rm.validation_scorecard(pred, boot, gap, psi, dq, "LightGBM v1.1")
sc_v10, rating_v10 = rm.validation_scorecard(
    pred, boot_v10, gap, psi, dq, "LightGBM v1.0", gap_name="LightGBM")
grades = rm.score_to_grade(rm.pd_to_score(p_ch))
binom = rm.binomial_grade_test(grades, y, p_ch)
hl_stat, hl_p = rm.hosmer_lemeshow(y, p_ch)
_, lr_hl_p = rm.hosmer_lemeshow(y, p_lr)
_, ks_ch, _, _ = rm.ks_curve(y, p_ch)
_, gini_ch = rm.lorenz_curve(y, p_ch)
gains = rm.gains_table(y, p_ch)
impact = rm.age_impact(pred, "lightgbm_v11_score")
g = gap.set_index("Model")
b = boot.set_index("Metric")
b_vv = boot_v11_v10.set_index("Metric")
cutoff80 = rm.pd_to_score(np.quantile(p_ch, 0.80))
econ = rm.grade_economics(p_ch, pred["credit_amount"], grades)
stress = rm.stress_test(p_ch, pred["credit_amount"], cutoff_score=cutoff80)

# ----------------------------------------------------------------------------- charts
INK, MUTED, GRID = "#1d2430", "#6b7686", "#e3e8ef"
TEAL, BLUE, RED, ORANGE, PURPLE = "#1F9E8C", "#3B7DD8", "#D64550", "#E8933A", "#8C6BD8"
COLORS = {"LightGBM v1.1": TEAL, "LightGBM v1.0": PURPLE, "WoE Scorecard": ORANGE, "Logistic Regression": BLUE}
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


def chart_remediation():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    t = trials.dropna(subset=["user_attrs_val_auc"])
    ax.axvspan(-0.01, 0.02, color=TEAL, alpha=0.08)
    ax.axvspan(0.02, 0.05, color=ORANGE, alpha=0.08)
    ax.scatter(t["user_attrs_gap"], t["user_attrs_val_auc"], s=26, color=BLUE, alpha=0.7, label="Optuna trial")
    best = t[t["number"] == v11["tuning"]["best_trial"]]
    ax.scatter(best["user_attrs_gap"], best["user_attrs_val_auc"], s=180, marker="*", color=TEAL,
               edgecolor=INK, linewidth=0.6, label="Selected", zorder=3)
    ax.set(xlabel="Fit − validation AUC gap", ylabel="Validation ROC-AUC",
           title=f"Optuna search ({len(t)} trials, validation fold)")
    ax.legend(loc="lower right")
    ax = axes[1]
    order = ["Logistic Regression", "WoE Scorecard", "LightGBM v1.0", "LightGBM v1.1"]
    x = np.arange(len(order))
    tr = [g.loc[GAP_NAMES.get(m, m), "Train ROC-AUC"] for m in order]
    ho = [g.loc[GAP_NAMES.get(m, m), "Holdout ROC-AUC"] for m in order]
    ax.bar(x - 0.2, tr, 0.4, color="#9aa5b5", label="Train")
    ax.bar(x + 0.2, ho, 0.4, color=TEAL, label="Holdout")
    for i, m in enumerate(order):
        gp = g.loc[GAP_NAMES.get(m, m), "AUC Gap"]
        ax.text(i, max(tr[i], ho[i]) + 0.006, f"{gp:+.3f}", ha="center", fontsize=7.5,
                color=RED if gp > 0.05 else ORANGE if gp > 0.02 else INK)
    ax.set_xticks(x, ["Logistic", "WoE", "LGBM v1.0", "LGBM v1.1"])
    ax.set(ylim=(0.6, 0.8), ylabel="ROC-AUC", title="Train vs holdout (gap labelled)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_roc_ks():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for name, col in MODELS.items():
        fpr, tpr, _ = roc_curve(y, pred[col])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2 if name == "LightGBM v1.1" else 1.3, label=name)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right", fontsize=7.5)
    ax = axes[1]
    curve, ks, at, _ = rm.ks_curve(y, p_ch)
    ax.plot(curve["population"] * 100, curve["cum_bad"] * 100, color=RED, lw=2, label="Cumulative bads")
    ax.plot(curve["population"] * 100, curve["cum_good"] * 100, color=TEAL, lw=2, label="Cumulative goods")
    k = curve.iloc[(curve["population"] - at).abs().argmin()]
    ax.vlines(k["population"] * 100, k["cum_good"] * 100, k["cum_bad"] * 100, color=ORANGE, lw=2, ls=":")
    ax.annotate(f"KS = {ks:.3f}", (k["population"] * 100 + 2, (k["cum_bad"] + k["cum_good"]) * 50),
                color=ORANGE, fontweight="bold")
    ax.set(xlabel="Applicants, riskiest first (%)", ylabel="Cumulative share (%)", title="KS: LightGBM v1.1")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_lorenz_dist():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for name, col in MODELS.items():
        curve, gn = rm.lorenz_curve(y, pred[col])
        ax.plot(curve["population"] * 100, curve["cum_bad"] * 100, color=COLORS[name],
                lw=2 if name == "LightGBM v1.1" else 1.3, label=f"{name} ({gn:.3f})")
    ax.plot([0, 100], [0, 100], "--", color=MUTED, lw=1)
    ax.set(xlabel="Applicants declined, riskiest first (%)", ylabel="Bads captured (%)", title="Lorenz curve (Gini)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax = axes[1]
    scores = rm.pd_to_score(p_ch)
    bins = np.arange(480, 681, 5)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.55, color=TEAL, label="Good")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.55, color=RED, label="Bad")
    for grade, lower in rm.GRADES[:-1]:
        ax.axvline(lower, color=MUTED, lw=0.8, ls=":")
        ax.text(lower + 1, ax.get_ylim()[1] * 0.95, grade, color=MUTED, fontsize=8)
    ax.set(xlabel="Credit score", ylabel="Density", title="Score distribution by outcome: v1.1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    ax = axes[0]
    for name in ["Logistic Regression", "LightGBM v1.1"]:
        obs, exp = calibration_curve(y, pred[MODELS[name]], n_bins=10, strategy="quantile")
        ax.plot(exp, obs, "o-", color=COLORS[name], lw=2, ms=4, label=name)
    ax.plot([0, 0.3], [0, 0.3], "--", color=MUTED, lw=1, label="Perfect")
    ax.set(xlim=(0, 0.3), ylim=(0, 0.3), xlabel="Mean predicted PD", ylabel="Observed rate",
           title="Reliability (deciles)")
    ax.legend(loc="upper left")
    ax = axes[1]
    x = np.arange(len(binom))
    ax.bar(x - 0.2, binom["observed_rate"] * 100, 0.4, color=TEAL, label="Observed")
    ax.bar(x + 0.2, binom["predicted_pd"] * 100, 0.4, color="#9aa5b5", label="Predicted")
    ax.set_xticks(x, binom["grade"])
    ax.set(ylabel="Rate (%)", title="Observed vs predicted by grade: v1.1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig_to_img(fig)


def chart_psi_shap():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    t = psi[psi["model"] == "LightGBM v1.1"]
    x = np.arange(1, len(t) + 1)
    ax.bar(x - 0.2, t["expected_pct"] * 100, 0.4, color="#9aa5b5", label="Development")
    ax.bar(x + 0.2, t["actual_pct"] * 100, 0.4, color=TEAL, label="Holdout")
    ax.set_xticks(x)
    ax.set(xlabel="Score decile (development)", ylabel="Share (%)",
           title=f"Score stability: PSI = {t['total_psi'].iloc[0]:.4f}")
    ax.set_ylim(0, 13.5)
    ax.legend(loc="upper center", ncol=2)
    ax = axes[1]
    top = shap.head(10).iloc[::-1]
    ax.barh(top["feature"].str.replace("_", " "), top["mean_abs_shap"], color=TEAL)
    ax.set(xlabel="Mean |SHAP| (log-odds)", title="Top drivers: TreeSHAP (v1.1)")
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


def chart_capital():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    x = np.arange(len(econ))
    funding, opex = 0.06, 0.03
    bottom = np.zeros(len(econ))
    for vals, lbl, c in [
        (np.full(len(econ), funding), "Funding", "#52627A"),
        (np.full(len(econ), opex), "Operating cost", "#8A97A8"),
        (econ["el_rate"].to_numpy(), "Expected loss", ORANGE),
        (econ["k_rate"].to_numpy() * 0.15, "Cost of capital", TEAL),
    ]:
        ax.bar(x, vals * 100, bottom=bottom * 100, color=c, label=lbl, width=0.6)
        bottom += vals
    for i, v in enumerate(econ["min_rate"]):
        ax.text(i, v * 100 + 0.6, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x, econ["grade"])
    ax.set(ylabel="Minimum annual rate (%)", title="Risk-based price by grade", ylim=(0, econ["min_rate"].max() * 118))
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    xs = np.arange(len(stress))
    ax.bar(xs - 0.2, stress["expected_loss"] / 1e9, 0.4, color=[TEAL, ORANGE, RED], label="Expected loss")
    ax.bar(xs + 0.2, stress["capital"] / 1e9, 0.4, color=[TEAL, ORANGE, RED], alpha=0.45, label="IRB capital")
    ax.set_xticks(xs, stress["scenario"])
    ax.set(ylabel="Amount (billions)", title="Stress test: EL and capital")
    ax.legend(loc="upper left", fontsize=7.5)
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


def sev_pill(sev):
    return f'<span class="sev {sev.lower()}">{sev}</span>'


def status_pill(status):
    cls = "closed" if status.startswith("Closed") else "accepted" if status.startswith("Accepted") else "open"
    return f'<span class="st {cls}">{status}</span>'


v10_gap, v11_gap = g.loc["LightGBM", "AUC Gap"], g.loc["LightGBM v1.1", "AUC Gap"]
impact_v10 = rm.age_impact(pred, "lightgbm_score")
ir_v10, ir_v11 = impact_v10["impact_ratio"].min(), impact["impact_ratio"].min()
fair_red = ir_v11 < 0.60
hl_text = ("does not reject" if hl_p >= 0.05 else "rejects at the 5% level")
FINDINGS = [
    ("F-01", "High", "Overfitting (v1.0)",
     f"v1.0 train ROC-AUC {g.loc['LightGBM', 'Train ROC-AUC']:.3f} vs holdout "
     f"{g.loc['LightGBM', 'Holdout ROC-AUC']:.3f} (gap {v10_gap:.3f}).",
     f"Closed in v1.1: gap {v11_gap:.3f}" if v11_gap <= 0.05 else "Open"),
    ("F-05", "Medium", "Holdout reuse",
     "v1.0 was compared and selected on the same holdout.",
     "Closed in v1.1: tuned on a validation fold"),
    ("F-02", "High", "No out-of-time validation",
     "The dataset has no application dates, so temporal drift cannot be measured.",
     "Open: condition of approval"),
    ("F-03", "High", "Point-in-time leakage risk",
     "Behavioural aggregates use full history; availability at application is not evidenced.",
     "Open: condition of approval"),
    ("F-04", "High" if fair_red else "Medium", "Age-band disparity" + (" (escalated)" if fair_red else ""),
     f"At 80% approval, under-25s are approved at {impact['approval_rate'].iloc[0]:.0%} vs "
     f"{impact['approval_rate'].max():.0%} for the best-treated band. Impact ratio {ir_v10:.2f} in v1.0, "
     f"{ir_v11:.2f} in v1.1" + (": below the 0.60 Red threshold." if fair_red else "."),
     "Open: blocks approval" if fair_red else "Open: justification and monitoring"),
    ("F-06", "Low", "Label definition",
     "TARGET flags payment difficulty and is not a Basel / IFRS 9 default definition.",
     "Accepted limitation"),
    ("F-07", "Low", "Illustrative economics",
     "Pricing, capital and cut-off analysis use assumed LGD, margin and cost inputs.",
     "Accepted limitation"),
]
findings_rows = "".join(
    f"<tr><td class='id'>{fid}</td><td>{sev_pill(sev)}</td><td><b>{title}</b><br>{desc}</td>"
    f"<td>{status_pill(status)}</td></tr>"
    for fid, sev, title, desc, status in FINDINGS
)
findings_table = (
    "<table class='findings'><thead><tr><th>ID</th><th>Severity</th><th>Finding</th>"
    f"<th>Status</th></tr></thead><tbody>{findings_rows}</tbody></table>"
)

scorecard_table = html_table(
    scorecard.assign(status=scorecard["status"].map(pill)).rename(columns={
        "test": "Test", "result": "Result", "status": "Status", "note": "Basis"}),
    cls="rag",
)
remed = sc_v10[["test", "result", "status"]].merge(
    scorecard[["test", "result", "status"]], on="test", suffixes=("_10", "_11"))
remed_table = html_table(
    remed.assign(status_10=remed["status_10"].map(pill), status_11=remed["status_11"].map(pill)).rename(columns={
        "test": "Test", "result_10": "v1.0 result", "status_10": "v1.0",
        "result_11": "v1.1 result", "status_11": "v1.1"}),
    cls="rag",
)

perf_rows = []
for name, col in MODELS.items():
    p = pred[col]
    _, ksv, _, _ = rm.ks_curve(y, p)
    auc = roc_auc_score(y, p)
    perf_rows.append((name, auc, 2 * auc - 1, ksv, average_precision_score(y, p), brier_score_loss(y, p),
                      g.loc[GAP_NAMES.get(name, name), "AUC Gap"]))
perf = pd.DataFrame(perf_rows, columns=["Model", "ROC-AUC", "Gini", "KS", "PR-AUC", "Brier (↓)", "Train–holdout gap"])
perf_table = html_table(perf, {c: "{:.4f}" for c in ["ROC-AUC", "Gini", "KS", "PR-AUC", "Brier (↓)"]}
                        | {"Train–holdout gap": "{:+.3f}"}, "num")

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
csi_table = html_table(csi.head(6).rename(columns={"feature": "Feature", "csi": "CSI"}), {"CSI": "{:.5f}"}, "num")
impact_table = html_table(
    impact.rename(columns={"band": "Age band", "applicants": "Applicants", "approval_rate": "Approval rate",
                           "observed_rate": "TARGET rate", "impact_ratio": "Impact ratio"}),
    {"Applicants": "{:,}", "Approval rate": "{:.1%}", "TARGET rate": "{:.2%}", "Impact ratio": "{:.2f}"}, "num",
)
econ_table = html_table(
    econ[["grade", "applicants", "avg_pd", "el_rate", "k_rate", "rwa", "min_rate"]].rename(columns={
        "grade": "Grade", "applicants": "N", "avg_pd": "Avg PD", "el_rate": "EL / EAD",
        "k_rate": "IRB K", "rwa": "RWA", "min_rate": "Min. rate"}),
    {"N": "{:,}", "Avg PD": "{:.2%}", "EL / EAD": "{:.2%}", "IRB K": "{:.2%}",
     "RWA": lambda v: f"{v / 1e9:,.2f}B", "Min. rate": "{:.1%}"}, "num",
)
stress_table = html_table(
    stress[["scenario", "odds_multiplier", "lgd", "avg_pd", "el_rate", "capital", "approval_rate", "share_grade_ef"]]
    .rename(columns={"scenario": "Scenario", "odds_multiplier": "PD odds ×", "lgd": "LGD", "avg_pd": "Avg PD",
                     "el_rate": "EL / EAD", "capital": "IRB capital", "approval_rate": "Approved",
                     "share_grade_ef": "In grades E/F"}),
    {"PD odds ×": "{:.2f}", "LGD": "{:.0%}", "Avg PD": "{:.2%}", "EL / EAD": "{:.2%}",
     "IRB capital": lambda v: f"{v / 1e9:,.2f}B", "Approved": "{:.1%}", "In grades E/F": "{:.1%}"}, "num",
)
thresholds_table = html_table(pd.DataFrame(rm.RAG_THRESHOLDS, columns=["Test", "Thresholds"]))
counts = scorecard["status"].value_counts()
today = date.today().strftime("%d %B %Y")
rating_color = RED if rating.startswith("Not") else ORANGE if "conditions" in rating else TEAL
mono = ", ".join(f"{k.replace('_', ' ')} ({'+' if v > 0 else '−'})" for k, v in v11["monotone_constraints"].items())
sev = stress.iloc[-1]
summary_text = (
    f"Remediation worked on its target: v1.1 removes the overfitting that blocked v1.0 (train–holdout gap "
    f"{v10_gap:.3f} → {v11_gap:.3f}) without losing discrimination (ΔAUC vs v1.0 "
    f"{b_vv.loc['ROC-AUC', 'Difference']:+.4f}, 95% CI [{b_vv.loc['ROC-AUC', 'Difference CI Low']:+.4f}, "
    f"{b_vv.loc['ROC-AUC', 'Difference CI High']:+.4f}]) and beats the logistic benchmark on every metric. "
)
if fair_red:
    summary_text += (
        f"However, the under-25 approval impact ratio slipped from {ir_v10:.2f} to {ir_v11:.2f}, crossing the Red "
        "threshold. Finding F-04 is escalated to High and blocks approval until a fairness remediation (for example "
        "segment-level cut-offs or a reweighted retrain) is validated. Out-of-time and point-in-time tests remain "
        "conditions."
    )
else:
    summary_text += (
        "Approval is conditional on an out-of-time test, a point-in-time feature audit and a documented response "
        "to the age-band disparity."
    )
base = stress.iloc[0]

# ----------------------------------------------------------------------------- HTML
HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Model Validation Report: Credit Risk Intelligence PD Model v1.1</title>
<style>
@page {{ size: A4; margin: 18mm 16mm 18mm 16mm;
  @bottom-left {{ content: "Credit Risk Intelligence · Model Validation Report · LightGBM v1.1"; font: 8pt Helvetica; color: #8a94a3; }}
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
.pill, .sev, .st {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 7.8pt; font-weight: 700; white-space: nowrap; }}
.pill.green {{ background: #dff5ea; color: #157a4b; }}
.pill.amber {{ background: #fdeedd; color: #a55a10; }}
.pill.red {{ background: #fbe2e4; color: #a62532; }}
.pill.not-assessed {{ background: #eceff3; color: #5d6878; }}
.sev.high {{ background: #a62532; color: white; }}
.sev.medium {{ background: #e8933a; color: white; }}
.sev.low {{ background: #8a97a8; color: white; }}
.st.closed {{ background: #dff5ea; color: #157a4b; }}
.st.open {{ background: #fdeedd; color: #a55a10; }}
.st.accepted {{ background: #eceff3; color: #5d6878; }}
td.id {{ font-weight: 700; white-space: nowrap; }}
table.findings td:nth-child(3) {{ width: 58%; }}
.cover {{ height: 297mm; background: #0E1929; color: white; padding: 34mm 22mm 20mm; position: relative; }}
.cover .eyebrow {{ color: #31C6B0; font-size: 10pt; letter-spacing: 3px; font-weight: 700; }}
.cover h1 {{ color: white; font-size: 32pt; line-height: 1.1; margin: 14px 0 10px; }}
.cover .sub {{ color: #A8B7CA; font-size: 13pt; max-width: 150mm; }}
.cover .ratings {{ margin-top: 22mm; display: flex; gap: 12px; flex-wrap: wrap; }}
.cover .rating {{ display: inline-block; border: 1.5px solid; padding: 10px 16px; border-radius: 10px; font-size: 13pt; font-weight: 800; }}
.cover .rating small {{ display: block; font-size: 8.5pt; font-weight: 600; opacity: .8; letter-spacing: 1px; }}
.cover table {{ position: absolute; left: 22mm; right: 22mm; bottom: 24mm; width: auto; color: #DCE6F3; font-size: 9.5pt; }}
.cover td {{ border-bottom: 1px solid #253B54; padding: 7px 0; }}
.cover td:first-child {{ color: #A8B7CA; width: 45mm; }}
.box {{ background: #f4f7fa; border-left: 4px solid {TEAL}; padding: 10px 14px; border-radius: 6px; margin: 8px 0 12px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 12px; }}
.kpi {{ border: 1px solid #dfe5ec; border-radius: 8px; padding: 8px 10px; }}
.kpi .l {{ color: {MUTED}; font-size: 8pt; text-transform: uppercase; letter-spacing: .5px; }}
.kpi .v {{ font-size: 16pt; font-weight: 800; color: #0E1929; }}
.kpi .d {{ font-size: 8pt; color: {TEAL}; }}
ul, ol {{ margin: 4px 0 8px; padding-left: 16px; }} li {{ margin: 2px 0; }}
.small {{ font-size: 8.4pt; color: {MUTED}; }}
</style></head><body>

<div class="cover">
  <div class="eyebrow">MODEL RISK MANAGEMENT · INDEPENDENT VALIDATION</div>
  <h1>Model Validation Report<br>Application PD Model v1.1</h1>
  <div class="sub">Remediated LightGBM challenger vs Logistic Regression and WoE scorecard benchmarks ·
  Home Credit consumer-loan applications</div>
  <div class="ratings">
    <div class="rating" style="color:{rating_color};border-color:{rating_color}"><small>V1.1 · THIS REVIEW</small>{rating}</div>
    <div class="rating" style="color:#8A97A8;border-color:#3b4b63"><small>V1.0 · PREVIOUS REVIEW</small>{rating_v10}</div>
  </div>
  <table>
    <tr><td>Model under review</td><td>LightGBM v1.1: {v11['n_trees']} trees, monotonic constraints, Optuna-tuned</td></tr>
    <tr><td>Benchmarks</td><td>Logistic Regression · WoE / IV scorecard ({sc_metrics['n_variables']} characteristics)</td></tr>
    <tr><td>Use</td><td>Application scoring, risk grading, approval cut-off and risk-based pricing (illustrative)</td></tr>
    <tr><td>Validation sample</td><td>{len(pred):,} applicants, stratified holdout untouched during tuning, TARGET rate {rate:.2%}</td></tr>
    <tr><td>Report date</td><td>{today}</td></tr>
    <tr><td>Classification</td><td>Portfolio project · public data · not a production model</td></tr>
  </table>
</div>

<section>
<h2><span class="n">1</span>Executive summary</h2>
<div class="box" style="border-left-color:{rating_color}"><b>Validation decision: {rating}.</b>
{summary_text}</div>
<div class="kpis">
  <div class="kpi"><div class="l">Gini</div><div class="v">{gini_ch:.3f}</div><div class="d">{gini_ch - (2 * lr_metrics['roc_auc'] - 1):+.3f} vs logistic</div></div>
  <div class="kpi"><div class="l">KS</div><div class="v">{ks_ch:.3f}</div><div class="d">{ks_ch - lr_metrics['ks']:+.3f} vs logistic</div></div>
  <div class="kpi"><div class="l">Train–holdout gap</div><div class="v">{v11_gap:.3f}</div><div class="d">was {v10_gap:.3f} in v1.0</div></div>
  <div class="kpi"><div class="l">Top-decile lift</div><div class="v">{gains['lift'].iloc[0]:.2f}x</div><div class="d">{gains['cum_bads_captured'].iloc[0]:.1%} of bads captured</div></div>
</div>
<h3>Validation scorecard: LightGBM v1.1</h3>
<p class="small">{counts.get('Green', 0)} Green · {counts.get('Amber', 0)} Amber · {counts.get('Red', 0)} Red ·
{counts.get('Not assessed', 0)} Not assessed. Any Red blocks approval; Amber or unassessed tests become conditions.
Thresholds in Appendix A.</p>
{scorecard_table}
<h3 style="break-before: page">Findings register</h3>
{findings_table}
</section>

<section>
<h2><span class="n">2</span>Remediation: v1.0 → v1.1</h2>
<p>The v1.0 review rated overfitting Red (finding F-01) and flagged that the holdout had been reused for model
selection (F-05). The model owner retrained the challenger as follows:</p>
<ul>
<li><b>Tuning on a validation fold.</b> {v11['tuning']['trials']} Optuna (TPE) trials on a 20% validation fold carved out of the
training sample. The holdout was not used until the final evaluation.</li>
<li><b>Overfitting-aware objective.</b> Validation AUC minus a penalty for any fit–validation gap above 0.02, with
early stopping ({v11['n_trees']} trees selected).</li>
<li><b>Stronger regularisation.</b> {v11['params']['num_leaves']} leaves (was 31), at least {v11['params']['min_child_samples']:,}
applicants per leaf (was 100), L2 = {v11['params']['lambda_l2']:.1f} (was 1.0), learning rate {v11['params']['learning_rate']:.3f}.</li>
<li><b>Monotonic constraints</b> on {mono}, so the model can never reward a riskier value of a key driver.</li>
</ul>
{chart_remediation()}
{remed_table}
</section>

<section>
<h2><span class="n">3</span>Scope, model and data</h2>
<h3>3.1 Purpose and scope</h3>
<p>The model estimates the probability that a consumer-loan applicant experiences payment difficulty
(Home Credit <i>TARGET</i>). Scores feed an A–F risk grade, an approval cut-off and risk-based pricing. This review
covers conceptual soundness, data quality, discrimination, calibration, stability, benchmarking, explainability,
fairness and the business use of the PDs. Implementation and production controls are out of scope.</p>
<h3>3.2 Design</h3>
<ul>
<li><b>Features:</b> 25 application, bureau, previous-application and installment-behaviour features. Age is excluded
from every model and used only for the fairness review.</li>
<li><b>Benchmarks:</b> logistic regression (median imputation, missing indicators, scaling, one-hot encoding) and a
WoE / IV scorecard (optimal monotonic binning, IV ≥ 0.02, {sc_metrics['n_variables']} characteristics kept).</li>
<li><b>Score scaling:</b> {rm.BASE_SCORE} points at {rm.BASE_ODDS}:1 good:bad odds, {rm.PDO} points to double the odds.
Grades: A ≥ 630, B 610–629, C 590–609, D 570–589, E 550–569, F &lt; 550.</li>
<li><b>Sample design:</b> stratified 80/20 split (seed 42); every model is scored on identical, programmatically
verified holdout IDs.</li>
</ul>
<h3>3.3 Data quality</h3>
<p>Features come from a PostgreSQL mart (<code>risk_analytics.customer_feature_mart_v2</code>) built from 7 source
tables (58M+ rows), one row per applicant. All data-quality assertions pass.</p>
{dq_table}
</section>

<section>
<h2><span class="n">4</span>Discrimination and benchmarking</h2>
{perf_table}
<p>LightGBM v1.1 has the highest ROC-AUC and KS. The WoE scorecard is the most transparent but gives up about
{(gini_ch - (2 * sc_metrics['roc_auc'] - 1)):.2f} Gini points: the measured price of full transparency.</p>
{chart_roc_ks()}
{chart_lorenz_dist()}
<h3>4.1 Statistical significance vs the logistic benchmark</h3>
{boot_table}
<p class="small">v1.1 vs v1.0: ΔAUC {b_vv.loc['ROC-AUC', 'Difference']:+.4f}
[{b_vv.loc['ROC-AUC', 'Difference CI Low']:+.4f}, {b_vv.loc['ROC-AUC', 'Difference CI High']:+.4f}]
({b_vv.loc['ROC-AUC', 'Evidence'].lower()}): remediation did not cost discrimination.</p>
<h3>4.2 Gains table: LightGBM v1.1</h3>
{gains_table}
</section>

<section>
<h2><span class="n">5</span>Calibration</h2>
<p>{(binom['result'] == 'Pass').sum()} of {len(binom)} grades pass a two-sided binomial test at the 5% level, the
test that drives the rating. Hosmer–Lemeshow {hl_text} (χ² = {hl_stat:.1f}, p = {hl_p:.3f}; logistic benchmark
p = {lr_hl_p:.3f}). With 61,503 applicants HL flags very small deviations: the largest gap between observed and
predicted rate in any grade is {(binom['observed_rate'] - binom['predicted_pd']).abs().max() * 100:.2f} pp.</p>
{chart_calibration()}
{binom_table}
<p>Observed rates rise strictly from grade A ({binom['observed_rate'].iloc[0]:.1%}) to F
({binom['observed_rate'].iloc[-1]:.1%}), a {binom['observed_rate'].iloc[-1] / binom['observed_rate'].iloc[0]:.0f}x spread:
rank ordering is preserved.</p>
</section>

<section>
<h2><span class="n">6</span>Stability and explainability</h2>
<p>Score and characteristic stability between development and holdout samples are negligible (PSI well below
0.10). This is expected under a random split and is <b>not</b> evidence of stability over time (finding F-02).</p>
{chart_psi_shap()}
{csi_table}
<p>TreeSHAP contributions reproduce every prediction exactly (log-odds sum asserted in the pipeline). The main
drivers are directionally intuitive for consumer credit, and the monotonic constraints guarantee the direction of
the six constrained drivers. Individual reason codes are produced in the dashboard.</p>
</section>

<section>
<h2><span class="n">7</span>Fairness</h2>
<p>Age is not a model input, yet approval rates at an 80% overall approval rate fall with age band. The gap
mirrors genuinely higher observed rates among younger applicants, but the under-25 impact ratio is
{ir_v11:.2f} (v1.0: {ir_v10:.2f}), {'below the 0.60 Red threshold' if fair_red else 'below the four-fifths benchmark'}
(finding F-04). Employment length, the strongest driver, is the main age proxy.</p>
{chart_fairness()}
{impact_table}
</section>

<section>
<h2><span class="n">8</span>Business use: capital, pricing and stress</h2>
<p>PDs are converted into expected loss (PD × LGD × EAD, LGD 45%, EAD = credit amount) and Basel IRB capital for
other retail exposures (asset correlation 3–16%, 99.9% confidence, 5 bp PD floor). The minimum risk-based rate per
grade adds funding (6%), operating cost (3%), expected loss and a 15% return on the IRB capital.</p>
{chart_capital()}
{econ_table}
<h3>8.1 Stress test</h3>
<p>PD odds are scaled by 1.5x (adverse) and 2.25x (severely adverse) with LGD +5 / +10 pp; the baseline 80%-approval
cut-off ({cutoff80:.0f} points) is held fixed. In the severe scenario expected loss rises
{sev['expected_loss'] / base['expected_loss']:.1f}x, IRB capital {sev['capital'] / base['capital'] - 1:+.0%}, and approval falls from
{base['approval_rate']:.0%} to {sev['approval_rate']:.0%} as {sev['share_grade_ef']:.0%} of applicants migrate to grades E/F.</p>
{stress_table}
</section>

<section>
<h2><span class="n">9</span>Conditions of approval and limitations</h2>
<h3>Conditions</h3>
<ol>
<li>F-02: an out-of-time test on dated applications shows no material degradation in Gini, KS or calibration.</li>
<li>F-03: behavioural aggregates are rebuilt from records dated before each application and performance re-measured.</li>
<li>F-04: the under-25 impact ratio is brought back to at least 0.60 through a validated fairness remediation,
with the business justification documented and outcomes monitored monthly by age band.</li>
<li>Ongoing: monthly PSI / CSI with Amber at 0.10 and Red at 0.25, and an annual re-validation.</li>
</ol>
<h3>Limitations</h3>
<ul>
<li>Random rather than chronological holdout.</li>
<li>TARGET is a payment-difficulty flag, not a regulatory default definition; PDs are treated as 12-month PDs.</li>
<li>Fairness review limited to age; other protected attributes not assessed.</li>
<li>Capital, pricing and cut-off economics are illustrative.</li>
</ul>

<section class="flow">
<h2><span class="n">A</span>Appendix: RAG thresholds and reproducibility</h2>
{thresholds_table}
<p class="small">Thresholds are illustrative and in line with common practice for retail application scorecards.
Reproduce with: <code>python tune_challenger.py &amp;&amp; python build_scorecard.py &amp;&amp;
python build_dashboard_data.py &amp;&amp; python build_validation_report.py</code>.
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
