import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import risk_metrics as rm
from risk_metrics import (
    BASE_ODDS, BASE_SCORE, GRADES, PDO, RAG_COLORS, RAG_THRESHOLDS,
    grade_range, pd_to_score, score_to_grade,
)

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DASH = OUT / "dashboard"
FILES = [
    OUT / "metrics.json",
    OUT / "lightgbm_metrics.json",
    OUT / "model_comparison.csv",
    OUT / "feature_importance.csv",
    OUT / "bootstrap_validation.csv",
    OUT / "generalization_gap.csv",
    DASH / "holdout_scored.parquet",
    DASH / "shap_sample.parquet",
    DASH / "shap_global.csv",
    DASH / "segment_risk.csv",
    DASH / "lightgbm_model.txt",
    DASH / "stability_score_psi.csv",
    DASH / "stability_feature_csi.csv",
    DASH / "data_quality.csv",
]
missing = [str(p.relative_to(ROOT)) for p in FILES if not p.exists()]
if missing:
    st.error(
        f"Missing project outputs: {missing}. "
        "Run the pipeline described in README.md "
        "(train_model → compare_models → validate_models → "
        "check_generalization → build_dashboard_data)."
    )
    st.stop()

# ---------------------------------------------------------------- constants
NAVY = "#0E1929"
PANEL = "#142238"
BORDER = "#253B54"
TEAL = "#31C6B0"
BLUE = "#69A8FF"
ORANGE = "#FFB36B"
RED = "#FF6B7A"
MUTED = "#A8B7CA"
TEXT = "#E7EEF8"

MODELS = {
    "LightGBM": ("lightgbm_score", TEAL),
    "Logistic Regression": ("logistic_score", BLUE),
}

GRADE_COLORS = {
    "A": "#2BB673", "B": TEAL, "C": BLUE, "D": ORANGE, "E": "#FF8A5B", "F": RED,
}

FEATURE_LABELS = {
    "annual_income": "Annual income",
    "credit_amount": "Credit amount",
    "annuity_amount": "Annuity amount",
    "contract_type": "Contract type",
    "income_type": "Income type",
    "education_type": "Education",
    "credit_to_income": "Credit-to-income ratio",
    "annuity_to_income": "Annuity-to-income ratio",
    "employment_years": "Employment length (years)",
    "employment_sentinel_flag": "Not-employed flag",
    "bureau_loan_count": "Bureau loans (all)",
    "active_loan_count": "Active bureau loans",
    "avg_bureau_credit": "Avg bureau credit amount",
    "previous_application_count": "Previous applications",
    "approved_count": "Previous approvals",
    "refused_count": "Previous refusals",
    "approval_rate": "Previous approval rate",
    "has_installment_history": "Has installment history",
    "installment_count": "Installments observed",
    "multiple_payment_count": "Installments paid in parts",
    "incomplete_installment_count": "Incomplete installments",
    "valid_installment_count": "Valid installments",
    "due_date_shortfall_count": "Installments short at due date",
    "due_date_shortfall_rate": "Due-date shortfall rate",
    "observed_underpaid_count": "Underpaid installments",
}


def label(feature):
    return FEATURE_LABELS.get(feature, feature)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    logistic = json.loads((OUT / "metrics.json").read_text())
    lightgbm = json.loads((OUT / "lightgbm_metrics.json").read_text())
    pred = pd.read_parquet(DASH / "holdout_scored.parquet")
    return {
        "logistic": logistic,
        "lightgbm": lightgbm,
        "pred": pred,
        "comparison": pd.read_csv(OUT / "model_comparison.csv"),
        "importance": pd.read_csv(OUT / "feature_importance.csv"),
        "bootstrap": pd.read_csv(OUT / "bootstrap_validation.csv"),
        "gap": pd.read_csv(OUT / "generalization_gap.csv"),
        "shap_sample": pd.read_parquet(DASH / "shap_sample.parquet"),
        "shap_global": pd.read_csv(DASH / "shap_global.csv"),
        "segments": pd.read_csv(DASH / "segment_risk.csv"),
        "psi": pd.read_csv(DASH / "stability_score_psi.csv"),
        "csi": pd.read_csv(DASH / "stability_feature_csi.csv"),
        "dq": pd.read_csv(DASH / "data_quality.csv"),
    }


@st.cache_resource
def load_booster():
    return lgb.Booster(model_file=str(DASH / "lightgbm_model.txt"))


data = load_data()
logistic, lightgbm = data["logistic"], data["lightgbm"]
pred = data["pred"]
booster = load_booster()
FEATURES = booster.feature_name()
PORTFOLIO_RATE = pred["target"].mean()


# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
.stApp {{ background: {NAVY}; color: {TEXT}; }}
.block-container {{ max-width: 1450px; padding-top: 3.5rem; }}
[data-testid="stMetric"] {{
    background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 18px 20px;
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
[data-testid="stMetricValue"] {{ color: #F3F7FC; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 20px; flex-wrap: wrap; }}
.stTabs [aria-selected="true"] {{ color: {TEAL} !important; }}
.hero-label {{ color: {TEAL}; font-size: 12px; font-weight: 700; letter-spacing: 2px; }}
.hero-title {{ font-size: 40px; font-weight: 750; letter-spacing: -1px; margin: 10px 0; line-height: 1.1; }}
.hero-subtitle {{ color: {MUTED}; margin-bottom: 18px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 26px; }}
.chip {{
    border: 1px solid {BORDER}; background: {PANEL}; color: {TEXT};
    border-radius: 999px; padding: 4px 12px; font-size: 13px;
}}
.insight {{
    background: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {TEAL};
    border-radius: 12px; padding: 16px 18px; height: 100%;
}}
.insight h4 {{ margin: 0 0 6px 0; font-size: 15px; color: {TEAL}; }}
.insight p {{ margin: 0; color: {TEXT}; font-size: 14px; line-height: 1.5; }}
.grade-badge {{
    display: inline-block; font-size: 44px; font-weight: 800; line-height: 1;
    padding: 14px 22px; border-radius: 14px; color: {NAVY};
}}
.reason {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px; font-size: 14px;
}}
.reason b {{ color: {ORANGE}; }}
.reason.good b {{ color: {TEAL}; }}
.pill {{
    display: inline-block; padding: 2px 10px; border-radius: 999px; border: 1px solid;
    font-size: 12px; font-weight: 700; white-space: nowrap;
}}
table.rag {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
table.rag th {{
    text-align: left; color: {MUTED}; font-weight: 600; font-size: 12px; text-transform: uppercase;
    letter-spacing: .6px; padding: 8px 10px; border-bottom: 1px solid {BORDER};
}}
table.rag td {{ padding: 9px 10px; border-bottom: 1px solid {BORDER}; vertical-align: top; }}
table.rag td.t {{ font-weight: 600; white-space: nowrap; }}
table.rag td.r {{ font-variant-numeric: tabular-nums; }}
table.rag td.n {{ color: {MUTED}; font-size: 13px; }}
.rating {{
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 14px;
}}
.rating .big {{ font-size: 22px; font-weight: 800; }}
.rating .meta {{ color: {MUTED}; font-size: 13px; }}
.rag-wrap {{ overflow-x: auto; }}
@media (max-width: 640px) {{ .hero-title {{ font-size: 30px; }} }}
</style>
""", unsafe_allow_html=True)


def insight(title, body):
    st.markdown(
        f'<div class="insight"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def chart_style(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color="#DCE6F3"),
        height=height,
        margin=dict(l=35, r=20, t=35, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#263A50", zeroline=False)
    return fig


def show(fig, key):
    st.plotly_chart(
        fig, width="stretch", theme=None, key=key,
        config={"displayModeBar": False, "displaylogo": False},
    )


# ---------------------------------------------------------------- analytics
def risk_deciles(score_column):
    tmp = pred[["target", score_column]].copy()
    tmp["decile"] = pd.qcut(
        tmp[score_column].rank(method="first"), 10, labels=range(1, 11)
    ).astype(int)
    return tmp.groupby("decile").agg(
        applicants=("target", "size"),
        cases=("target", "sum"),
        target_rate=("target", "mean"),
        avg_score=(score_column, "mean"),
    ).reset_index()


@st.cache_data
def strategy_curve(score_column, margin, lgd):
    """Approve the lowest-risk applicants first and accumulate outcomes."""
    order = np.argsort(pred[score_column].to_numpy(), kind="stable")
    y = pred["target"].to_numpy()[order]
    credit = pred["credit_amount"].to_numpy(dtype=float)[order]
    n = len(y)
    approved = np.arange(1, n + 1)
    bads = np.cumsum(y)
    pnl = np.where(y == 1, -lgd * credit, margin * credit)
    return pd.DataFrame({
        "approval_rate": approved / n,
        "bad_rate": bads / approved,
        "profit": np.cumsum(pnl),
        "cutoff_pd": pred[score_column].to_numpy()[order],
    })


def at_approval(curve, rate):
    idx = int(np.clip(round(rate * len(curve)) - 1, 0, len(curve) - 1))
    return curve.iloc[idx]


def fmt_money(value):
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}{value / 1e6:.1f}M"
    return f"{sign}{value:,.0f}"


# ---------------------------------------------------------------- charts
def decile_chart(deciles):
    fig = go.Figure(go.Bar(
        x=deciles["decile"],
        y=deciles["target_rate"] * 100,
        marker_color=[BLUE] * 9 + [TEAL],
        name="TARGET = 1 rate",
        hovertemplate="Decile %{x}<br>Rate %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(
        y=PORTFOLIO_RATE * 100, line_color=ORANGE, line_dash="dash",
        annotation_text="Portfolio average", annotation_font_color=ORANGE,
    )
    fig.update_layout(
        xaxis_title="Risk decile (10 = highest score)",
        yaxis_title="Observed TARGET = 1 (%)",
    )
    fig.update_xaxes(dtick=1)
    return chart_style(fig)


def roc_chart():
    fig = go.Figure()
    for name, (column, color) in reversed(MODELS.items()):
        fpr, tpr, _ = roc_curve(pred["target"], pred[column])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=name, line=dict(color=color, width=3)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
        line=dict(color=MUTED, dash="dash"),
    ))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return chart_style(fig)


def pr_chart():
    fig = go.Figure()
    for name, (column, color) in reversed(MODELS.items()):
        precision, recall, _ = precision_recall_curve(pred["target"], pred[column])
        fig.add_trace(go.Scatter(
            x=recall[::-1], y=precision[::-1], name=name, mode="lines",
            line=dict(color=color, width=3),
        ))
    fig.add_hline(y=PORTFOLIO_RATE, line_dash="dash", line_color=MUTED)
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return chart_style(fig)


def calibration_chart():
    fig = go.Figure()
    for name, (column, color) in reversed(MODELS.items()):
        observed, expected = calibration_curve(
            pred["target"], pred[column], n_bins=10, strategy="quantile"
        )
        fig.add_trace(go.Scatter(
            x=expected, y=observed, name=name, mode="lines+markers",
            line=dict(color=color, width=3),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 0.4], y=[0, 0.4], name="Perfect calibration", mode="lines",
        line=dict(color=MUTED, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Mean predicted probability", yaxis_title="Observed TARGET rate"
    )
    return chart_style(fig)


def bootstrap_chart(boot):
    fig = go.Figure()
    for _, row in boot.iterrows():
        better_low = row["Metric"] == "Brier Score"
        good = row["Difference CI High"] < 0 if better_low else row["Difference CI Low"] > 0
        fig.add_trace(go.Scatter(
            x=[row["Difference"]],
            y=[row["Metric"] + (" (lower is better)" if better_low else "")],
            mode="markers",
            marker=dict(size=13, color=TEAL if good else ORANGE),
            error_x=dict(
                type="data", symmetric=False,
                array=[row["Difference CI High"] - row["Difference"]],
                arrayminus=[row["Difference"] - row["Difference CI Low"]],
                color=TEAL if good else ORANGE, thickness=3, width=8,
            ),
            showlegend=False,
            hovertemplate=(
                f"{row['Metric']}<br>Δ = {row['Difference']:+.4f}<br>"
                f"95% CI [{row['Difference CI Low']:+.4f}, "
                f"{row['Difference CI High']:+.4f}]<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line_color=MUTED, line_dash="dash")
    fig.update_layout(xaxis_title="LightGBM − Logistic (95% bootstrap CI)")
    return chart_style(fig, 300)


def gap_chart(gap):
    fig = go.Figure()
    for col, color in [("Train ROC-AUC", MUTED), ("Holdout ROC-AUC", TEAL)]:
        fig.add_trace(go.Bar(
            x=gap["Model"], y=gap[col], name=col, marker_color=color,
            text=gap[col].map(lambda v: f"{v:.3f}"), textposition="outside",
        ))
    fig.update_layout(barmode="group", yaxis_title="ROC-AUC")
    fig.update_yaxes(range=[0.5, 0.82])
    return chart_style(fig, 320)


def ks_chart(column, color):
    curve, ks, at_pop, _ = rm.ks_curve(pred["target"], pred[column])
    k = curve.iloc[(curve["population"] - at_pop).abs().argmin()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["population"] * 100, y=curve["cum_bad"] * 100, name="Cumulative bads",
        line=dict(color=RED, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=curve["population"] * 100, y=curve["cum_good"] * 100, name="Cumulative goods",
        line=dict(color=color, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[k["population"] * 100] * 2, y=[k["cum_good"] * 100, k["cum_bad"] * 100],
        mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dot"),
        marker=dict(size=7, color=ORANGE), name=f"KS = {ks:.3f}",
    ))
    fig.add_annotation(
        x=k["population"] * 100, y=(k["cum_bad"] + k["cum_good"]) * 50,
        text=f"KS {ks:.3f}<br>at {at_pop:.0%} of applicants", showarrow=False,
        xanchor="left", xshift=8, font=dict(color=ORANGE),
    )
    fig.update_layout(
        xaxis_title="Applicants ranked riskiest → safest (%)", yaxis_title="Cumulative share (%)",
    )
    return chart_style(fig, 380)


def lorenz_chart():
    fig = go.Figure()
    bad_rate = PORTFOLIO_RATE
    fig.add_trace(go.Scatter(
        x=[0, bad_rate * 100, 100], y=[0, 100, 100], name="Perfect model",
        line=dict(color=MUTED, dash="dot", width=1.5),
    ))
    for name, (column, color) in reversed(MODELS.items()):
        curve, gini = rm.lorenz_curve(pred["target"], pred[column])
        fig.add_trace(go.Scatter(
            x=curve["population"] * 100, y=curve["cum_bad"] * 100,
            name=f"{name} · Gini {gini:.3f}", line=dict(color=color, width=3),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], name="Random", line=dict(color=MUTED, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Applicants declined, riskiest first (%)", yaxis_title="Bads captured (%)",
    )
    return chart_style(fig, 380)


def score_distribution_chart(column, color):
    scores = pd_to_score(pred[column])
    fig = go.Figure()
    for outcome, name, c in [(0, "No difficulty (good)", color), (1, "Payment difficulty (bad)", RED)]:
        fig.add_trace(go.Histogram(
            x=scores[pred["target"].to_numpy() == outcome], name=name,
            histnorm="percent", xbins=dict(start=480, end=680, size=5),
            marker_color=c, opacity=0.6,
        ))
    for grade, lower in GRADES[:-1]:
        fig.add_vline(x=lower, line_color=BORDER, line_dash="dot",
                      annotation_text=grade, annotation_position="top right",
                      annotation_font_color=MUTED)
    fig.update_layout(barmode="overlay", xaxis_title="Credit score", yaxis_title="Share of group (%)")
    return chart_style(fig, 380)


def rag_pill(status):
    return (
        f'<span class="pill" style="background:{RAG_COLORS[status]}22;'
        f'color:{RAG_COLORS[status]};border-color:{RAG_COLORS[status]}66">{status}</span>'
    )


def rag_table_html(table):
    rows = "".join(
        f"<tr><td class='t'>{r.test}</td><td class='r'>{r.result}</td>"
        f"<td>{rag_pill(r.status)}</td><td class='n'>{r.note}</td></tr>"
        for r in table.itertuples()
    )
    return (
        "<table class='rag'><thead><tr><th>Test</th><th>Result</th><th>Status</th>"
        f"<th>Basis</th></tr></thead><tbody>{rows}</tbody></table>"
    )


@st.cache_data
def scorecard(model):
    return rm.validation_scorecard(
        pred, data["bootstrap"], data["gap"], data["psi"], data["dq"], model
    )


# ---------------------------------------------------------------- header
best = data["bootstrap"].set_index("Metric")
st.markdown(f"""
<div class="hero-label">RISK ANALYTICS · MODEL VALIDATION · CREDIT STRATEGY</div>
<div class="hero-title">Credit Risk Intelligence</div>
<div class="hero-subtitle">
End-to-end probability-of-default modelling on 307,511 Home Credit applicants —
from a PostgreSQL feature mart to a validated challenger model and an approval strategy.
</div>
<div class="chips">
  <span class="chip">PostgreSQL feature mart</span>
  <span class="chip">58M+ source rows</span>
  <span class="chip">Logistic Regression vs LightGBM</span>
  <span class="chip">Paired bootstrap CIs</span>
  <span class="chip">TreeSHAP reason codes</span>
  <span class="chip">Score scaling & cut-off strategy</span>
</div>
""", unsafe_allow_html=True)

(
    overview, insights_tab, models_tab, strategy_tab, policy_tab, applicant_tab, validation_tab
) = st.tabs([
    "Overview",
    "Portfolio Insights",
    "Model Comparison",
    "Credit Strategy",
    "Policy Simulator",
    "Applicant Explorer",
    "Model Validation",
])

# ================================================================= OVERVIEW
with overview:
    deciles = risk_deciles("lightgbm_score")
    top = deciles.iloc[-1]
    lift = top["target_rate"] / PORTFOLIO_RATE
    capture = top["cases"] / pred["target"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", "307,511", border=True)
    c2.metric("Holdout applicants", f"{len(pred):,}", border=True)
    c3.metric("TARGET = 1 rate", f"{PORTFOLIO_RATE:.2%}", border=True)
    c4.metric(
        "LightGBM ROC-AUC", f"{lightgbm['roc_auc']:.4f}",
        delta=f"+{lightgbm['roc_auc'] - logistic['roc_auc']:.4f}",
        border=True,
    )

    st.subheader("Key findings")
    lgb_curve = strategy_curve("lightgbm_score", 0.10, 0.50)
    log_curve = strategy_curve("logistic_score", 0.10, 0.50)
    lgb80, log80 = at_approval(lgb_curve, 0.80), at_approval(log_curve, 0.80)
    avoided = (log80["bad_rate"] - lgb80["bad_rate"]) * 10_000
    auc_ci = best.loc["ROC-AUC"]
    gap = data["gap"].set_index("Model")

    i1, i2, i3 = st.columns(3)
    with i1:
        insight(
            "Challenger beats baseline — with evidence",
            f"LightGBM improves ROC-AUC by <b>{auc_ci['Difference']:+.3f}</b> "
            f"(95% CI {auc_ci['Difference CI Low']:+.3f} to "
            f"{auc_ci['Difference CI High']:+.3f}). All four metrics improve "
            "and every bootstrap interval excludes zero.",
        )
    with i2:
        insight(
            "Business value at an 80% approval rate",
            f"Bad rate among approved applicants falls from "
            f"<b>{log80['bad_rate']:.2%}</b> to <b>{lgb80['bad_rate']:.2%}</b> — "
            f"about <b>{avoided:.0f} fewer problem loans per 10,000 approvals</b> "
            "at the same volume.",
        )
    with i3:
        insight(
            "Overfitting signal to fix next",
            f"LightGBM train AUC is <b>{gap.loc['LightGBM', 'Train ROC-AUC']:.3f}</b> "
            f"vs <b>{gap.loc['LightGBM', 'Holdout ROC-AUC']:.3f}</b> on holdout "
            f"(gap {gap.loc['LightGBM', 'AUC Gap']:.3f}); the logistic baseline "
            "shows no gap. Early stopping and stronger regularisation are next.",
        )

    st.subheader("Risk concentration & discrimination")
    left, right = st.columns(2)
    with left:
        show(decile_chart(deciles), "overview_deciles")
    with right:
        show(roc_chart(), "overview_roc")

    c1, c2, c3 = st.columns(3)
    c1.metric("Top-decile risk lift", f"{lift:.2f}x", border=True)
    c2.metric("Positive cases in top decile", f"{capture:.1%}", border=True)
    c3.metric("LightGBM KS", f"{lightgbm['ks']:.4f}", border=True)

    with st.expander("How the pipeline works"):
        st.markdown("""
1. **Load & model the data in PostgreSQL** – the Home Credit tables (58M+ rows) are
   loaded into Postgres and aggregated into one-row-per-applicant views and
   materialised views (`sql/`). Mart v2 combines application, bureau, previous
   applications and installment behaviour; bureau-balance and credit-card features
   are already materialised for the next iteration. Data-quality assertions check
   grain, label completeness, sentinel values and ratio ranges.
2. **Baseline** – regularised Logistic Regression with median imputation,
   missing-value indicators, scaling and one-hot encoding (`train_model.py`).
3. **Challenger** – LightGBM with native categorical handling on the *identical*
   holdout IDs, verified programmatically (`compare_models.py`).
4. **Validation** – paired stratified bootstrap (500 resamples) for metric
   differences, train-vs-holdout gap, calibration (`validate_models.py`,
   `check_generalization.py`).
5. **Explainability & strategy** – TreeSHAP contributions, score scaling, grade
   bands and approval-rate economics (`build_dashboard_data.py`, this app).
""")
    st.caption(
        "Age is deliberately excluded from both models and only used for the fairness review. "
        "All statistics are measured on the held-out sample."
    )

# ================================================================= INSIGHTS
with insights_tab:
    st.subheader("Where does risk sit in the portfolio?")
    st.caption(
        "Descriptive TARGET rates across all 307,511 applicants, computed from the "
        "PostgreSQL feature mart. Segments with fewer than 100 applicants are hidden."
    )
    segments = data["segments"]
    segments = segments[segments["applicants"] >= 100]
    dimensions = list(dict.fromkeys(segments["dimension"]))
    dimension = st.pills(
        "Segment by", dimensions, default="Installment shortfall rate",
        key="segment_dimension",
    ) or dimensions[0]
    seg = segments[segments["dimension"] == dimension].sort_values("order")

    fig = go.Figure(go.Bar(
        x=seg["segment"],
        y=seg["target_rate"] * 100,
        marker_color=[
            RED if r > PORTFOLIO_RATE * 1.25 else TEAL if r < PORTFOLIO_RATE * 0.8 else BLUE
            for r in seg["target_rate"]
        ],
        customdata=np.stack([seg["applicants"], seg["cases"]], axis=1),
        text=(seg["target_rate"] * 100).map(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate=(
            "%{x}<br>TARGET rate %{y:.2f}%<br>"
            "Applicants %{customdata[0]:,}<br>Cases %{customdata[1]:,}<extra></extra>"
        ),
    ))
    fig.add_hline(
        y=PORTFOLIO_RATE * 100, line_color=ORANGE, line_dash="dash",
        annotation_text="Portfolio average", annotation_font_color=ORANGE,
    )
    fig.update_layout(xaxis_title=dimension, yaxis_title="Observed TARGET = 1 (%)")
    fig.update_yaxes(rangemode="tozero")
    show(chart_style(fig, 400), "segment_chart")

    left, right = st.columns([3, 2])
    with left:
        table = seg[["segment", "applicants", "cases", "target_rate"]].copy()
        table["share"] = table["applicants"] / table["applicants"].sum()
        table["relative_risk"] = table["target_rate"] / PORTFOLIO_RATE
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "segment": "Segment",
                "applicants": st.column_config.NumberColumn("Applicants", format="%d"),
                "cases": st.column_config.NumberColumn("TARGET = 1", format="%d"),
                "target_rate": st.column_config.NumberColumn("Rate", format="percent"),
                "share": st.column_config.ProgressColumn(
                    "Share of portfolio", format="percent", min_value=0, max_value=1
                ),
                "relative_risk": st.column_config.NumberColumn("vs average", format="%.2fx"),
            },
        )
    with right:
        rates = segments.groupby("dimension")["target_rate"].agg(["min", "max"])
        rates["spread"] = rates["max"] / rates["min"]
        strongest = rates["spread"].idxmax()
        insight(
            "Reading the segments",
            "Payment behaviour is the clearest single signal: applicants who missed "
            "the due-date amount on 30%+ of past installments show roughly "
            "<b>2x</b> the default-difficulty rate of those who never did. "
            "Younger applicants and short employment tenure also carry elevated risk; "
            "pensioners and applicants with higher education sit well below average. "
            f"Largest high/low spread among dimensions shown: <b>{strongest}</b>.",
        )

# ================================================================= MODELS
with models_tab:
    st.subheader("Baseline vs challenger")
    st.dataframe(
        data["comparison"],
        width="stretch",
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in ["Logistic Regression", "LightGBM", "Difference"]
        },
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**ROC curve**")
        show(roc_chart(), "comparison_roc")
    with right:
        st.markdown("**Precision–recall curve**")
        show(pr_chart(), "comparison_pr")

    left, right = st.columns(2)
    with left:
        st.markdown("**Is the improvement real? Paired bootstrap, 500 resamples**")
        show(bootstrap_chart(data["bootstrap"]), "bootstrap_chart")
        st.caption(
            "Both models are scored on the same resampled applicants, so the "
            "interval reflects the uncertainty of the *difference*. "
            "Teal = interval excludes zero in LightGBM's favour."
        )
    with right:
        st.markdown("**Does it generalise? Train vs holdout**")
        show(gap_chart(data["gap"]), "gap_chart")
        st.caption(
            "LightGBM fits the training data noticeably better than the holdout. "
            "It still wins on holdout, but the gap argues for early stopping, "
            "lower tree complexity and an out-of-time test before any deployment."
        )

    st.subheader("Credit-risk diagnostics")
    st.caption(
        "The standard views a credit-risk or model-validation team reviews before "
        "accepting a scorecard: separation (KS), concentration (Lorenz / Gini), score "
        "distributions and the decile gains table."
    )
    diag_model = st.segmented_control(
        "Model", list(MODELS), default="LightGBM", key="diag_model"
    ) or "LightGBM"
    diag_col, diag_color = MODELS[diag_model]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Kolmogorov–Smirnov: {diag_model}**")
        show(ks_chart(diag_col, diag_color), "ks_chart")
    with right:
        st.markdown("**Lorenz curve: share of bads captured**")
        show(lorenz_chart(), "lorenz_chart")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Score distribution by outcome: {diag_model}**")
        show(score_distribution_chart(diag_col, diag_color), "score_dist")
    with right:
        st.markdown(f"**Gains table: {diag_model}** (decile 1 = riskiest 10%)")
        gains = rm.gains_table(pred["target"], pred[diag_col])
        gains["score_range"] = [
            f"{pd_to_score(hi):.0f} – {pd_to_score(lo):.0f}"
            for lo, hi in zip(gains["min_pd"], gains["max_pd"])
        ]
        st.dataframe(
            gains[["decile", "score_range", "applicants", "bads", "bad_rate",
                   "cum_bads_captured", "lift", "cum_lift"]],
            hide_index=True, width="stretch", height=388,
            column_config={
                "decile": "Decile",
                "score_range": "Score",
                "applicants": st.column_config.NumberColumn("Applicants", format="%d"),
                "bads": st.column_config.NumberColumn("Bads", format="%d"),
                "bad_rate": st.column_config.NumberColumn("Bad rate", format="percent"),
                "cum_bads_captured": st.column_config.ProgressColumn(
                    "Cum. bads captured", format="percent", min_value=0, max_value=1
                ),
                "lift": st.column_config.NumberColumn("Lift", format="%.2fx"),
                "cum_lift": st.column_config.NumberColumn("Cum. lift", format="%.2fx"),
            },
        )


# ================================================================= STRATEGY
with strategy_tab:
    st.subheader("From probability to decision")
    st.caption(
        f"PDs are converted to a points-based score: {BASE_SCORE} points at "
        f"good:bad odds of {BASE_ODDS}:1, and every {PDO} points doubles the odds "
        "(the standard scorecard scaling used in retail credit)."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
    with ctrl1:
        model_choice = st.segmented_control(
            "Scoring model", list(MODELS), default="LightGBM", key="strategy_model"
        ) or "LightGBM"
    with ctrl2:
        margin = st.slider(
            "Net revenue on a good loan (% of credit)", 1, 30, 10, key="margin"
        ) / 100
    with ctrl3:
        lgd = st.slider(
            "Loss on a problem loan (% of credit)", 10, 100, 50, key="lgd"
        ) / 100

    column, color = MODELS[model_choice]
    scores = pd_to_score(pred[column])
    grades = score_to_grade(scores)

    # ---- grade table
    grade_df = (
        pd.DataFrame({"grade": grades, "target": pred["target"], "pd": pred[column]})
        .groupby("grade")
        .agg(applicants=("target", "size"), bad_rate=("target", "mean"), avg_pd=("pd", "mean"))
        .reindex([g for g, _ in GRADES])
        .dropna()
        .reset_index()
    )
    grade_df["score_range"] = grade_df["grade"].map(grade_range)
    grade_df["share"] = grade_df["applicants"] / grade_df["applicants"].sum()
    monotonic = grade_df["bad_rate"].is_monotonic_increasing

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grade_df["grade"], y=grade_df["bad_rate"] * 100, name="Observed rate",
            marker_color=[GRADE_COLORS[g] for g in grade_df["grade"]],
            text=(grade_df["bad_rate"] * 100).map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=grade_df["grade"], y=grade_df["avg_pd"] * 100, name="Average predicted PD",
            mode="lines+markers", line=dict(color=TEXT, dash="dot"),
        ))
        fig.update_layout(xaxis_title="Risk grade", yaxis_title="TARGET = 1 (%)")
        show(chart_style(fig, 340), "grade_chart")
    with right:
        st.dataframe(
            grade_df[["grade", "score_range", "applicants", "share", "bad_rate", "avg_pd"]],
            hide_index=True,
            width="stretch",
            column_config={
                "grade": "Grade",
                "score_range": "Score",
                "applicants": st.column_config.NumberColumn("Applicants", format="%d"),
                "share": st.column_config.NumberColumn("Share", format="percent"),
                "bad_rate": st.column_config.NumberColumn("Observed", format="percent"),
                "avg_pd": st.column_config.NumberColumn("Predicted", format="percent"),
            },
        )
        if monotonic:
            st.success("Observed rate rises monotonically from grade A to F.")
        else:
            st.warning("Observed rate is not monotonic across grades — review band cut-offs.")

    # ---- approval strategy
    st.subheader("Approval-rate trade-off")
    curve = strategy_curve(column, margin, lgd)
    other_name = next(m for m in MODELS if m != model_choice)
    other_curve = strategy_curve(MODELS[other_name][0], margin, lgd)
    best_idx = int(curve["profit"].idxmax())
    best_row = curve.iloc[best_idx]

    approval = st.slider(
        "Target approval rate", 0.50, 1.00, 0.80, 0.01, format="%.2f", key="approval"
    )
    row = at_approval(curve, approval)
    other_row = at_approval(other_curve, approval)
    approve_all = curve.iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Score cut-off",
        f"{pd_to_score(row['cutoff_pd']):.0f}",
        help="Approve applicants scoring at or above this value",
    )
    k2.metric(
        "Bad rate of approved", f"{row['bad_rate']:.2%}",
        delta=f"{(row['bad_rate'] - other_row['bad_rate']) * 100:+.2f} pp vs {other_name}",
        delta_color="inverse",
    )
    k3.metric(
        "Problem loans declined",
        f"{1 - row['bad_rate'] * approval / PORTFOLIO_RATE:.1%}",
        help="Share of all TARGET = 1 applicants that this cut-off rejects",
    )
    k4.metric(
        "Profit vs approve-all",
        fmt_money(row["profit"] - approve_all["profit"]),
        help="Illustrative, using credit amount and the revenue/loss assumptions above",
    )

    sample = curve.iloc[np.linspace(0, len(curve) - 1, 400).astype(int)]
    other_sample = other_curve.iloc[np.linspace(0, len(other_curve) - 1, 400).astype(int)]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=other_sample["approval_rate"] * 100, y=other_sample["bad_rate"] * 100,
            name=other_name, line=dict(color=MODELS[other_name][1], width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=sample["approval_rate"] * 100, y=sample["bad_rate"] * 100,
            name=model_choice, line=dict(color=color, width=3),
        ))
        fig.add_vline(x=approval * 100, line_color=ORANGE, line_dash="dash")
        fig.update_layout(
            xaxis_title="Approval rate (%)", yaxis_title="Bad rate of approved (%)",
            hovermode="x unified",
        )
        fig.update_xaxes(range=[40, 100])
        fig.update_yaxes(range=[0, PORTFOLIO_RATE * 100 * 1.1])
        show(chart_style(fig, 340), "strategy_badrate")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample["approval_rate"] * 100, y=sample["profit"],
            name="Cumulative profit", line=dict(color=color, width=3),
            fill="tozeroy", fillcolor="rgba(49,198,176,0.12)",
        ))
        fig.add_trace(go.Scatter(
            x=[best_row["approval_rate"] * 100], y=[best_row["profit"]],
            mode="markers+text", marker=dict(size=12, color=ORANGE),
            text=[f"Optimum {best_row['approval_rate']:.0%}"], textposition="top center",
            name="Profit-maximising approval rate",
        ))
        fig.add_vline(x=approval * 100, line_color=ORANGE, line_dash="dash")
        fig.update_layout(
            xaxis_title="Approval rate (%)", yaxis_title="Illustrative profit",
            showlegend=False,
        )
        fig.update_xaxes(range=[40, 100])
        show(chart_style(fig, 340), "strategy_profit")

    st.caption(
        f"With these assumptions the break-even PD is {margin / (margin + lgd):.1%}; the "
        f"profit-maximising approval rate is {best_row['approval_rate']:.0%}. "
        "Economics are illustrative — TARGET marks payment difficulty, not realised loss."
    )

    # ---- fairness
    st.subheader("Fairness review by age band")
    approved_mask = pred[column].to_numpy() <= row["cutoff_pd"]
    age_band = pd.cut(
        pred["age_years"], [0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"], right=False,
    )
    fair = (
        pd.DataFrame({"band": age_band, "approved": approved_mask, "target": pred["target"]})
        .groupby("band", observed=True)
        .agg(
            applicants=("approved", "size"),
            approval_rate=("approved", "mean"),
            observed_rate=("target", "mean"),
        )
        .reset_index()
    )
    fair["impact_ratio"] = fair["approval_rate"] / fair["approval_rate"].max()

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fair["band"].astype(str), y=fair["approval_rate"] * 100,
            name="Approval rate",
            marker_color=[RED if r < 0.8 else TEAL for r in fair["impact_ratio"]],
            text=(fair["approval_rate"] * 100).map(lambda v: f"{v:.0f}%"),
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Age band", yaxis_title="Approval rate (%)")
        fig.update_yaxes(range=[0, 110])
        show(chart_style(fig, 320), "fairness_chart")
    with right:
        st.dataframe(
            fair, hide_index=True, width="stretch",
            column_config={
                "band": "Age band",
                "applicants": st.column_config.NumberColumn("Applicants", format="%d"),
                "approval_rate": st.column_config.NumberColumn("Approved", format="percent"),
                "observed_rate": st.column_config.NumberColumn("TARGET rate", format="percent"),
                "impact_ratio": st.column_config.NumberColumn("Impact ratio", format="%.2f"),
            },
        )
        flagged = fair.loc[fair["impact_ratio"] < 0.8, "band"].astype(str).tolist()
        if flagged:
            st.warning(
                f"Impact ratio below 0.8 for {', '.join(flagged)}. Age is not a model "
                "input, so the gap comes through correlated features (e.g. employment "
                "length) and mirrors genuinely higher observed rates — it still needs a "
                "documented business justification before use."
            )
        else:
            st.success("All age bands keep an impact ratio of at least 0.8 at this cut-off.")

# ================================================================= POLICY
with policy_tab:
    st.subheader("Policy simulator")
    st.caption(
        "Combine a score cut-off with knock-out rules, as a credit-policy team would, and see "
        "the effect on approval rate, bad rate and profit. Rules apply in order; each row of "
        "the waterfall shows only the applicants that rule declines on top of the earlier ones."
    )

    controls, results = st.columns([1, 2.2], gap="large")
    with controls:
        pol_model = st.segmented_control(
            "Scoring model", list(MODELS), default="LightGBM", key="pol_model"
        ) or "LightGBM"
        pol_col = MODELS[pol_model][0]
        pol_scores = pd_to_score(pred[pol_col])
        min_score = st.slider("Minimum credit score", 520, 640, 570, 1, key="pol_score")

        st.markdown("**Knock-out rules**")
        rule_specs = [
            ("shortfall", "Due-date shortfall rate above", "due_date_shortfall_rate",
             0.0, 1.0, 0.30, 0.05, ">", True, "%.2f"),
            ("refused", "Previous refusals at least", "refused_count",
             1, 15, 4, 1, ">=", True, "%d"),
            ("active", "Active bureau loans at least", "active_loan_count",
             1, 15, 6, 1, ">=", False, "%d"),
            ("cti", "Credit-to-income above", "credit_to_income",
             1.0, 20.0, 10.0, 0.5, ">", False, "%.1f"),
            ("ati", "Annuity-to-income above", "annuity_to_income",
             0.05, 1.0, 0.40, 0.05, ">", False, "%.2f"),
        ]
        rules = []
        for key, text, column, lo, hi, default, step, op, on, fmt in rule_specs:
            enabled = st.checkbox(f"Decline if {text.lower()}", value=on, key=f"rule_{key}")
            threshold = st.slider(
                text, lo, hi, default, step, format=fmt, key=f"thr_{key}",
                disabled=not enabled, label_visibility="collapsed",
            )
            if enabled:
                values = pred[column].to_numpy(dtype=float)
                hit = (values > threshold) if op == ">" else (values >= threshold)
                hit = np.nan_to_num(hit, nan=False).astype(bool)
                shown = f"{threshold:.2f}" if isinstance(threshold, float) else f"{threshold}"
                rules.append((f"{text} {shown}", hit))
        with st.expander("Economics assumptions"):
            pol_margin = st.slider("Net revenue on a good loan (%)", 1, 30, 10, key="pol_margin") / 100
            pol_lgd = st.slider("Loss on a problem loan (%)", 10, 100, 50, key="pol_lgd") / 100
        st.caption("Age-based rules are deliberately not offered: age is a protected attribute.")

    y_all = pred["target"].to_numpy()
    credit = pred["credit_amount"].to_numpy(dtype=float)
    n_all = len(y_all)

    def pnl(mask):
        return float(np.where(y_all[mask] == 1, -pol_lgd, pol_margin) @ credit[mask])

    approved = np.ones(n_all, dtype=bool)
    steps = [("Score below " + str(min_score), pol_scores < min_score)] + rules
    waterfall = []
    for name, hit in steps:
        declined_now = approved & hit
        waterfall.append({
            "rule": name,
            "declined": int(declined_now.sum()),
            "bad_rate_declined": y_all[declined_now].mean() if declined_now.any() else np.nan,
        })
        approved &= ~hit
    waterfall = pd.DataFrame(waterfall)

    appr_rate = approved.mean()
    bad_rate_appr = y_all[approved].mean() if approved.any() else np.nan
    bads_declined = 1 - y_all[approved].sum() / y_all.sum()
    profit_delta = pnl(approved) - pnl(np.ones(n_all, dtype=bool))

    # Benchmark: score-only policy at the same approval rate
    k = max(int(round(appr_rate * n_all)), 1)
    score_only = np.zeros(n_all, dtype=bool)
    score_only[np.argsort(pred[pol_col].to_numpy(), kind="stable")[:k]] = True
    score_only_bad = y_all[score_only].mean()

    with results:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Approval rate", f"{appr_rate:.1%}", border=True)
        k2.metric(
            "Bad rate of approved", f"{bad_rate_appr:.2%}",
            delta=f"{(bad_rate_appr - PORTFOLIO_RATE) * 100:+.2f} pp vs portfolio",
            delta_color="inverse", border=True,
        )
        k3.metric("Problem loans declined", f"{bads_declined:.1%}", border=True)
        k4.metric("Profit vs approve-all", fmt_money(profit_delta), border=True)

        diff_pp = (bad_rate_appr - score_only_bad) * 100
        if rules:
            if diff_pp > 0.02:
                verdict = (
                    f"The knock-out rules <b>cost</b> accuracy: a pure score cut-off approving the "
                    f"same {appr_rate:.0%} would reach a bad rate of <b>{score_only_bad:.2%}</b> "
                    f"({diff_pp:+.2f} pp). The score already prices this information in."
                )
            elif diff_pp < -0.02:
                verdict = (
                    f"The knock-out rules <b>add</b> value beyond the score: a pure cut-off at the same "
                    f"{appr_rate:.0%} approval gives <b>{score_only_bad:.2%}</b> ({diff_pp:+.2f} pp)."
                )
            else:
                verdict = (
                    f"The rules are <b>neutral</b>: a pure score cut-off at {appr_rate:.0%} approval gives "
                    f"a similar bad rate ({score_only_bad:.2%})."
                )
            insight("Do the rules beat the score alone?", verdict)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=waterfall["rule"][::-1], x=waterfall["declined"][::-1], orientation="h",
            marker_color=[ORANGE] * (len(waterfall) - 1) + [MODELS[pol_model][1]],
            text=[
                f"{d:,} · {b:.1%} bad" if d else "none"
                for d, b in zip(waterfall["declined"][::-1], waterfall["bad_rate_declined"][::-1])
            ],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} applicants declined<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Decline waterfall", font=dict(size=14)),
            xaxis_title="Additional applicants declined",
        )
        fig.update_yaxes(automargin=True)
        chart_style(fig, 120 + 48 * len(waterfall)).update_layout(margin=dict(l=35, r=110, t=45, b=45))
        show(fig, "policy_waterfall")
        st.caption(
            f"Portfolio TARGET rate is {PORTFOLIO_RATE:.2%}. A rule is earning its place when the "
            "bad rate of the applicants it declines is well above that."
        )

    # ---- swap-set
    st.subheader("Swap-set analysis: benchmark vs challenger")
    st.caption(
        "At the same approval rate, which applicants does switching from the logistic benchmark "
        "to LightGBM newly approve (swap-in) or newly decline (swap-out)? A good challenger swaps "
        "out riskier applicants than it swaps in."
    )
    swap_rate = st.slider("Approval rate for both models", 0.50, 0.95, 0.80, 0.01,
                          format="%.2f", key="swap_rate")
    k = int(round(swap_rate * n_all))
    lr_ok = np.zeros(n_all, dtype=bool)
    gb_ok = np.zeros(n_all, dtype=bool)
    lr_ok[np.argsort(pred["logistic_score"].to_numpy(), kind="stable")[:k]] = True
    gb_ok[np.argsort(pred["lightgbm_score"].to_numpy(), kind="stable")[:k]] = True
    cells = {
        "Approved by both": lr_ok & gb_ok,
        "Swap-in (LightGBM only)": ~lr_ok & gb_ok,
        "Swap-out (benchmark only)": lr_ok & ~gb_ok,
        "Declined by both": ~lr_ok & ~gb_ok,
    }
    swap = pd.DataFrame([
        {"segment": name, "applicants": int(m.sum()),
         "share": m.mean(), "bad_rate": y_all[m].mean() if m.any() else np.nan}
        for name, m in cells.items()
    ])

    left, right = st.columns([2, 3])
    with left:
        # rows: LightGBM approves / declines; columns: benchmark approves / declines
        z = [[swap.loc[0, "applicants"], swap.loc[1, "applicants"]],
             [swap.loc[2, "applicants"], swap.loc[3, "applicants"]]]
        br = [[swap.loc[0, "bad_rate"], swap.loc[1, "bad_rate"]],
              [swap.loc[2, "bad_rate"], swap.loc[3, "bad_rate"]]]
        fig = go.Figure(go.Heatmap(
            z=br, x=["Benchmark approves", "Benchmark declines"],
            y=["LightGBM approves", "LightGBM declines"],
            text=[[f"{z[i][j]:,}<br>bad rate {br[i][j]:.1%}" for j in range(2)] for i in range(2)],
            texttemplate="%{text}", colorscale=[[0, TEAL], [0.5, ORANGE], [1, RED]],
            showscale=False, hoverinfo="skip",
        ))
        fig.update_yaxes(autorange="reversed")
        show(chart_style(fig, 300), "swap_heatmap")
    with right:
        swap_in, swap_out = swap.loc[1], swap.loc[2]
        insight(
            "Swap-set verdict",
            f"Switching to LightGBM at {swap_rate:.0%} approval swaps in "
            f"<b>{swap_in['applicants']:,}</b> applicants with a bad rate of "
            f"<b>{swap_in['bad_rate']:.1%}</b> and swaps out <b>{swap_out['applicants']:,}</b> "
            f"with a bad rate of <b>{swap_out['bad_rate']:.1%}</b>. "
            + ("The challenger trades riskier applicants for safer ones at the same volume."
               if swap_out["bad_rate"] > swap_in["bad_rate"]
               else "The challenger does not improve the approved population at this volume."),
        )
        st.dataframe(
            swap, hide_index=True, width="stretch",
            column_config={
                "segment": "Segment",
                "applicants": st.column_config.NumberColumn("Applicants", format="%d"),
                "share": st.column_config.NumberColumn("Share", format="percent"),
                "bad_rate": st.column_config.NumberColumn("Bad rate", format="percent"),
            },
        )

# ================================================================= APPLICANT
with applicant_tab:
    st.subheader("Why did the model score this applicant this way?")
    st.caption(
        "TreeSHAP decomposes each LightGBM prediction (in log-odds) into the "
        "contribution of every feature. A random sample of 5,000 holdout applicants is available."
    )
    shap_sample = data["shap_sample"]
    sample_pred = pred.set_index("applicant_id").loc[shap_sample["applicant_id"]]
    sample_pred = sample_pred.assign(score=pd_to_score(sample_pred["lightgbm_score"]))

    tiers = {
        "Highest risk": sample_pred.sort_values("lightgbm_score", ascending=False),
        "Lowest risk": sample_pred.sort_values("lightgbm_score"),
        "Defaulted (TARGET = 1)": sample_pred[sample_pred["target"] == 1],
        "All sampled": sample_pred,
    }
    pick1, pick2 = st.columns([1, 2])
    with pick1:
        tier = st.selectbox("Browse", list(tiers), key="applicant_tier")
    with pick2:
        options = tiers[tier].index[:300].tolist()
        applicant_id = st.selectbox(
            "Applicant ID",
            options,
            format_func=lambda i: (
                f"{i} · score {sample_pred.loc[i, 'score']:.0f} · "
                f"PD {sample_pred.loc[i, 'lightgbm_score']:.1%}"
            ),
            key="applicant_id",
        )

    applicant = pred[pred["applicant_id"] == applicant_id].iloc[0]
    contrib = shap_sample.set_index("applicant_id").loc[applicant_id]
    base_value = float(contrib["base_value"])
    contrib = contrib[FEATURES].astype(float)
    pd_value = float(applicant["lightgbm_score"])
    score_value = float(pd_to_score(pd_value))
    grade = score_to_grade(score_value)[0]
    decile = int(np.ceil((pred["lightgbm_score"] < pd_value).mean() * 10)) or 1

    def fmt_value(feature, value):
        if pd.isna(value):
            return "missing"
        if isinstance(value, str):
            return value
        if feature in ("annual_income", "credit_amount", "annuity_amount", "avg_bureau_credit"):
            return f"{value:,.0f}"
        if feature in ("due_date_shortfall_rate", "approval_rate"):
            return f"{value:.0%}"
        if feature in ("credit_to_income", "annuity_to_income"):
            return f"{value:.2f}x"
        if feature == "employment_years":
            return f"{value:.1f} years"
        return f"{value:,.0f}"

    g1, g2, g3, g4 = st.columns([1, 1.3, 1.3, 1.3])
    with g1:
        st.markdown(
            f'<div class="grade-badge" style="background:{GRADE_COLORS[grade]}">{grade}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Risk grade")
    g2.metric("Probability (PD)", f"{pd_value:.1%}", border=True)
    g3.metric("Credit score", f"{score_value:.0f}", border=True)
    g4.metric(
        "Actual outcome",
        "Difficulty" if applicant["target"] == 1 else "No issue",
        help="Observed TARGET on the holdout — never seen by the model during training",
        border=True,
    )

    left, right = st.columns([3, 2])
    with left:
        order = contrib.abs().sort_values(ascending=False)
        top_feats = order.index[:9].tolist()
        rest = contrib.drop(top_feats).sum()
        names = [f"{label(f)} = {fmt_value(f, applicant[f])}" for f in top_feats]
        values = contrib[top_feats].tolist()
        if abs(rest) > 1e-9:
            names.append(f"{len(FEATURES) - 9} other features")
            values.append(rest)
        avg_pd = 1 / (1 + np.exp(-base_value))
        fig = go.Figure(go.Bar(
            orientation="h",
            y=names[::-1],
            x=values[::-1],
            marker_color=[RED if v > 0 else TEAL for v in values[::-1]],
            text=[f"{v:+.2f}" for v in values[::-1]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=MUTED)
        fig.update_layout(
            title=dict(
                text=(
                    f"Average applicant {avg_pd:.1%}  →  this applicant {pd_value:.1%}"
                ),
                font=dict(size=14),
            ),
            xaxis_title="Log-odds contribution (red raises risk)",
        )
        fig.update_yaxes(automargin=True)
        chart_style(fig, 440).update_layout(margin=dict(l=35, r=45, t=55, b=45))
        show(fig, "shap_contributions")
    with right:
        st.markdown("**Top reasons — risk raising**")
        raising = contrib[contrib > 0].sort_values(ascending=False).head(4)
        if raising.empty:
            st.caption("No feature raises this applicant's risk above average.")
        for f, v in raising.items():
            st.markdown(
                f'<div class="reason"><b>▲ {label(f)}</b> = {fmt_value(f, applicant[f])} '
                f"<span style='color:{MUTED}'>({v:+.2f})</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("**Top reasons — risk lowering**")
        lowering = contrib[contrib < 0].sort_values().head(3)
        for f, v in lowering.items():
            st.markdown(
                f'<div class="reason good"><b>▼ {label(f)}</b> = {fmt_value(f, applicant[f])} '
                f"<span style='color:{MUTED}'>({v:+.2f})</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(f"Holdout risk decile: {decile} of 10.")

    # ---- what-if simulator
    st.subheader("What-if simulator")
    st.caption(
        "Change the application and re-score it live with the saved LightGBM model. "
        "Credit-to-income and annuity-to-income are recomputed automatically."
    )

    def num(feature, default=0.0):
        value = applicant[feature]
        return default if pd.isna(value) else float(value)

    w1, w2, w3 = st.columns(3)
    with w1:
        credit = st.number_input(
            "Credit amount", 10_000, 5_000_000, int(num("credit_amount")),
            step=25_000, format="%d", key=f"wi_credit_{applicant_id}",
        )
        income = st.number_input(
            "Annual income", 10_000, 10_000_000, int(num("annual_income")),
            step=10_000, format="%d", key=f"wi_income_{applicant_id}",
        )
    with w2:
        employment = st.slider(
            "Employment length (years)", 0.0, 45.0, min(num("employment_years"), 45.0),
            0.5, key=f"wi_emp_{applicant_id}",
            disabled=bool(applicant["employment_sentinel_flag"]),
        )
        active = st.slider(
            "Active bureau loans", 0, 20, int(num("active_loan_count")),
            key=f"wi_active_{applicant_id}",
        )
    with w3:
        shortfall = st.slider(
            "Due-date shortfall rate", 0.0, 1.0, num("due_date_shortfall_rate"), 0.01,
            key=f"wi_short_{applicant_id}",
            disabled=not bool(applicant["has_installment_history"]),
        )
        refused = st.slider(
            "Previous refusals", 0, 20, int(min(num("refused_count"), 20)),
            key=f"wi_refused_{applicant_id}",
        )

    original = pred.loc[pred["applicant_id"] == applicant_id, FEATURES].copy()
    scenario = original.copy()
    annuity_ratio = num("annuity_amount") / max(num("credit_amount"), 1.0)
    annuity = credit * annuity_ratio
    scenario["credit_amount"] = credit
    scenario["annual_income"] = income
    scenario["annuity_amount"] = annuity
    scenario["credit_to_income"] = round(credit / income, 4)
    scenario["annuity_to_income"] = round(annuity / income, 4)
    if not applicant["employment_sentinel_flag"]:
        scenario["employment_years"] = employment
    scenario["active_loan_count"] = active
    scenario["bureau_loan_count"] = max(int(num("bureau_loan_count")), active)
    if applicant["has_installment_history"]:
        scenario["due_date_shortfall_rate"] = shortfall
    scenario["refused_count"] = refused
    scenario["previous_application_count"] = max(
        int(num("previous_application_count")), int(num("approved_count")) + refused
    )

    base_pd = float(booster.predict(original)[0])
    new_pd = float(booster.predict(scenario)[0])
    new_score = float(pd_to_score(new_pd))
    new_grade = score_to_grade(new_score)[0]

    r1, r2, r3 = st.columns(3)
    changed = abs(new_pd - base_pd) >= 0.0005
    r1.metric(
        "Scenario PD", f"{new_pd:.1%}",
        delta=f"{(new_pd - base_pd) * 100:+.1f} pp" if changed else None,
        delta_color="inverse", border=True,
    )
    r2.metric(
        "Scenario score", f"{new_score:.0f}",
        delta=f"{new_score - pd_to_score(base_pd):+.0f} pts" if changed else None,
        border=True,
    )
    r3.metric("Scenario grade", new_grade, delta=f"from {grade}" if new_grade != grade else None,
              delta_color="off", border=True)
    st.caption(
        "The simulator shows model sensitivity, not causal effects. Tree models are "
        "piecewise constant, so small changes can leave the score unchanged."
    )

# ================================================================= VALIDATION
with validation_tab:
    st.subheader("Validation scorecard")
    st.caption(
        "Each test is rated Green / Amber / Red against pre-defined thresholds, in the style "
        "of a bank model-risk review. Any Red or unassessed test blocks production approval."
    )
    val_model = st.segmented_control(
        "Model under review", list(MODELS), default="LightGBM", key="val_model"
    ) or "LightGBM"
    sc_table, rating = scorecard(val_model)
    counts = sc_table["status"].value_counts()
    rating_color = (
        RAG_COLORS["Red"] if rating.startswith("Not")
        else RAG_COLORS["Amber"] if "conditions" in rating else RAG_COLORS["Green"]
    )
    tally = " · ".join(
        f"{counts.get(s, 0)} {s}" for s in ["Green", "Amber", "Red", "Not assessed"]
    )
    st.markdown(
        f'<div class="rating"><div><div class="meta">Overall rating · {val_model}</div>'
        f'<div class="big" style="color:{rating_color}">{rating}</div></div>'
        f'<div class="meta">{tally}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="rag-wrap">{rag_table_html(sc_table)}</div>', unsafe_allow_html=True)
    with st.expander("RAG thresholds"):
        st.dataframe(
            pd.DataFrame(RAG_THRESHOLDS, columns=["Test", "Thresholds"]),
            hide_index=True, width="stretch",
        )
        st.caption(
            "Thresholds are illustrative, in line with common practice for retail "
            "application scorecards; a real model-risk policy sets its own."
        )

    pdf_report = ROOT / "docs" / "Model_Validation_Report.pdf"
    if pdf_report.exists():
        st.download_button(
            "Download full validation report (PDF)",
            pdf_report.read_bytes(),
            file_name="Model_Validation_Report.pdf",
            mime="application/pdf",
            type="primary",
            key="download_pdf_report",
        )

    st.subheader("Calibration")
    left, right = st.columns([3, 2])
    with left:
        show(calibration_chart(), "validation_calibration")
    with right:
        val_col = MODELS[val_model][0]
        binom = rm.binomial_grade_test(
            score_to_grade(pd_to_score(pred[val_col])), pred["target"], pred[val_col]
        )
        hl_stat, hl_p = rm.hosmer_lemeshow(pred["target"], pred[val_col])
        st.markdown(f"**Binomial test by grade: {val_model}**")
        st.dataframe(
            binom[["grade", "applicants", "observed_rate", "predicted_pd", "p_value", "result"]],
            hide_index=True, width="stretch",
            column_config={
                "grade": "Grade",
                "applicants": st.column_config.NumberColumn("N", format="%d"),
                "observed_rate": st.column_config.NumberColumn("Observed", format="percent"),
                "predicted_pd": st.column_config.NumberColumn("Predicted", format="percent"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.3f"),
                "result": "Result",
            },
        )
        st.caption(
            f"Hosmer–Lemeshow (10 groups): χ² = {hl_stat:.1f}, p = {hl_p:.3f}. "
            "With 61k applicants HL rejects small deviations easily, so the per-grade "
            "binomial test drives the rating."
        )

    st.subheader("Stability")
    left, right = st.columns([3, 2])
    psi = data["psi"][data["psi"]["model"] == val_model]
    with left:
        st.markdown(f"**Score PSI = {psi['total_psi'].iloc[0]:.4f}** · development vs holdout")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(psi) + 1)), y=psi["expected_pct"] * 100,
            name="Development (train)", marker_color=MUTED,
        ))
        fig.add_trace(go.Bar(
            x=list(range(1, len(psi) + 1)), y=psi["actual_pct"] * 100,
            name="Holdout", marker_color=MODELS[val_model][1],
        ))
        fig.update_layout(
            barmode="group", xaxis_title="Score bin (development deciles)",
            yaxis_title="Share of applicants (%)",
        )
        fig.update_xaxes(dtick=1)
        show(chart_style(fig, 340), "psi_chart")
    with right:
        st.markdown("**Characteristic stability (CSI), top features**")
        st.dataframe(
            data["csi"].head(8).assign(feature=lambda d: d["feature"].map(label)),
            hide_index=True, width="stretch",
            column_config={
                "feature": "Feature",
                "csi": st.column_config.NumberColumn("CSI", format="%.5f"),
            },
        )
        st.caption(
            "PSI < 0.10 is stable. A random split makes this near zero by construction; "
            "the same code would be run monthly on new applications in production."
        )

    st.subheader("Data quality")
    st.dataframe(
        data["dq"].assign(passed=lambda d: d["passed"].map({True: "✅ Pass", False: "❌ Fail"})),
        hide_index=True, width="stretch",
        column_config={
            "check": "Assertion", "expected": "Expected",
            "observed": "Observed", "passed": "Result",
        },
    )

    st.subheader("What drives LightGBM?")
    left, right = st.columns(2)
    shap_global = data["shap_global"].head(15).iloc[::-1]
    with left:
        fig = go.Figure(go.Bar(
            x=shap_global["mean_abs_shap"], y=shap_global["feature"].map(label),
            orientation="h", marker_color=TEAL,
        ))
        fig.update_layout(xaxis_title="Mean |SHAP| (log-odds)", title="TreeSHAP importance")
        show(chart_style(fig, 490), "validation_shap")
    with right:
        top_features = data["importance"].nlargest(15, "importance").sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top_features["importance"], y=top_features["feature"].map(label),
            orientation="h", marker_color=BLUE,
        ))
        fig.update_layout(xaxis_title="Total split gain", title="Gain importance")
        show(chart_style(fig, 490), "validation_importance")
    st.caption(
        "SHAP measures how much each feature moves individual predictions; gain measures "
        "how useful a feature was for splitting during training. Neither establishes causality."
    )

    st.subheader("Bootstrap validation results")
    st.dataframe(
        data["bootstrap"], hide_index=True, width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in data["bootstrap"].columns if c not in ("Metric", "Evidence")
        },
    )

    report_file = OUT / "model_validation_report.md"
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Export model comparison (CSV)",
            data["comparison"].to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
    if report_file.exists():
        report_text = report_file.read_text(encoding="utf-8")
        with d2:
            st.download_button(
                "Download validation summary (Markdown)",
                data=report_text,
                file_name="model_validation_report.md",
                mime="text/markdown",
                key="download_validation_report",
            )
        with st.expander("Read the full validation report"):
            st.markdown(report_text)

st.divider()
st.caption(
    "Research and portfolio demonstration only · "
    "TARGET is a payment-difficulty indicator, not a regulatory default definition · "
    "Not an approved credit decisioning system"
)
