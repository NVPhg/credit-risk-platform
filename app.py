import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
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
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="auto",
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
    DASH / "lightgbm_v11_model.txt",
    OUT / "lightgbm_v11_metrics.json",
    OUT / "scorecard_metrics.json",
    OUT / "bootstrap_v11_vs_logistic.csv",
    OUT / "bootstrap_v11_vs_v10.csv",
    OUT / "tuning_trials.csv",
    OUT / "scorecard" / "points_table.csv",
    OUT / "scorecard" / "iv_table.csv",
    DASH / "feature_importance_challenger.csv",
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
NAVY = "#0E1929"          # brand ink (headings, dark accents)
BG = "#F4F6FA"            # page background
PANEL = "#FFFFFF"         # cards and charts
BORDER = "#DDE3EC"
GRIDLINE = "#E8EDF3"
TEAL = "#0F9F8A"
BLUE = "#2F6FD6"
ORANGE = "#E08A2E"
RED = "#D64550"
MUTED = "#5F6B7D"
TEXT = "#1B2433"

PURPLE = "#7E5BD6"
CHALLENGER = "LightGBM v1.1"
CH_COL = "lightgbm_v11_score"
MODELS = {
    "LightGBM v1.1": ("lightgbm_v11_score", TEAL),
    "LightGBM v1.0": ("lightgbm_score", PURPLE),
    "WoE Scorecard": ("scorecard_score", ORANGE),
    "Logistic Regression": ("logistic_score", BLUE),
}
GAP_NAMES = {"LightGBM v1.0": "LightGBM"}

GRADE_COLORS = {
    "A": "#1F9E5A", "B": TEAL, "C": BLUE, "D": ORANGE, "E": "#E5683A", "F": RED,
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
        "lightgbm_v10": lightgbm,
        "lightgbm": json.loads((OUT / "lightgbm_v11_metrics.json").read_text()),
        "scorecard_metrics": json.loads((OUT / "scorecard_metrics.json").read_text()),
        "pred": pred,
        "importance": pd.read_csv(DASH / "feature_importance_challenger.csv"),
        "bootstrap_v10": pd.read_csv(OUT / "bootstrap_validation.csv"),
        "bootstrap": pd.read_csv(OUT / "bootstrap_v11_vs_logistic.csv"),
        "bootstrap_v11_v10": pd.read_csv(OUT / "bootstrap_v11_vs_v10.csv"),
        "trials": pd.read_csv(OUT / "tuning_trials.csv"),
        "points": pd.read_csv(OUT / "scorecard" / "points_table.csv"),
        "iv": pd.read_csv(OUT / "scorecard" / "iv_table.csv"),
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
    return lgb.Booster(model_file=str(DASH / "lightgbm_v11_model.txt"))


data = load_data()
logistic, lightgbm = data["logistic"], data["lightgbm"]
pred = data["pred"]
booster = load_booster()
FEATURES = booster.feature_name()
PORTFOLIO_RATE = pred["target"].mean()


# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp {{ background: {BG}; color: {TEXT}; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"], .stApp [data-baseweb="tab"], .stApp .stCaption,
.stApp [data-testid="stMarkdownContainer"] {{ font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}
.stApp h3 {{ font-weight: 700; letter-spacing: -0.3px; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDER}; }}
.card, .kpi, .insight, .decision, .rating, .reason, [data-testid="stMetric"], .chip, .badge {{
    box-shadow: 0 1px 2px rgba(16, 24, 40, .05);
}}
.stApp h1, .stApp h2, .stApp h3, .hero-title {{ color: {NAVY}; }}
.brand {{ display: flex; align-items: center; gap: 12px; }}
.brand svg {{ flex: none; }}
.hero {{ display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); gap: 28px; align-items: stretch; margin-bottom: 26px; }}
@media (max-width: 1000px) {{ .hero {{ grid-template-columns: 1fr; }} }}
.card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 20px 22px; }}
.mc-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }}
.mc-title {{ color: {MUTED}; font-size: 12px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; }}
.status {{ font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 999px; border: 1px solid; }}
.mc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; margin: 14px 0 12px; }}
.mc-grid .k {{ color: {MUTED}; font-size: 12px; }}
.mc-grid .v {{ color: {TEXT}; font-size: 14px; font-weight: 600; }}
.dots {{ display: flex; gap: 6px; margin: 6px 0 6px; flex-wrap: wrap; }}
.dot {{ width: 22px; height: 8px; border-radius: 4px; }}
.mc-foot {{ color: {MUTED}; font-size: 12px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 4px 0 8px; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: var(--accent, {TEAL}); }}
.kpi .l {{ color: {MUTED}; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; }}
.kpi .v {{ color: {NAVY}; font-size: clamp(22px, 2.1vw, 30px); font-weight: 800; letter-spacing: -0.8px; margin: 4px 0 2px; font-variant-numeric: tabular-nums; }}
.kpi .s {{ color: {MUTED}; font-size: 13px; }}
.kpi .s b {{ color: var(--accent, {TEAL}); font-weight: 700; }}
.section {{ margin: 26px 0 12px; }}
.section .eyebrow {{ color: {TEAL}; font-size: 11.5px; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; }}
.section h3 {{ margin: 4px 0 4px; font-size: 24px; padding: 0; }}
.section p {{ color: {MUTED}; margin: 0; font-size: 14.5px; max-width: 980px; }}
.decision {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 22px; margin: 6px 0 16px; border-left: 5px solid var(--dc); }}
.decision .top {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
.decision .lbl {{ color: {MUTED}; font-size: 12px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; }}
.decision .val {{ color: var(--dc); font-size: 26px; font-weight: 800; }}
.decision .why {{ color: {MUTED}; font-size: 13.5px; }}
.decision ol {{ margin: 10px 0 0 0; padding-left: 20px; }}
.decision li {{ margin: 3px 0; }}
.side-links a {{ color: {TEAL} !important; text-decoration: none; }}
.side-note {{ color: {MUTED}; font-size: 12.5px; line-height: 1.5; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
.badge {{ font-size: 11.5px; padding: 3px 9px; border-radius: 6px; background: {PANEL}; border: 1px solid {BORDER}; color: {TEXT}; }}
.block-container {{ max-width: 1450px; padding-top: 3.5rem; }}
[data-testid="stMetric"] {{
    background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 18px 20px;
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; }}
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
    padding: 14px 22px; border-radius: 14px; color: #FFFFFF;
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


LOGO_SVG = f"""<svg width="{{s}}" height="{{s}}" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="{{gid}}" x1="0" y1="1" x2="1" y2="0">
<stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{TEAL}"/></linearGradient></defs>
<rect x="1.5" y="1.5" width="37" height="37" rx="10" fill="{PANEL}" stroke="{BORDER}"/>
<rect x="9" y="22" width="5" height="9" rx="1.5" fill="url(#{{gid}})" opacity=".55"/>
<rect x="17.5" y="16" width="5" height="15" rx="1.5" fill="url(#{{gid}})" opacity=".8"/>
<rect x="26" y="10" width="5" height="21" rx="1.5" fill="url(#{{gid}})"/>
<path d="M8 17 L18 11 L24 14 L32 7" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def logo(size=40, gid="logo"):
    return LOGO_SVG.replace("{s}", str(size)).replace("{gid}", f"grad-{gid}")


def section(eyebrow, title, description=None):
    desc = f"<p>{description}</p>" if description else ""
    st.markdown(
        f'<div class="section"><div class="eyebrow">{eyebrow}</div><h3>{title}</h3>{desc}</div>',
        unsafe_allow_html=True,
    )


def kpi_cards(items):
    """items: list of (label, value, sub_html, accent_color)."""
    cards = "".join(
        f'<div class="kpi" style="--accent:{accent}"><div class="l">{lbl}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>'
        for lbl, value, sub, accent in items
    )
    st.markdown(f'<div class="kpis">{cards}</div>', unsafe_allow_html=True)


ADVERSE_REASONS = {
    "employment_years": "Short length of current employment",
    "employment_sentinel_flag": "Not currently in employment",
    "refused_count": "Number of previously refused applications",
    "approval_rate": "Low approval rate on previous applications",
    "approved_count": "Limited history of approved applications",
    "previous_application_count": "Number of previous applications",
    "credit_amount": "Size of the requested credit amount",
    "annuity_amount": "Size of the monthly instalment",
    "credit_to_income": "Credit amount is high relative to income",
    "annuity_to_income": "Instalment is high relative to income",
    "annual_income": "Level of declared income",
    "due_date_shortfall_rate": "History of paying less than the amount due by the due date",
    "due_date_shortfall_count": "Number of instalments short at the due date",
    "observed_underpaid_count": "Underpaid instalments in repayment history",
    "multiple_payment_count": "Instalments repaid in several partial payments",
    "installment_count": "Limited length of repayment history",
    "valid_installment_count": "Limited length of repayment history",
    "incomplete_installment_count": "Incomplete instalment records",
    "has_installment_history": "No repayment history with the lender",
    "active_loan_count": "Number of active loans at other lenders",
    "bureau_loan_count": "Number of loans reported to the credit bureau",
    "avg_bureau_credit": "Average credit amount held at other lenders",
    "education_type": "Education level",
    "income_type": "Source of income",
    "contract_type": "Type of credit product requested",
}


FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
pio.templates["cri"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=12, color=TEXT),
        colorway=[TEAL, BLUE, ORANGE, RED, PURPLE, "#4CAF50"],
        title=dict(font=dict(family=FONT, size=14, color=TEXT), x=0, xanchor="left"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
        xaxis=dict(title=dict(font=dict(color=MUTED, size=12)), tickfont=dict(color=MUTED),
                   linecolor=BORDER, ticks=""),
        yaxis=dict(title=dict(font=dict(color=MUTED, size=12)), tickfont=dict(color=MUTED),
                   linecolor=BORDER, ticks=""),
        legend=dict(font=dict(color=TEXT, size=12), bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
)
pio.templates.default = "plotly_white+cri"


def chart_style(fig, height=360):
    fig.update_layout(
        template="plotly_white+cri",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        height=height,
        margin=dict(l=35, r=20, t=35, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if fig.layout.title.text:
        # Title on its own line above the legend
        fig.update_layout(
            margin=dict(t=78),
            title=dict(x=0.01, xanchor="left", y=0.97, yanchor="top", yref="container"),
        )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False)
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


@st.cache_data
def model_metrics_table():
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    gap = data["gap"].set_index("Model")
    rows = []
    for name, (column, _) in MODELS.items():
        y, p = pred["target"], pred[column]
        _, ks, _, _ = rm.ks_curve(y, p)
        auc = roc_auc_score(y, p)
        rows.append({
            "Model": name, "ROC-AUC": auc, "Gini": 2 * auc - 1, "KS": ks,
            "PR-AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
            "Train–holdout gap": float(gap.loc[GAP_NAMES.get(name, name), "AUC Gap"]),
        })
    return pd.DataFrame(rows)


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
    fig.update_layout(xaxis_title="Challenger − Logistic (95% bootstrap CI)")
    return chart_style(fig, 300)


def gap_chart(gap):
    names = {v: k for k, v in GAP_NAMES.items()}
    order = [m for m in reversed(list(MODELS))]
    gap = gap.assign(Model=gap["Model"].map(lambda m: names.get(m, m)))
    gap = gap.set_index("Model").reindex([m for m in order if m in set(gap["Model"])]).reset_index()
    fig = go.Figure()
    for col, color in [("Train ROC-AUC", MUTED), ("Holdout ROC-AUC", TEAL)]:
        fig.add_trace(go.Bar(
            x=gap["Model"], y=gap[col], name=col, marker_color=color,
            text=gap[col].map(lambda v: f"{v:.3f}"), textposition="outside",
        ))
    for i, r in gap.iterrows():
        fig.add_annotation(
            x=r["Model"], y=max(r["Train ROC-AUC"], r["Holdout ROC-AUC"]) + 0.03,
            text=f"gap {r['AUC Gap']:+.3f}", showarrow=False,
            font=dict(color=RED if r["AUC Gap"] > 0.05 else ORANGE if r["AUC Gap"] > 0.02 else TEAL),
        )
    fig.update_layout(barmode="group", yaxis_title="ROC-AUC")
    fig.update_yaxes(range=[0.5, 0.84])
    return chart_style(fig, 340)


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
        x=[0, bad_rate * 100, 100], y=[0, 100, 100], name="Perfect model", mode="lines",
        line=dict(color=MUTED, dash="dot", width=1.5),
    ))
    for name, (column, color) in reversed(MODELS.items()):
        curve, gini = rm.lorenz_curve(pred["target"], pred[column])
        fig.add_trace(go.Scatter(
            x=curve["population"] * 100, y=curve["cum_bad"] * 100,
            name=f"{name} · Gini {gini:.3f}", line=dict(color=color, width=3),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], name="Random", mode="lines", line=dict(color=MUTED, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Applicants declined, riskiest first (%)", yaxis_title="Bads captured (%)",
    )
    chart_style(fig, 380)
    fig.update_layout(legend=dict(orientation="v", x=0.98, y=0.04, xanchor="right", yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1))
    return fig


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
    boots = {"LightGBM v1.1": data["bootstrap"], "LightGBM v1.0": data["bootstrap_v10"]}
    return rm.validation_scorecard(
        pred, boots.get(model), data["gap"], data["psi"], data["dq"], model,
        col=MODELS[model][0], gap_name=GAP_NAMES.get(model, model),
    )


# ---------------------------------------------------------------- header
best = data["bootstrap"].set_index("Metric")
card_table, card_rating = scorecard(CHALLENGER)
card_color = (
    RAG_COLORS["Red"] if card_rating.startswith("Not")
    else RAG_COLORS["Amber"] if "conditions" in card_rating else RAG_COLORS["Green"]
)
card_counts = card_table["status"].value_counts()
dots = "".join(
    f'<span class="dot" title="{r.test}: {r.status}" style="background:{RAG_COLORS[r.status]}"></span>'
    for r in card_table.itertuples()
)
_, card_gini = rm.lorenz_curve(pred["target"], pred[CH_COL])

with st.sidebar:
    st.markdown(
        f'<div class="brand">{logo(36, "side")}<div><b>Credit Risk Intelligence</b><br>'
        f'<span class="side-note">Application PD model · v1.0</span></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="side-note">An end-to-end credit-risk project: PostgreSQL feature mart → '
        "Logistic Regression benchmark vs LightGBM challenger → independent validation → "
        "credit-policy decisioning.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-links" style="margin-top:14px">'
        '<a href="https://github.com/NVPhg/credit-risk-platform" target="_blank">↗ Source code on GitHub</a></div>',
        unsafe_allow_html=True,
    )
    side_pdf = ROOT / "docs" / "Model_Validation_Report.pdf"
    if side_pdf.exists():
        st.download_button(
            "Validation report (PDF)", side_pdf.read_bytes(),
            file_name="Model_Validation_Report.pdf", mime="application/pdf",
            icon=":material/description:", width="stretch", key="side_pdf",
        )
    st.markdown("**Built with**")
    st.markdown(
        '<div class="badges">' + "".join(
            f'<span class="badge">{t}</span>' for t in [
                "PostgreSQL", "SQL", "Python", "pandas", "scikit-learn", "LightGBM",
                "TreeSHAP", "SciPy", "Plotly", "Streamlit",
            ]
        ) + "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="side-note">Data: Home Credit Default Risk (Kaggle), 307,511 applicants. '
        "TARGET is a payment-difficulty flag, not a regulatory default definition. "
        "Portfolio project, not a production decisioning system.</div>",
        unsafe_allow_html=True,
    )

st.markdown(f"""
<div class="hero">
  <div>
    <div class="brand">{logo(44, "hero")}
      <div class="hero-label">RISK ANALYTICS · MODEL VALIDATION · CREDIT POLICY</div>
    </div>
    <div class="hero-title">Credit Risk Intelligence</div>
    <div class="hero-subtitle">
      End-to-end probability-of-default modelling on 307,511 consumer-loan applicants,
      from a PostgreSQL feature mart to an independently validated challenger model and a
      credit-policy decision framework.
    </div>
    <div class="chips">
      <span class="chip">PostgreSQL feature mart · 58M+ rows</span>
      <span class="chip">Logistic Regression vs LightGBM</span>
      <span class="chip">Bootstrap · KS · Gini · PSI</span>
      <span class="chip">TreeSHAP reason codes</span>
      <span class="chip">Scorecard & cut-off strategy</span>
      <span class="chip">Policy simulator</span>
    </div>
  </div>
  <div class="card">
    <div class="mc-head">
      <div class="mc-title">Model card</div>
      <span class="status" style="color:{card_color};border-color:{card_color}66;background:{card_color}1A">{card_rating}</span>
    </div>
    <div class="mc-grid">
      <div><div class="k">Model</div><div class="v">LightGBM challenger v1.1</div></div>
      <div><div class="k">Benchmark</div><div class="v">Logistic Regression</div></div>
      <div><div class="k">Use</div><div class="v">Application PD · grading · cut-off</div></div>
      <div><div class="k">Validation sample</div><div class="v">{len(pred):,} applicants</div></div>
      <div><div class="k">Gini / KS</div><div class="v">{card_gini:.3f} / {lightgbm['ks']:.3f}</div></div>
      <div><div class="k">Bad rate (holdout)</div><div class="v">{PORTFOLIO_RATE:.2%}</div></div>
    </div>
    <div class="mc-title" style="font-size:11px">Validation tests</div>
    <div class="dots">{dots}</div>
    <div class="mc-foot">{card_counts.get('Green', 0)} green · {card_counts.get('Amber', 0)} amber ·
      {card_counts.get('Red', 0)} red · {card_counts.get('Not assessed', 0)} not assessed.
      Details in the Model Validation tab.</div>
  </div>
</div>
""", unsafe_allow_html=True)

(
    overview, insights_tab, models_tab, strategy_tab, capital_tab, policy_tab, applicant_tab,
    validation_tab,
) = st.tabs([
    ":material/space_dashboard: Overview",
    ":material/pie_chart: Portfolio Insights",
    ":material/compare_arrows: Model Comparison",
    ":material/tune: Credit Strategy",
    ":material/account_balance: Capital & Stress",
    ":material/rule: Policy Simulator",
    ":material/person_search: Applicant Explorer",
    ":material/verified: Model Validation",
])

# ================================================================= OVERVIEW
with overview:
    deciles = risk_deciles(CH_COL)
    top = deciles.iloc[-1]
    lift = top["target_rate"] / PORTFOLIO_RATE
    capture = top["cases"] / pred["target"].sum()

    lgb_curve = strategy_curve(CH_COL, 0.10, 0.50)
    log_curve = strategy_curve("logistic_score", 0.10, 0.50)
    lgb80, log80 = at_approval(lgb_curve, 0.80), at_approval(log_curve, 0.80)
    avoided = (log80["bad_rate"] - lgb80["bad_rate"]) * 10_000
    auc_ci = best.loc["ROC-AUC"]
    gap = data["gap"].set_index("Model")

    kpi_cards([
        ("Applicants modelled", "307,511", f"{len(pred):,} in the validation holdout", BLUE),
        ("Portfolio bad rate", f"{PORTFOLIO_RATE:.2%}", "TARGET = 1 on the holdout", ORANGE),
        ("Challenger Gini (v1.1)", f"{card_gini:.3f}",
         f"<b>{card_gini - rm.lorenz_curve(pred['target'], pred['logistic_score'])[1]:+.3f}</b> vs benchmark", TEAL),
        ("Bad rate at 80% approval", f"{lgb80['bad_rate']:.2%}",
         f"<b>{(lgb80['bad_rate'] - log80['bad_rate']) * 100:+.2f} pp</b> vs benchmark", TEAL),
    ])

    section("Executive summary", "Key findings")
    i1, i2, i3 = st.columns(3)
    with i1:
        insight(
            "Challenger beats baseline — with evidence",
            f"LightGBM v1.1 improves ROC-AUC by <b>{auc_ci['Difference']:+.3f}</b> "
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
            "Overfitting found, then fixed",
            f"Validation flagged v1.0 for overfitting (train–holdout AUC gap "
            f"<b>{gap.loc['LightGBM', 'AUC Gap']:.3f}</b>). Optuna tuning on a separate fold, "
            f"early stopping and monotonic constraints cut it to "
            f"<b>{gap.loc['LightGBM v1.1', 'AUC Gap']:.3f}</b> in v1.1 while holdout AUC "
            f"moved from {gap.loc['LightGBM', 'Holdout ROC-AUC']:.3f} to "
            f"<b>{gap.loc['LightGBM v1.1', 'Holdout ROC-AUC']:.3f}</b>.",
        )

    section(
        "Performance", "Risk concentration & discrimination",
        "Applicants ranked by LightGBM v1.1 score: the riskiest decile carries more than "
        "a quarter of all payment-difficulty cases.",
    )
    left, right = st.columns(2)
    with left:
        show(decile_chart(deciles), "overview_deciles")
    with right:
        show(roc_chart(), "overview_roc")

    kpi_cards([
        ("Top-decile lift", f"{lift:.2f}x", "Bad rate of riskiest 10% vs portfolio", TEAL),
        ("Bads in top decile", f"{capture:.1%}", "Share of all cases captured", TEAL),
        ("KS statistic", f"{lightgbm['ks']:.3f}",
         f"<b>{lightgbm['ks'] - logistic['ks']:+.3f}</b> vs benchmark", BLUE),
        ("ROC-AUC", f"{lightgbm['roc_auc']:.3f}",
         f"<b>{lightgbm['roc_auc'] - logistic['roc_auc']:+.3f}</b> vs benchmark", BLUE),
    ])

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
    section(
        "Portfolio insights", "Where does risk sit in the portfolio?",
        "Descriptive TARGET rates across all 307,511 applicants, computed from the PostgreSQL "
        "feature mart. Segments with fewer than 100 applicants are hidden.",
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
    section(
        "Benchmarking", "Four models, one holdout",
        "The logistic benchmark, a traditional WoE scorecard, the original LightGBM v1.0 and "
        "the remediated LightGBM v1.1 are all scored on the identical 61,503-applicant holdout.",
    )
    st.dataframe(
        model_metrics_table(),
        width="stretch",
        hide_index=True,
        column_config={
            "Model": "Model",
            **{c: st.column_config.NumberColumn(format="%.4f")
               for c in ["ROC-AUC", "Gini", "KS", "PR-AUC", "Brier"]},
            "Train–holdout gap": st.column_config.NumberColumn(format="%+.3f"),
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
            "LightGBM v1.1 vs the logistic benchmark. Teal = interval excludes zero in the challenger's favour."
        )
    with right:
        st.markdown("**Does it generalise? Train vs holdout**")
        show(gap_chart(data["gap"]), "gap_chart")
        st.caption(
            "LightGBM v1.0 fitted the training data far better than the holdout. The "
            "remediated v1.1 closes most of that gap; the WoE scorecard and logistic "
            "benchmark show almost none."
        )

    section(
        "Diagnostics", "Credit-risk diagnostics",
        "The standard views a credit-risk or model-validation team reviews before accepting a "
        "scorecard: separation (KS), concentration (Lorenz / Gini), score distributions and the gains table.",
    )
    diag_model = st.segmented_control(
        "Model", list(MODELS), default=CHALLENGER, key="diag_model"
    ) or CHALLENGER
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


    # ---- traditional WoE scorecard
    section(
        "Traditional scorecard", "WoE / IV application scorecard",
        "The transparent benchmark most banks still run: optimal monotonic binning, "
        "Weight-of-Evidence logistic regression and a points table on the same 600 / 20:1 / 20 PDO scale.",
    )
    scm = data["scorecard_metrics"]
    kpi_cards([
        ("Scorecard Gini", f"{2 * scm['roc_auc'] - 1:.3f}",
         f"vs <b>{2 * data['lightgbm']['roc_auc'] - 1:.3f}</b> for LightGBM v1.1", ORANGE),
        ("KS", f"{scm['ks']:.3f}", f"vs <b>{data['lightgbm']['ks']:.3f}</b> for v1.1", ORANGE),
        ("Variables kept", f"{scm['n_variables']}", "IV ≥ 0.02 after optimal binning", BLUE),
        ("Train–holdout gap", f"{scm['auc_gap']:+.3f}", "Fully transparent, no overfitting", TEAL),
    ])
    left, right = st.columns([2, 3])
    iv = data["iv"].copy()
    iv = iv[iv["iv"] > 0].sort_values("iv").tail(15)
    with left:
        def iv_color(v):
            return RED if v < 0.02 else MUTED if v < 0.1 else BLUE if v < 0.3 else TEAL
        fig = go.Figure(go.Bar(
            x=iv["iv"], y=iv["name"].map(label), orientation="h",
            marker_color=[iv_color(v) for v in iv["iv"]],
            text=iv["iv"].map(lambda v: f"{v:.3f}"), textposition="outside", cliponaxis=False,
        ))
        fig.add_vline(x=0.02, line_color=RED, line_dash="dot")
        fig.add_vline(x=0.1, line_color=MUTED, line_dash="dot")
        fig.update_layout(title=dict(text="Information Value by characteristic", font=dict(size=14)),
                          xaxis_title="IV (0.02 weak · 0.1 medium · 0.3 strong)")
        fig.update_yaxes(automargin=True)
        show(chart_style(fig, 470).update_layout(margin=dict(l=35, r=50, t=45, b=45)), "iv_chart")
    with right:
        points = data["points"].copy()
        variables = [v for v in data["iv"].sort_values("iv", ascending=False)["name"] if v in set(points["Variable"])]
        var = st.selectbox("Characteristic", variables, format_func=label, key="sc_var")
        pv = points[points["Variable"] == var].copy()
        pv = pv[pv["Count (%)"] > 0]
        fig = go.Figure(go.Bar(
            x=pv["Bin"].astype(str), y=pv["Points"],
            marker_color=[TEAL if p >= pv["Points"].median() else ORANGE for p in pv["Points"]],
            text=pv["Points"].map(lambda v: f"{v:.0f}"), textposition="outside",
            customdata=np.stack([pv["Count (%)"] * 100, pv["Event rate"] * 100], axis=1),
            hovertemplate="%{x}<br>Points %{y:.0f}<br>Share %{customdata[0]:.1f}%<br>"
                          "Bad rate %{customdata[1]:.1f}%<extra></extra>",
        ))
        fig.update_layout(title=dict(text=f"Points by attribute: {label(var)}", font=dict(size=14)),
                          xaxis_title="Attribute (bin)", yaxis_title="Points")
        show(chart_style(fig, 300), "points_chart")
        st.dataframe(
            pv[["Bin", "Count (%)", "Event rate", "WoE", "Points"]], hide_index=True, width="stretch",
            column_config={
                "Count (%)": st.column_config.NumberColumn("Share", format="percent"),
                "Event rate": st.column_config.NumberColumn("Bad rate", format="percent"),
                "WoE": st.column_config.NumberColumn(format="%.3f"),
                "Points": st.column_config.NumberColumn(format="%.0f"),
            },
        )

# ================================================================= STRATEGY
with strategy_tab:
    section("Decisioning", "From probability to decision")
    st.caption(
        f"PDs are converted to a points-based score: {BASE_SCORE} points at "
        f"good:bad odds of {BASE_ODDS}:1, and every {PDO} points doubles the odds "
        "(the standard scorecard scaling used in retail credit)."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
    with ctrl1:
        model_choice = st.segmented_control(
            "Scoring model", list(MODELS), default=CHALLENGER, key="strategy_model"
        ) or CHALLENGER
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
    section("Cut-off strategy", "Approval-rate trade-off")
    curve = strategy_curve(column, margin, lgd)
    other_name = "Logistic Regression" if model_choice != "Logistic Regression" else CHALLENGER
    other_curve = strategy_curve(MODELS[other_name][0], margin, lgd)
    best_idx = int(curve["profit"].idxmax())
    best_row = curve.iloc[best_idx]

    approval = st.slider(
        "Target approval rate", 0.50, 1.00, 0.80, 0.01, format="%.2f", key="approval"
    )
    row = at_approval(curve, approval)
    other_row = at_approval(other_curve, approval)
    approve_all = curve.iloc[-1]

    bad_diff = (row["bad_rate"] - other_row["bad_rate"]) * 100
    profit_diff = row["profit"] - approve_all["profit"]
    kpi_cards([
        ("Score cut-off", f"{pd_to_score(row['cutoff_pd']):.0f}",
         "Approve at or above this score", BLUE),
        ("Bad rate of approved", f"{row['bad_rate']:.2%}",
         f"<b>{bad_diff:+.2f} pp</b> vs {other_name}", TEAL if bad_diff <= 0 else RED),
        ("Problem loans declined", f"{1 - row['bad_rate'] * approval / PORTFOLIO_RATE:.1%}",
         "Share of all TARGET = 1 rejected", ORANGE),
        ("Profit vs approve-all", fmt_money(profit_diff),
         "Illustrative economics", TEAL if profit_diff >= 0 else RED),
    ])

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
            fill="tozeroy", fillcolor="rgba(15,159,138,0.12)",
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
    section("Fair lending", "Fairness review by age band")
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

# ================================================================= CAPITAL
with capital_tab:
    section(
        "Basel IRB · pricing", "Capital, expected loss and risk-based pricing",
        "Each grade's expected loss (PD × LGD × EAD) and Basel IRB capital for retail "
        "exposures, turned into the minimum interest rate the grade must earn.",
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        cap_model = st.selectbox("Model", list(MODELS), index=0, key="cap_model")
    with c2:
        cap_lgd = st.slider("LGD (%)", 20, 90, 45, 5, key="cap_lgd") / 100
    with c3:
        cap_funding = st.slider("Funding cost (%)", 2.0, 12.0, 6.0, 0.5, key="cap_funding") / 100
    with c4:
        cap_hurdle = st.slider("Hurdle ROE (%)", 5, 30, 15, key="cap_hurdle") / 100
    with c5:
        cap_cap = st.slider("Maximum annual rate (%)", 15, 60, 35, key="cap_rate_cap") / 100
    cap_opex = 0.03

    cap_col = MODELS[cap_model][0]
    cap_pd = pred[cap_col].to_numpy()
    cap_grades = score_to_grade(pd_to_score(cap_pd))
    econ = rm.grade_economics(
        cap_pd, pred["credit_amount"], cap_grades,
        lgd=cap_lgd, funding=cap_funding, opex=cap_opex, hurdle=cap_hurdle,
    )
    total_ead = econ["ead"].sum()
    kpi_cards([
        ("Exposure (EAD)", fmt_money(total_ead), f"{len(pred):,} holdout loans", BLUE),
        ("Expected loss", fmt_money(econ["el"].sum()), f"<b>{econ['el'].sum() / total_ead:.2%}</b> of EAD", ORANGE),
        ("IRB capital (K)", fmt_money(econ["capital"].sum()),
         f"<b>{econ['capital'].sum() / total_ead:.2%}</b> of EAD", TEAL),
        ("RWA density", f"{econ['rwa'].sum() / total_ead:.0%}", "Risk-weighted assets / EAD", BLUE),
    ])

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=econ["grade"], y=[cap_funding * 100] * len(econ), name="Funding",
                             marker_color="#B8C2D0"))
        fig.add_trace(go.Bar(x=econ["grade"], y=[cap_opex * 100] * len(econ), name="Operating cost",
                             marker_color="#D5DCE6"))
        fig.add_trace(go.Bar(x=econ["grade"], y=econ["el_rate"] * 100, name="Expected loss",
                             marker_color=ORANGE))
        fig.add_trace(go.Bar(x=econ["grade"], y=econ["k_rate"] * cap_hurdle * 100,
                             name="Cost of capital", marker_color=TEAL))
        fig.add_hline(y=cap_cap * 100, line_color=RED, line_dash="dash",
                      annotation_text=f"Rate cap {cap_cap:.0%}", annotation_font_color=RED)
        for _, r in econ.iterrows():
            fig.add_annotation(x=r["grade"], y=r["min_rate"] * 100 + 1.5, text=f"{r['min_rate']:.1%}",
                               showarrow=False, font=dict(color=TEXT))
        fig.update_layout(barmode="stack", xaxis_title="Risk grade",
                          yaxis_title="Minimum annual rate (%)",
                          title=dict(text="Risk-based price build-up by grade", font=dict(size=14)))
        show(chart_style(fig, 400), "pricing_chart")
    with right:
        econ_view = econ[["grade", "applicants", "avg_pd", "el_rate", "k_rate", "min_rate"]].copy()
        econ_view["viable"] = np.where(econ_view["min_rate"] <= cap_cap, "✅ Lend", "❌ Above cap")
        st.dataframe(
            econ_view, hide_index=True, width="stretch",
            column_config={
                "grade": "Grade",
                "applicants": st.column_config.NumberColumn("N", format="%d"),
                "avg_pd": st.column_config.NumberColumn("PD", format="percent"),
                "el_rate": st.column_config.NumberColumn("EL", format="percent"),
                "k_rate": st.column_config.NumberColumn("IRB K", format="percent"),
                "min_rate": st.column_config.NumberColumn("Min. rate", format="percent"),
                "viable": "At cap",
            },
        )
        unviable = econ_view.loc[econ_view["min_rate"] > cap_cap, "grade"].tolist()
        insight(
            "Pricing verdict",
            (f"Grades <b>{', '.join(unviable)}</b> cannot be priced profitably under a "
             f"{cap_cap:.0%} cap: decline them or require collateral / a co-signer."
             if unviable else
             f"Every grade can be priced profitably under a {cap_cap:.0%} cap; "
             "grade F still needs the widest margin.")
            + f" Price spread from grade A to F: <b>{econ['min_rate'].iloc[0]:.1%} → "
              f"{econ['min_rate'].iloc[-1]:.1%}</b>.",
        )
    st.caption(
        "Basel IRB formula for other retail exposures: correlation between 3% and 16%, 99.9% "
        "confidence, no maturity adjustment, 5 bp PD floor. PD is treated as a 12-month PD and "
        "EAD as the credit amount. Operating cost fixed at 3%. Illustrative, not a regulatory return."
    )

    # ---- stress test
    section(
        "Stress testing", "What happens in a downturn?",
        "PDs are stressed by scaling the odds of default (rank order preserved) and LGD is "
        "raised; the score cut-off for 80% approval under the baseline is held fixed.",
    )
    st_cutoff = pd_to_score(np.quantile(cap_pd, 0.80))
    stress = rm.stress_test(cap_pd, pred["credit_amount"], lgd=cap_lgd, cutoff_score=st_cutoff)
    base_row = stress.iloc[0]
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        colors = [TEAL, ORANGE, RED]
        fig.add_trace(go.Bar(
            x=stress["scenario"], y=stress["expected_loss"], name="Expected loss",
            marker_color=colors, text=[fmt_money(v) for v in stress["expected_loss"]],
            textposition="outside", offsetgroup=0,
        ))
        fig.add_trace(go.Bar(
            x=stress["scenario"], y=stress["capital"], name="IRB capital",
            marker_color=["rgba(15,159,138,.4)", "rgba(224,138,46,.4)", "rgba(214,69,80,.4)"],
            text=[fmt_money(v) for v in stress["capital"]], textposition="outside", offsetgroup=1,
        ))
        fig.update_layout(barmode="group", yaxis_title="Amount",
                          title=dict(text="Expected loss and capital by scenario", font=dict(size=14)))
        show(chart_style(fig, 380), "stress_chart")
    with right:
        view = stress[["scenario", "odds_multiplier", "lgd", "avg_pd", "el_rate",
                       "approval_rate", "share_grade_ef"]]
        st.dataframe(
            view, hide_index=True, width="stretch",
            column_config={
                "scenario": "Scenario",
                "odds_multiplier": st.column_config.NumberColumn("PD odds ×", format="%.2f"),
                "lgd": st.column_config.NumberColumn("LGD", format="percent"),
                "avg_pd": st.column_config.NumberColumn("Avg PD", format="percent"),
                "el_rate": st.column_config.NumberColumn("EL / EAD", format="percent"),
                "approval_rate": st.column_config.NumberColumn("Approved", format="percent"),
                "share_grade_ef": st.column_config.NumberColumn("In E/F", format="percent"),
            },
        )
        severe = stress.iloc[-1]
        insight(
            "Stress verdict",
            f"In the severely adverse scenario, expected loss rises "
            f"<b>{severe['expected_loss'] / base_row['expected_loss']:.1f}x</b> and IRB capital "
            f"<b>{severe['capital'] / base_row['capital'] - 1:+.0%}</b>. With the cut-off unchanged, approval "
            f"falls from {base_row['approval_rate']:.0%} to <b>{severe['approval_rate']:.0%}</b> "
            f"as {severe['share_grade_ef']:.0%} of applicants migrate to grades E/F.",
        )

# ================================================================= POLICY
with policy_tab:
    section("Credit policy", "Policy simulator")
    st.caption(
        "Combine a score cut-off with knock-out rules, as a credit-policy team would, and see "
        "the effect on approval rate, bad rate and profit. Rules apply in order; each row of "
        "the waterfall shows only the applicants that rule declines on top of the earlier ones."
    )

    controls, results = st.columns([1, 2.2], gap="large")
    with controls:
        pol_model = st.segmented_control(
            "Scoring model", list(MODELS), default=CHALLENGER, key="pol_model"
        ) or CHALLENGER
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
        kpi_cards([
            ("Approval rate", f"{appr_rate:.1%}", f"{int(approved.sum()):,} applicants", BLUE),
            ("Bad rate approved", f"{bad_rate_appr:.2%}",
             f"<b>{(bad_rate_appr - PORTFOLIO_RATE) * 100:+.2f} pp</b> vs portfolio", TEAL),
            ("Bads declined", f"{bads_declined:.1%}", "of all TARGET = 1", ORANGE),
            ("Profit vs all", fmt_money(profit_delta), "vs approving everyone",
             TEAL if profit_delta >= 0 else RED),
        ])

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
    section("Champion vs challenger", "Swap-set analysis")
    st.caption(
        "At the same approval rate, which applicants does switching from the logistic benchmark "
        "to LightGBM v1.1 newly approve (swap-in) or newly decline (swap-out)? A good challenger swaps "
        "out riskier applicants than it swaps in."
    )
    swap_rate = st.slider("Approval rate for both models", 0.50, 0.95, 0.80, 0.01,
                          format="%.2f", key="swap_rate")
    k = int(round(swap_rate * n_all))
    lr_ok = np.zeros(n_all, dtype=bool)
    gb_ok = np.zeros(n_all, dtype=bool)
    lr_ok[np.argsort(pred["logistic_score"].to_numpy(), kind="stable")[:k]] = True
    gb_ok[np.argsort(pred[CH_COL].to_numpy(), kind="stable")[:k]] = True
    cells = {
        "Approved by both": lr_ok & gb_ok,
        "Swap-in (LightGBM v1.1 only)": ~lr_ok & gb_ok,
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
            y=["v1.1 approves", "v1.1 declines"],
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
            f"Switching to LightGBM v1.1 at {swap_rate:.0%} approval swaps in "
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
    section("Explainability", "Why did the model score this applicant this way?")
    st.caption(
        "TreeSHAP decomposes each LightGBM v1.1 prediction (in log-odds) into the "
        "contribution of every feature. A random sample of 5,000 holdout applicants is available."
    )
    shap_sample = data["shap_sample"]
    sample_pred = pred.set_index("applicant_id").loc[shap_sample["applicant_id"]]
    sample_pred = sample_pred.assign(score=pd_to_score(sample_pred[CH_COL]))

    tiers = {
        "Highest risk": sample_pred.sort_values(CH_COL, ascending=False),
        "Lowest risk": sample_pred.sort_values(CH_COL),
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
                f"PD {sample_pred.loc[i, CH_COL]:.1%}"
            ),
            key="applicant_id",
        )

    applicant = pred[pred["applicant_id"] == applicant_id].iloc[0]
    contrib = shap_sample.set_index("applicant_id").loc[applicant_id]
    base_value = float(contrib["base_value"])
    contrib = contrib[FEATURES].astype(float)
    pd_value = float(applicant[CH_COL])
    score_value = float(pd_to_score(pd_value))
    grade = score_to_grade(score_value)[0]
    decile = int(np.ceil((pred[CH_COL] < pd_value).mean() * 10)) or 1

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

    # ---- credit decision (illustrative policy: 80% approval cut-off, 10-point refer band)
    cutoff_score = pd_to_score(np.quantile(pred[CH_COL], 0.80))
    if score_value >= cutoff_score:
        decision, dcolor = "Approve", "#1F9E5A"
        why = f"Score {score_value:.0f} is at or above the cut-off of {cutoff_score:.0f}."
    elif score_value >= cutoff_score - 10:
        decision, dcolor = "Refer to underwriter", ORANGE
        why = (f"Score {score_value:.0f} is within 10 points below the cut-off of "
               f"{cutoff_score:.0f}: manual review.")
    else:
        decision, dcolor = "Decline", RED
        why = f"Score {score_value:.0f} is below the cut-off of {cutoff_score:.0f}."
    top_reasons = contrib[contrib > 0].sort_values(ascending=False).head(4)
    reasons_html = "".join(
        f"<li>{ADVERSE_REASONS.get(f, label(f))} "
        f"<span style='color:{MUTED}'>({label(f)} = {fmt_value(f, applicant[f])})</span></li>"
        for f in top_reasons.index
    )
    reasons_block = (
        f"<div class='lbl' style='margin-top:12px'>"
        f"{'Principal reasons' if decision != 'Approve' else 'Main risk factors to note'}</div>"
        f"<ol>{reasons_html}</ol>" if reasons_html else ""
    )
    st.markdown(
        f'<div class="decision" style="--dc:{dcolor}"><div class="top">'
        f'<span class="lbl">Credit decision</span><span class="val">{decision}</span>'
        f'<span class="why">{why}</span></div>{reasons_block}</div>',
        unsafe_allow_html=True,
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
    section("Sensitivity", "What-if simulator")
    st.caption(
        "Change the application and re-score it live with the saved LightGBM v1.1 model. "
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
    section("Model risk", "Validation scorecard")
    st.caption(
        "Each test is rated Green / Amber / Red against pre-defined thresholds, in the style "
        "of a bank model-risk review. Any Red blocks approval; Amber or unassessed tests become conditions of approval."
    )
    val_model = st.segmented_control(
        "Model under review", list(MODELS), default=CHALLENGER, key="val_model"
    ) or CHALLENGER
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

    # ---- remediation v1.0 -> v1.1
    section(
        "Remediation", "From v1.0 to v1.1: closing the findings",
        "Validation of v1.0 raised a Red overfitting finding. The model owner retrained it; "
        "the same tests were re-run on the untouched holdout.",
    )
    t10, r10 = scorecard("LightGBM v1.0")
    t11, r11 = scorecard("LightGBM v1.1")

    def rating_color(r):
        return RAG_COLORS["Red"] if r.startswith("Not") else RAG_COLORS["Amber"] if "conditions" in r else RAG_COLORS["Green"]

    rc1, rc2 = st.columns(2)
    for col_, name, rating_, table_ in [(rc1, "LightGBM v1.0", r10, t10), (rc2, "LightGBM v1.1", r11, t11)]:
        cnt = table_["status"].value_counts()
        col_.markdown(
            f'<div class="rating"><div><div class="meta">{name}</div>'
            f'<div class="big" style="color:{rating_color(rating_)}">{rating_}</div></div>'
            f'<div class="meta">{cnt.get("Green", 0)} Green · {cnt.get("Amber", 0)} Amber · '
            f'{cnt.get("Red", 0)} Red · {cnt.get("Not assessed", 0)} Not assessed</div></div>',
            unsafe_allow_html=True,
        )
    compare = t10[["test", "result", "status"]].merge(
        t11[["test", "result", "status"]], on="test", suffixes=(" v1.0", " v1.1"))
    rows_html = "".join(
        f"<tr><td class='t'>{r['test']}</td><td class='r'>{r['result v1.0']}</td>"
        f"<td>{rag_pill(r['status v1.0'])}</td><td class='r'>{r['result v1.1']}</td>"
        f"<td>{rag_pill(r['status v1.1'])}</td></tr>"
        for _, r in compare.iterrows()
    )
    st.markdown(
        "<div class='rag-wrap'><table class='rag'><thead><tr><th>Test</th><th>v1.0 result</th>"
        "<th>v1.0</th><th>v1.1 result</th><th>v1.1</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    gap_tbl = data["gap"].set_index("Model")
    v11 = data["lightgbm"]
    left, right = st.columns([3, 2])
    with left:
        trials = data["trials"].dropna(subset=["user_attrs_val_auc"])
        best_n = v11["tuning"]["best_trial"]
        fig = go.Figure()
        fig.add_vrect(x0=-0.01, x1=0.02, fillcolor=TEAL, opacity=0.07, line_width=0)
        fig.add_vrect(x0=0.02, x1=0.05, fillcolor=ORANGE, opacity=0.07, line_width=0)
        fig.add_trace(go.Scatter(
            x=trials["user_attrs_gap"], y=trials["user_attrs_val_auc"], mode="markers",
            marker=dict(size=10, color=trials["value"], colorscale=[[0, BORDER], [1, BLUE]],
                        line=dict(width=1, color=PANEL)),
            name="Optuna trial",
            customdata=np.stack([trials["number"], trials["user_attrs_best_iteration"]], axis=1),
            hovertemplate="Trial %{customdata[0]}<br>Validation AUC %{y:.4f}<br>"
                          "Gap %{x:.4f}<br>Trees %{customdata[1]}<extra></extra>",
        ))
        b = trials[trials["number"] == best_n]
        fig.add_trace(go.Scatter(
            x=b["user_attrs_gap"], y=b["user_attrs_val_auc"], mode="markers+text",
            marker=dict(size=18, color=TEAL, symbol="star", line=dict(width=1, color=PANEL)),
            text=["Selected"], textposition="top center", name="Selected (v1.1)",
        ))
        fig.update_layout(
            title=dict(text=f"Optuna search: {len(trials)} trials on a validation fold", font=dict(size=14)),
            xaxis_title="Fit − validation AUC gap (green < 0.02, amber < 0.05)",
            yaxis_title="Validation ROC-AUC",
        )
        show(chart_style(fig, 380), "optuna_chart")
    with right:
        b11 = data["bootstrap_v11_v10"].set_index("Metric").loc["ROC-AUC"]
        findings = pd.DataFrame([
            ("F-01", "Overfitting",
             f"Closed: gap {gap_tbl.loc['LightGBM', 'AUC Gap']:.3f} → {gap_tbl.loc['LightGBM v1.1', 'AUC Gap']:.3f}"
             if gap_tbl.loc["LightGBM v1.1", "AUC Gap"] <= 0.05 else "Open"),
            ("F-05", "Holdout reuse", "Closed: tuned on a validation fold; holdout untouched"),
            ("F-02", "No out-of-time test", "Open: condition of approval"),
            ("F-03", "Point-in-time audit", "Open: condition of approval"),
            ("F-04", "Age-band disparity",
             "Escalated: impact ratio " + t10.set_index("test").loc["Fairness (age impact ratio)", "result"].split()[0]
             + " → " + t11.set_index("test").loc["Fairness (age impact ratio)", "result"].split()[0]
             + ", blocks approval"
             if t11.set_index("test").loc["Fairness (age impact ratio)", "status"] == "Red"
             else "Open: monitor; justification documented"),
            ("F-06", "Label definition", "Accepted limitation"),
        ], columns=["ID", "Finding", "Status after remediation"])
        st.dataframe(findings, hide_index=True, width="stretch")
        insight(
            "What changed",
            f"Monotonic constraints on {len(v11['monotone_constraints'])} risk drivers, "
            f"{v11['n_trees']} trees with {v11['params']['num_leaves']} leaves and at least "
            f"{v11['params']['min_child_samples']} applicants per leaf. Against v1.0, ROC-AUC changes by "
            f"<b>{b11['Difference']:+.4f}</b> (95% CI {b11['Difference CI Low']:+.4f} to "
            f"{b11['Difference CI High']:+.4f}): discrimination is kept while the overfitting is removed. "
            "The trade-off: the under-25 approval impact ratio slipped, so fairness is now the blocking finding.",
        )

    section("Accuracy of PDs", "Calibration")
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

    section("Population drift", "Stability")
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

    section("Inputs", "Data quality")
    st.dataframe(
        data["dq"].assign(passed=lambda d: d["passed"].map({True: "✅ Pass", False: "❌ Fail"})),
        hide_index=True, width="stretch",
        column_config={
            "check": "Assertion", "expected": "Expected",
            "observed": "Observed", "passed": "Result",
        },
    )

    section("Explainability", "What drives LightGBM v1.1?")
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

    section("Evidence", "Bootstrap validation results")
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
            model_metrics_table().to_csv(index=False),
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
