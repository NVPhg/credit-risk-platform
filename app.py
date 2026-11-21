import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

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

# Score scaling: 600 points at good:bad odds of 20:1, 20 points to double the odds
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 20
FACTOR = PDO / np.log(2)
GRADES = [  # (grade, lower bound inclusive)
    ("A", 630), ("B", 610), ("C", 590), ("D", 570), ("E", 550), ("F", -np.inf),
]
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


def pd_to_score(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    score = BASE_SCORE + FACTOR * np.log(((1 - p) / p) / BASE_ODDS)
    return float(score) if score.ndim == 0 else score


def score_to_grade(score):
    score = np.atleast_1d(score)
    out = np.full(score.shape, "F", dtype=object)
    for grade, lower in reversed(GRADES):
        out[score >= lower] = grade
    return out


def grade_range(grade):
    bounds = dict(GRADES)
    names = [g for g, _ in GRADES]
    i = names.index(grade)
    lower = bounds[grade]
    upper = bounds[names[i - 1]] if i > 0 else None
    if upper is None:
        return f"≥ {lower:.0f}"
    if np.isinf(lower):
        return f"< {upper:.0f}"
    return f"{lower:.0f} – {upper - 1:.0f}"


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
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


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
    overview, insights_tab, models_tab, strategy_tab, applicant_tab, validation_tab
) = st.tabs([
    "Overview",
    "Portfolio Insights",
    "Model Comparison",
    "Credit Strategy",
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
    st.subheader("Probability calibration")
    show(calibration_chart(), "validation_calibration")
    st.caption(
        "Both models track observed rates closely in the bulk of the distribution; "
        "Brier score improves from "
        f"{logistic['brier_score']:.4f} to {lightgbm['brier_score']:.4f}."
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

    st.subheader("Validation status")
    status = pd.DataFrame([
        ("Common holdout, verified identical applicant IDs", "Done"),
        ("ROC-AUC, PR-AUC, KS, Brier, calibration", "Done"),
        ("Paired stratified bootstrap confidence intervals", "Done"),
        ("Train vs holdout generalisation check", "Done — LightGBM gap flagged"),
        ("Explainability: TreeSHAP global + reason codes", "Done"),
        ("Fairness review by age band (exploratory)", "Done"),
        ("Point-in-time / leakage audit of installment features", "Outstanding"),
        ("Out-of-time validation", "Outstanding — no application dates in data"),
        ("Hyper-parameter tuning on a separate validation fold", "Outstanding"),
        ("Independent validation & stability monitoring (PSI)", "Outstanding"),
    ], columns=["Check", "Status"])
    st.dataframe(status, hide_index=True, width="stretch")

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
                "Download validation report (Markdown)",
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
