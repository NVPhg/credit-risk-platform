# Credit Risk Model Validation Report

## 1. Executive Summary

- Holdout sample: 61,503 applicants
- Observed TARGET=1 rate: 8.07%
- Baseline: Logistic Regression
- Challenger: LightGBM
- Validation: stratified random holdout (not out-of-time)

## 2. Bootstrap Performance Comparison

Paired stratified bootstrap with 500 resamples, 95% percentile confidence intervals.

| Metric | Logistic | LightGBM | Difference | 95% CI for difference |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.6827 | 0.7023 | +0.0196 | [+0.0154, +0.0233] |
| Average Precision | 0.1628 | 0.1846 | +0.0218 | [+0.0163, +0.0267] |
| KS | 0.2690 | 0.3009 | +0.0319 | [+0.0206, +0.0401] |
| Brier Score | 0.0716 | 0.0707 | -0.0009 | [-0.0011, -0.0007] |

Positive differences favor LightGBM except for Brier Score, where a negative difference favors LightGBM.

## 3. Risk Concentration

| Indicator | Logistic | LightGBM |
|---|---:|---:|
| Top 10% TARGET rate | 19.49% | 21.48% |
| Top 10% lift | 2.41x | 2.66x |
| Positive-case capture | 24.15% | 26.61% |

## 4. Validation Limitations

- The holdout is random, not chronological.
- The same holdout has been examined during model comparison; subsequent model selection can introduce selection bias.
- Historical feature availability at application time has not been independently established.
- In particular, installment-derived aggregates require a point-in-time leakage audit.
- Bootstrap intervals quantify sampling uncertainty under the chosen resampling scheme. They do not resolve selection bias, data leakage, or distribution shift.
- Calibration, fairness, stability, independent validation and external validation are not yet complete.
- Home Credit TARGET is a payment-difficulty indicator, not automatically a regulatory probability of default.

## 5. Validation Decision

LightGBM is a promising challenger based on observed holdout performance. Approval for production use is not recommended without further validation.

## 6. Reproducibility

- Bootstrap seed: 42
- Bootstrap replications: 500
- Input: outputs/model_predictions.csv
- Detailed results: outputs/bootstrap_validation.csv

Research and portfolio demonstration only.
