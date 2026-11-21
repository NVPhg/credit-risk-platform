-- Level 3: modelling mart, exactly one row per applicant (307,511 rows).

-- Mart v1: applicant + bureau + previous applications.
CREATE OR REPLACE VIEW risk_analytics.customer_feature_mart AS
 SELECT a.applicant_id,
    a.target,
    a.annual_income,
    a.credit_amount,
    a.annuity_amount,
    a.contract_type,
    a.income_type,
    a.education_type,
    a.credit_to_income,
    a.annuity_to_income,
    a.age_years,
    a.employment_years,
    a.employment_sentinel_flag,
    COALESCE(b.bureau_loan_count, 0::bigint) AS bureau_loan_count,
    COALESCE(b.active_loan_count, 0::bigint) AS active_loan_count,
    b.avg_bureau_credit,
    COALESCE(p.previous_application_count, 0::bigint) AS previous_application_count,
    COALESCE(p.approved_count, 0::bigint) AS approved_count,
    COALESCE(p.refused_count, 0::bigint) AS refused_count,
    p.approval_rate
   FROM risk_analytics.applicant_features a
     LEFT JOIN risk_analytics.bureau_features b ON a.applicant_id = b.applicant_id
     LEFT JOIN risk_analytics.previous_application_features p ON a.applicant_id = p.applicant_id;

-- Mart v2 (used by the models): v1 + installment_features_v2.
CREATE OR REPLACE VIEW risk_analytics.customer_feature_mart_v2 AS
 SELECT m.applicant_id,
    m.target,
    m.annual_income,
    m.credit_amount,
    m.annuity_amount,
    m.contract_type,
    m.income_type,
    m.education_type,
    m.credit_to_income,
    m.annuity_to_income,
    m.age_years,
    m.employment_years,
    m.employment_sentinel_flag,
    m.bureau_loan_count,
    m.active_loan_count,
    m.avg_bureau_credit,
    m.previous_application_count,
    m.approved_count,
    m.refused_count,
    m.approval_rate,
        CASE
            WHEN i.applicant_id IS NULL THEN 0
            ELSE 1
        END AS has_installment_history,
    COALESCE(i.installment_count, 0::bigint) AS installment_count,
    COALESCE(i.multiple_payment_count, 0::bigint) AS multiple_payment_count,
    COALESCE(i.incomplete_installment_count, 0::bigint) AS incomplete_installment_count,
    COALESCE(i.valid_installment_count, 0::bigint) AS valid_installment_count,
    COALESCE(i.due_date_shortfall_count, 0::bigint) AS due_date_shortfall_count,
    i.due_date_shortfall_rate,
    COALESCE(i.observed_underpaid_count, 0::bigint) AS observed_underpaid_count
   FROM risk_analytics.customer_feature_mart m
     LEFT JOIN risk_analytics.installment_features_v2 i ON m.applicant_id = i.applicant_id;

