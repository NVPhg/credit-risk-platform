-- Data-quality assertions run before modelling. Every query should return the expected value in the comment.

-- 1. Grain: exactly one row per applicant (expect 307511 / 307511)
SELECT COUNT(*) AS row_count, COUNT(DISTINCT applicant_id) AS distinct_applicants
FROM risk_analytics.customer_feature_mart_v2;

-- 2. Label completeness (expect 0)
SELECT COUNT(*) AS missing_target
FROM risk_analytics.customer_feature_mart_v2
WHERE target IS NULL;

-- 3. Portfolio TARGET rate (expect ~8.07%)
SELECT ROUND(AVG(target)::numeric, 4) AS target_rate
FROM risk_analytics.customer_feature_mart_v2;

-- 4. Employment sentinel handled (expect 0 rows with employment_years for sentinel applicants)
SELECT COUNT(*) AS sentinel_leak
FROM risk_analytics.customer_feature_mart_v2
WHERE employment_sentinel_flag = 1 AND employment_years IS NOT NULL;

-- 5. Ratios inside a valid range (expect 0)
SELECT COUNT(*) AS invalid_rates
FROM risk_analytics.customer_feature_mart_v2
WHERE due_date_shortfall_rate NOT BETWEEN 0 AND 1
   OR approval_rate NOT BETWEEN 0 AND 1;

-- 6. Coverage of each source (share of applicants with history)
SELECT
    ROUND(AVG((bureau_loan_count > 0)::int)::numeric, 4)          AS bureau_coverage,
    ROUND(AVG((previous_application_count > 0)::int)::numeric, 4) AS previous_app_coverage,
    ROUND(AVG(has_installment_history)::numeric, 4)                AS installment_coverage
FROM risk_analytics.customer_feature_mart_v2;
