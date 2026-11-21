-- Level 2: behavioural aggregates, materialised because the source tables hold tens of millions of rows.
-- Refresh with: REFRESH MATERIALIZED VIEW risk_analytics.<name>;

-- Installment v2: collapse partial payments to one row per installment before measuring due-date shortfall.
DROP MATERIALIZED VIEW IF EXISTS risk_analytics.installment_features_v2 CASCADE;
CREATE MATERIALIZED VIEW risk_analytics.installment_features_v2 AS
 WITH installment_level AS (
         SELECT installments_payments."SK_ID_CURR" AS applicant_id,
            count(*) AS payment_records,
            max((installments_payments."AMT_INSTALMENT")::numeric) AS amount_due,
            count(*) FILTER (WHERE ((installments_payments."DAYS_ENTRY_PAYMENT" IS NULL) OR (installments_payments."AMT_PAYMENT" IS NULL))) AS incomplete_records,
            COALESCE(sum((installments_payments."AMT_PAYMENT")::numeric) FILTER (WHERE (installments_payments."DAYS_ENTRY_PAYMENT" <= installments_payments."DAYS_INSTALMENT")), (0)::numeric) AS paid_by_due_date,
            COALESCE(sum((installments_payments."AMT_PAYMENT")::numeric) FILTER (WHERE (installments_payments."DAYS_ENTRY_PAYMENT" IS NOT NULL)), (0)::numeric) AS total_paid_observed
           FROM installments_payments
          GROUP BY installments_payments."SK_ID_CURR", installments_payments."SK_ID_PREV", installments_payments."NUM_INSTALMENT_VERSION", installments_payments."NUM_INSTALMENT_NUMBER"
        )
 SELECT applicant_id,
    count(*) AS installment_count,
    count(*) FILTER (WHERE (payment_records > 1)) AS multiple_payment_count,
    count(*) FILTER (WHERE (incomplete_records > 0)) AS incomplete_installment_count,
    count(*) FILTER (WHERE ((incomplete_records = 0) AND (amount_due > (0)::numeric))) AS valid_installment_count,
    count(*) FILTER (WHERE ((incomplete_records = 0) AND (amount_due > (0)::numeric) AND (paid_by_due_date < (amount_due * 0.999)))) AS due_date_shortfall_count,
    round(((count(*) FILTER (WHERE ((incomplete_records = 0) AND (amount_due > (0)::numeric) AND (paid_by_due_date < (amount_due * 0.999)))))::numeric / (NULLIF(count(*) FILTER (WHERE ((incomplete_records = 0) AND (amount_due > (0)::numeric))), 0))::numeric), 4) AS due_date_shortfall_rate,
    count(*) FILTER (WHERE ((incomplete_records = 0) AND (amount_due > (0)::numeric) AND (total_paid_observed < (amount_due * 0.999)))) AS observed_underpaid_count
   FROM installment_level
  GROUP BY applicant_id;

-- Bureau monthly balance, step 1: status history per bureau loan.
DROP MATERIALIZED VIEW IF EXISTS risk_analytics.bureau_balance_loan_features CASCADE;
CREATE MATERIALIZED VIEW risk_analytics.bureau_balance_loan_features AS
 SELECT "SK_ID_BUREAU" AS bureau_loan_id,
    count(*) AS balance_month_count,
    count(*) FILTER (WHERE (("STATUS")::text = ANY ((ARRAY['0'::character varying, '1'::character varying, '2'::character varying, '3'::character varying, '4'::character varying, '5'::character varying])::text[]))) AS graded_month_count,
    count(*) FILTER (WHERE (("STATUS")::text = ANY ((ARRAY['1'::character varying, '2'::character varying, '3'::character varying, '4'::character varying, '5'::character varying])::text[]))) AS delinquent_status_month_count,
    count(*) FILTER (WHERE (("STATUS")::text = ANY ((ARRAY['4'::character varying, '5'::character varying])::text[]))) AS severe_status_month_count,
    count(*) FILTER (WHERE (("STATUS")::text = 'C'::text)) AS closed_month_count,
    count(*) FILTER (WHERE (("STATUS")::text = 'X'::text)) AS unknown_month_count,
    count(*) FILTER (WHERE ((("MONTHS_BALANCE" >= '-11'::integer) AND ("MONTHS_BALANCE" <= 0)) AND (("STATUS")::text = ANY ((ARRAY['0'::character varying, '1'::character varying, '2'::character varying, '3'::character varying, '4'::character varying, '5'::character varying])::text[])))) AS recent12_graded_month_count,
    count(*) FILTER (WHERE ((("MONTHS_BALANCE" >= '-11'::integer) AND ("MONTHS_BALANCE" <= 0)) AND (("STATUS")::text = ANY ((ARRAY['1'::character varying, '2'::character varying, '3'::character varying, '4'::character varying, '5'::character varying])::text[])))) AS recent12_delinquent_month_count,
    max(
        CASE
            WHEN (("STATUS")::text = ANY ((ARRAY['0'::character varying, '1'::character varying, '2'::character varying, '3'::character varying, '4'::character varying, '5'::character varying])::text[])) THEN ("STATUS")::integer
            ELSE NULL::integer
        END) AS max_status_code
   FROM bureau_balance
  GROUP BY "SK_ID_BUREAU";

-- Bureau monthly balance, step 2: roll loan-level history up to the applicant.
DROP MATERIALIZED VIEW IF EXISTS risk_analytics.bureau_balance_customer_features CASCADE;
CREATE MATERIALIZED VIEW risk_analytics.bureau_balance_customer_features AS
 SELECT b."SK_ID_CURR" AS applicant_id,
    count(*) AS loans_with_monthly_history,
    sum(bb.balance_month_count) AS balance_month_count,
    sum(bb.graded_month_count) AS graded_month_count,
    sum(bb.delinquent_status_month_count) AS delinquent_month_count,
    sum(bb.severe_status_month_count) AS severe_month_count,
    sum(bb.closed_month_count) AS closed_month_count,
    sum(bb.unknown_month_count) AS unknown_month_count,
    sum(bb.recent12_graded_month_count) AS recent12_graded_month_count,
    sum(bb.recent12_delinquent_month_count) AS recent12_delinquent_month_count,
    max(bb.max_status_code) AS max_bureau_status,
    round((sum(bb.delinquent_status_month_count) / NULLIF(sum(bb.graded_month_count), (0)::numeric)), 4) AS delinquency_rate,
    round((sum(bb.severe_status_month_count) / NULLIF(sum(bb.graded_month_count), (0)::numeric)), 4) AS severe_status_rate,
    round((sum(bb.recent12_delinquent_month_count) / NULLIF(sum(bb.recent12_graded_month_count), (0)::numeric)), 4) AS recent12_delinquency_rate
   FROM (bureau b
     JOIN risk_analytics.bureau_balance_loan_features bb ON ((b."SK_ID_BUREAU" = bb.bureau_loan_id)))
  GROUP BY b."SK_ID_CURR";

-- Credit card behaviour: DPD months, utilisation and over-limit months.
DROP MATERIALIZED VIEW IF EXISTS risk_analytics.credit_card_customer_features CASCADE;
CREATE MATERIALIZED VIEW risk_analytics.credit_card_customer_features AS
 SELECT "SK_ID_CURR" AS applicant_id,
    count(DISTINCT "SK_ID_PREV") AS card_account_count,
    count(*) AS card_month_count,
    count("SK_DPD") AS valid_dpd_month_count,
    count(*) FILTER (WHERE ("SK_DPD" > 0)) AS card_dpd_month_count,
    count(*) FILTER (WHERE ("SK_DPD" >= 30)) AS card_dpd30_month_count,
    round(((count(*) FILTER (WHERE ("SK_DPD" >= 30)))::numeric / (NULLIF(count("SK_DPD"), 0))::numeric), 4) AS card_dpd30_month_rate,
    round(avg(("AMT_BALANCE")::numeric), 2) AS avg_card_balance,
    round(avg(
        CASE
            WHEN (("AMT_CREDIT_LIMIT_ACTUAL" > 0) AND ("AMT_BALANCE" >= (0)::double precision)) THEN (("AMT_BALANCE")::numeric / ("AMT_CREDIT_LIMIT_ACTUAL")::numeric)
            ELSE NULL::numeric
        END), 4) AS avg_credit_utilization,
    round(max(
        CASE
            WHEN (("AMT_CREDIT_LIMIT_ACTUAL" > 0) AND ("AMT_BALANCE" >= (0)::double precision)) THEN (("AMT_BALANCE")::numeric / ("AMT_CREDIT_LIMIT_ACTUAL")::numeric)
            ELSE NULL::numeric
        END), 4) AS max_credit_utilization,
    count(*) FILTER (WHERE ("AMT_CREDIT_LIMIT_ACTUAL" <= 0)) AS nonpositive_limit_month_count,
    count(*) FILTER (WHERE (("AMT_CREDIT_LIMIT_ACTUAL" > 0) AND ("AMT_BALANCE" > ("AMT_CREDIT_LIMIT_ACTUAL")::double precision))) AS overlimit_month_count
   FROM credit_card_balance
  WHERE ("MONTHS_BALANCE" <= 0)
  GROUP BY "SK_ID_CURR";

