-- Level 1: one row per applicant from each raw source table.
-- Raw Kaggle tables are loaded into the public schema with their original column names.

-- Application-level attributes, affordability ratios and the DAYS_EMPLOYED = 365243 sentinel fix.
CREATE OR REPLACE VIEW risk_analytics.applicant_features AS
 SELECT "SK_ID_CURR" AS applicant_id,
    "TARGET" AS target,
    "AMT_INCOME_TOTAL"::numeric AS annual_income,
    "AMT_CREDIT"::numeric AS credit_amount,
    "AMT_ANNUITY"::numeric AS annuity_amount,
    "NAME_CONTRACT_TYPE" AS contract_type,
    "NAME_INCOME_TYPE" AS income_type,
    "NAME_EDUCATION_TYPE" AS education_type,
        CASE
            WHEN "AMT_INCOME_TOTAL" > 0::double precision THEN round("AMT_CREDIT"::numeric / "AMT_INCOME_TOTAL"::numeric, 4)
            ELSE NULL::numeric
        END AS credit_to_income,
        CASE
            WHEN "AMT_INCOME_TOTAL" > 0::double precision THEN round("AMT_ANNUITY"::numeric / "AMT_INCOME_TOTAL"::numeric, 4)
            ELSE NULL::numeric
        END AS annuity_to_income,
        CASE
            WHEN "DAYS_BIRTH" < 0 THEN round((- "DAYS_BIRTH"::numeric) / 365.25, 1)
            ELSE NULL::numeric
        END AS age_years,
        CASE
            WHEN "DAYS_EMPLOYED" = 365243 OR "DAYS_EMPLOYED" > 0 THEN NULL::numeric
            ELSE round((- "DAYS_EMPLOYED"::numeric) / 365.25, 1)
        END AS employment_years,
        CASE
            WHEN "DAYS_EMPLOYED" = 365243 THEN 1
            ELSE 0
        END AS employment_sentinel_flag
   FROM application_train;

-- Credit Bureau: loan counts and average credit amount per applicant.
CREATE OR REPLACE VIEW risk_analytics.bureau_features AS
 SELECT "SK_ID_CURR" AS applicant_id,
    count(*) AS bureau_loan_count,
    count(*) FILTER (WHERE "CREDIT_ACTIVE"::text = 'Active'::text) AS active_loan_count,
    round(avg("AMT_CREDIT_SUM"::numeric), 2) AS avg_bureau_credit
   FROM bureau
  GROUP BY "SK_ID_CURR";

-- Previous Home Credit applications: approved / refused history.
CREATE OR REPLACE VIEW risk_analytics.previous_application_features AS
 SELECT "SK_ID_CURR" AS applicant_id,
    count(*) AS previous_application_count,
    count(*) FILTER (WHERE "NAME_CONTRACT_STATUS"::text = 'Approved'::text) AS approved_count,
    count(*) FILTER (WHERE "NAME_CONTRACT_STATUS"::text = 'Refused'::text) AS refused_count,
    round(count(*) FILTER (WHERE "NAME_CONTRACT_STATUS"::text = 'Approved'::text)::numeric / NULLIF(count(*), 0)::numeric, 4) AS approval_rate
   FROM previous_application
  GROUP BY "SK_ID_CURR";

-- Installment payments v1 (row-level lateness; superseded by installment_features_v2).
CREATE OR REPLACE VIEW risk_analytics.installment_features AS
 SELECT "SK_ID_CURR" AS applicant_id,
    count(*) AS payment_record_count,
    count(*) FILTER (WHERE "DAYS_INSTALMENT" IS NOT NULL AND "DAYS_ENTRY_PAYMENT" IS NOT NULL) AS dated_payment_records,
    count(*) FILTER (WHERE "DAYS_ENTRY_PAYMENT" > "DAYS_INSTALMENT") AS late_payment_records,
    round(count(*) FILTER (WHERE "DAYS_ENTRY_PAYMENT" > "DAYS_INSTALMENT")::numeric / NULLIF(count(*) FILTER (WHERE "DAYS_INSTALMENT" IS NOT NULL AND "DAYS_ENTRY_PAYMENT" IS NOT NULL), 0)::numeric, 4) AS late_payment_record_rate,
    round(avg("DAYS_ENTRY_PAYMENT"::numeric - "DAYS_INSTALMENT"::numeric) FILTER (WHERE "DAYS_ENTRY_PAYMENT" > "DAYS_INSTALMENT"), 2) AS avg_late_days
   FROM installments_payments
  GROUP BY "SK_ID_CURR";

