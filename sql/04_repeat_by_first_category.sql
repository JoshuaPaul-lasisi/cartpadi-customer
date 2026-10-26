-- 04_repeat_by_first_category.sql
-- 180-day repeat rate by the category of the first order, with a 95% Wilson
-- interval, for categories with at least 1,000 customers. Only customers whose
-- first purchase was 180+ days before the snapshot are counted.

WITH base AS (
    SELECT first_lead_category AS category,
           COUNT(*)            AS customers,
           SUM(repeat_180d)    AS repeaters
    FROM customers_clean
    WHERE repeat_180d_observed
    GROUP BY 1
    HAVING COUNT(*) >= 1000
),
rates AS (
    SELECT *, repeaters / customers AS p, 1.96 AS z FROM base
)
SELECT
    category,
    customers,
    repeaters,
    ROUND(p, 4) AS repeat_rate,
    ROUND((p + z * z / (2 * customers) - z * sqrt(p * (1 - p) / customers + z * z / (4 * customers * customers)))
          / (1 + z * z / customers), 4) AS ci_low,
    ROUND((p + z * z / (2 * customers) + z * sqrt(p * (1 - p) / customers + z * z / (4 * customers * customers)))
          / (1 + z * z / customers), 4) AS ci_high
FROM rates
ORDER BY repeat_rate DESC;
