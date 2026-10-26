-- 03_cohort_retention.sql
-- Cohort matrix in long format: share of each first-purchase cohort buying in
-- each later calendar month. Months with no buyers appear as 0, and months
-- after the snapshot are left out (they haven't happened yet).

WITH cohorts AS (
    SELECT customer_unique_id, date_trunc('month', first_purchase) AS cohort_month
    FROM customers_clean
),
cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_size FROM cohorts GROUP BY cohort_month
),
grid AS (
    -- every cohort x every month from 0 up to the snapshot month
    SELECT s.cohort_month, s.cohort_size, k.k
    FROM cohort_size s,
         LATERAL range(0, date_diff('month', s.cohort_month, DATE '2018-08-01') + 1) AS k(k)
),
activity AS (
    SELECT
        c.cohort_month,
        date_diff('month', c.cohort_month, date_trunc('month', o.purchase_ts)) AS k,
        COUNT(DISTINCT o.customer_unique_id) AS active_customers
    FROM orders_clean o
    JOIN cohorts c USING (customer_unique_id)
    WHERE o.completed
    GROUP BY 1, 2
)
SELECT
    strftime(g.cohort_month, '%Y-%m')               AS cohort_month,
    g.cohort_size,
    g.k                                             AS months_since_first_purchase,
    COALESCE(a.active_customers, 0)                 AS active_customers,
    ROUND(COALESCE(a.active_customers, 0) / g.cohort_size, 5) AS retention_rate
FROM grid g
LEFT JOIN activity a ON a.cohort_month = g.cohort_month AND a.k = g.k
ORDER BY g.cohort_month, g.k;
