-- 06_rfm_segments.sql
-- RFM scores and segments, using the adapted scheme from notebook 02:
-- R and M are quintiles (5 = best), F is one purchase day or two-plus.
-- Ties at quintile edges are broken by customer id, so counts can differ
-- from the notebook by a handful of customers.

WITH base AS (
    SELECT
        customer_unique_id,
        floor(date_diff('second', last_purchase, TIMESTAMP '2018-09-01') / 86400)::INT AS recency_days,
        n_purchase_days,
        total_spend
    FROM customers_clean
),
scored AS (
    SELECT *,
        6 - NTILE(5) OVER (ORDER BY recency_days, customer_unique_id) AS r,
        NTILE(5) OVER (ORDER BY total_spend, customer_unique_id)      AS m,
        CASE WHEN n_purchase_days >= 2 THEN 2 ELSE 1 END              AS f
    FROM base
)
SELECT
    CASE
        WHEN f = 2 AND r >= 3 THEN 'Repeat, active'
        WHEN f = 2            THEN 'Repeat, lapsed'
        WHEN r >= 4 AND m >= 4 THEN 'New, high value'
        WHEN r >= 4            THEN 'New'
        WHEN r >= 2 AND m >= 4 THEN 'Cooling, high value'
        WHEN r >= 2            THEN 'Cooling'
        WHEN m >= 4            THEN 'Dormant, high value'
        ELSE                        'Dormant'
    END                                   AS segment,
    COUNT(*)                              AS customers,
    ROUND(AVG(recency_days))              AS avg_recency_days,
    ROUND(AVG(total_spend), 2)            AS avg_spend_brl,
    ROUND(SUM(total_spend), 2)            AS revenue_brl,
    ROUND(SUM(total_spend) / SUM(SUM(total_spend)) OVER (), 4) AS revenue_share
FROM scored
GROUP BY 1
ORDER BY revenue_brl DESC;
