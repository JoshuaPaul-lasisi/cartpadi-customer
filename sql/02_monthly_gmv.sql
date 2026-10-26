-- 02_monthly_gmv.sql
-- GMV, orders, average order value and new customers by month.

WITH monthly AS (
    SELECT
        date_trunc('month', purchase_ts) AS month,
        COUNT(*)                         AS orders,
        SUM(order_value)                 AS gmv
    FROM orders_clean
    WHERE completed
    GROUP BY 1
),
new_customers AS (
    SELECT date_trunc('month', first_purchase) AS month, COUNT(*) AS new_customers
    FROM customers_clean
    GROUP BY 1
)
SELECT
    strftime(m.month, '%Y-%m')         AS month,
    m.orders,
    ROUND(m.gmv, 2)                    AS gmv_brl,
    ROUND(m.gmv / m.orders, 2)         AS avg_order_value_brl,
    n.new_customers
FROM monthly m
LEFT JOIN new_customers n USING (month)
ORDER BY m.month;
