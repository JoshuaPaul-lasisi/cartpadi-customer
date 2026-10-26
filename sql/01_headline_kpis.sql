-- 01_headline_kpis.sql
-- The headline numbers: customers, orders, GMV, and how many customers come back.

WITH split_carts AS (
    -- Customers whose only "second order" was placed on the same day as the first.
    SELECT COUNT(*) AS n
    FROM (
        SELECT customer_unique_id, COUNT(*) AS orders_on_first_day
        FROM (
            SELECT customer_unique_id, purchase_ts,
                   CAST(purchase_ts AS DATE) = MIN(CAST(purchase_ts AS DATE)) OVER (PARTITION BY customer_unique_id) AS on_first_day
            FROM orders_clean
            WHERE completed
        )
        WHERE on_first_day
        GROUP BY customer_unique_id
        HAVING COUNT(*) > 1
    )
)
SELECT
    (SELECT COUNT(*) FROM customers_clean)                                   AS customers,
    (SELECT COUNT(*) FROM orders_clean WHERE completed)                      AS completed_orders,
    (SELECT ROUND(SUM(order_value), 2) FROM orders_clean WHERE completed)    AS gmv_brl,
    (SELECT ROUND(AVG(order_value), 2) FROM orders_clean WHERE completed)    AS avg_order_value_brl,
    (SELECT ROUND(AVG((second_purchase IS NOT NULL)::INT), 4) FROM customers_clean) AS repeat_ever_rate,
    (SELECT ROUND(AVG(repeat_180d), 4) FROM customers_clean WHERE repeat_180d_observed) AS repeat_180d_rate,
    (SELECT COUNT(*) FROM customers_clean WHERE repeat_180d_observed)        AS customers_with_180d_window,
    (SELECT n FROM split_carts)                                              AS customers_with_same_day_orders;
