-- 00_views.sql
-- Views over the raw Olist CSVs, then the two clean tables every query uses.
-- Same rules as notebook 01:
--   * a customer is customer_unique_id (customer_id is issued per order)
--   * orders placed 2017-01-01 to 2018-08-31 only
--   * completed = has items and is not canceled or unavailable
--   * order value = items + freight; lead category and seller = most expensive item
--     (ties on price go to the lower order_item_id)
--   * a repeat purchase is a completed order on a later calendar day than the first

CREATE OR REPLACE VIEW raw_customers   AS SELECT * FROM read_csv('data/raw/olist_customers_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_orders      AS SELECT * FROM read_csv('data/raw/olist_orders_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_order_items AS SELECT * FROM read_csv('data/raw/olist_order_items_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_payments    AS SELECT * FROM read_csv('data/raw/olist_order_payments_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_reviews     AS SELECT * FROM read_csv('data/raw/olist_order_reviews_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_products    AS SELECT * FROM read_csv('data/raw/olist_products_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_sellers     AS SELECT * FROM read_csv('data/raw/olist_sellers_dataset.csv', header = true);
CREATE OR REPLACE VIEW raw_category_en AS SELECT * FROM read_csv('data/raw/product_category_name_translation.csv', header = true);

-- One row per order: totals over its items.
CREATE OR REPLACE VIEW order_totals AS
SELECT
    order_id,
    SUM(price)                AS items_value,
    SUM(freight_value)        AS freight_value,
    COUNT(*)                  AS n_items,
    COUNT(DISTINCT seller_id) AS n_sellers
FROM raw_order_items
GROUP BY order_id;

-- Category and seller state of each order's most expensive item.
CREATE OR REPLACE VIEW order_lead AS
WITH ranked AS (
    SELECT
        i.order_id,
        COALESCE(
            t.product_category_name_english,
            CASE p.product_category_name
                WHEN 'pc_gamer' THEN 'pc_gamer'
                WHEN 'portateis_cozinha_e_preparadores_de_alimentos' THEN 'kitchen_portables_and_food_prep'
            END,
            'unknown')        AS lead_category,
        s.seller_state,
        ROW_NUMBER() OVER (PARTITION BY i.order_id ORDER BY i.price DESC, i.order_item_id) AS rn
    FROM raw_order_items i
    LEFT JOIN raw_products p    ON p.product_id = i.product_id
    LEFT JOIN raw_category_en t ON t.product_category_name = p.product_category_name
    LEFT JOIN raw_sellers s     ON s.seller_id = i.seller_id
)
SELECT order_id, lead_category, seller_state FROM ranked WHERE rn = 1;

-- Main payment type (the one that paid the most) and instalments.
CREATE OR REPLACE VIEW order_payment AS
WITH ranked AS (
    SELECT
        order_id,
        payment_type,
        MAX(payment_installments) OVER (PARTITION BY order_id)                        AS installments,
        MAX(CASE WHEN payment_type = 'voucher' THEN 1 ELSE 0 END) OVER (PARTITION BY order_id) AS used_voucher,
        ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY payment_value DESC)        AS rn
    FROM raw_payments
)
SELECT order_id, payment_type, installments, used_voucher FROM ranked WHERE rn = 1;

-- Latest answered review per order.
CREATE OR REPLACE VIEW order_review AS
SELECT order_id, review_score
FROM (
    SELECT order_id, review_score,
           ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_answer_timestamp DESC) AS rn
    FROM raw_reviews
)
WHERE rn = 1;

-- Clean order table (matches data/processed/orders_enriched.csv).
CREATE OR REPLACE VIEW orders_clean AS
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_purchase_timestamp                                   AS purchase_ts,
    o.order_status,
    o.order_delivered_customer_date                              AS delivered_ts,
    o.order_estimated_delivery_date                              AS estimated_delivery_ts,
    (o.order_status NOT IN ('canceled', 'unavailable') AND t.order_id IS NOT NULL) AS completed,
    COALESCE(t.items_value, 0) + COALESCE(t.freight_value, 0)    AS order_value,
    t.n_items,
    t.n_sellers,
    l.lead_category,
    l.seller_state,
    pm.payment_type,
    pm.installments,
    pm.used_voucher,
    r.review_score,
    date_diff('second', o.order_purchase_timestamp, o.order_delivered_customer_date) / 86400.0      AS delivery_days,
    date_diff('second', o.order_estimated_delivery_date, o.order_delivered_customer_date) / 86400.0 AS delay_days,
    CASE WHEN o.order_delivered_customer_date IS NULL THEN NULL
         WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1
         ELSE 0 END                                              AS is_late
FROM raw_orders o
JOIN raw_customers c       ON c.customer_id = o.customer_id
LEFT JOIN order_totals t   ON t.order_id = o.order_id
LEFT JOIN order_lead l     ON l.order_id = o.order_id
LEFT JOIN order_payment pm ON pm.order_id = o.order_id
LEFT JOIN order_review r   ON r.order_id = o.order_id
WHERE o.order_purchase_timestamp >= TIMESTAMP '2017-01-01'
  AND o.order_purchase_timestamp <  TIMESTAMP '2018-09-01';

-- Clean customer table (matches data/processed/customers.csv), built from completed orders.
CREATE OR REPLACE VIEW customers_clean AS
WITH done AS (
    SELECT *,
           CAST(purchase_ts AS DATE) AS purchase_day,
           ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts, order_id) AS seq
    FROM orders_clean
    WHERE completed
),
firsts AS (
    SELECT * FROM done WHERE seq = 1
),
agg AS (
    SELECT
        customer_unique_id,
        MIN(purchase_ts)             AS first_purchase,
        MAX(purchase_ts)             AS last_purchase,
        COUNT(*)                     AS n_orders,
        COUNT(DISTINCT purchase_day) AS n_purchase_days,
        SUM(order_value)             AS total_spend
    FROM done
    GROUP BY customer_unique_id
),
second AS (
    SELECT d.customer_unique_id, MIN(d.purchase_ts) AS second_purchase
    FROM done d
    JOIN firsts f USING (customer_unique_id)
    WHERE d.purchase_day > f.purchase_day
    GROUP BY d.customer_unique_id
)
SELECT
    a.*,
    f.customer_state,
    f.order_id        AS first_order_id,
    f.order_value     AS first_order_value,
    f.lead_category   AS first_lead_category,
    f.payment_type    AS first_payment_type,
    f.review_score    AS first_review_score,
    f.delivery_days   AS first_delivery_days,
    f.delay_days      AS first_delay_days,
    f.is_late         AS first_is_late,
    s.second_purchase,
    date_diff('second', a.first_purchase, s.second_purchase) / 86400.0 AS days_to_second,
    a.first_purchase < TIMESTAMP '2018-09-01' - INTERVAL 180 DAY      AS repeat_180d_observed,
    CASE WHEN a.first_purchase < TIMESTAMP '2018-09-01' - INTERVAL 180 DAY
         THEN COALESCE(date_diff('second', a.first_purchase, s.second_purchase) / 86400.0 <= 180, false)::INT
    END                                                                AS repeat_180d
FROM agg a
JOIN firsts f USING (customer_unique_id)
LEFT JOIN second s USING (customer_unique_id);
