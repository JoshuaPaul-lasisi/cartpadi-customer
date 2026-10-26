-- 05_delivery_and_reviews.sql
-- Two views of the delivery problem.
-- Part 1: share of 1 and 2-star reviews by how early or late the parcel arrived.

SELECT
    CASE
        WHEN delay_days <= -10 THEN '1. 10+ days early'
        WHEN delay_days <= -5  THEN '2. 5 to 10 early'
        WHEN delay_days <= 0   THEN '3. 0 to 5 early'
        WHEN delay_days <= 3   THEN '4. 1 to 3 late'
        WHEN delay_days <= 7   THEN '5. 4 to 7 late'
        WHEN delay_days <= 14  THEN '6. 8 to 14 late'
        ELSE                        '7. 15+ late'
    END                                                   AS delay_band,
    COUNT(*)                                              AS orders,
    ROUND(AVG(review_score), 2)                           AS avg_review,
    ROUND(AVG((review_score <= 2)::INT), 3)               AS bad_review_share
FROM orders_clean
WHERE completed AND delivered_ts IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- Part 2: late rate and average delivery time by customer state (1,000+ delivered orders).

SELECT
    customer_state,
    COUNT(*)                         AS delivered_orders,
    ROUND(AVG(is_late), 3)           AS late_rate,
    ROUND(AVG(delivery_days), 1)     AS avg_delivery_days
FROM orders_clean
WHERE completed AND delivered_ts IS NOT NULL
GROUP BY 1
HAVING COUNT(*) >= 1000
ORDER BY late_rate DESC;
