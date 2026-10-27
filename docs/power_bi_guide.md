# Power BI dashboard build guide

A step-by-step guide to building the five-page dashboard from the tables the notebooks write to `data/processed/`. Each page ends with the numbers it should show, so you can check your work against the notebooks.

Before you start: run notebooks 01 to 05 once so that `data/processed/` holds `orders_enriched.csv`, `customers.csv`, `rfm_segments.csv`, `cohort_retention.csv`, `repeat_scores.csv` and `clv_scores.csv`.

## 1. Load and shape the data

### 1.1 Get the files

Home, Get data, Text/CSV. Load these six files, then click Transform data to open Power Query.

### 1.2 Fix types in Power Query

Power BI guesses types from the first rows, and some guesses are wrong. Set these by hand.

| Query | Column | Type |
|---|---|---|
| orders_enriched | purchase_ts, delivered_ts, estimated_delivery_ts | Date/Time |
| orders_enriched | order_value, items_value, freight_value, delivery_days, delay_days, review_score, is_late | Decimal number |
| orders_enriched | completed | Whole number |
| customers | first_purchase, last_purchase, second_purchase | Date/Time |
| customers | total_spend, days_to_second, repeat_180d, first_review_score | Decimal number |
| customers | repeat_ever, repeat_180d_observed | Whole number |
| cohort_retention | retention_rate, cumulative_spend_per_customer_brl | Decimal number |

If your Windows region uses a comma as the decimal separator, set each CSV's locale to English (United States) in the Changed Type step, or the decimals will load wrong.

Leave empty cells as null. `repeat_180d` is empty on purpose for customers whose 180-day window hadn't closed by the snapshot, and `delivered_ts` is empty for orders that never arrived.

### 1.3 Add columns in Power Query

1. In orders_enriched, add a column `purchase_date`: Add Column, Custom Column, `Date.From([purchase_ts])`.
2. In cohort_retention, add a column `cohort_date`: `Date.FromText([cohort_month] & "-01")`, typed as Date.
3. Rename the queries: orders_enriched to Orders, customers to Customers, cohort_retention to Cohorts.

### 1.4 Merge the customer tables into one

The four customer files share `customer_unique_id`. Fold them into Customers so the model has one customer table, not four.

1. Select Customers in the Queries pane on the left. Home, Merge Queries (the top half of the button, not "as New").
2. In the Merge window, Customers is already at the top. In the dropdown below it pick rfm_segments. Click the `customer_unique_id` header in both tables so both turn green. Join kind: Left Outer. The bar at the bottom should say 94,703 of 94,703 rows matched. Click OK.
3. A new column appears at the far right of Customers, named rfm_segments, with "Table" in every cell. Each cell holds that customer's matching row from rfm_segments, folded up. Expanding unfolds it into ordinary columns.
4. Click the icon with two arrows pointing apart in that column's header. In the list that opens, untick (Select All Columns), then tick only `segment`, `R` and `M`. Untick "Use original column name as prefix" at the bottom, or the columns arrive as `rfm_segments.segment`. Click OK.
5. Repeat steps 1 to 4 with repeat_scores, ticking only `p_repeat_180d` and `return_tier`.
6. Repeat with clv_scores, ticking only `expected_repeat_value_180d` and `customer_value`.
7. Check the row count at the bottom left of Power Query: Customers should still have 94,703 rows. More rows would mean a merge matched some customers twice.
8. Right-click rfm_segments, repeat_scores and clv_scores in the Queries pane and untick Enable load. They stay as sources but don't clutter the model.

Only tick the columns listed. The score files also carry `customer_state`, `total_spend` and other columns that Customers already has, and expanding them would create duplicates.

Close and Apply.

### 1.5 Date table and relationships

Modeling, New table:

```
Date =
ADDCOLUMNS (
    CALENDAR ( DATE ( 2017, 1, 1 ), DATE ( 2018, 8, 31 ) ),
    "Month start", DATE ( YEAR ( [Date] ), MONTH ( [Date] ), 1 ),
    "Month", FORMAT ( [Date], "mmm yyyy" ),
    "Month number", YEAR ( [Date] ) * 100 + MONTH ( [Date] )
)
```

Select the Month column and set Sort by column to Month number. Mark the table as a date table (Table tools, Mark as date table, column Date).

In Model view, create two relationships, both one-to-many with single-direction filtering:

| From (one side) | To (many side) |
|---|---|
| Customers[customer_unique_id] | Orders[customer_unique_id] |
| Date[Date] | Orders[purchase_date] |

Cohorts stands alone. It is already a summary table.

### 1.6 Theme

View, Themes, Browse for themes, and pick `docs/olist_theme.json`. It sets the same colours as the notebook charts: blue for the main series, orange for a comparison, grey for context, a white card background and light borders.

## 2. Measures

Create a table to hold them: Home, Enter data, name it `_Measures`, load it, then add each measure below to it with New measure. Format each one as shown.

```
Customers = COUNTROWS ( Customers )

Completed Orders =
CALCULATE ( COUNTROWS ( Orders ), Orders[completed] = 1 )

GMV =
CALCULATE ( SUM ( Orders[order_value] ), Orders[completed] = 1 )

Average Order Value = DIVIDE ( [GMV], [Completed Orders] )

Labelled Customers =
CALCULATE ( COUNTROWS ( Customers ), Customers[repeat_180d_observed] = 1 )

Repeat Rate 180d =
CALCULATE ( AVERAGE ( Customers[repeat_180d] ), Customers[repeat_180d_observed] = 1 )

Repeat Rate Ever = AVERAGE ( Customers[repeat_ever] )

Repeat Customer Revenue Share =
DIVIDE (
    CALCULATE ( SUM ( Customers[total_spend] ), Customers[repeat_ever] = 1 ),
    SUM ( Customers[total_spend] )
)

Delivered Orders =
CALCULATE (
    COUNTROWS ( Orders ),
    Orders[completed] = 1,
    NOT ISBLANK ( Orders[delivered_ts] )
)

Late Rate =
CALCULATE ( AVERAGE ( Orders[is_late] ), Orders[completed] = 1 )

Average Delivery Days =
CALCULATE ( AVERAGE ( Orders[delivery_days] ), Orders[completed] = 1 )

Reviewed Orders =
CALCULATE (
    COUNTROWS ( Orders ),
    Orders[completed] = 1,
    NOT ISBLANK ( Orders[review_score] )
)

Bad Review Share =
DIVIDE (
    CALCULATE (
        COUNTROWS ( Orders ),
        Orders[completed] = 1,
        NOT ISBLANK ( Orders[review_score] ),
        Orders[review_score] <= 2
    ),
    [Reviewed Orders]
)
```

Formats: Customers, Completed Orders, Labelled Customers, Delivered Orders and Reviewed Orders as whole numbers with a thousands separator. GMV and Average Order Value as currency, R$, no decimals for GMV. Every rate and share as a percentage with one decimal place (two for the repeat rates, which sit near 2%).

Two traps these measures avoid:

1. In DAX a blank is treated as zero in comparisons, so `review_score <= 2` is true for orders with no review. That is why Bad Review Share filters out blanks first. Without it the share comes out too high.
2. AVERAGE skips blanks. Late Rate and Repeat Rate 180d rely on that: undelivered orders have no `is_late`, and recent customers have no `repeat_180d`, so they drop out of the average instead of counting as zero.

### Calculated column for delay bands

On the Orders table, New column:

```
Delay band =
SWITCH (
    TRUE (),
    ISBLANK ( Orders[delay_days] ), "8. Not delivered",
    Orders[delay_days] <= -10, "1. 10+ days early",
    Orders[delay_days] <= -5, "2. 5 to 10 days early",
    Orders[delay_days] <= 0, "3. 0 to 5 days early",
    Orders[delay_days] <= 3, "4. 1 to 3 days late",
    Orders[delay_days] <= 7, "5. 4 to 7 days late",
    Orders[delay_days] <= 14, "6. 8 to 14 days late",
    "7. 15+ days late"
)
```

The number at the front keeps the bands in order on an axis.

## 3. Pages

Every page gets the same layout: a title text box top left, a one-line takeaway under it, KPI cards across the top, charts below. Keep to one or two colours per chart; grey for context.

### Page 1: Overview

1. Title: "Olist customer retention". Takeaway: "97.8% of customers buy once. Revenue rests on new customers."
2. Four cards: Customers, GMV, Repeat Rate 180d, Repeat Customer Revenue Share.
3. Line chart: X axis Date[Month], Y axis GMV. Add a text annotation at November 2017: "Black Friday".
4. Clustered bar chart: Customers by Customers[n_purchase_days]. Filter the visual to show 1, 2, 3 and 4. Data labels on.
5. Slicer: Customers[customer_state] as a dropdown.
6. Small text box in the footer: "Source: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0)."

Check: 94,703 customers; GMV R$15.7M; Repeat Rate 180d 1.98%; Repeat Customer Revenue Share 4.2%; November 2017 GMV about R$1.17M.

### Page 2: Orders and geography

1. Takeaway: "São Paulo state has 42% of customers and the fastest delivery."
2. Cards: Completed Orders, Average Order Value, Average Delivery Days.
3. Bar chart: GMV by Orders[lead_category]. Visual filter: Top N, 15, by GMV.
4. Bar chart: Customers by Customers[customer_state], sorted descending. Top N, 12.
5. Bar chart: Average Delivery Days by Orders[customer_state], sorted ascending, filtered to states with Delivered Orders at least 1,000.
6. Donut or bar: Completed Orders by Orders[payment_type]. A bar reads better than a donut with four slices this uneven.

Check: 97,905 completed orders; average order value R$160; average delivery 12.5 days; top categories health_beauty (R$1.43M), watches_gifts (R$1.30M), bed_bath_table (R$1.24M); credit card 75.5% of orders, boleto 19.9%.

### Page 3: Delivery and reviews

This is the strongest page. Give the main chart the most space.

1. Takeaway: "Late parcels get bad reviews: past three days late, most customers leave one or two stars."
2. Cards: Late Rate, Bad Review Share.
3. Column chart: X axis Orders[Delay band], Y axis Bad Review Share. Filter out "8. Not delivered". Data labels on. Use conditional formatting on the columns: blue for the three early bands, red (#e34948) for the late ones (Format, Columns, Colors, fx, rules on the Delay band text).
4. Bar chart: Late Rate by Orders[customer_state], sorted descending, filtered to Delivered Orders at least 1,000.
5. Table: customer_state, Delivered Orders, Late Rate, Average Delivery Days.

Check: Late Rate 8.1%; Bad Review Share 13.8% overall; bad-review share about 9% for the early bands, 61% at 4 to 7 days late and 78% past a week late; worst late rates CE 15.4%, BA 14.0%, RJ 13.5%; SP 5.9%.

### Page 4: Cohorts and repeat buying

1. Takeaway: "Under 1% of any cohort buys again in a later month."
2. Matrix: Rows Cohorts[cohort_date] (set to show month and year, not the date hierarchy), Columns Cohorts[months_since_first_purchase], Values Cohorts[retention_rate] (Sum). Visual filter: months_since_first_purchase between 1 and 12. Turn off row and column totals.
3. Conditional formatting on the matrix values: Background colour, Gradient, lowest value #cde2fb, highest value #0d366b, with the maximum set to a fixed number 0.008 so a single high cell doesn't wash out the rest. Format values as percentage with one decimal.
4. Bar chart: Repeat Rate 180d by Customers[first_lead_category]. Visual filter: Labelled Customers is at least 1,000. Sort descending.
5. Card: Repeat Rate Ever.

Check: month 1 cells mostly 0.2% to 0.7%; bed_bath_table 2.7%, sports_leisure 2.6% to 2.7%, cool_stuff 1.0% at the bottom; Repeat Rate Ever 2.17%.

### Page 5: Segments and targeting

1. Takeaway: "The model's top 10% returns at 1.8 times the average. That group is the target for a second-order test."
2. Bar chart: Customers by Customers[segment].
3. Bar chart: SUM(Customers[total_spend]) by Customers[segment]. Put it beside the first so size and revenue sit side by side.
4. Bar chart: Customers by Customers[return_tier], in the order High, Medium, Low (sort by a small helper column or set the order manually).
5. Card with a fixed text box under it: "Out-of-time test: top 10% returned at 3.2% against 1.8% on average (notebook 04)." These numbers come from the model test and can't be rebuilt from the dashboard tables, because the tiers on labelled customers are in-sample.
6. Table, the action list: customer_unique_id, customer_state, first_lead_category, total_spend, p_repeat_180d. Filters on the visual: return_tier is High, repeat_180d_observed is 0. These are recent customers the model ranks highest, the ones a campaign can still reach. Sort by p_repeat_180d descending.

Check: High 9,471, Medium 18,940, Low 66,292 customers; the action list holds 4,750 customers; "New, high value" holds 28.5% of revenue and "Cooling, high value" 27.2%.

## 4. Finish and publish

1. Turn off visual headers you don't need (Format, Visual header) and align everything to the grid.
2. Add page navigation buttons, or rename the page tabs clearly.
3. Save as `dashboard/olist_retention.pbix`.
4. Export a screenshot of each page (File, Export, Export to PDF, or a screen capture) into `reports/figures/` as `dashboard_1_overview.png` to `dashboard_5_segments.png`.
5. Commit the .pbix and the screenshots. The .pbix contains a copy of the data, which is fine under Olist's licence as long as the source line on page 1 stays.
