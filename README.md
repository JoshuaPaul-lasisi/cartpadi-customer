# CartPadi customer churn and retention analysis

Who buys again, who drops off, and how much revenue is at stake when they do: a churn and retention study of a Nigerian e-commerce marketplace.

## Company brief

CartPadi Nigeria Ltd. is a fictional pan-Nigerian B2C e-commerce marketplace. It sells electronics, phones, fashion, beauty, home and sports products to customers in 24 states, with Lagos and Abuja as its biggest markets. The dataset covers every customer who signed up between 1 January 2024 and 31 December 2025, their orders, the items in those orders, the product catalogue, and a sample of site sessions.

## Data disclaimer

All data in this project is synthetic. CartPadi is not a real company and has no relation to any existing brand. Product brands are invented; any match to a real name is a coincidence. The generation script, `scripts/01_data_generation.py`, is the provenance of the data. It runs with a fixed seed (42), so anyone can rebuild the same five CSV files.

## Project structure

```
cartpadi-customer/
  scripts/
    01_data_generation.py    builds the five raw CSVs
    validate_patterns.py     checks the injected patterns in the raw data
    build_notebook_0X.py     writes each notebook from source (edit these, not the .ipynb)
  src/
    cartpadi.py              shared paths, data cleaning and chart style
  data/
    raw/                     generated CSVs (not committed, rebuild with the script)
    processed/               notebook outputs (RFM, cohorts, churn scores, CLV)
  notebooks/                 01 to 05 analysis notebooks
  models/                    trained churn model
  dashboard/                 Power BI file
  reports/figures/           exported charts and the executive summary
  requirements.txt
```

## The data

Snapshot date is 31 December 2025. A customer is churned when they have placed no non-cancelled order in the 90 days up to the snapshot.

| File | Rows | Grain |
|---|---|---|
| customers.csv | 25,000 | one row per customer |
| orders.csv | about 121,000 | one row per order |
| order_items.csv | about 275,000 | one row per order line |
| products.csv | 2,000 | one row per product |
| sessions.csv | 480,000 | one row per site session (about a 15% sample of the full log) |

Notes on the columns:

1. `order_total_ngn` is the sum of line values after discount. `discount_ngn` is the naira value of those discounts. Both reconcile to `order_items.csv` to the kobo.
2. `discount_pct` is a percentage (15.0 means 15%).
3. `unit_price_ngn` is the list price on the order date. `base_price_ngn` in products is the list price at the snapshot; prices rise about 28% across the two years.
4. `delivery_days` is empty for cancelled orders.
5. `customer_id` in sessions is empty for anonymous visitors (about 31% of sessions).
6. Sessions are a sample, so not every order has a matching converted session.

The raw files carry two known quality issues for the EDA audit to catch: about 0.15% of order rows are exported twice, and about 0.8% of customers have no city.

## Patterns in the data (seed 42)

These are the behaviours the generator builds in, measured on the output by `validate_patterns.py`. Ninety-day retention here means a second order within 90 days of the first, for customers whose first order was at least 90 days before the snapshot.

| Pattern | Target | Generated |
|---|---|---|
| Customers with 1 / 2 to 4 / 5+ orders | 55% / 25% / 20% | 55.9% / 22.5% / 21.6% |
| Median days to second order | 28 | 28 |
| 90-day retention, referral | 45% | 41.9% |
| 90-day retention, email | 40% | 39.1% |
| 90-day retention, direct | 38% | 35.3% |
| 90-day retention, organic search | 32% | 31.6% |
| 90-day retention, paid social | 20% | 19.0% |
| 90-day retention, CartPadi wallet first order | 40% | 42.2% |
| 90-day retention, card | 28% | 32.3% |
| 90-day retention, bank transfer / USSD | 25% | 30.4% / 28.0% |
| 90-day retention, pay on delivery | 17% | 22.0% |
| Q4 signups vs Q1 to Q2 signups, 90-day retention | 30% lower | 31% lower |
| Nov / Dec / Apr / Jan orders vs a normal month (2025) | 2.5 / 2.2 / 1.4 / 0.7 | 2.56 / 2.22 / 1.42 / 0.68 |
| Electronics vs Fashion first-buyer LTV | 3.2x | 3.4x |
| Highest category repurchase rate | Beauty and Skincare | Beauty and Skincare |
| Highest average order value | Phones and Accessories | Phones and Accessories |
| Lagos share of customers / revenue | 43% / 51% | 42.9% / 51.8% |
| Abuja share of customers / revenue | 14% / 17% | 13.7% / 17.0% |
| Mean delivery days, Lagos vs outside top 5 states | 3 to 5 vs 8 to 14 | 4.0 vs 11.5 |
| Session conversion rate | about 3.5% | 3.7% |

Channel and payment effects overlap in the same customers, so both sets of targets can't land exactly at once. The order of best to worst holds for both. The 2024 seasonal peaks look sharper than 2025 because the customer base starts from zero in January 2024.

## How to run

```
pip install -r requirements.txt
python scripts/01_data_generation.py
python scripts/validate_patterns.py
```

The generator takes about 10 seconds. Then run the notebooks in order, 01 to 05, from the `notebooks` folder. Each notebook is written by a script in `scripts/`, so to change one, edit its builder and run:

```
python scripts/build_notebook_02.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_rfm_segmentation.ipynb
```

## Tools and libraries

Python (pandas, NumPy, scikit-learn, XGBoost, Matplotlib, seaborn), SQL, Power BI.

## Build status

1. Notebook 01, data generation and pattern checks: done
2. Notebook 02, EDA and RFM segmentation: done
3. Notebook 03, cohort retention: next
4. Notebook 04, churn prediction model
5. Notebook 05, CLV and recommendations
6. Power BI dashboard, five pages

## Contact

GitHub: [JoshuaPaul-lasisi](https://github.com/JoshuaPaul-lasisi)
