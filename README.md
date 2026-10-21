# Olist customer retention analysis

Why do 98% of customers on a Brazilian e-commerce platform never buy a second time, and what in their first order predicts whether they will? A retention study on 94,703 real customers and 97,905 orders.

## The business and the data

Olist is a Brazilian company that connects small shops to large online marketplaces. Shops list through Olist, customers buy on the marketplace, and Olist manages the order through to delivery.

The data is the "Brazilian E-Commerce Public Dataset by Olist", published by Olist on Kaggle: about 100,000 real, anonymised orders from September 2016 to October 2018, across nine linked tables (customers, orders, items, payments, reviews, products, sellers, category names and geolocation). Values are in Brazilian reais (R$).

Licence: CC BY-NC-SA 4.0. The data is free for non-commercial use with attribution to Olist, and anything derived from it is shared under the same licence. The raw files are not stored in this repository.

## Key findings

1. 97.8% of customers bought on a single day. Only 1.98% came back within 180 days, and about 2.2% ever did. Under 1% of any monthly cohort buys again in any later month.
2. About a third of apparent second orders are split carts placed the same day. Counting customers by `customer_id` hides repeat buying completely, because Olist issues a new `customer_id` for every order.
3. Late parcels wreck reviews. More than three days late, 60% to 76% of customers leave one or two stars. Late orders are 8% of deliveries but a third of all bad reviews.
4. Nothing in the first order moves the repeat rate much. The first order's category is the strongest signal: bed, bath and table, sports and furniture buyers return at up to 2.7 times the rate of gadget and car-part buyers. Review score and delivery speed show smaller real gaps, mostly explained by geography.
5. The first order says little about whether a customer will return. The best model (random forest, tested out of time on later customers) reaches a ROC AUC of 0.58. It is useful for one thing: its top 10% of scores returned at 1.8 times the average rate.
6. Customer value is almost entirely the first order. Raising the 180-day repeat rate from 2% to 3% would add about R$129,000 of GMV a year (1%), so retention spend has to be cheap: email, timing and category targeting, with any voucher tested against a hold-out group first.

![Summary](reports/figures/05_executive_summary.png)

## Project structure

```
  notebooks/
    01_data_audit_and_preparation.ipynb   audit of the raw tables, builds the clean tables
    02_eda_rfm_segmentation.ipynb         revenue, geography, delivery and reviews; RFM segments
    03_cohort_retention_analysis.ipynb    cohort heatmap, time to second purchase, what predicts a return
    04_repeat_purchase_model.ipynb        predicts a second purchase within 180 days from the first order
    05_clv_and_recommendations.ipynb      customer value, size of the prize, playbook, one-page summary
  scripts/
    build_notebook_0X.py                  writes each notebook from source (edit these, not the .ipynb)
  src/
    olist.py                              shared paths, loading, chart style and helpers
  data/
    raw/                                  Olist CSVs from Kaggle (not committed)
    processed/                            clean tables and notebook outputs (not committed, rebuilt by the notebooks)
  reports/figures/                        every chart the notebooks save
  models/                                 trained model
```

## How to run

1. Download the dataset from Kaggle: search "Brazilian E-Commerce Public Dataset by Olist" (olistbr/brazilian-ecommerce).
2. Unzip the CSV files into `data/raw/`.
3. Install and run:

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_audit_and_preparation.ipynb
```

Run the notebooks in order, 01 first: it writes the clean tables the others read. To change a notebook, edit its builder in `scripts/` and rebuild:

```
python scripts/build_notebook_02.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_rfm_segmentation.ipynb
```

## Tools and libraries

Python (pandas, NumPy, scikit-learn, XGBoost, Matplotlib), SQL, Power BI.

## Build status

1. Notebook 01, data audit and preparation: done
2. Notebook 02, EDA and RFM segmentation: done
3. Notebook 03, cohort retention: done
4. Notebook 04, repeat-purchase model: done
5. Notebook 05, CLV and recommendations: done
6. Power BI dashboard: next. It reads the tables in `data/processed/` (customers, orders_enriched, rfm_segments, cohort_retention, repeat_scores, clv_scores).

## Contact

GitHub: [JoshuaPaul-lasisi](https://github.com/JoshuaPaul-lasisi)
