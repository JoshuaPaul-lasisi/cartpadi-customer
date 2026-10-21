"""Writes notebooks/01_data_audit_and_preparation.ipynb from the cells below."""

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_audit_and_preparation.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# 01 Data audit and preparation

## The business

Olist is a Brazilian company that connects small shops to the big online marketplaces. A shop lists its products through Olist, the customer buys on the marketplace, and Olist handles the order through to delivery.

The question for this project: why do so few Olist customers ever buy a second time, and what in their first order predicts whether they will?

## The data

Olist published about 100,000 real, anonymised orders placed between September 2016 and October 2018 on Kaggle as the "Brazilian E-Commerce Public Dataset by Olist". The licence is CC BY-NC-SA 4.0: free to use for non-commercial work with attribution, and derived data must be shared on the same terms. The raw files are not stored in this repository; download them from Kaggle into `data/raw/` (see the README).

Values are in Brazilian reais (R$).

This notebook checks the raw tables, fixes what needs fixing, and writes two clean tables that every later notebook reads:

1. `data/processed/orders_enriched.csv`: one row per order, with payment, review, delivery and product details joined on.
2. `data/processed/customers.csv`: one row per real customer, with their first-order details and whether they came back.
""")

md("""
## Data model

```mermaid
erDiagram
    customers ||--|| orders : "customer_id (one per order)"
    orders ||--|{ order_items : contains
    orders ||--|{ payments : "paid by"
    orders ||--o{ reviews : "reviewed in"
    products ||--o{ order_items : "sold as"
    sellers ||--o{ order_items : "sold by"
    customers {
        string customer_id PK
        string customer_unique_id
        string customer_state
    }
    orders {
        string order_id PK
        string customer_id FK
        string order_status
        datetime order_purchase_timestamp
        datetime order_delivered_customer_date
        datetime order_estimated_delivery_date
    }
    order_items {
        string order_id FK
        int order_item_id
        string product_id FK
        string seller_id FK
        float price
        float freight_value
    }
    payments {
        string order_id FK
        string payment_type
        int payment_installments
        float payment_value
    }
    reviews {
        string review_id
        string order_id FK
        int review_score
    }
```

The geolocation file is left out. It maps zip prefixes to coordinates, and state and city are already on the customer and seller tables.
""")

code("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd

import olist as ol

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.max_columns", 30)

raw = ol.load_raw()
pd.DataFrame({name: [len(df), df.shape[1]] for name, df in raw.items()}, index=["rows", "columns"]).T
""")

md("""
## 1. Audit

### 1.1 Who is a customer?

This is the most important check in the project. In Olist's data, `customer_id` is not a person. A new `customer_id` is created for every order, so each one appears exactly once. The real person is `customer_unique_id`. Counting customers by `customer_id` would show a 0% repeat rate and make the whole retention question disappear.
""")

code("""
customers, orders = raw["customers"], raw["orders"]
print(f"customer_id values:        {customers['customer_id'].nunique():,}")
print(f"customer_unique_id values: {customers['customer_unique_id'].nunique():,}")
print(f"Orders per customer_id:    {orders.groupby('customer_id').size().max()} at most")
per_person = customers.groupby("customer_unique_id").size()
print(f"People with more than one customer_id: {(per_person > 1).sum():,}")
""")

md("""
### 1.2 Nulls
""")

code("""
nulls = pd.concat({name: df.isna().sum() for name, df in raw.items()}).rename("nulls")
nulls = nulls[nulls > 0].to_frame()
nulls["share"] = [n / len(raw[t]) for (t, _), n in zip(nulls.index, nulls["nulls"])]
nulls
""")

md("""
Most gaps follow the order's status: an order that was never shipped has no carrier or delivery date. Review titles and messages are optional, so most are empty. The real gap is 610 products with no category, which become "unknown".
""")

code("""
delivered = orders[orders["order_status"] == "delivered"]
print("Delivered orders with no delivery date:", delivered["order_delivered_customer_date"].isna().sum())
print("Delivered orders with no approval date:", delivered["order_approved_at"].isna().sum())
print()
print(orders["order_status"].value_counts().to_string())
""")

md("""
### 1.3 Date range

The first months are thin: Olist was starting up in late 2016 and has almost no orders for November and December that year. The export also stops partway through September 2018, and nearly everything after August is cancelled. The analysis window is January 2017 to August 2018 (20 full months), with 31 August 2018 as the snapshot date.
""")

code("""
by_month = orders["order_purchase_timestamp"].dt.to_period("M").value_counts().sort_index()
outside = by_month[(by_month.index < "2017-01") | (by_month.index > "2018-08")]
print("Orders outside the window, by month:")
print(outside.to_string())
late = orders[orders["order_purchase_timestamp"] >= ol.SNAPSHOT_END]
print("\\nStatus of orders after the snapshot:", late["order_status"].value_counts().to_dict())
""")

md("""
### 1.4 Payments

An order can have several payment rows: a customer may pay part with a voucher and the rest by card. Each order gets one main payment type (the one that paid the most), its largest instalment count, and a flag for any voucher used. The payment total should match items plus freight.
""")

code("""
pay = raw["payments"]
print("Payment rows per order:", pay.groupby("order_id").size().value_counts().sort_index().head(6).to_dict())
print("Payment types:", pay["payment_type"].value_counts().to_dict())

items = raw["order_items"]
order_value = items.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)
paid = pay.groupby("order_id")["payment_value"].sum()
diff = (paid - order_value).dropna()
print(f"\\nOrders where payment differs from items + freight by more than R$1: {(diff.abs() > 1).sum():,} of {len(diff):,}")
print(f"Median difference on those: R${diff[diff.abs() > 1].median():,.2f}")
inst = pay.groupby("order_id")["payment_installments"].max()
print(f"Share of those paid in instalments: {(inst.reindex(diff[diff.abs() > 1].index) > 1).mean():.0%} "
      f"(all orders: {(inst > 1).mean():.0%})")
no_items = orders[~orders["order_id"].isin(items["order_id"])]
print(f"\\nOrders with no items: {len(no_items):,}", no_items["order_status"].value_counts().to_dict())
""")

md("""
A few hundred orders were paid more than their items and freight, mostly by small amounts. Nearly all of them were paid in instalments, against about half of all orders, so this is most likely card interest. Order value in this project is items plus freight, so interest is not counted as spend.

775 orders have no items. Almost all are cancelled or unavailable; the 8 others (created, invoiced or shipped with nothing in them) are treated as not completed.
""")

md("""
### 1.5 Reviews
""")

code("""
rev = raw["reviews"]
print("Reviews per order:", rev.groupby("order_id").size().value_counts().sort_index().to_dict())
print("Duplicated review_id values:", rev["review_id"].duplicated().sum())
print("Orders with no review:", (~orders["order_id"].isin(rev["order_id"])).sum())
print("\\nScore distribution:")
print(rev["review_score"].value_counts(normalize=True).sort_index().map("{:.1%}".format).to_string())
""")

md("""
A few hundred orders have more than one review (the customer reviewed, then reviewed again), and some review IDs are shared across orders that were reviewed together. For each order the most recent answered review is kept.
""")

md("""
### 1.6 Products and categories
""")

code("""
prod, tr = raw["products"], raw["category_translation"]
print("Products with no category:", prod["product_category_name"].isna().sum())
missing_tr = set(prod["product_category_name"].dropna()) - set(tr["product_category_name"])
print("Categories with no English name:", sorted(missing_tr))
print("Distinct categories:", prod["product_category_name"].nunique())
""")

md("""
Two categories have no English name in the translation file; they get one by hand. Products with no category are labelled "unknown".
""")

md("""
### 1.7 Split carts

Some customers place several orders within minutes of each other, usually because their basket had products from different sellers or they added something just after checking out. These are one shopping trip, not a customer coming back. A repeat purchase in this project means a completed order on a later calendar day than the first.
""")

code("""
m = orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
m = m[~m["order_status"].isin(["canceled", "unavailable"])].sort_values("order_purchase_timestamp")
m["seq"] = m.groupby("customer_unique_id").cumcount()
first_ts = m[m["seq"] == 0].set_index("customer_unique_id")["order_purchase_timestamp"]
second_ts = m[m["seq"] == 1].set_index("customer_unique_id")["order_purchase_timestamp"]
gap_hours = (second_ts - first_ts.reindex(second_ts.index)).dt.total_seconds() / 3600
print(f"People with a second completed order: {len(gap_hours):,}")
print(f"  of which within 24 hours of the first: {(gap_hours < 24).sum():,}")
print(f"  of which on the same calendar day:     {(second_ts.dt.date == first_ts.reindex(second_ts.index).dt.date).sum():,}")
""")

md("""
## 2. Build the clean tables

Rules applied:

1. Orders placed from 1 January 2017 to 31 August 2018 only.
2. A completed order is any order with items that was not cancelled and not unavailable. Orders still shipping at the snapshot count: the customer has paid.
3. Order value is items plus freight.
4. Lead category and seller are those of the most expensive item in the order.
5. Delivery days run from purchase to delivery; delay days from the estimated date to the actual one (positive means late).
""")

code("""
TRANSLATE_EXTRA = {"pc_gamer": "pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos": "kitchen_portables_and_food_prep"}
cat_en = dict(zip(tr["product_category_name"], tr["product_category_name_english"])) | TRANSLATE_EXTRA

o = orders.merge(customers, on="customer_id")
o = o[(o["order_purchase_timestamp"] >= ol.WINDOW_START) & (o["order_purchase_timestamp"] < ol.SNAPSHOT_END)].copy()

it = items.merge(prod[["product_id", "product_category_name"]], on="product_id", how="left")
it = it.merge(raw["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left")
it["category"] = it["product_category_name"].map(cat_en).fillna("unknown")
agg = it.groupby("order_id").agg(items_value=("price", "sum"), freight_value=("freight_value", "sum"),
                                 n_items=("order_item_id", "size"), n_sellers=("seller_id", "nunique"),
                                 n_categories=("category", "nunique"))
lead = it.sort_values(["order_id", "price"], ascending=[True, False]).drop_duplicates("order_id").set_index("order_id")

pay_main = pay.sort_values(["order_id", "payment_value"], ascending=[True, False]).drop_duplicates("order_id")
pay_agg = pay.groupby("order_id").agg(installments=("payment_installments", "max"),
                                      used_voucher=("payment_type", lambda s: int((s == "voucher").any())))
review = (rev.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
          .set_index("order_id")["review_score"])

oe = pd.DataFrame({
    "order_id": o["order_id"].to_numpy(),
    "customer_unique_id": o["customer_unique_id"].to_numpy(),
    "customer_state": o["customer_state"].to_numpy(),
    "customer_city": o["customer_city"].to_numpy(),
    "purchase_ts": o["order_purchase_timestamp"].to_numpy(),
    "order_status": o["order_status"].to_numpy(),
    "delivered_ts": o["order_delivered_customer_date"].to_numpy(),
    "estimated_delivery_ts": o["order_estimated_delivery_date"].to_numpy(),
})
oe = oe.join(agg, on="order_id")
oe["completed"] = ((~oe["order_status"].isin(["canceled", "unavailable"])) & oe["n_items"].notna()).astype(int)
oe[["items_value", "freight_value", "n_items", "n_sellers", "n_categories"]] = (
    oe[["items_value", "freight_value", "n_items", "n_sellers", "n_categories"]].fillna(0))
oe["order_value"] = oe["items_value"] + oe["freight_value"]
oe["freight_share"] = np.where(oe["order_value"] > 0, oe["freight_value"] / oe["order_value"], np.nan)
oe["lead_category"] = oe["order_id"].map(lead["category"])
oe["seller_state"] = oe["order_id"].map(lead["seller_state"])
oe["same_state_seller"] = (oe["seller_state"] == oe["customer_state"]).astype(int)
oe["payment_type"] = oe["order_id"].map(pay_main.set_index("order_id")["payment_type"])
oe = oe.join(pay_agg, on="order_id")
oe["review_score"] = oe["order_id"].map(review)
oe["delivery_days"] = (oe["delivered_ts"] - oe["purchase_ts"]).dt.total_seconds() / 86400
oe["delay_days"] = (oe["delivered_ts"] - oe["estimated_delivery_ts"]).dt.total_seconds() / 86400
oe["promised_days"] = (oe["estimated_delivery_ts"] - oe["purchase_ts"]).dt.total_seconds() / 86400
oe["is_late"] = np.where(oe["delivered_ts"].notna(), (oe["delay_days"] > 0).astype(float), np.nan)
oe = oe.sort_values("purchase_ts").reset_index(drop=True)

print(f"Orders in window: {len(oe):,}  (completed {oe['completed'].sum():,})")
oe.head(3).T
""")

md("""
### 2.1 Customer table

One row per `customer_unique_id`, built from completed orders only. The first order's details (category, payment, delivery, review) are carried on the customer row, because the first order is the only thing every customer has in common.

The repeat flag, `repeat_180d`, is 1 if the customer placed a completed order on a later day within 180 days of their first. It is only defined (`repeat_180d_observed = 1`) for customers whose first order was at least 180 days before the snapshot; for later customers the window hasn't closed yet.
""")

code("""
done = oe[oe["completed"] == 1].copy()
done["purchase_day"] = done["purchase_ts"].dt.normalize()

g = done.groupby("customer_unique_id")
cust = pd.DataFrame({
    "customer_state": g["customer_state"].first(),
    "customer_city": g["customer_city"].first(),
    "first_purchase": g["purchase_ts"].min(),
    "last_purchase": g["purchase_ts"].max(),
    "n_orders": g["order_id"].size(),
    "n_purchase_days": g["purchase_day"].nunique(),
    "total_spend": g["order_value"].sum(),
})

first_orders = done.drop_duplicates("customer_unique_id").set_index("customer_unique_id")
for col in ["order_id", "order_value", "freight_share", "n_items", "n_sellers", "lead_category", "seller_state",
            "same_state_seller", "payment_type", "installments", "used_voucher", "review_score",
            "delivery_days", "delay_days", "promised_days", "is_late"]:
    cust[f"first_{col}"] = first_orders[col]

first_day = cust["first_purchase"].dt.normalize()
later = done[done["purchase_day"] > done["customer_unique_id"].map(first_day)]
next_ts = later.groupby("customer_unique_id")["purchase_ts"].min()
cust["second_purchase"] = next_ts
cust["days_to_second"] = (cust["second_purchase"] - cust["first_purchase"]).dt.total_seconds() / 86400
cust["repeat_ever"] = cust["second_purchase"].notna().astype(int)
cust["repeat_180d_observed"] = (cust["first_purchase"] < ol.SNAPSHOT_END - pd.Timedelta(days=ol.REPEAT_WINDOW_DAYS)).astype(int)
cust["repeat_180d"] = np.where(cust["repeat_180d_observed"] == 1,
                               (cust["days_to_second"] <= ol.REPEAT_WINDOW_DAYS).astype(int), np.nan)
cust = cust.reset_index()

print(f"Customers: {len(cust):,}")
print(f"Ever made a repeat purchase (later day): {cust['repeat_ever'].mean():.2%}  ({cust['repeat_ever'].sum():,} people)")
print(f"Repeat within 180 days (customers with a full window): {cust['repeat_180d'].mean():.2%} "
      f"of {cust['repeat_180d_observed'].sum():,}")
print(f"Median days to second purchase: {cust['days_to_second'].median():.0f}")
""")

md("""
### 2.2 Sanity checks and export
""")

code("""
assert oe["order_id"].is_unique
assert cust["customer_unique_id"].is_unique
assert (cust["n_orders"] >= cust["n_purchase_days"]).all()
assert cust["total_spend"].sum().round(2) == done["order_value"].sum().round(2)
assert (cust["first_purchase"] >= ol.WINDOW_START).all() and (cust["last_purchase"] < ol.SNAPSHOT_END).all()

ol.PROCESSED.mkdir(parents=True, exist_ok=True)
oe.to_csv(ol.PROCESSED / "orders_enriched.csv", index=False)
cust.to_csv(ol.PROCESSED / "customers.csv", index=False)
print(f"orders_enriched.csv  {len(oe):,} rows, {oe.shape[1]} columns")
print(f"customers.csv        {len(cust):,} rows, {cust.shape[1]} columns")
""")

md("""
## What the audit found

1. `customer_id` is one per order. The real customer is `customer_unique_id`; using the wrong one hides every repeat purchase.
2. About a third of apparent second orders are split carts placed the same day. Counting them would overstate repeat buying by roughly half.
3. The usable window is January 2017 to August 2018. Late 2016 is a launch trickle and the September 2018 tail is mostly cancellations.
4. Payments, reviews and categories each needed a rule to get to one value per order: main payment type, latest review, English category name with "unknown" for blanks.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, NB)
print(f"Wrote {NB}")
