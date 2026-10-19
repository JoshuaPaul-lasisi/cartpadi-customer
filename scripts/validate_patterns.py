"""
Checks that the injected patterns show up in data/raw/*.csv.

Run after scripts/01_data_generation.py:
    python scripts/validate_patterns.py
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SNAPSHOT = pd.Timestamp("2025-12-31")

customers = pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"])
orders = pd.read_csv(RAW / "orders.csv", parse_dates=["order_date"]).drop_duplicates()
items = pd.read_csv(RAW / "order_items.csv")
products = pd.read_csv(RAW / "products.csv")
sessions = pd.read_csv(RAW / "sessions.csv", parse_dates=["session_date"])

live = orders[orders["order_status"] != "cancelled"]


def pct(x):
    return f"{100 * x:5.1f}%"


print("Row counts (orders after de-duplication)")
for name, df in [("customers", customers), ("orders", orders), ("order_items", items),
                 ("products", products), ("sessions", sessions)]:
    print(f"  {name:<12} {len(df):>9,}")

print("\nOrders per customer (all statuses)")
n = orders.groupby("customer_id").size().reindex(customers["customer_id"], fill_value=0)
print(f"  1 order    {pct((n == 1).mean())}   target 55%")
print(f"  2-4 orders {pct(n.between(2, 4).mean())}   target 25%")
print(f"  5+ orders  {pct((n >= 5).mean())}   target 20%")

o = orders.sort_values("order_date")
o["seq"] = o.groupby("customer_id").cumcount()
first = o[o["seq"] == 0].set_index("customer_id")
second = o[o["seq"] == 1].set_index("customer_id")["order_date"]
gap = (second - first["order_date"]).dt.days
print(f"\nMedian days to second order: {gap.median():.0f}   target 28")

# 90-day retention: second order within 90 days of the first, for customers
# whose first order is at least 90 days before the snapshot.
mature = first[first["order_date"] <= SNAPSHOT - pd.Timedelta(days=90)].copy()
mature["ret90"] = gap.reindex(mature.index).le(90).fillna(False)
mature = mature.join(customers.set_index("customer_id"))
print(f"\n90-day retention overall: {pct(mature['ret90'].mean())}")

print("\n90-day retention by acquisition channel")
targets = {"referral": 45, "email": 40, "direct": 38, "organic_search": 32, "influencer": None,
           "paid_social": 20}
r = mature.groupby("acquisition_channel")["ret90"].mean()
for ch, t in targets.items():
    print(f"  {ch:<16} {pct(r[ch])}   target {t if t else '-'}{'%' if t else ''}")

print("\n90-day retention by first-order payment method")
targets = {"CartPadi_wallet": 40, "card": 28, "bank_transfer": 25, "USSD": 25, "pay_on_delivery": 17}
r = mature.groupby("payment_method")["ret90"].mean()
for pm, t in targets.items():
    print(f"  {pm:<16} {pct(r[pm])}   target {t}%")

q = mature["signup_date"].dt.quarter
q12 = mature.loc[q.isin([1, 2]), "ret90"].mean()
q4 = mature.loc[q == 4, "ret90"].mean()
print(f"\nSignup Q1-Q2 retention {pct(q12)}, Q4 {pct(q4)}, Q4 is {100 * (1 - q4 / q12):.0f}% lower   target 30%")

print("\nMonthly orders index (normal months = 1.0)")
m = orders.groupby([orders["order_date"].dt.year, orders["order_date"].dt.month]).size()
for year in (2024, 2025):
    s = m.loc[year]
    base = s.loc[[3, 5, 6, 7, 8, 9]].mean()
    print(f"  {year}: Jan {s[1] / base:.2f}  Apr {s[4] / base:.2f}  Nov {s[11] / base:.2f}  Dec {s[12] / base:.2f}")
print(f"  YoY order growth 2025 vs 2024: {100 * (m.loc[2025].sum() / m.loc[2024].sum() - 1):.0f}%")

# Category effects. First category = category of the first line of the first order.
it = items.merge(products[["product_id", "category"]], on="product_id")
lead = it.sort_values("item_id").groupby("order_id")["category"].first()
first["first_category"] = first["order_id"].map(lead)
spend = live.groupby("customer_id")["order_total_ngn"].sum()
cat = first[["first_category"]].join(spend.rename("ltv")).join(n.rename("n_orders"))
cat["ltv"] = cat["ltv"].fillna(0)
by = cat.groupby("first_category").agg(ltv=("ltv", "mean"), repeat=("n_orders", lambda x: (x >= 2).mean()))
print("\nBy first category: mean LTV (NGN) and repeat rate")
for c, row in by.sort_values("ltv", ascending=False).iterrows():
    print(f"  {c:<24} {row['ltv']:>12,.0f}   repeat {pct(row['repeat'])}")
print(f"  Electronics / Fashion LTV ratio: {by.loc['Electronics', 'ltv'] / by.loc['Fashion', 'ltv']:.2f}   target 3.2")

# Category repurchase: bought the first category again on a later order.
cat_orders = it.merge(orders[["order_id", "customer_id"]], on="order_id")[["customer_id", "order_id", "category"]]
cat_orders = cat_orders.drop_duplicates(["customer_id", "order_id", "category"])
k = cat_orders.groupby(["customer_id", "category"]).size()
first_cat = first["first_category"]
again = pd.Series([k.get((cid, c), 0) >= 2 for cid, c in first_cat.items()], index=first_cat.index)
print("\nCategory repurchase rate (bought first category again)")
for c, v in again.groupby(first_cat).mean().sort_values(ascending=False).items():
    print(f"  {c:<24} {pct(v)}")

aov = live.assign(cat=live["order_id"].map(lead)).groupby("cat")["order_total_ngn"].mean()
print("\nAverage order value by lead category (NGN)")
for c, v in aov.sort_values(ascending=False).items():
    print(f"  {c:<24} {v:>12,.0f}")

print("\nGeography")
cust_share = customers["state"].value_counts(normalize=True)
rev = live.merge(customers[["customer_id", "state"]], on="customer_id").groupby("state")["order_total_ngn"].sum()
rev_share = rev / rev.sum()
for st in cust_share.index[:6]:
    print(f"  {st:<10} customers {pct(cust_share[st])}   revenue {pct(rev_share[st])}")
top5 = ["Lagos", "Abuja", "Rivers", "Kano", "Oyo"]
dd = orders.groupby(orders["delivery_state"].isin(top5))["delivery_days"].mean()
print(f"  Mean delivery days: Lagos {orders.loc[orders['delivery_state'] == 'Lagos', 'delivery_days'].mean():.1f}, "
      f"top-5 {dd[True]:.1f}, outside top-5 {dd[False]:.1f}")
in_top5 = mature["state"].isin(top5)
print(f"  90-day retention: top-5 states {pct(mature.loc[in_top5, 'ret90'].mean())}, "
      f"others {pct(mature.loc[~in_top5, 'ret90'].mean())}")

print("\nOther checks")
print(f"  Churn rate: {pct(customers['is_churned'].mean())}")
print(f"  Order status mix: {orders['order_status'].value_counts(normalize=True).round(3).to_dict()}")
print(f"  Session conversion rate: {pct(sessions['converted'].mean())}")
print(f"  Anonymous sessions: {pct(sessions['customer_id'].isna().mean())}")
gmv = live["order_total_ngn"].sum()
print(f"  GMV (non-cancelled): NGN {gmv / 1e9:,.2f}bn, AOV NGN {live['order_total_ngn'].mean():,.0f}")
