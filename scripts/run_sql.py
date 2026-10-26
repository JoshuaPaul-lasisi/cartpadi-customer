"""
Runs the SQL in sql/ against the raw Olist CSVs with DuckDB, prints each
result, and checks the key numbers against the notebook outputs.

Run from the repository root, after the notebooks have been run once
(the check reads data/processed/):
    python scripts/run_sql.py            print every query result
    python scripts/run_sql.py --check    also compare with the notebooks
"""

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"
PROCESSED = ROOT / "data" / "processed"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def statements(path):
    """Split a .sql file into statements (no semicolons inside strings in this project)."""
    text = "\n".join(line for line in path.read_text().splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in text.split(";") if s.strip()]


def run_all(con):
    for stmt in statements(SQL / "00_views.sql"):
        con.execute(stmt)
    results = {}
    for path in sorted(SQL.glob("0[1-9]_*.sql")):
        frames = [con.execute(s).df() for s in statements(path)]
        results[path.stem] = frames
        for i, df in enumerate(frames, 1):
            part = f" (part {i})" if len(frames) > 1 else ""
            print(f"\n=== {path.name}{part}: {len(df):,} rows ===")
            print(df.head(25).to_string(index=False))
    return results


def check(results):
    cust = pd.read_csv(PROCESSED / "customers.csv")
    orders = pd.read_csv(PROCESSED / "orders_enriched.csv")
    cohort = pd.read_csv(PROCESSED / "cohort_retention.csv")
    rfm = pd.read_csv(PROCESSED / "rfm_segments.csv")
    done = orders[orders["completed"] == 1]
    ok = True

    def compare(name, sql_value, nb_value, tol=0.0):
        nonlocal ok
        good = abs(sql_value - nb_value) <= tol
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {name:<38} SQL {sql_value:>14,.4f}   notebook {nb_value:>14,.4f}")

    print("\n=== Checks against the notebook outputs ===")
    k = results["01_headline_kpis"][0].iloc[0]
    obs = cust[cust["repeat_180d_observed"] == 1]
    compare("customers", k["customers"], len(cust))
    compare("completed orders", k["completed_orders"], len(done))
    compare("GMV (R$)", k["gmv_brl"], done["order_value"].sum(), tol=0.05)
    compare("repeat rate, ever", k["repeat_ever_rate"], cust["repeat_ever"].mean(), tol=0.0001)
    compare("repeat rate, 180 days", k["repeat_180d_rate"], obs["repeat_180d"].mean(), tol=0.0001)

    c = results["03_cohort_retention"][0].merge(cohort, on=["cohort_month", "months_since_first_purchase"],
                                                suffixes=("_sql", "_nb"))
    compare("cohort cells matched", len(c), len(cohort))
    compare("cohort max retention difference", (c["retention_rate_sql"] - c["retention_rate_nb"]).abs().max(), 0,
            tol=0.00001)

    # When two items in an order share the top price, SQL takes the lower item number and pandas
    # may take the other one. That moves 46 customers' first category, so rates can differ in
    # the fourth decimal place.
    cat = results["04_repeat_by_first_category"][0].set_index("category")
    nb_cat = obs.groupby("first_lead_category")["repeat_180d"].mean()
    compare("category max repeat-rate difference", (cat["repeat_rate"] - nb_cat.reindex(cat.index)).abs().max(), 0,
            tol=0.001)

    late = results["05_delivery_and_reviews"][1].set_index("customer_state")
    dl = done[done["delivered_ts"].notna()]
    nb_late = dl.groupby("customer_state")["is_late"].mean()
    compare("late rate max difference by state", (late["late_rate"] - nb_late.reindex(late.index)).abs().max(), 0,
            tol=0.0006)

    seg = results["06_rfm_segments"][0].set_index("segment")["customers"]
    nb_seg = rfm["segment"].value_counts()
    compare("RFM segment max count difference", (seg - nb_seg.reindex(seg.index)).abs().max(), 0, tol=50)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return ok


if __name__ == "__main__":
    os.chdir(ROOT)  # the views read data/raw/ by relative path
    con = duckdb.connect()
    res = run_all(con)
    if "--check" in sys.argv:
        sys.exit(0 if check(res) else 1)
