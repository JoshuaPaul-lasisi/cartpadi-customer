"""Writes notebooks/01_data_generation.ipynb from the cells below."""

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_generation.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# 01 Data generation

## Company brief

CartPadi Nigeria Ltd. is a fictional pan-Nigerian B2C e-commerce marketplace. It sells electronics, phones, fashion, beauty, home and sports products, ships to 24 states, and takes card, bank transfer, USSD, pay on delivery and its own CartPadi wallet. Lagos and Abuja are its biggest markets.

The business question for this project: which customers stop buying, why, and how much revenue walks out with them?

## Data disclaimer

Every row in this project is synthetic. CartPadi is not a real company and has no relation to any existing brand; product brands are invented. The script `scripts/01_data_generation.py` is the provenance of the data. It runs on a fixed seed (42), so the same five CSV files come out every time. This notebook runs that script and shows what it produced."""),
    md("""## Data model

```mermaid
erDiagram
    customers ||--o{ orders : places
    customers ||--o{ sessions : "browses (when logged in)"
    orders ||--|{ order_items : contains
    products ||--o{ order_items : "sold as"
    customers {
        string customer_id PK
        date signup_date
        string state
        string acquisition_channel
        int is_churned
    }
    orders {
        string order_id PK
        string customer_id FK
        datetime order_date
        string order_status
        string payment_method
        float order_total_ngn
    }
    order_items {
        string item_id PK
        string order_id FK
        string product_id FK
        int quantity
        float unit_price_ngn
    }
    products {
        string product_id PK
        string category
        float base_price_ngn
    }
    sessions {
        string session_id PK
        string customer_id FK
        datetime session_date
        int converted
    }
```

Snapshot date is 31 December 2025. A customer is churned when they have placed no non-cancelled order in the 90 days up to the snapshot."""),
    md("""## How the generator works

1. Calendar. Each day gets a trading intensity: November 2.4, December 1.95, April 1.65, January 0.85 and about 1.0 for normal months, with a small weekend lift. Repeat-purchase gaps run on this calendar, so a 28-day gap in a normal month takes about 11 days in November.
2. Customers. 25,000 signups spread over 2024 and 2025, following a softer version of the same calendar plus 30% growth. Each gets a state (Lagos 43%, Abuja 14%), channel, device, preferred payment method and a favourite category.
3. Repeat odds. Each customer's chance of a second order within 90 days is the channel's target rate, scaled by payment method, signup quarter (Q4 deal-hunters 36% lower than Q1 and Q2 before other effects), region and favourite category. A bad first order (cancelled, returned or slow) cuts those odds.
4. Orders. Repeaters split into occasional buyers (two to four orders) and loyal buyers who keep a personal buying rhythm until they drop out. About four in ten repeaters are seasonal returners who only come back months later.
5. Order lines. One to four lines per order. The lead line usually comes from the favourite category; add-ons lean cheap. Prices rise about 28% over the two years and discounts peak in November.
6. Sessions. A 14.5% sample of orders gets a converted session; the rest of the 480,000 sessions are browsing by known customers (tailing off after their last order) and anonymous visitors.
7. Noise. About 0.15% of order rows are written twice and about 0.8% of customers lose their city, so the notebook 02 audit has real problems to find."""),
    code("""import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().parent
result = subprocess.run([sys.executable, str(ROOT / "scripts" / "01_data_generation.py")],
                        capture_output=True, text=True, check=True)
print(result.stdout)"""),
    code("""import pandas as pd

RAW = ROOT / "data" / "raw"
for name in ["customers", "orders", "order_items", "products", "sessions"]:
    df = pd.read_csv(RAW / f"{name}.csv", nrows=5)
    print(f"{name}.csv")
    display(df)"""),
    md("""## Pattern check

`scripts/validate_patterns.py` measures each injected pattern on the files just written. The later notebooks rediscover these from the data; this check only confirms the generator did its job."""),
    code("""result = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_patterns.py")],
                        capture_output=True, text=True, check=True)
print(result.stdout)"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, NB)
print(f"Wrote {NB}")
