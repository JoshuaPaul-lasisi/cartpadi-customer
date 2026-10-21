"""
Shared helpers for the Olist retention notebooks: paths, analysis window,
loading the prepared tables, and one chart style for every figure.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

# Analysis window. Olist's first months (Sep to Dec 2016) are a handful of test
# orders, and the export stops partway through September 2018.
WINDOW_START = pd.Timestamp("2017-01-01")
SNAPSHOT = pd.Timestamp("2018-08-31")
SNAPSHOT_END = SNAPSHOT + pd.Timedelta(days=1)
REPEAT_WINDOW_DAYS = 180

# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------
# Categorical slots, used in this fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b9b8b0"  # context marks (everything that is not the focus)
# Single-hue sequential ramp (blue), light to dark.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("cartpadi_blue", BLUE_RAMP)


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def brl(x, decimals=1):
    """Short Brazilian real label: R$15.4M, R$160.6k, R$137."""
    for div, unit in [(1e9, "bn"), (1e6, "M"), (1e3, "k")]:
        if abs(x) >= div:
            return f"R${x / div:,.{decimals}f}{unit}"
    return f"R${x:,.0f}"


def brl_axis(ax, axis="y", decimals=0):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: brl(v, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def pct_axis(ax, axis="y", decimals=0):
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def hbar(ax, labels, values, color=SERIES[0], fmt=None, highlight=None):
    """Horizontal bars, largest at the top, value labels at the bar ends.
    `highlight` keeps one label in colour and greys out the rest."""
    labels, values = list(labels), np.asarray(values, dtype=float)
    y = np.arange(len(labels))[::-1]
    colors = [color if highlight is None or lab == highlight else NEUTRAL for lab in labels]
    ax.barh(y, values, color=colors, height=0.68, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    if fmt is not None:
        pad = values.max() * 0.01
        for yi, v in zip(y, values):
            ax.text(v + pad, yi, fmt(v), va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, values.max() * 1.15)
    return ax


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
ORDER_DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"]


def load_raw():
    """The Olist tables as published on Kaggle (geolocation is not used)."""
    t = {k: pd.read_csv(RAW / f) for k, f in RAW_FILES.items()}
    for col in ORDER_DATES:
        t["orders"][col] = pd.to_datetime(t["orders"][col])
    for col in ["review_creation_date", "review_answer_timestamp"]:
        t["reviews"][col] = pd.to_datetime(t["reviews"][col])
    return t


def load_orders():
    """Order-level table built in notebook 01: one row per order in the window."""
    return pd.read_csv(PROCESSED / "orders_enriched.csv",
                       parse_dates=["purchase_ts", "delivered_ts", "estimated_delivery_ts"])


def load_customers():
    """Customer-level table built in notebook 01: one row per real person."""
    return pd.read_csv(PROCESSED / "customers.csv", parse_dates=["first_purchase", "last_purchase"])


def wilson(successes, n, z=1.96):
    """95% Wilson confidence interval for a proportion. Returns (low, high)."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(df, by, flag, min_n=0):
    """Rate of a 0/1 flag by group, with counts and a 95% interval."""
    g = df.groupby(by, observed=True)[flag].agg(["sum", "size"])
    g = g[g["size"] >= min_n]
    g["rate"] = g["sum"] / g["size"]
    g["low"], g["high"] = wilson(g["sum"], g["size"])
    return g.rename(columns={"sum": "repeaters", "size": "customers"})


def rate_chart(ax, table, title, base=None, highlight_top=None):
    """Horizontal dot-and-interval chart of rates, highest at the top."""
    t = table.sort_values("rate")
    y = np.arange(len(t))
    ax.hlines(y, t["low"], t["high"], color=AXIS, linewidth=2.5, zorder=2)
    ax.scatter(t["rate"], y, s=40, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for yi, (idx, row) in zip(y, t.iterrows()):
        ax.text(row["high"] + 0.0006, yi, f"{row['rate']:.1%}  (n={int(row['customers']):,})", va="center",
                fontsize=8.5, color=INK_2)
    if base is not None:
        ax.axvline(base, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y, t.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    pct_axis(ax, axis="x", decimals=1)
    ax.set_xlim(0, t["high"].max() * 1.35)
    ax.set_title(title)
    return ax
