"""
Shared helpers for the CartPadi notebooks: paths, data loading and cleaning,
and one chart style so every figure in the project looks the same.
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

SNAPSHOT = pd.Timestamp("2025-12-31")
SNAPSHOT_END = SNAPSHOT + pd.Timedelta(days=1)  # orders are timestamps up to 23:59 on the snapshot
CHURN_WINDOW_DAYS = 90
TOP5_STATES = ["Lagos", "Abuja", "Rivers", "Kano", "Oyo"]

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


def naira(x, decimals=1):
    """Short naira label: N8.8bn, N412.5M, N76.8k."""
    for div, unit in [(1e9, "bn"), (1e6, "M"), (1e3, "k")]:
        if abs(x) >= div:
            return f"N{x / div:,.{decimals}f}{unit}"
    return f"N{x:,.0f}"


def naira_axis(ax, axis="y", decimals=0):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: naira(v, decimals))
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
def load_raw():
    """The five raw tables exactly as the generator wrote them."""
    return {
        "customers": pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"]),
        "orders": pd.read_csv(RAW / "orders.csv", parse_dates=["order_date"]),
        "order_items": pd.read_csv(RAW / "order_items.csv"),
        "products": pd.read_csv(RAW / "products.csv"),
        "sessions": pd.read_csv(RAW / "sessions.csv", parse_dates=["session_date"]),
    }


def clean(tables):
    """Apply the fixes found in the notebook 02 audit:
    1. drop order rows exported twice
    2. fill missing city with 'Unknown'
    3. store delivery_days as a nullable integer (empty for cancelled orders)"""
    t = {k: v.copy() for k, v in tables.items()}
    t["orders"] = t["orders"].drop_duplicates().reset_index(drop=True)
    t["orders"]["delivery_days"] = t["orders"]["delivery_days"].astype("Int64")
    t["customers"]["city"] = t["customers"]["city"].fillna("Unknown")
    return t


def load_clean():
    return clean(load_raw())


def first_order_category(order_items, products):
    """Category of the first line of each order (the order's lead category)."""
    it = order_items.merge(products[["product_id", "category"]], on="product_id")
    return it.sort_values("item_id").groupby("order_id")["category"].first()
