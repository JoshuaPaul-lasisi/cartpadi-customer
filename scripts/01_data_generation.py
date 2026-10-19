"""
CartPadi Nigeria Ltd. synthetic data generator.

CartPadi is a FICTIONAL pan-Nigerian B2C e-commerce marketplace. It has no
relation to any real company or brand. Every row this script writes is
synthetic. This script is the provenance of the data: run it with the same
seed and you get the same five CSVs, byte for byte.

Outputs (data/raw/):
    customers.csv     25,000 customers who signed up between Jan 2024 and Dec 2025
    products.csv      2,000 products in 6 categories
    orders.csv        one row per order
    order_items.csv   one row per order line
    sessions.csv      about a 15% random sample of the site session log

Snapshot date: 2025-12-31. A customer is churned when they have placed no
non-cancelled order in the 90 days up to the snapshot.

Usage:
    python scripts/01_data_generation.py
"""

from pathlib import Path
import uuid

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

START = pd.Timestamp("2024-01-01")
SNAPSHOT = pd.Timestamp("2025-12-31")
END = SNAPSHOT + pd.Timedelta(days=1)  # exclusive upper bound for timestamps
CHURN_WINDOW_DAYS = 90

N_CUSTOMERS = 25_000
N_PRODUCTS = 2_000
SESSION_SAMPLE_RATE = 0.145  # share of the full session log exported
TARGET_SESSIONS = 480_000


# =============================================================================
# 1. Calendar: seasonality and time warping
# =============================================================================
# Monthly order multipliers relative to a normal trading month (= 1.0).
MONTH_MULT = {1: 0.85, 2: 0.95, 3: 1.0, 4: 1.65, 5: 1.0, 6: 0.95,
              7: 0.95, 8: 1.0, 9: 1.0, 10: 1.05, 11: 2.4, 12: 1.95}
DOW_MULT = np.array([0.95, 0.95, 0.97, 1.0, 1.08, 1.06, 0.99])  # Mon..Sun

DAYS = pd.date_range(START, SNAPSHOT, freq="D")
N_DAYS = len(DAYS)
day_intensity = (np.array([MONTH_MULT[m] for m in DAYS.month])
                 * DOW_MULT[DAYS.dayofweek])
# Business growth: signups run about 30% higher by late 2025.
growth = np.linspace(1.0, 1.3, N_DAYS)

# Cumulative intensity lets repeat-purchase gaps run on a "busy calendar":
# a 28-day gap in a normal month takes about 11 calendar days in November.
cum_intensity = np.concatenate([[0.0], np.cumsum(day_intensity)])
day_edges = np.arange(N_DAYS + 1, dtype=float)


def to_warped(day_float):
    """Calendar day offset (days since START) -> warped time."""
    return np.interp(day_float, day_edges, cum_intensity)


def from_warped(w):
    """Warped time -> calendar day offset."""
    return np.interp(w, cum_intensity, day_edges)


W_END = cum_intensity[-1]

# Signups follow a softer version of the order calendar.
SIGNUP_SEASONALITY = 0.6

# Hour-of-day profile for order and session timestamps (Lagos time).
HOUR_W = np.array([0.6, 0.35, 0.2, 0.15, 0.15, 0.3, 0.8, 1.6, 2.4, 3.0, 3.4, 3.8,
                   4.4, 4.6, 4.0, 3.6, 3.6, 4.0, 4.6, 5.2, 5.6, 5.4, 4.2, 2.0])
HOUR_W = HOUR_W / HOUR_W.sum()


def random_time_of_day(n):
    hours = rng.choice(24, size=n, p=HOUR_W)
    secs = rng.integers(0, 3600, size=n)
    return hours * 3600 + secs  # seconds after midnight


def price_index(day_offset):
    """List prices rise about 28% from Jan 2024 to Dec 2025 (1.0 at snapshot)."""
    return 0.78 + 0.22 * (np.asarray(day_offset) / (N_DAYS - 1))


# =============================================================================
# 2. Reference data: geography, channels, payment
# =============================================================================
STATES = {
    # state: (customer share, cities, region tier)
    "Lagos": (0.43, ["Ikeja", "Lekki", "Surulere", "Yaba", "Ikorodu", "Ajah",
                     "Victoria Island", "Festac", "Alimosho", "Agege", "Gbagada",
                     "Ojodu"], "lagos"),
    "Abuja": (0.14, ["Garki", "Wuse", "Maitama", "Gwarinpa", "Kubwa", "Lugbe",
                     "Jabi", "Asokoro"], "abuja"),
    "Rivers": (0.075, ["Port Harcourt", "Obio-Akpor", "Eleme"], "top5"),
    "Kano": (0.065, ["Kano", "Nassarawa", "Fagge", "Tarauni"], "top5"),
    "Oyo": (0.06, ["Ibadan", "Ogbomosho", "Oyo"], "top5"),
    "Ogun": (0.035, ["Abeokuta", "Ota", "Sagamu", "Ijebu-Ode"], "other"),
    "Enugu": (0.025, ["Enugu", "Nsukka"], "other"),
    "Delta": (0.022, ["Warri", "Asaba", "Sapele"], "other"),
    "Edo": (0.02, ["Benin City", "Auchi"], "other"),
    "Kaduna": (0.02, ["Kaduna", "Zaria"], "other"),
    "Anambra": (0.018, ["Onitsha", "Awka", "Nnewi"], "other"),
    "Akwa Ibom": (0.012, ["Uyo", "Eket"], "other"),
    "Imo": (0.012, ["Owerri", "Orlu"], "other"),
    "Kwara": (0.011, ["Ilorin", "Offa"], "other"),
    "Plateau": (0.01, ["Jos", "Bukuru"], "other"),
    "Abia": (0.009, ["Aba", "Umuahia"], "other"),
    "Ondo": (0.008, ["Akure", "Ondo"], "other"),
    "Osun": (0.008, ["Osogbo", "Ile-Ife"], "other"),
    "Cross River": (0.007, ["Calabar"], "other"),
    "Benue": (0.005, ["Makurdi"], "other"),
    "Ekiti": (0.004, ["Ado-Ekiti"], "other"),
    "Bauchi": (0.004, ["Bauchi"], "other"),
    "Sokoto": (0.004, ["Sokoto"], "other"),
    "Borno": (0.002, ["Maiduguri"], "other"),
}
STATE_NAMES = list(STATES)
STATE_P = np.array([v[0] for v in STATES.values()])
STATE_P = STATE_P / STATE_P.sum()
STATE_TIER = {s: v[2] for s, v in STATES.items()}

# Repeat-purchase multiplier and basket "affluence" by region tier.
TIER_RETENTION = {"lagos": 1.10, "abuja": 1.08, "top5": 0.95, "other": 0.72}
TIER_PRICE_ALPHA = {"lagos": 0.52, "abuja": 0.46, "top5": 0.55, "other": 0.62}
# Delivery days range (inclusive) by region tier, outside peak season.
TIER_DELIVERY = {"lagos": (2, 5), "abuja": (2, 6), "top5": (4, 8), "other": (8, 14)}

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_P = [0.24, 0.41, 0.21, 0.10, 0.04]
GENDERS = ["F", "M", "Prefer not to say"]
GENDER_P = [0.52, 0.44, 0.04]

CHANNELS = ["organic_search", "paid_social", "referral", "email", "influencer", "direct"]
CHANNEL_P = [0.24, 0.24, 0.13, 0.10, 0.10, 0.19]
# Target share of customers who buy again within 90 days of their first order.
CHANNEL_RET90 = {"referral": 0.49, "email": 0.42, "direct": 0.38,
                 "organic_search": 0.33, "influencer": 0.24, "paid_social": 0.20}

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_P = [0.72, 0.23, 0.05]

PAYMENTS = ["card", "bank_transfer", "USSD", "pay_on_delivery", "CartPadi_wallet"]
PAYMENT_P_LAGOS = [0.38, 0.18, 0.07, 0.20, 0.17]
PAYMENT_P_OTHER = [0.30, 0.20, 0.12, 0.29, 0.09]
PAYMENT_RET90 = {"CartPadi_wallet": 0.37, "card": 0.28, "bank_transfer": 0.255,
                 "USSD": 0.255, "pay_on_delivery": 0.19}

# Signup quarter effect: Q4 deal-hunters retain 30% worse than Q1-Q2 signups.
QUARTER_RETENTION = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.64}


# =============================================================================
# 3. Product catalogue
# =============================================================================
CATEGORIES = ["Electronics", "Fashion", "Beauty and Skincare", "Home and Kitchen",
              "Phones and Accessories", "Sports and Fitness"]

# category: {subcategory: (min price, max price, catalogue weight, item nouns)}
# Nouns run cheapest to dearest; a product gets the noun that fits its price.
SUBCATEGORIES = {
    "Electronics": {
        "Earphones": (9_000, 65_000, 0.17, ["Neckband Earphones", "Wireless Earbuds", "Pro Earbuds"]),
        "Headphones": (25_000, 260_000, 0.10, ["Over-Ear Headphones", "Noise Cancelling Headphones"]),
        "Bluetooth Speakers": (18_000, 190_000, 0.13, ["Mini Speaker", "Portable Speaker", "Party Speaker"]),
        "Power Banks": (12_000, 48_000, 0.14, ["Slim Power Bank", "Power Bank 20000mAh", "Power Bank 30000mAh"]),
        "Smart Watches": (28_000, 240_000, 0.12, ["Fitness Smart Watch", "Smart Watch"]),
        "Laptops": (420_000, 1_950_000, 0.12, ["Laptop 14in", "Laptop 15.6in", "Ultrabook"]),
        "Televisions": (210_000, 1_350_000, 0.13, ["LED TV 32in", "Smart TV 43in", "Smart TV 55in"]),
        "Gaming": (180_000, 950_000, 0.09, ["Gaming Controller Bundle", "Game Console"]),
    },
    "Phones and Accessories": {
        "Smartphones": (115_000, 1_650_000, 0.52, ["Smartphone 64GB", "Smartphone 128GB", "Smartphone 256GB"]),
        "Feature Phones": (12_000, 38_000, 0.08, ["Dual SIM Feature Phone", "Torch Phone"]),
        "Tablets": (140_000, 850_000, 0.12, ["Kids Tablet 8in", "Tablet 10in"]),
        "Chargers and Cables": (3_500, 28_000, 0.12, ["Fast Charger", "USB-C Cable", "Car Charger"]),
        "Phone Cases": (2_500, 15_000, 0.10, ["Silicone Case", "Rugged Case", "Flip Cover"]),
        "Screen Protectors": (2_000, 8_500, 0.06, ["Tempered Glass", "Privacy Screen Guard"]),
    },
    "Fashion": {
        "Men's Shoes": (14_000, 90_000, 0.14, ["Leather Loafers", "Sneakers", "Corporate Shoes"]),
        "Women's Shoes": (12_000, 80_000, 0.14, ["Block Heels", "Flat Sandals", "Sneakers"]),
        "Dresses": (9_000, 68_000, 0.14, ["Ankara Midi Dress", "Bodycon Dress", "Kaftan"]),
        "Men's Shirts": (6_500, 38_000, 0.13, ["Senator Shirt", "Oxford Shirt", "Polo Shirt"]),
        "Ankara Fabrics": (8_000, 48_000, 0.12, ["Ankara 6 Yards", "Adire Fabric", "Lace Fabric 5 Yards"]),
        "Bags": (9_000, 95_000, 0.12, ["Tote Bag", "Crossbody Bag", "Backpack"]),
        "Jewellery and Watches": (5_000, 65_000, 0.09, ["Beaded Necklace", "Wristwatch", "Earring Set"]),
        "T-Shirts": (4_500, 19_000, 0.12, ["Cotton T-Shirt", "Graphic Tee", "Oversized Tee"]),
    },
    "Beauty and Skincare": {
        "Moisturisers": (3_500, 30_000, 0.18, ["Shea Butter Moisturiser", "Daily Face Cream", "Body Lotion"]),
        "Serums": (5_500, 40_000, 0.14, ["Vitamin C Serum", "Niacinamide Serum", "Hyaluronic Serum"]),
        "Sunscreen": (4_500, 27_000, 0.12, ["SPF 50 Sunscreen", "Tinted Sunscreen"]),
        "Hair Care": (3_000, 32_000, 0.18, ["Hair Growth Oil", "Leave-In Conditioner", "Edge Control"]),
        "Fragrances": (9_000, 98_000, 0.12, ["Eau de Parfum", "Body Mist", "Oud Perfume Oil"]),
        "Makeup": (3_000, 42_000, 0.16, ["Matte Lipstick", "Foundation", "Setting Powder"]),
        "Body Wash": (2_500, 16_000, 0.10, ["Black Soap Body Wash", "Exfoliating Scrub"]),
    },
    "Home and Kitchen": {
        "Blenders": (26_000, 125_000, 0.14, ["Blender 1.5L", "Smoothie Blender", "Grinder Blender"]),
        "Air Fryers": (58_000, 230_000, 0.10, ["Air Fryer 4L", "Air Fryer 6L"]),
        "Cookware Sets": (22_000, 155_000, 0.14, ["Non-Stick Pot Set", "Stainless Pot Set"]),
        "Bedding": (13_000, 85_000, 0.16, ["Duvet Set", "Bedsheet Set", "Pillow Pair"]),
        "Storage": (3_000, 26_000, 0.16, ["Food Flask Set", "Storage Containers", "Laundry Basket"]),
        "Rechargeable Fans": (26_000, 95_000, 0.12, ["Rechargeable Fan 16in", "Solar Fan"]),
        "Kitchen Utensils": (2_500, 21_000, 0.18, ["Knife Set", "Mortar and Pestle", "Utensil Set"]),
    },
    "Sports and Fitness": {
        "Dumbbells": (11_000, 95_000, 0.16, ["Dumbbell Pair 5kg", "Adjustable Dumbbells"]),
        "Yoga Mats": (6_500, 32_000, 0.16, ["Yoga Mat 6mm", "Exercise Mat"]),
        "Jerseys": (13_000, 48_000, 0.20, ["Football Jersey", "Training Jersey"]),
        "Treadmills": (380_000, 1_250_000, 0.07, ["Folding Treadmill", "Motorised Treadmill"]),
        "Sports Shoes": (16_000, 98_000, 0.22, ["Running Shoes", "Football Boots", "Training Shoes"]),
        "Fitness Trackers": (16_000, 85_000, 0.19, ["Fitness Band", "Heart Rate Tracker"]),
    },
}
CATEGORY_PRODUCT_SHARE = {"Electronics": 0.17, "Fashion": 0.26, "Beauty and Skincare": 0.18,
                          "Home and Kitchen": 0.16, "Phones and Accessories": 0.13,
                          "Sports and Fitness": 0.10}

# Fictional brands. Any match to a real brand is coincidental.
BRANDS = {
    "Electronics": ["Voltani", "Harmattan Audio", "Lumora", "Okapi Tech", "Zenvo", "Kpakpa Sound", "Arewa Digital"],
    "Phones and Accessories": ["Zumari", "Kelvo", "Ayaba Mobile", "Novik", "Ranti", "Odera"],
    "Fashion": ["Adire House", "Ankara Lane", "Eko Threads", "Aso Studio", "Owambe Wear", "Gidi Street"],
    "Beauty and Skincare": ["Shea Glow", "Ori Naturals", "Melanin Lab", "Adunni Skin", "Ewa Botanics", "Zobo Beauty"],
    "Home and Kitchen": ["Calabash Home", "Ile Essentials", "Pepperpot", "Ofada Living", "Kitchenmate"],
    "Sports and Fitness": ["Savannah Sport", "Kinetix", "Sahel Active", "Ogun Runners", "Eagle Fit"],
}
MODEL_WORDS = ["Pro", "Lite", "Max", "Plus", "Classic", "Air", "Prime", "Neo", "Go", "X"]

# First-purchase category mix (category affinity) and its effect on behaviour.
AFFINITY_P = {"Fashion": 0.26, "Beauty and Skincare": 0.18, "Phones and Accessories": 0.17,
              "Home and Kitchen": 0.16, "Electronics": 0.14, "Sports and Fitness": 0.09}
CATEGORY_RETENTION = {"Beauty and Skincare": 1.30, "Electronics": 1.32, "Fashion": 1.00,
                      "Home and Kitchen": 0.92, "Phones and Accessories": 0.85,
                      "Sports and Fitness": 0.85}
# Order-rate multiplier for repeat buyers (Fashion and Beauty buy often).
CATEGORY_FREQUENCY = {"Fashion": 1.35, "Beauty and Skincare": 1.25, "Home and Kitchen": 0.9,
                      "Sports and Fitness": 0.85, "Electronics": 0.95, "Phones and Accessories": 0.7}
# Chance that a later order's lead item comes from the affinity category.
CATEGORY_STICKINESS = {"Electronics": 0.65, "Beauty and Skincare": 0.80, "Fashion": 0.85,
                       "Phones and Accessories": 0.45, "Home and Kitchen": 0.55,
                       "Sports and Fitness": 0.55}
# Shift on the price-aversion exponent for lead lines: negative = pricier basket.
CATEGORY_ALPHA_SHIFT = {"Phones and Accessories": -0.25, "Electronics": -0.15, "Fashion": 0.55,
                        "Beauty and Skincare": 0.10, "Home and Kitchen": 0.0, "Sports and Fitness": 0.05}


def build_products():
    rows = []
    n_by_cat = {c: int(round(N_PRODUCTS * s)) for c, s in CATEGORY_PRODUCT_SHARE.items()}
    n_by_cat["Fashion"] += N_PRODUCTS - sum(n_by_cat.values())
    pid = 1
    for cat in CATEGORIES:
        subs = SUBCATEGORIES[cat]
        sub_names = list(subs)
        sub_w = np.array([subs[s][2] for s in sub_names])
        sub_pick = rng.choice(sub_names, size=n_by_cat[cat], p=sub_w / sub_w.sum())
        for sub in sub_pick:
            lo, hi, _, nouns = subs[sub]
            q = rng.uniform()
            price = np.exp(np.log(lo) + q * (np.log(hi) - np.log(lo)))
            noun = nouns[min(int(q * len(nouns)), len(nouns) - 1)]
            price = round(price / 100) * 100 - 1 if price > 10_000 else round(price / 50) * 50
            brand = rng.choice(BRANDS[cat])
            name = f"{brand} {noun} {rng.choice(MODEL_WORDS)}"
            rows.append((f"PRD-{pid:05d}", name, cat, sub, brand, float(price)))
            pid += 1
    products = pd.DataFrame(rows, columns=["product_id", "product_name", "category",
                                           "subcategory", "brand", "base_price_ngn"])
    # Identical names happen (same brand, noun, model word). Add a variant code.
    dup = products.duplicated("product_name", keep=False)
    products.loc[dup, "product_name"] += " " + products.loc[dup].groupby("product_name").cumcount().add(1).map(
        lambda i: f"V{i}")
    # Bestseller effect: a few products sell far more than the rest.
    products["_popularity"] = rng.lognormal(0, 0.8, len(products))
    return products


# =============================================================================
# 4. Customers
# =============================================================================
def build_customers():
    signup_w = day_intensity ** SIGNUP_SEASONALITY * growth
    signup_day = rng.choice(N_DAYS, size=N_CUSTOMERS, p=signup_w / signup_w.sum())
    signup_day.sort()

    state = rng.choice(STATE_NAMES, size=N_CUSTOMERS, p=STATE_P)
    city = np.array([rng.choice(STATES[s][1]) for s in state], dtype=object)

    customers = pd.DataFrame({
        "customer_id": [str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63)),
                                      version=4)) for _ in range(N_CUSTOMERS)],
        "signup_date": DAYS[signup_day],
        "state": state,
        "city": city,
        "age_group": rng.choice(AGE_GROUPS, size=N_CUSTOMERS, p=AGE_P),
        "gender": rng.choice(GENDERS, size=N_CUSTOMERS, p=GENDER_P),
        "acquisition_channel": rng.choice(CHANNELS, size=N_CUSTOMERS, p=CHANNEL_P),
        "first_device": rng.choice(DEVICES, size=N_CUSTOMERS, p=DEVICE_P),
    })
    customers["_signup_day"] = signup_day
    customers["_tier"] = customers["state"].map(STATE_TIER)

    is_lagos_abuja = customers["_tier"].isin(["lagos", "abuja"]).to_numpy()
    pay = np.where(is_lagos_abuja,
                   rng.choice(PAYMENTS, size=N_CUSTOMERS, p=PAYMENT_P_LAGOS),
                   rng.choice(PAYMENTS, size=N_CUSTOMERS, p=PAYMENT_P_OTHER))
    customers["_pref_payment"] = pay

    aff_names = list(AFFINITY_P)
    aff_p = np.array(list(AFFINITY_P.values()))
    customers["_affinity"] = rng.choice(aff_names, size=N_CUSTOMERS, p=aff_p / aff_p.sum())
    return customers


def retention_propensity(customers):
    """Probability of a second order within 90 days of the first (before the
    first-order experience adjustment). Each factor is normalised to a
    population mean of 1, so channel marginals land near CHANNEL_RET90."""
    def norm_factor(values):
        values = np.asarray(values, dtype=float)
        return values / values.mean()

    quarter = DAYS[customers["_signup_day"]].quarter
    f_channel = customers["acquisition_channel"].map(CHANNEL_RET90).to_numpy()
    f_pay = norm_factor(customers["_pref_payment"].map(PAYMENT_RET90))
    f_quarter = norm_factor(pd.Series(quarter).map(QUARTER_RETENTION))
    f_tier = norm_factor(customers["_tier"].map(TIER_RETENTION))
    f_cat = norm_factor(customers["_affinity"].map(CATEGORY_RETENTION))
    return f_channel * f_pay * f_quarter * f_tier * f_cat


# =============================================================================
# 5. Orders (headers)
# =============================================================================
# Time to second order. Most repeaters come back within a few weeks
# (lognormal on the trading calendar); a minority are "seasonal returners"
# who only come back 4 to 14 months later, often for the next sale season.
SECOND_GAP_MEDIAN, SECOND_GAP_SIGMA = 17.0, 0.95
P_LATE_RETURNER = 0.42
LATE_GAP_RANGE = (110, 420)
P_SECOND_WITHIN_90 = 0.60   # calibrated share of repeaters back within 90 days
P_LOYAL_GIVEN_REPEAT = 0.47
LOYAL_DROPOUT = 0.035       # per-order chance a 5+ order customer stops buying


def first_order_experience(tier, status, delivery_days):
    """A bad first order (cancelled, returned, slow) dents the urge to return."""
    if status == "cancelled":
        return 0.55
    if status == "returned":
        return 0.65
    if delivery_days <= 4:
        return 1.08
    if delivery_days <= 8:
        return 1.0
    return 0.88


def draw_status(payment, lead_category):
    p_cancel = 0.11 if payment == "pay_on_delivery" else 0.045
    p_return = 0.085 if lead_category == "Fashion" else 0.035
    u = rng.random()
    if u < p_cancel:
        return "cancelled"
    if u < p_cancel + p_return:
        return "returned"
    return "delivered"


def draw_delivery_days(tier, day_offset):
    lo, hi = TIER_DELIVERY[tier]
    d = int(rng.integers(lo, hi + 1))
    month = DAYS[min(int(day_offset), N_DAYS - 1)].month
    day = DAYS[min(int(day_offset), N_DAYS - 1)].day
    if (month == 11 and day >= 15) or month == 12:  # peak-season backlog
        d += int(rng.integers(1, 4))
    return d


def draw_payment(pref, tier):
    if rng.random() < 0.72:
        return pref
    p = PAYMENT_P_LAGOS if tier in ("lagos", "abuja") else PAYMENT_P_OTHER
    return PAYMENTS[rng.choice(len(PAYMENTS), p=p)]


def draw_lead_category(affinity, first):
    if first or rng.random() < CATEGORY_STICKINESS[affinity]:
        return affinity
    names = list(AFFINITY_P)
    p = np.array(list(AFFINITY_P.values()))
    return names[rng.choice(len(names), p=p / p.sum())]


def build_orders(customers):
    p90 = retention_propensity(customers)
    order_rows = []

    cols = zip(customers["_signup_day"], customers["_tier"], customers["_pref_payment"],
               customers["_affinity"])
    for i, (signup_day, tier, pref, aff) in enumerate(cols):
        # First order: same day for most, otherwise within a couple of weeks.
        lag = 0.0 if rng.random() < 0.62 else rng.exponential(4.0)
        first_day = min(signup_day + lag, N_DAYS - 1)
        order_days = [first_day]

        # First-order attributes decide part of the repeat odds.
        pay0 = pref if rng.random() < 0.85 else draw_payment(pref, tier)
        status0 = draw_status(pay0, aff)
        dd0 = draw_delivery_days(tier, first_day)
        p_ret90 = p90[i] * first_order_experience(tier, status0, dd0)
        p_repeat = min(p_ret90 / P_SECOND_WITHIN_90, 0.97)

        if rng.random() < p_repeat:
            freq = CATEGORY_FREQUENCY[aff]
            loyalty = np.clip(np.sqrt(p_ret90 / 0.31), 0.5, 1.6)
            loyal = rng.random() < min(P_LOYAL_GIVEN_REPEAT * loyalty, 0.9)
            if rng.random() < P_LATE_RETURNER:
                gap = rng.uniform(*LATE_GAP_RANGE)
            else:
                gap = rng.lognormal(np.log(SECOND_GAP_MEDIAN), SECOND_GAP_SIGMA)
            w = to_warped(first_day) + gap
            if loyal:
                cadence = rng.lognormal(np.log(18 / freq), 0.35)  # personal buying rhythm
                n = 1
                while w < W_END:
                    order_days.append(from_warped(w))
                    n += 1
                    if n >= 5 and rng.random() < LOYAL_DROPOUT:
                        break
                    w += rng.lognormal(np.log(cadence), 0.55)
            else:
                n_total = rng.choice([2, 3, 4], p=[0.52, 0.30, 0.18])
                while w < W_END and len(order_days) < n_total:
                    order_days.append(from_warped(w))
                    w += rng.lognormal(np.log(48 / freq), 0.9)

        for k, d in enumerate(order_days):
            if k == 0:
                pay, status, dd = pay0, status0, dd0
                lead = aff
            else:
                pay = draw_payment(pref, tier)
                lead = draw_lead_category(aff, first=False)
                status = draw_status(pay, lead)
                dd = draw_delivery_days(tier, d)
            order_rows.append((i, d, k, pay, status, dd, lead))

    orders = pd.DataFrame(order_rows, columns=["_cust_idx", "_day", "_seq", "payment_method",
                                               "order_status", "delivery_days", "_lead_category"])
    # Timestamps: calendar day + time of day. First orders cannot precede signup.
    tod = random_time_of_day(len(orders))
    day_int = np.floor(orders["_day"].to_numpy()).astype(int)
    orders["order_date"] = START + pd.to_timedelta(day_int, unit="D") + pd.to_timedelta(tod, unit="s")
    orders = orders.sort_values(["order_date", "_cust_idx"]).reset_index(drop=True)
    orders["_day_int"] = (orders["order_date"].dt.normalize() - START).dt.days

    cust = customers.iloc[orders["_cust_idx"]].reset_index(drop=True)
    orders["customer_id"] = cust["customer_id"]
    orders["_tier"] = cust["_tier"]
    orders["_affinity"] = cust["_affinity"]

    # Most orders ship to the customer's own state; a few are gifts elsewhere.
    gift = rng.random(len(orders)) < 0.04
    orders["delivery_state"] = np.where(gift, rng.choice(STATE_NAMES, size=len(orders), p=STATE_P),
                                        cust["state"])
    # Gift deliveries follow the destination's delivery times.
    for idx in np.flatnonzero(gift):
        orders.at[idx, "delivery_days"] = draw_delivery_days(STATE_TIER[orders.at[idx, "delivery_state"]],
                                                             orders.at[idx, "_day_int"])
    orders.loc[orders["order_status"] == "cancelled", "delivery_days"] = np.nan
    orders["delivery_days"] = orders["delivery_days"].astype("Int64")
    orders["order_id"] = [f"CP-{i:07d}" for i in range(1, len(orders) + 1)]
    return orders


# =============================================================================
# 6. Order items
# =============================================================================
ITEMS_PER_ORDER_P = [0.30, 0.30, 0.22, 0.18]  # 1, 2, 3, 4 lines


def item_discount_pct(day_int):
    """Discount percentage per line, deepest around Black Friday."""
    n = len(day_int)
    dates = DAYS[np.minimum(day_int, N_DAYS - 1)]
    month = dates.month.to_numpy()
    p_disc = np.select([month == 11, month == 12, month == 4], [0.62, 0.45, 0.32], 0.18)
    hi = np.select([month == 11, month == 12, month == 4], [40, 25, 20], 15)
    lo = np.where(month == 11, 10, 5)
    pct = np.round(rng.uniform(lo, hi) / 5) * 5
    return np.where(rng.random(n) < p_disc, pct, 0.0)


def build_order_items(orders, products):
    n_lines = rng.choice([1, 2, 3, 4], size=len(orders), p=ITEMS_PER_ORDER_P)
    line_order = np.repeat(np.arange(len(orders)), n_lines)
    line_no = np.concatenate([np.arange(k) for k in n_lines])

    lead_cat = orders["_lead_category"].to_numpy()[line_order]
    tier = orders["_tier"].to_numpy()[line_order]

    # Lines after the first mostly stay in the lead category, the rest roam.
    names = list(AFFINITY_P)
    p = np.array(list(AFFINITY_P.values()))
    roam = rng.choice(names, size=len(line_order), p=p / p.sum())
    stay = rng.random(len(line_order)) < 0.65
    line_cat = np.where((line_no == 0) | stay, lead_cat, roam)

    product_idx = np.empty(len(line_order), dtype=int)
    prices = products["base_price_ngn"].to_numpy()
    pop = products["_popularity"].to_numpy()
    for cat in CATEGORIES:
        cat_idx = np.flatnonzero(products["category"].to_numpy() == cat)
        for t, alpha in TIER_PRICE_ALPHA.items():
            for is_lead in (True, False):
                mask = (line_cat == cat) & (tier == t) & ((line_no == 0) == is_lead)
                if not mask.any():
                    continue
                # Add-on lines lean cheaper (accessories, small items).
                a = alpha + CATEGORY_ALPHA_SHIFT[cat] if is_lead else alpha + 0.7
                w = pop[cat_idx] * prices[cat_idx] ** (-a)
                product_idx[mask] = rng.choice(cat_idx, size=mask.sum(), p=w / w.sum())

    unit_base = prices[product_idx]
    cheap_cat = np.isin(products["category"].to_numpy()[product_idx], ["Fashion", "Beauty and Skincare"])
    qty = np.where(cheap_cat,
                   rng.choice([1, 2, 3], size=len(line_order), p=[0.74, 0.19, 0.07]),
                   rng.choice([1, 2, 3], size=len(line_order), p=[0.92, 0.07, 0.01]))
    qty = np.where(unit_base > 150_000, 1, qty)

    day_int = orders["_day_int"].to_numpy()[line_order]
    noise = rng.uniform(0.97, 1.03, len(line_order))
    unit_price = np.round(unit_base * price_index(day_int) * noise / 50) * 50
    disc = item_discount_pct(day_int)

    items = pd.DataFrame({
        "item_id": [f"IT-{i:08d}" for i in range(1, len(line_order) + 1)],
        "order_id": orders["order_id"].to_numpy()[line_order],
        "product_id": products["product_id"].to_numpy()[product_idx],
        "quantity": qty,
        "unit_price_ngn": unit_price,
        "discount_pct": disc,
    })
    gross = items["quantity"] * items["unit_price_ngn"]
    items["_discount_ngn"] = gross * items["discount_pct"] / 100
    items["_net_ngn"] = gross - items["_discount_ngn"]
    return items


# =============================================================================
# 7. Sessions (about 15% sample of the session log)
# =============================================================================
SESSION_CHANNEL_P_ANON = [0.30, 0.34, 0.08, 0.03, 0.10, 0.15]
SESSION_CHANNEL_P_KNOWN = [0.22, 0.16, 0.06, 0.24, 0.06, 0.26]


def build_sessions(customers, orders):
    # Converted sessions: one per sampled order, ending at checkout.
    conv = orders.sample(frac=SESSION_SAMPLE_RATE, random_state=SEED)
    n_conv = len(conv)
    conv_pages = 5 + rng.poisson(8, n_conv)
    conv_dur = np.round(conv_pages * rng.lognormal(np.log(1.1), 0.35, n_conv), 1)
    cust_conv = customers.set_index("customer_id").loc[conv["customer_id"]]
    conv_device = np.where(rng.random(n_conv) < 0.6, cust_conv["first_device"],
                           rng.choice(DEVICES, size=n_conv, p=[0.55, 0.38, 0.07]))
    conv_channel = np.where(rng.random(n_conv) < 0.4, cust_conv["acquisition_channel"],
                            rng.choice(CHANNELS, size=n_conv, p=SESSION_CHANNEL_P_KNOWN))
    converted = pd.DataFrame({
        "customer_id": conv["customer_id"].to_numpy(),
        "session_date": (conv["order_date"] - pd.to_timedelta(conv_dur, unit="m")).dt.floor("s").to_numpy(),
        "channel": conv_channel,
        "device": conv_device,
        "pages_viewed": conv_pages,
        "duration_min": conv_dur,
        "converted": 1,
    })

    # Browsing sessions that did not convert.
    n_browse = TARGET_SESSIONS - n_conv
    n_anon = int(n_browse * 0.32)
    n_known = n_browse - n_anon

    stats = orders[orders["order_status"] != "cancelled"].groupby("customer_id").agg(
        n_orders=("order_id", "size"), last_day=("_day_int", "max"))
    stats = customers.set_index("customer_id")[["_signup_day", "first_device", "acquisition_channel"]].join(stats)
    stats["n_orders"] = stats["n_orders"].fillna(1)
    stats["last_day"] = stats["last_day"].fillna(stats["_signup_day"])
    # Browsing tails off after the last order; lapsed customers drift away.
    tail = rng.exponential(45, len(stats))
    stats["browse_end"] = np.minimum(stats["last_day"] + tail, N_DAYS - 1)
    weight = (2 + stats["n_orders"]) * (1 + (stats["browse_end"] - stats["_signup_day"]) / 120)
    k = rng.multinomial(n_known, (weight / weight.sum()).to_numpy())

    idx = np.repeat(np.arange(len(stats)), k)
    w_lo = to_warped(stats["_signup_day"].to_numpy()[idx])
    w_hi = to_warped(stats["browse_end"].to_numpy()[idx] + 1)
    known_day = np.floor(from_warped(rng.uniform(w_lo, w_hi))).astype(int)

    anon_w = day_intensity * growth
    anon_day = rng.choice(N_DAYS, size=n_anon, p=anon_w / anon_w.sum())

    browse_day = np.concatenate([known_day, anon_day])
    n_b = len(browse_day)
    known_ids = stats.index.to_numpy()[idx]
    known_dev = stats["first_device"].to_numpy()[idx]
    known_ch = stats["acquisition_channel"].to_numpy()[idx]
    device = np.concatenate([
        np.where(rng.random(n_known) < 0.7, known_dev, rng.choice(DEVICES, size=n_known, p=[0.76, 0.19, 0.05])),
        rng.choice(DEVICES, size=n_anon, p=[0.80, 0.16, 0.04]),
    ])
    channel = np.concatenate([
        np.where(rng.random(n_known) < 0.35, known_ch,
                 rng.choice(CHANNELS, size=n_known, p=SESSION_CHANNEL_P_KNOWN)),
        rng.choice(CHANNELS, size=n_anon, p=SESSION_CHANNEL_P_ANON),
    ])
    pages = np.where(rng.random(n_b) < 0.32, 1, 1 + rng.geometric(0.28, n_b))
    dur = np.round(np.maximum(0.1, pages * rng.lognormal(np.log(0.8), 0.5, n_b)), 1)
    browse = pd.DataFrame({
        "customer_id": np.concatenate([known_ids, np.full(n_anon, None, dtype=object)]),
        "session_date": START + pd.to_timedelta(browse_day, unit="D")
        + pd.to_timedelta(random_time_of_day(n_b), unit="s"),
        "channel": channel,
        "device": device,
        "pages_viewed": pages,
        "duration_min": dur,
        "converted": 0,
    })

    sessions = pd.concat([converted, browse], ignore_index=True)
    sessions = sessions[sessions["session_date"] >= START]
    sessions = sessions.sort_values("session_date").reset_index(drop=True)
    sessions.insert(0, "session_id", [f"SES-{i:07d}" for i in range(1, len(sessions) + 1)])
    return sessions


# =============================================================================
# 8. Assemble, add realistic data-quality noise, export
# =============================================================================
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Building products ...")
    products = build_products()
    print("Building customers ...")
    customers = build_customers()
    print("Building orders ...")
    orders = build_orders(customers)
    print("Building order items ...")
    items = build_order_items(orders, products)

    totals = items.groupby("order_id")[["_net_ngn", "_discount_ngn"]].sum()
    orders = orders.join(totals, on="order_id")
    orders["order_total_ngn"] = orders["_net_ngn"].round(2)
    orders["discount_ngn"] = orders["_discount_ngn"].round(2)

    print("Building sessions ...")
    sessions = build_sessions(customers, orders)

    # Churn flag: no non-cancelled order in the 90 days up to the snapshot.
    live = orders[orders["order_status"] != "cancelled"]
    last_order = live.groupby("customer_id")["order_date"].max()
    cutoff = END - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    customers["is_churned"] = (customers["customer_id"].map(last_order).fillna(START) < cutoff).astype(int)

    # Known data-quality issues for the EDA audit to find:
    #   1. about 0.15% of order rows exported twice (a known ETL bug)
    #   2. about 0.8% of customers have no city captured
    dupes = orders.sample(frac=0.0015, random_state=SEED + 1)
    orders_out = pd.concat([orders, dupes]).sort_values("order_date", kind="stable")
    no_city = rng.random(len(customers)) < 0.008
    customers.loc[no_city, "city"] = None

    customers_out = customers[["customer_id", "signup_date", "state", "city", "age_group", "gender",
                               "acquisition_channel", "first_device", "is_churned"]]
    orders_out = orders_out[["order_id", "customer_id", "order_date", "order_status", "payment_method",
                             "order_total_ngn", "discount_ngn", "delivery_days", "delivery_state"]]
    items_out = items[["item_id", "order_id", "product_id", "quantity", "unit_price_ngn", "discount_pct"]]
    products_out = products[["product_id", "product_name", "category", "subcategory", "brand",
                             "base_price_ngn"]]

    customers_out.to_csv(RAW_DIR / "customers.csv", index=False, date_format="%Y-%m-%d")
    products_out.to_csv(RAW_DIR / "products.csv", index=False)
    orders_out.to_csv(RAW_DIR / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    items_out.to_csv(RAW_DIR / "order_items.csv", index=False)
    sessions.to_csv(RAW_DIR / "sessions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    print("\nRows written")
    for name, df in [("customers", customers_out), ("products", products_out), ("orders", orders_out),
                     ("order_items", items_out), ("sessions", sessions)]:
        print(f"  {name:<12} {len(df):>9,}")
    print(f"\nFiles in {RAW_DIR}")


if __name__ == "__main__":
    main()
