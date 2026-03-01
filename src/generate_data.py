"""
Synthetic data generator — matches Kaggle priyamchoksi credit card schema.
Generates ~350K transactions across 500 users with realistic impulse patterns.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os, random

fake = Faker()
np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
N_USERS        = 500
N_TRANSACTIONS = 350_000
START_DATE     = datetime(2022, 1, 1)
END_DATE       = datetime(2024, 12, 31)

CATEGORIES = [
    "grocery_pos", "shopping_net", "entertainment", "gas_transport",
    "food_dining", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_pos", "travel"
]

# Impulse-prone categories → higher base probability of being generated
IMPULSE_CATS = {
    "shopping_net": 0.15, "entertainment": 0.12, "misc_net": 0.10,
    "shopping_pos": 0.12, "food_dining": 0.10, "travel": 0.06,
    "personal_care": 0.05, "misc_pos": 0.07, "gas_transport": 0.05,
    "grocery_pos": 0.10, "health_fitness": 0.04, "home": 0.03, "kids_pets": 0.01
}
CAT_WEIGHTS = list(IMPULSE_CATS.values())

JOBS = [
    "Software Engineer", "Marketing Manager", "Student", "Freelancer",
    "Sales Executive", "Data Analyst", "Teacher", "Graphic Designer",
    "Product Manager", "Content Creator", "Consultant", "Nurse"
]

AMT_PARAMS = {
    "grocery_pos":    (800,  400),
    "shopping_net":   (1800, 1200),
    "entertainment":  (600,  400),
    "gas_transport":  (400,  150),
    "food_dining":    (500,  300),
    "health_fitness": (900,  500),
    "home":           (2500, 2000),
    "kids_pets":      (700,  400),
    "misc_net":       (1200, 800),
    "misc_pos":       (600,  400),
    "personal_care":  (500,  250),
    "shopping_pos":   (1500, 1000),
    "travel":         (5000, 4000),
}


def generate_users(n: int) -> pd.DataFrame:
    rows = []
    for uid in range(1, n + 1):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=32)
        profile = np.random.choice(
            ["conservative", "night_owl", "weekend_splurger", "high_risk"],
            p=[0.30, 0.25, 0.25, 0.20]
        )
        rows.append({
            "cc_num":  f"CARD{uid:05d}",
            "gender":  np.random.choice(["M", "F"]),
            "dob":     dob.strftime("%Y-%m-%d"),
            "job":     np.random.choice(JOBS),
            "city":    fake.city(),
            "state":   fake.state_abbr(),
            "profile": profile,
        })
    return pd.DataFrame(rows)


def sample_hour(profile: str, is_weekend: bool) -> int:
    """Sample transaction hour based on user behavioural profile."""
    if profile == "night_owl":
        weights = [0.005]*18 + [0.03, 0.05, 0.08, 0.12, 0.15, 0.13]  # 18–23
        weights += [0.08, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]         # 0–6 next day → flatten to 24
        weights = (np.array(weights[:24]) / sum(weights[:24])).tolist()
    elif profile == "weekend_splurger" and is_weekend:
        weights = [0.01]*9 + [0.06, 0.08, 0.09, 0.10, 0.10, 0.09, 0.08, 0.07, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01]
        weights = (np.array(weights[:24]) / sum(weights[:24])).tolist()
    elif profile == "high_risk":
        weights = [0.03]*7 + [0.04, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.07, 0.07, 0.07, 0.07, 0.05, 0.04, 0.04, 0.03]
        weights = (np.array(weights[:24]) / sum(weights[:24])).tolist()
    else:  # conservative
        weights = [0.005]*7 + [0.04, 0.08, 0.10, 0.11, 0.10, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.005]
        weights = (np.array(weights[:24]) / sum(weights[:24])).tolist()
    return np.random.choice(range(24), p=weights)


def generate_amount(category: str, profile: str) -> float:
    mu, sigma = AMT_PARAMS[category]
    if profile == "high_risk":
        mu *= 1.6
    elif profile == "conservative":
        mu *= 0.7
    amt = np.random.normal(mu, sigma)
    return max(round(abs(amt), 2), 10.0)


def generate_transactions(users: pd.DataFrame, n: int) -> pd.DataFrame:
    total_days = (END_DATE - START_DATE).days
    rows = []

    for _ in range(n):
        user = users.sample(1).iloc[0]
        day_offset = np.random.randint(0, total_days)
        base_date  = START_DATE + timedelta(days=day_offset)
        is_weekend = base_date.weekday() >= 5
        hour       = sample_hour(user["profile"], is_weekend)
        minute     = np.random.randint(0, 60)
        second     = np.random.randint(0, 60)
        ts         = base_date.replace(hour=hour, minute=minute, second=second)

        cat = np.random.choice(CATEGORIES, p=CAT_WEIGHTS)
        amt = generate_amount(cat, user["profile"])

        rows.append({
            "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num":    user["cc_num"],
            "merchant":  fake.company(),
            "category":  cat,
            "amt":       amt,
            "gender":    user["gender"],
            "dob":       user["dob"],
            "job":       user["job"],
            "city":      user["city"],
            "state":     user["state"],
            "is_fraud":  0,
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data", exist_ok=True)
    print("⚙️  Generating users...")
    users = generate_users(N_USERS)
    users.to_csv("data/users.csv", index=False)
    print(f"   ✅ {len(users)} users saved → data/users.csv")

    print("⚙️  Generating transactions...")
    txns = generate_transactions(users, N_TRANSACTIONS)
    txns = txns.sort_values("trans_date_trans_time").reset_index(drop=True)
    txns.to_csv("data/transactions.csv", index=False)
    print(f"   ✅ {len(txns):,} transactions saved → data/transactions.csv")


if __name__ == "__main__":
    main()