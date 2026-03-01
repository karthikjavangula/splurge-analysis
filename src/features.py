"""
Feature engineering pipeline.
Reads data/transactions.csv → computes behavioural features → saves data/features.csv
"""

import pandas as pd
import numpy as np
import os

# ── Category risk lookup (domain-defined, 0–1) ────────────────────────────────
CATEGORY_RISK = {
    "shopping_net":   0.90,
    "misc_net":       0.85,
    "entertainment":  0.80,
    "shopping_pos":   0.75,
    "food_dining":    0.65,
    "travel":         0.70,
    "personal_care":  0.55,
    "misc_pos":       0.60,
    "gas_transport":  0.30,
    "grocery_pos":    0.20,
    "health_fitness": 0.25,
    "home":           0.20,
    "kids_pets":      0.15,
}
IMPULSE_CATEGORIES = {"shopping_net", "misc_net", "entertainment",
                       "shopping_pos", "food_dining", "travel"}


def load_transactions(path: str = "data/transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    print(f"   Loaded {len(df):,} transactions")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["trans_date_trans_time"]

    # ── Temporal features ─────────────────────────────────────────────────────
    df["hour"]          = ts.dt.hour
    df["day_of_week"]   = ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"]  = ts.dt.day
    df["month"]         = ts.dt.month
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_late_night"] = (df["hour"].between(22, 23) | df["hour"].between(0, 2)).astype(int)
    df["is_end_of_month"]  = (df["day_of_month"] >= 25).astype(int)
    df["is_post_salary"]   = (df["day_of_month"] <= 3).astype(int)

    # ── Category features ─────────────────────────────────────────────────────
    df["category_risk"]   = df["category"].map(CATEGORY_RISK).fillna(0.4)
    df["is_impulse_cat"]  = df["category"].isin(IMPULSE_CATEGORIES).astype(int)

    # ── User-level aggregates (rolling) ───────────────────────────────────────
    df["date_only"] = ts.dt.date

    # per-user average spend
    user_avg = df.groupby("cc_num")["amt"].transform("mean")
    df["spend_vs_mean_ratio"] = (df["amt"] / user_avg.replace(0, 1)).clip(upper=5)

    # daily txn count per user
    daily_counts = df.groupby(["cc_num", "date_only"])["amt"].transform("count")
    df["daily_txn_count"] = daily_counts

    # user's historical % of purchases in impulse categories
    user_impulse_freq = df.groupby("cc_num")["is_impulse_cat"].transform("mean")
    df["impulse_cat_freq"] = user_impulse_freq

    # ── Impulse Risk Score (rule-based, 0–100) ────────────────────────────────
    df["impulse_risk_score"] = (
          0.30 * df["category_risk"]
        + 0.20 * df["is_late_night"]
        + 0.20 * (df["spend_vs_mean_ratio"] / 5)
        + 0.10 * df["is_weekend"]
        + 0.10 * df["is_end_of_month"]
        + 0.10 * df["impulse_cat_freq"]
    ) * 100

    df["impulse_risk_score"] = df["impulse_risk_score"].clip(0, 100).round(2)

    # ── Binary ground-truth label ─────────────────────────────────────────────
    df["impulse_label"] = (df["impulse_risk_score"] >= 40).astype(int)

    # ── Risk band ─────────────────────────────────────────────────────────────
    df["risk_band"] = pd.cut(
        df["impulse_risk_score"],
        bins=[-1, 29, 59, 79, 100],
        labels=["Low", "Moderate", "High", "Critical"]
    )

    df = df.drop(columns=["date_only"])
    return df


def main():
    os.makedirs("data", exist_ok=True)
    print("⚙️  Engineering features...")
    df = load_transactions()
    df = engineer_features(df)
    df.to_csv("data/features.csv", index=False)
    print(f"   ✅ Features saved → data/features.csv  (shape: {df.shape})")
    print(f"   Impulse rate: {df['impulse_label'].mean():.1%}")
    print(f"   Avg risk score: {df['impulse_risk_score'].mean():.1f}")


if __name__ == "__main__":
    main()