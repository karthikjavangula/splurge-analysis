"""
Impulse Risk Score engine.
Can be imported by the dashboard for on-the-fly scoring of a single transaction.
"""

import numpy as np
import pandas as pd

CATEGORY_RISK = {
    "shopping_net":   0.90, "misc_net":       0.85,
    "entertainment":  0.80, "shopping_pos":   0.75,
    "food_dining":    0.65, "travel":         0.70,
    "personal_care":  0.55, "misc_pos":       0.60,
    "gas_transport":  0.30, "grocery_pos":    0.20,
    "health_fitness": 0.25, "home":           0.20,
    "kids_pets":      0.15,
}

IMPULSE_CATEGORIES = {
    "shopping_net", "misc_net", "entertainment",
    "shopping_pos", "food_dining", "travel"
}


def compute_risk_score(
    category: str,
    amt: float,
    hour: int,
    day_of_week: int,
    day_of_month: int,
    user_avg_spend: float,
    impulse_hist_pct: float,
) -> dict:
    """
    Compute the Impulse Risk Score for a single prospective transaction.
    Returns a dict with the score, band, and per-trigger breakdown.
    """
    cat_risk      = CATEGORY_RISK.get(category, 0.40)
    is_late_night = int(hour >= 22 or hour <= 2)
    is_weekend    = int(day_of_week >= 5)
    is_eom        = int(day_of_month >= 25)
    is_post_sal   = int(day_of_month <= 3)
    svmr          = min(amt / max(user_avg_spend, 1), 5) / 5   # normalised 0-1
    imp_freq      = impulse_hist_pct / 100                      # convert % → 0-1

    score = (
          0.30 * cat_risk
        + 0.20 * is_late_night
        + 0.20 * svmr
        + 0.10 * is_weekend
        + 0.10 * is_eom
        + 0.10 * imp_freq
    ) * 100

    score = round(min(max(score, 0), 100), 1)

    if score < 30:
        band, color = "Low Risk",      "#2ecc71"
    elif score < 60:
        band, color = "Moderate Risk", "#f1c40f"
    elif score < 80:
        band, color = "High Risk",     "#e67e22"
    else:
        band, color = "Critical Risk", "#e74c3c"

    triggers = {
        "Category Risk":          round(cat_risk * 100, 1),
        "Late Night Purchase":    is_late_night * 100,
        "Spend vs Personal Avg":  round(svmr * 100, 1),
        "Weekend Purchase":       is_weekend * 100,
        "End-of-Month Pressure":  is_eom * 100,
        "Impulse History":        round(imp_freq * 100, 1),
    }

    active_triggers = [k for k, v in triggers.items() if v > 0]

    return {
        "score":           score,
        "band":            band,
        "color":           color,
        "triggers":        triggers,
        "active_triggers": active_triggers,
        "is_late_night":   bool(is_late_night),
        "is_weekend":      bool(is_weekend),
        "is_eom":          bool(is_eom),
        "is_post_salary":  bool(is_post_sal),
        "category":        category,
        "is_impulse_cat":  category in IMPULSE_CATEGORIES,
    }


def score_dataframe(df: pd.DataFrame) -> pd.Series:
    """Batch-score a features dataframe."""
    scores = (
          0.30 * df["category_risk"]
        + 0.20 * df["is_late_night"]
        + 0.20 * (df["spend_vs_mean_ratio"] / 5)
        + 0.10 * df["is_weekend"]
        + 0.10 * df["is_end_of_month"]
        + 0.10 * df["impulse_cat_freq"]
    ) * 100
    return scores.clip(0, 100).round(2)