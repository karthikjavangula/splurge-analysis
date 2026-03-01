"""
Personalized behavioural nudge engine.
Generates 3 targeted, actionable nudges per user based on their risk profile.
"""

from typing import Literal

NUDGE_LIBRARY = {
    "late_night": [
        "🌙 Set a no-spend rule after 10 PM for 7 days — track the difference.",
        "🌙 Enable a spending lock on your card between 10 PM and 6 AM.",
        "🌙 Before any late-night purchase, write down *why* you need it. Sleep on it.",
    ],
    "entertainment": [
        "🎮 Review your entertainment subscriptions — cancel 1 unused service this week.",
        "🎬 Set a monthly entertainment budget and track it with a simple note.",
        "🎵 Try a 30-day challenge: switch 3 paid entertainment habits to free alternatives.",
    ],
    "end_of_month": [
        "📅 Use the 50/30/20 rule — track spending weekly, not just at month-end.",
        "📅 Create an end-of-month spending 'freeze window' for the last 5 days.",
        "📅 Set a calendar reminder on the 20th to review remaining budget.",
    ],
    "post_salary": [
        "💰 Auto-transfer 20% to savings the moment your salary hits your account.",
        "💰 Write your top 3 financial goals before spending post-salary.",
        "💰 Wait 24 hours before any non-essential purchase after payday.",
    ],
    "high_spend_ratio": [
        "📊 Apply the 48-hour rule for any purchase more than 2× your usual spend.",
        "📊 Screenshot your bank balance before every purchase above ₹1,000.",
        "📊 Compare this purchase against your monthly savings target before buying.",
    ],
    "shopping_online": [
        "🛒 Remove saved card details from shopping apps — add friction to impulse buys.",
        "🛒 Use a wishlist: if it's still there in 48 hours, then buy it.",
        "🛒 Unsubscribe from 3 promotional email lists today.",
    ],
    "weekend_splurge": [
        "🗓️ Plan weekend activities in advance to avoid unplanned spending.",
        "🗓️ Set a weekend cash envelope — when it's gone, it's gone.",
        "🗓️ Schedule a free activity for each weekend this month.",
    ],
    "general": [
        "✅ Review yesterday's transactions every morning — awareness reduces impulse spend.",
        "✅ Identify your top 3 spending triggers and write a counter-plan for each.",
        "✅ Try a 'spend nothing' day once a week and track how it feels.",
    ],
}


def get_nudges(
    profile_name: str,
    top_category: str,
    late_night_pct: float,
    eom_pct: float,
    spend_ratio: float,
    n: int = 3,
) -> list[dict]:
    """
    Return n nudges tailored to the user's strongest triggers.

    Args:
        profile_name:  e.g. "Night Owl Impulse Buyer"
        top_category:  most-spent category string
        late_night_pct: % of transactions that are late-night (0–100)
        eom_pct:        % of transactions at end-of-month
        spend_ratio:    average spend-vs-mean ratio
        n:              number of nudges to return

    Returns:
        List of dicts: {"trigger": str, "nudge": str, "priority": int}
    """
    import random, numpy as np

    candidates = []

    if late_night_pct > 15 or "Night Owl" in profile_name:
        candidates.append(("late_night",     late_night_pct, 1))
    if top_category in ("shopping_net", "misc_net", "shopping_pos"):
        candidates.append(("shopping_online", 80, 2))
    if top_category == "entertainment":
        candidates.append(("entertainment",  70, 2))
    if eom_pct > 20:
        candidates.append(("end_of_month",   eom_pct, 3))
    if spend_ratio > 1.5:
        candidates.append(("high_spend_ratio", spend_ratio * 20, 3))
    if "Weekend" in profile_name:
        candidates.append(("weekend_splurge", 60, 2))
    if "High-Risk" in profile_name:
        candidates.append(("post_salary",    75, 1))

    # Always include a general nudge as fallback
    candidates.append(("general", 10, 4))

    # Sort by priority (lower = more important), then score desc
    candidates.sort(key=lambda x: (x[2], -x[1]))

    seen_triggers = set()
    results = []
    for trigger, score, priority in candidates:
        if trigger in seen_triggers:
            continue
        seen_triggers.add(trigger)
        nudge_text = random.choice(NUDGE_LIBRARY[trigger])
        trigger_label = trigger.replace("_", " ").title()
        results.append({
            "trigger": trigger_label,
            "nudge":   nudge_text,
            "priority": priority,
        })
        if len(results) == n:
            break

    return results


def get_realtime_nudge(risk_result: dict) -> str:
    """
    Single contextual nudge for the Risk Predictor tab,
    based on the computed risk result from risk_engine.
    """
    score    = risk_result["score"]
    triggers = risk_result["active_triggers"]

    if score < 30:
        return "✅ This looks like a planned, low-risk purchase. Go for it!"

    messages = []
    if risk_result.get("is_late_night"):
        messages.append("🌙 You're buying late at night — sleep on this one.")
    if risk_result.get("is_eom"):
        messages.append("📅 End-of-month pressure is real. Is this in your budget?")
    if risk_result.get("is_post_salary"):
        messages.append("💰 Post-salary euphoria can inflate spending — pause first.")
    if risk_result.get("is_weekend"):
        messages.append("🗓️ Weekend purchases are 40% more likely to be regretted.")
    if risk_result.get("is_impulse_cat"):
        messages.append("🛒 This category has a high impulse purchase rate.")

    if not messages:
        messages.append("📊 Your spend is significantly above your personal average.")

    if score >= 80:
        messages.append("🔴 CRITICAL: Apply the 48-hour rule before completing this purchase.")
    elif score >= 60:
        messages.append("🟠 HIGH RISK: Ask yourself — will I still value this in a week?")

    return " | ".join(messages[:2])