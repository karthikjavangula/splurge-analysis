"""
Model training:
  1. XGBoost binary classifier → predicts impulse_label
  2. KMeans clustering (k=4) → behavioural profiles
Saves artefacts to models/
"""

import pandas as pd
import numpy as np
import joblib, json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
import xgboost as xgb

FEATURE_COLS = [
    "hour", "day_of_week", "day_of_month", "month",
    "is_weekend", "is_late_night", "is_end_of_month", "is_post_salary",
    "category_risk", "is_impulse_cat",
    "spend_vs_mean_ratio", "daily_txn_count", "impulse_cat_freq",
    "amt"
]

CLUSTER_FEATURES = [
    "impulse_cat_freq", "is_late_night", "is_weekend",
    "spend_vs_mean_ratio", "is_end_of_month", "daily_txn_count"
]

PROFILE_LABELS = {
    0: "Conservative Spender",
    1: "Night Owl Impulse Buyer",
    2: "Weekend Splurger",
    3: "High-Risk Impulse Buyer",
}

PROFILE_ICONS = {
    "Conservative Spender":      "🛡️",
    "Night Owl Impulse Buyer":   "🌙",
    "Weekend Splurger":          "🎉",
    "High-Risk Impulse Buyer":   "🔥",
}


def train_classifier(df: pd.DataFrame):
    print("   Training XGBoost classifier...")
    X = df[FEATURE_COLS].fillna(0)
    y = df["impulse_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    acc    = accuracy_score(y_test, y_pred)

    print(f"   XGBoost → AUC: {auc:.4f}  |  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Planned", "Impulse"]))

    # Feature importance
    fi = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return clf, fi, {"auc": round(auc, 4), "accuracy": round(acc, 4)}


def train_clustering(df: pd.DataFrame):
    print("   Training KMeans clustering (k=4)...")
    user_agg = df.groupby("cc_num")[CLUSTER_FEATURES].mean().reset_index()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_agg[CLUSTER_FEATURES])

    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    user_agg["cluster"] = km.fit_predict(X_scaled)

    # Map clusters to descriptive profiles by late-night spend signature
    cluster_means = user_agg.groupby("cluster")[CLUSTER_FEATURES].mean()
    rank = cluster_means.rank()
    
    # Assign profiles: highest late_night → night_owl, etc.
    scores = {
        c: (rank.loc[c, "is_late_night"] * 2 +
            rank.loc[c, "spend_vs_mean_ratio"] +
            rank.loc[c, "impulse_cat_freq"])
        for c in range(4)
    }
    sorted_clusters = sorted(scores, key=scores.get)
    mapping = {
        sorted_clusters[0]: 0,  # Conservative
        sorted_clusters[1]: 2,  # Weekend Splurger (mid)
        sorted_clusters[2]: 1,  # Night Owl (mid-high)
        sorted_clusters[3]: 3,  # High-Risk
    }
    user_agg["profile_id"]   = user_agg["cluster"].map(mapping)
    user_agg["profile_name"] = user_agg["profile_id"].map(PROFILE_LABELS)

    return km, scaler, user_agg[["cc_num", "cluster", "profile_id", "profile_name"]]


def main():
    os.makedirs("models", exist_ok=True)
    print("⚙️  Loading features...")
    df = pd.read_csv("data/features.csv")
    print(f"   Loaded {len(df):,} rows")

    # ── Classifier ─────────────────────────────────────────────────────────────
    clf, fi, metrics = train_classifier(df)
    joblib.dump(clf, "models/xgb_model.pkl")
    fi.to_csv("models/feature_importance.csv", index=False)
    print("   ✅ XGBoost model → models/xgb_model.pkl")

    # ── Clustering ─────────────────────────────────────────────────────────────
    km, scaler, user_profiles = train_clustering(df)
    joblib.dump(km,     "models/kmeans.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    user_profiles.to_csv("models/user_profiles.csv", index=False)
    print("   ✅ KMeans + Scaler → models/")
    print("   Profile distribution:")
    print(user_profiles["profile_name"].value_counts().to_string())

    # ── Metadata ───────────────────────────────────────────────────────────────
    meta = {
        "model":    "XGBoostClassifier",
        "clusters": 4,
        "features": FEATURE_COLS,
        **metrics
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("   ✅ Metadata → models/model_meta.json")


if __name__ == "__main__":
    main()