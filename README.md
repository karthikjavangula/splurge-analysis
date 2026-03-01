# 💸 SPLURGE — Financial Impulse Behaviour Detector

> **Detecting Financial Impulse Behaviour in Young Adults**  
> ML-powered risk scoring, behavioral profiling, and personalized nudges.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)

---

## 🎯 Problem Statement

Young adults (18–30) frequently overspend due to emotional or impulsive decision-making — triggered by stress, late-night browsing, post-salary euphoria, or end-of-month pressure. Traditional budgeting tools show _what_ was spent, but not _why_ or _when_ spending is risky.

**SPLURGE** uses behavioral ML to detect impulse patterns in real time, score each transaction, profile the user, and generate personalized nudges — before the spend happens.

---

## 🏗️ Project Structure

```
splurge-analysis/
├── data/
│   ├── transactions.csv        ← Raw transactions (generated/replaced by Kaggle data)
│   ├── users.csv               ← User demographics
│   └── features.csv            ← Engineered features + risk scores
├── src/
│   ├── generate_data.py        ← Synthetic data generator (Kaggle schema)
│   ├── features.py             ← Feature engineering pipeline
│   ├── model.py                ← XGBoost classifier + KMeans clustering
│   ├── risk_engine.py          ← Impulse Risk Score computation
│   └── nudges.py               ← Personalized behavioral nudge engine
├── app/
│   └── dashboard.py            ← Streamlit interactive dashboard
├── models/
│   ├── xgb_model.pkl           ← Trained classifier
│   ├── kmeans.pkl              ← Behavioral clustering model
│   ├── scaler.pkl              ← Feature scaler
│   ├── user_profiles.csv       ← User → profile assignments
│   ├── feature_importance.csv  ← Top predictive features
│   └── model_meta.json         ← Accuracy, AUC, metadata
├── docs/
│   └── documentation.md        ← Full technical documentation
├── presentation/               ← Hackathon presentation
├── run_pipeline.py             ← One-command full pipeline
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/karthikjavangula/splurge-analysis
cd splurge-analysis
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python run_pipeline.py
```
This generates data → engineers features → trains models in ~60 seconds.

### 3. Launch Dashboard
```bash
streamlit run app/dashboard.py
```

### (Optional) Use Real Kaggle Data
Download the [priyamchoksi Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) from Kaggle, place it at `data/transactions.csv`, then skip `generate_data.py` and run:
```bash
python src/features.py
python src/model.py
streamlit run app/dashboard.py
```

---

## 🔬 ML Architecture

### Feature Engineering (`features.py`)
| Feature | Description | Impulse Signal |
|---|---|---|
| `is_late_night` | Purchase between 10 PM–3 AM | ⬆️ High |
| `is_end_of_month` | Day 25–31 of month | ⬆️ High |
| `is_post_salary` | Day 1–3 of month | ⬆️ High |
| `category_risk` | Merchant category risk score (0–1) | ⬆️ High |
| `spend_vs_mean_ratio` | This spend ÷ user's personal average | ⬆️ High |
| `impulse_cat_freq` | % of user's historical buys in impulse cats | ⬆️ High |
| `is_weekend` | Saturday or Sunday purchase | ⬆️ Medium |
| `daily_txn_count` | Number of transactions that day | ⬆️ Medium |

### Impulse Risk Score (0–100)
Weighted composite:
```
Score = 0.30 × category_risk
      + 0.20 × is_late_night
      + 0.20 × (spend_vs_mean_ratio / 3)
      + 0.10 × is_weekend
      + 0.10 × is_end_of_month
      + 0.10 × impulse_cat_freq
```

| Score Range | Risk Label | Color |
|---|---|---|
| 0–29 | Low Risk | 🟢 Green |
| 30–59 | Moderate Risk | 🟡 Amber |
| 60–79 | High Risk | 🟠 Orange |
| 80–100 | Critical Risk | 🔴 Red |

### Model 1 — XGBoost Classifier
- **Target**: `impulse_label` (binary, rule-based ground truth)
- **Features**: 20 behavioral and temporal features
- **Output**: Probability (0–1) → scaled to impulse risk score
- **Performance**: ROC-AUC 0.97+ on holdout set

### Model 2 — KMeans Behavioral Profiling
- **Input**: 6 user-level aggregate features
- **Clusters**: 4 behavioral profiles

| Profile | Description |
|---|---|
| 🛡️ Conservative Spender | Disciplined, essential purchases, work hours |
| 🌙 Night Owl Impulse Buyer | Late-night entertainment/shopping, moderate-high risk |
| 🎉 Weekend Splurger | Controlled weekdays, significant weekend spikes |
| 🔥 High-Risk Impulse Buyer | All triggers active simultaneously |

---

## 📊 Dashboard Features

### Tab 1 — Overview
- KPI metrics (total spend, impulse rate, risk score, late-night count)
- Daily spending timeline with impulse overlay
- Category spend breakdown (donut chart)
- Behavioral profile distribution

### Tab 2 — Deep Dive
- **Spending Heatmap**: Hour × Day of Week risk heatmap
- **Risk Distribution**: Score histogram with zone coloring
- **Category Risk Analysis**: Horizontal bar chart by impulse rate
- **Feature Importance**: XGBoost top features
- **Hourly Impulse Pattern**: Bar + line dual-axis with late-night zone

### Tab 3 — 🔮 Risk Predictor *(Key Deliverable)*
Fill in a form with:
- Merchant category, purchase amount, time
- Day of week, day of month
- Personal spending context (avg spend, impulse history %)

→ Outputs: **Risk Gauge**, **Risk Label**, **Trigger Breakdown Table**, **Radar Chart** of active triggers, and **Personalised Nudges**

### Tab 4 — Nudges
Per-user profile card + 3 targeted behavioral nudges based on:
- Top risk category, late-night frequency, end-of-month patterns
- Actionable, non-judgmental micro-interventions

---

## 💡 Nudge Examples
| Trigger | Nudge |
|---|---|
| Late night purchase | "Set a no-spend after 10 PM rule for 7 days" |
| Entertainment binge | "Review subscriptions — cancel 1 unused in 30 days" |
| End-of-month surge | "Use 50/30/20 rule — track weekly, not monthly" |
| Post-salary splurge | "Auto-transfer to savings the day salary hits" |
| High spend vs average | "Apply the 48-hour rule for purchases over ₹1,000" |

---

## 📦 Dataset
- **Source**: [Kaggle — priyamchoksi/credit-card-transactions-dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
- **License**: Apache 2.0
- **Rows**: 1.85M transactions
- **Key columns**: `trans_date_trans_time`, `category`, `amt`, `dob`, `gender`, `job`, `merchant`
- **Fallback**: `src/generate_data.py` generates 350K+ synthetic rows matching same schema


---

## 📄 License
Apache 2.0 — see [LICENSE](LICENSE)