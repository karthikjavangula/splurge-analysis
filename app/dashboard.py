"""
SPLURGE — Streamlit Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.risk_engine import compute_risk_score
from src.nudges     import get_nudges, get_realtime_nudge

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPLURGE — Impulse Detector",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg":       "#0f1117",
    "card":     "#1a1d27",
    "accent":   "#7c5cbf",
    "green":    "#2ecc71",
    "amber":    "#f1c40f",
    "orange":   "#e67e22",
    "red":      "#e74c3c",
    "text":     "#e8e8f0",
}

CATEGORY_RISK = {
    "shopping_net": 0.90, "misc_net": 0.85, "entertainment": 0.80,
    "shopping_pos": 0.75, "food_dining": 0.65, "travel": 0.70,
    "personal_care": 0.55, "misc_pos": 0.60, "gas_transport": 0.30,
    "grocery_pos": 0.20, "health_fitness": 0.25, "home": 0.20, "kids_pets": 0.15,
}

CATEGORIES = sorted(CATEGORY_RISK.keys())

PROFILE_COLORS = {
    "Conservative Spender":    "#2ecc71",
    "Night Owl Impulse Buyer": "#9b59b6",
    "Weekend Splurger":        "#3498db",
    "High-Risk Impulse Buyer": "#e74c3c",
}

BAND_COLOR = {
    "Low":      "#2ecc71",
    "Moderate": "#f1c40f",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d27;
        border-radius: 8px;
        padding: 8px 20px;
        color: #e8e8f0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7c5cbf !important;
        color: white !important;
    }
    .metric-card {
        background: #1a1d27;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2d3a;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c5cbf; }
    .metric-label { font-size: 0.85rem; color: #8888aa; margin-top: 4px; }
    .nudge-card {
        background: #1a1d27;
        border-left: 4px solid #7c5cbf;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .trigger-tag {
        background: #2a1f4a;
        color: #b09fe0;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .risk-badge-low      { color: #2ecc71; font-weight: 700; font-size: 1.6rem; }
    .risk-badge-moderate { color: #f1c40f; font-weight: 700; font-size: 1.6rem; }
    .risk-badge-high     { color: #e67e22; font-weight: 700; font-size: 1.6rem; }
    .risk-badge-critical { color: #e74c3c; font-weight: 700; font-size: 1.6rem; }
    h1, h2, h3 { color: #e8e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df       = pd.read_csv("data/features.csv", parse_dates=["trans_date_trans_time"])
    profiles = pd.read_csv("models/user_profiles.csv")
    fi       = pd.read_csv("models/feature_importance.csv")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    df = df.merge(profiles[["cc_num", "profile_name"]], on="cc_num", how="left")
    df["profile_name"] = df["profile_name"].fillna("Conservative Spender")
    return df, profiles, fi, meta


@st.cache_resource(show_spinner=False)
def load_models():
    clf    = joblib.load("models/xgb_model.pkl")
    km     = joblib.load("models/kmeans.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return clf, km, scaler


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, color: str = "#7c5cbf"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def risk_gauge(score: float, band: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 42, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1d27",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#1a2d1f"},
                {"range": [30, 60], "color": "#2d2a18"},
                {"range": [60, 80], "color": "#2d1f10"},
                {"range": [80,100], "color": "#2d1010"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score
            }
        },
        title={"text": f"<b>{band}</b>", "font": {"size": 18, "color": color}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f1117",
        font={"color": "#e8e8f0"},
    )
    return fig


def radar_chart(triggers: dict):
    cats   = list(triggers.keys())
    values = list(triggers.values())
    values.append(values[0])
    cats.append(cats[0])

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=cats,
        fill="toself",
        fillcolor="rgba(124,92,191,0.25)",
        line=dict(color="#7c5cbf", width=2),
        marker=dict(color="#7c5cbf", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1a1d27",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#333", tickcolor="#666"),
            angularaxis=dict(gridcolor="#333", tickcolor="#aaa"),
        ),
        showlegend=False,
        paper_bgcolor="#0f1117",
        height=320,
        margin=dict(t=30, b=30),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 💸 SPLURGE")
        st.markdown("*Financial Impulse Detector*")
        st.divider()

        with st.spinner("Loading data..."):
            try:
                df, profiles, fi, meta = load_data()
                clf, km, scaler = load_models()
                data_ok = True
            except FileNotFoundError as e:
                data_ok = False
                st.error(f"Run `python run_pipeline.py` first.\n\n{e}")
                st.stop()

        # User selector
        all_users = sorted(df["cc_num"].unique())
        selected_user = st.selectbox("👤 Select User", all_users[:100],
                                      index=0, key="user_sel")
        user_df = df[df["cc_num"] == selected_user]

        st.divider()
        st.markdown(f"**Model AUC:** `{meta.get('auc', 'N/A')}`")
        st.markdown(f"**Accuracy:**  `{meta.get('accuracy', 'N/A')}`")
        st.markdown(f"**Total Txns:** `{len(df):,}`")
        st.divider()
        st.caption("SPLURGE · ML-Powered Risk Scoring")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 💸 SPLURGE — Financial Impulse Behaviour Detector")
    st.markdown("*Behavioural ML · Risk Scoring · Personalized Nudges*")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Deep Dive",
        "🔮 Risk Predictor",
        "💡 Nudges",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"### Overview — User `{selected_user}`")

        user_profile = profiles[profiles["cc_num"] == selected_user]
        profile_name = user_profile["profile_name"].values[0] if len(user_profile) else "Unknown"
        icon = {"Conservative Spender": "🛡️", "Night Owl Impulse Buyer": "🌙",
                "Weekend Splurger": "🎉", "High-Risk Impulse Buyer": "🔥"}.get(profile_name, "👤")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi_card("Total Spend",    f"₹{user_df['amt'].sum():,.0f}")
        with c2: kpi_card("Transactions",   f"{len(user_df):,}")
        with c3: kpi_card("Impulse Rate",   f"{user_df['impulse_label'].mean():.0%}",
                           PALETTE["orange"])
        with c4: kpi_card("Avg Risk Score", f"{user_df['impulse_risk_score'].mean():.1f}",
                           PALETTE["amber"])
        with c5: kpi_card(f"{icon} Profile", profile_name.split()[0],
                           PROFILE_COLORS.get(profile_name, "#7c5cbf"))

        st.markdown("---")
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("#### 📈 Daily Spending Timeline")
            daily = (user_df.groupby(user_df["trans_date_trans_time"].dt.date)
                     .agg(total_spend=("amt", "sum"),
                          impulse_spend=("amt", lambda x: x[user_df.loc[x.index, "impulse_label"] == 1].sum()))
                     .reset_index())
            daily.columns = ["date", "total_spend", "impulse_spend"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["total_spend"],
                mode="lines", name="Total Spend",
                line=dict(color="#7c5cbf", width=2),
                fill="tozeroy", fillcolor="rgba(124,92,191,0.15)"
            ))
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["impulse_spend"],
                mode="lines", name="Impulse Spend",
                line=dict(color="#e74c3c", width=1.5, dash="dot"),
            ))
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font=dict(color="#e8e8f0"), height=280,
                legend=dict(bgcolor="#1a1d27", bordercolor="#333"),
                margin=dict(t=10, b=40, l=10, r=10),
                xaxis=dict(gridcolor="#2a2d3a"),
                yaxis=dict(gridcolor="#2a2d3a", title="₹ Amount"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### 🍩 Category Breakdown")
            cat_spend = user_df.groupby("category")["amt"].sum().reset_index()
            cat_spend.columns = ["category", "total"]
            cat_spend = cat_spend.sort_values("total", ascending=False).head(8)

            fig = px.pie(cat_spend, values="total", names="category",
                         hole=0.55, color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              marker=dict(line=dict(color="#0f1117", width=2)))
            fig.update_layout(
                paper_bgcolor="#0f1117", font=dict(color="#e8e8f0"),
                showlegend=False, height=280,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Profile distribution (all users)
        st.markdown("#### 👥 Behavioural Profile Distribution (All Users)")
        prof_counts = profiles["profile_name"].value_counts().reset_index()
        prof_counts.columns = ["Profile", "Count"]
        prof_counts["color"] = prof_counts["Profile"].map(PROFILE_COLORS)

        fig = px.bar(prof_counts, x="Profile", y="Count", color="Profile",
                     color_discrete_map=PROFILE_COLORS)
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font=dict(color="#e8e8f0"), showlegend=False, height=280,
            margin=dict(t=10, b=40),
            xaxis=dict(gridcolor="#2a2d3a"),
            yaxis=dict(gridcolor="#2a2d3a"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(f"### Deep Dive — User `{selected_user}`")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🔥 Spending Heatmap (Hour × Day)")
            heatmap = user_df.groupby(["hour", "day_of_week"])["impulse_risk_score"].mean().reset_index()
            heatmap_pivot = heatmap.pivot(index="hour", columns="day_of_week", values="impulse_risk_score").fillna(0)
            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            heatmap_pivot.columns = [day_labels[c] for c in heatmap_pivot.columns]

            fig = px.imshow(
                heatmap_pivot,
                color_continuous_scale=[[0, "#1a1d27"], [0.3, "#2d1f10"],
                                         [0.6, "#7c5cbf"], [1.0, "#e74c3c"]],
                aspect="auto",
                labels=dict(x="Day of Week", y="Hour", color="Avg Risk"),
            )
            fig.update_layout(
                paper_bgcolor="#0f1117", font=dict(color="#e8e8f0"), height=340,
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 📊 Risk Score Distribution")
            band_counts = user_df["risk_band"].value_counts().reindex(
                ["Low", "Moderate", "High", "Critical"]).fillna(0)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=user_df["impulse_risk_score"],
                nbinsx=40,
                marker=dict(
                    color=user_df["impulse_risk_score"],
                    colorscale=[[0, "#2ecc71"], [0.3, "#f1c40f"],
                                [0.6, "#e67e22"], [1.0, "#e74c3c"]],
                    line=dict(width=0),
                ),
                opacity=0.85,
            ))
            for x, label, color in [(30, "Low→Mod", "#f1c40f"),
                                     (60, "Mod→High", "#e67e22"),
                                     (80, "High→Crit", "#e74c3c")]:
                fig.add_vline(x=x, line=dict(color=color, dash="dash", width=1.5),
                              annotation_text=label, annotation_font_color=color)
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font=dict(color="#e8e8f0"), height=340,
                xaxis=dict(title="Risk Score", gridcolor="#2a2d3a"),
                yaxis=dict(title="Count",      gridcolor="#2a2d3a"),
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### 📦 Category Impulse Rate")
            cat_ir = (user_df.groupby("category")
                      .agg(impulse_rate=("impulse_label", "mean"),
                           txn_count=("amt", "count"))
                      .reset_index()
                      .sort_values("impulse_rate", ascending=True))

            fig = px.bar(cat_ir, x="impulse_rate", y="category",
                         orientation="h",
                         color="impulse_rate",
                         color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                         labels={"impulse_rate": "Impulse Rate", "category": ""},
                         text=cat_ir["impulse_rate"].map("{:.0%}".format))
            fig.update_traces(textposition="outside")
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font=dict(color="#e8e8f0"), height=360,
                xaxis=dict(tickformat=".0%", gridcolor="#2a2d3a"),
                coloraxis_showscale=False,
                margin=dict(t=10, b=40, l=10, r=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.markdown("#### 🧠 XGBoost Feature Importance")
            fi_top = fi.head(10).sort_values("importance", ascending=True)

            fig = px.bar(fi_top, x="importance", y="feature",
                         orientation="h",
                         color="importance",
                         color_continuous_scale=["#4a3a7a", "#7c5cbf", "#b09fe0"],
                         labels={"importance": "Importance", "feature": ""},
                         text=fi_top["importance"].map("{:.3f}".format))
            fig.update_traces(textposition="outside")
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font=dict(color="#e8e8f0"), height=360,
                xaxis=dict(gridcolor="#2a2d3a"),
                coloraxis_showscale=False,
                margin=dict(t=10, b=40, l=10, r=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Hourly impulse pattern
        st.markdown("#### ⏰ Hourly Impulse Pattern")
        hourly = user_df.groupby("hour").agg(
            txn_count=("amt", "count"),
            avg_risk=("impulse_risk_score", "mean"),
            impulse_rate=("impulse_label", "mean"),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=hourly["hour"], y=hourly["txn_count"],
            name="Transaction Count",
            marker_color=["#e74c3c" if (h >= 22 or h <= 2) else "#7c5cbf"
                          for h in hourly["hour"]],
            opacity=0.7,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["avg_risk"],
            name="Avg Risk Score",
            mode="lines+markers",
            line=dict(color="#f1c40f", width=2.5),
            marker=dict(size=6),
        ), secondary_y=True)
        fig.add_vrect(x0=21.5, x1=23.5, fillcolor="#e74c3c",
                      opacity=0.08, line_width=0, annotation_text="Late Night Zone",
                      annotation_font_color="#e74c3c")
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font=dict(color="#e8e8f0"), height=320,
            xaxis=dict(title="Hour of Day", dtick=1, gridcolor="#2a2d3a"),
            legend=dict(bgcolor="#1a1d27"),
            margin=dict(t=10, b=40),
        )
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False,
                         gridcolor="#2a2d3a")
        fig.update_yaxes(title_text="Avg Risk Score", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — RISK PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔮 Risk Predictor — Evaluate a Future Purchase")
        st.markdown("Fill in the details below to get an instant impulse risk assessment.")
        st.markdown("---")

        with st.form("risk_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🏷️ Purchase Details**")
                category = st.selectbox("Merchant Category", CATEGORIES,
                                         index=CATEGORIES.index("shopping_net"))
                amount   = st.number_input("Purchase Amount (₹)", min_value=10.0,
                                            max_value=100000.0, value=1500.0, step=50.0)

            with col2:
                st.markdown("**🕐 Timing**")
                hour     = st.slider("Hour of Purchase", 0, 23, 21)
                dow      = st.selectbox("Day of Week",
                                         ["Monday","Tuesday","Wednesday","Thursday",
                                          "Friday","Saturday","Sunday"],
                                         index=5)
                dom      = st.slider("Day of Month", 1, 31, 27)

            with col3:
                st.markdown("**👤 Your Spending Context**")
                user_avg  = st.number_input("Your Average Spend per Txn (₹)",
                                             min_value=100.0, value=800.0, step=50.0)
                imp_hist  = st.slider("Your Impulse Purchase History (%)", 0, 100, 35)

            submitted = st.form_submit_button("⚡ Calculate Risk", use_container_width=True)

        if submitted:
            dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                       "Friday":4,"Saturday":5,"Sunday":6}
            result = compute_risk_score(
                category=category,
                amt=amount,
                hour=hour,
                day_of_week=dow_map[dow],
                day_of_month=dom,
                user_avg_spend=user_avg,
                impulse_hist_pct=imp_hist,
            )
            score, band, color = result["score"], result["band"], result["color"]
            nudge = get_realtime_nudge(result)

            st.markdown("---")
            # Score + gauge + badge
            gc, bc = st.columns([1, 1])
            with gc:
                st.plotly_chart(risk_gauge(score, band, color), use_container_width=True)
            with bc:
                badge_class = band.split()[0].lower()
                st.markdown(f"""
                <br><br>
                <div style="text-align:center;">
                    <div class="risk-badge-{badge_class}">{band}</div>
                    <div style="font-size:3.5rem; font-weight:800; color:{color}; margin-top:8px;">
                        {score:.0f} / 100
                    </div>
                    <div style="color:#8888aa; font-size:0.9rem; margin-top:8px;">
                        Impulse Risk Score
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="nudge-card" style="margin-top:20px; border-left-color:{color};">
                    {nudge}
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            tc, rc = st.columns([1, 1])

            with tc:
                st.markdown("#### 📋 Trigger Breakdown")
                trig_df = pd.DataFrame([
                    {"Trigger": k, "Score": v,
                     "Active": "✅" if v > 0 else "—"}
                    for k, v in result["triggers"].items()
                ]).sort_values("Score", ascending=False)

                fig = px.bar(trig_df, x="Score", y="Trigger",
                             orientation="h",
                             color="Score",
                             color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
                             text="Active",
                             labels={"Score": "Contribution", "Trigger": ""})
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                    font=dict(color="#e8e8f0"), height=320,
                    xaxis=dict(range=[0, 105], gridcolor="#2a2d3a"),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=20, l=10, r=60),
                )
                st.plotly_chart(fig, use_container_width=True)

            with rc:
                st.markdown("#### 🕸️ Risk Radar")
                st.plotly_chart(radar_chart(result["triggers"]), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — NUDGES
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"### 💡 Personalized Nudges — User `{selected_user}`")

        user_profile_row = profiles[profiles["cc_num"] == selected_user]
        pname = user_profile_row["profile_name"].values[0] if len(user_profile_row) else "Conservative Spender"
        picon = {"Conservative Spender": "🛡️", "Night Owl Impulse Buyer": "🌙",
                 "Weekend Splurger": "🎉", "High-Risk Impulse Buyer": "🔥"}.get(pname, "👤")
        pcolor = PROFILE_COLORS.get(pname, "#7c5cbf")

        # Profile card
        top_cat   = user_df.groupby("category")["amt"].sum().idxmax()
        late_pct  = user_df["is_late_night"].mean() * 100
        eom_pct   = user_df["is_end_of_month"].mean() * 100
        avg_ratio = user_df["spend_vs_mean_ratio"].mean()
        avg_risk  = user_df["impulse_risk_score"].mean()

        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {pcolor}; text-align:left; margin-bottom:24px;">
            <div style="font-size:2rem; margin-bottom:6px;">{picon} {pname}</div>
            <div style="display:flex; gap:32px; flex-wrap:wrap; margin-top:12px;">
                <div><span style="color:#8888aa;">Top Category</span><br>
                     <b style="color:{pcolor};">{top_cat}</b></div>
                <div><span style="color:#8888aa;">Late Night Txns</span><br>
                     <b style="color:{pcolor};">{late_pct:.1f}%</b></div>
                <div><span style="color:#8888aa;">End-of-Month Txns</span><br>
                     <b style="color:{pcolor};">{eom_pct:.1f}%</b></div>
                <div><span style="color:#8888aa;">Avg Spend Ratio</span><br>
                     <b style="color:{pcolor};">{avg_ratio:.2f}×</b></div>
                <div><span style="color:#8888aa;">Avg Risk Score</span><br>
                     <b style="color:{pcolor};">{avg_risk:.1f}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

        nudges = get_nudges(
            profile_name=pname,
            top_category=top_cat,
            late_night_pct=late_pct,
            eom_pct=eom_pct,
            spend_ratio=avg_ratio,
            n=3,
        )

        st.markdown("#### Your 3 Targeted Nudges")
        for i, n in enumerate(nudges, 1):
            st.markdown(f"""
            <div class="nudge-card">
                <div class="trigger-tag">{n['trigger']}</div>
                <div style="font-size:1.05rem; color:#e8e8f0;">{n['nudge']}</div>
            </div>""", unsafe_allow_html=True)

        # All-user nudge stats
        st.markdown("---")
        st.markdown("#### 📊 Nudge Trigger Frequency (All Users)")
        all_late   = df["is_late_night"].mean() * 100
        all_eom    = df["is_end_of_month"].mean() * 100
        all_psal   = df["is_post_salary"].mean() * 100
        all_wknd   = df["is_weekend"].mean() * 100
        all_imphist = df["impulse_cat_freq"].mean() * 100

        trigger_summary = pd.DataFrame({
            "Trigger": ["Late Night", "End of Month", "Post Salary",
                        "Weekend", "Impulse Category"],
            "% Users Affected": [all_late, all_eom, all_psal, all_wknd, all_imphist]
        }).sort_values("% Users Affected", ascending=True)

        fig = px.bar(trigger_summary, x="% Users Affected", y="Trigger",
                     orientation="h",
                     color="% Users Affected",
                     color_continuous_scale=["#4a3a7a", "#7c5cbf", "#e74c3c"],
                     text=trigger_summary["% Users Affected"].map("{:.1f}%".format))
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font=dict(color="#e8e8f0"), height=300,
            xaxis=dict(range=[0, 115], gridcolor="#2a2d3a", ticksuffix="%"),
            coloraxis_showscale=False,
            margin=dict(t=10, b=20, l=10, r=60),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()