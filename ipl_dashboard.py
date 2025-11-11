import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import openai

# =======================
# 1. PAGE CONFIG
# =======================
st.set_page_config(page_title="IPL Deep Dive Dashboard", layout="wide")

st.markdown("""
    <style>
        body {background-color: #0E1117; color: #FAFAFA;}
        .stApp {background-color: #0E1117;}
        h1, h2, h3, h4 {color: #FF6F00;}
        .stMetric {background-color: #1E1E1E; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

# =======================
# 2. LOAD DATA & MODELS
# =======================
@st.cache_data
def load_data():
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    
    # Normalize match ID column
    match_id_col = None
    for col in ["id", "match_id", "matchid"]:
        if col in matches.columns:
            match_id_col = col
            break
    if match_id_col is None:
        st.error("No match ID column found in all_matches.csv.")
        st.stop()

    matches.rename(columns={match_id_col: "match_id"}, inplace=True)
    return matches, deliveries

@st.cache_resource
def load_model():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        return model, team_encoder, venue_encoder, toss_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None

matches, deliveries = load_data()
model, team_encoder, venue_encoder, toss_encoder = load_model()

if model is None:
    st.stop()

# =======================
# 3. PREDICT FUNCTION
# =======================
def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    """Returns predicted winner and win probabilities (8 features)."""
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]

    input_df = pd.DataFrame({
        "team1_enc": [team1_enc],
        "team2_enc": [team2_enc],
        "venue_enc": [venue_enc],
        "toss_enc": [toss_enc],
        "team1_form": [team1_form],
        "team2_form": [team2_form],
        "team1_win_pct": [team1_win_pct],
        "team2_win_pct": [team2_win_pct]
    })

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    winner = team1 if pred == 1 else team2
    win_probs = {team1: proba[1] * 100, team2: proba[0] * 100}
    return winner, win_probs

# =======================
# 4. SIDEBAR INPUTS
# =======================
st.sidebar.header("Select Match Parameters")

team_list = sorted(matches["team1"].unique())
venue_list = sorted(matches["venue"].unique())

team1 = st.sidebar.selectbox("Team 1", team_list, index=0)
team2 = st.sidebar.selectbox("Team 2", [t for t in team_list if t != team1], index=1)
venue = st.sidebar.selectbox("Venue", venue_list)
toss_winner = st.sidebar.selectbox("Toss Winner", [team1, team2])
team1_form = st.sidebar.slider(f"{team1} Form (0-1)", 0.0, 1.0, 0.5)
team2_form = st.sidebar.slider(f"{team2} Form (0-1)", 0.0, 1.0, 0.5)

# =======================
# 5. LIVE PREDICTION + VISUALS
# =======================
try:
    # --- H2H Stats ---
    h2h = matches[
        ((matches["team1"] == team1) & (matches["team2"] == team2)) |
        ((matches["team1"] == team2) & (matches["team2"] == team1))
    ]
    team1_wins = (h2h["winner"] == team1).sum()
    team2_wins = (h2h["winner"] == team2).sum()
    total = len(h2h)
    team1_win_pct = team1_wins / total if total > 0 else 0.5
    team2_win_pct = team2_wins / total if total > 0 else 0.5

    # --- Prediction ---
    winner, win_probs = predict_match_winner(
        model, team_encoder, venue_encoder, toss_encoder,
        team1, team2, venue, toss_winner,
        team1_form, team2_form, team1_win_pct, team2_win_pct
    )

    st.subheader(f"🏆 Predicted Winner: {winner}")

    # --- Animated Pie Chart (Plotly) ---
    fig = go.Figure(data=[
        go.Pie(
            labels=[team1, team2],
            values=[win_probs[team1], win_probs[team2]],
            textinfo="label+percent",
            marker=dict(colors=["#FF6F00", "#1E90FF"]),
            hole=0.4
        )
    ])
    fig.update_traces(sort=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- H2H Summary ---
    st.markdown("### 📊 Head-to-Head Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matches", total)
    c2.metric(f"{team1} Wins", team1_wins)
    c3.metric(f"{team2} Wins", team2_wins)

    # --- Venue Averages ---
    st.markdown("---")
    st.subheader(f"🏟️ Team Averages at {venue}")

    def get_avg_stats(team):
        venue_matches = matches[matches["venue"] == venue]
        relevant_matches = venue_matches[
            (venue_matches["team1"] == team) | (venue_matches["team2"] == team)
        ]
        if relevant_matches.empty:
            return 0, 0
        match_ids = relevant_matches["match_id"].unique()
        innings = deliveries[deliveries["match_id"].isin(match_ids) & (deliveries["batting_team"] == team)]
        if innings.empty:
            return 0, 0
        avg_runs = innings.groupby("match_id")["total_runs"].sum().mean()
        avg_wkts = innings.groupby("match_id")["is_wicket"].sum().mean()
        return round(avg_runs, 1), round(avg_wkts, 1)

    t1_runs, t1_wkts = get_avg_stats(team1)
    t2_runs, t2_wkts = get_avg_stats(team2)

    c4, c5 = st.columns(2)
    c4.metric(f"{team1} Avg Score", f"{t1_runs} / {t1_wkts}")
    c5.metric(f"{team2} Avg Score", f"{t2_runs} / {t2_wkts}")

    # --- Key Players (Against Opponent) ---
    st.markdown("---")
    st.subheader("🔥 Key Player Stats (Against Opponent)")

    vs_team1 = deliveries[(deliveries["batting_team"] == team1) & (deliveries["bowling_team"] == team2)]
    vs_team2 = deliveries[(deliveries["batting_team"] == team2) & (deliveries["bowling_team"] == team1)]

    st.write(f"**Top 5 Batters ({team1} vs {team2})**")
    top_batters1 = vs_team1.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_batters1)

    st.write(f"**Top 5 Batters ({team2} vs {team1})**")
    top_batters2 = vs_team2.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_batters2)

    st.write(f"**Top 5 Bowlers ({team1} vs {team2})**")
    top_bowlers1 = vs_team1[vs_team1["is_wicket"] == 1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_bowlers1)

    st.write(f"**Top 5 Bowlers ({team2} vs {team1})**")
    top_bowlers2 = vs_team2[vs_team2["is_wicket"] == 1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_bowlers2)

except Exception as e:
    st.error(f"Error in prediction: {e}")

# =======================
# 6. SIMPLE AI CHATBOT (Google Gemini)
# =======================
import google.generativeai as genai

st.markdown("---")
st.header("🤖 IPL Insights Chatbot (Gemini)")

# Configure Gemini API
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Google API configuration error: {e}")

user_query = st.text_input("Ask about IPL teams, venues, or players:")

if user_query:
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(
            f"You are an IPL cricket analyst. Answer concisely and factually.\n\nUser question: {user_query}"
        )
        st.success(response.text)
    except Exception as e:
        st.error(f"Chatbot error: {e}")
