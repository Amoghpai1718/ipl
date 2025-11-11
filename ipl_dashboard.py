import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from googleapiclient.discovery import build
import os

st.set_page_config(page_title="🏏 IPL Match Predictor & Deep Dive Dashboard", layout="wide")

# ======================================================
# 1. LOAD MODEL AND ENCODERS
# ======================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        return model, team_encoder, venue_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None

model, team_encoder, venue_encoder = load_model()

# ======================================================
# 2. LOAD DATASETS
# ======================================================
@st.cache_data
def load_data():
    try:
        deliveries = pd.read_csv("all_deliveries.csv")
        matches = pd.read_csv("all_matches.csv")
        return deliveries, matches
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

deliveries, matches = load_data()

# ======================================================
# 3. PREDICTION FUNCTION
# ======================================================
def predict_match_winner(model, team_encoder, venue_encoder, team1, team2, venue,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    if model is None:
        return None, {team1: 0, team2: 0}

    try:
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]

        input_df = pd.DataFrame({
            "team1_enc": [team1_enc],
            "team2_enc": [team2_enc],
            "venue_enc": [venue_enc],
            "team1_form": [team1_form],
            "team2_form": [team2_form],
            "team1_win_pct": [team1_win_pct],
            "team2_win_pct": [team2_win_pct]
        })

        proba = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]

        winner = team1 if pred == 1 else team2
        win_probs = {team1: round(proba[1]*100, 2), team2: round(proba[0]*100, 2)}
        return winner, win_probs

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, {team1: 0, team2: 0}

# ======================================================
# 4. UI - TEAM SELECTION
# ======================================================
st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

if model is None:
    st.stop()

teams = list(team_encoder.classes_)
venues = list(venue_encoder.classes_)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", teams)
    team1_form = st.slider(f"{team1} Recent Form (0–10)", 0, 10, 5)
    team1_win_pct = st.slider(f"{team1} Overall Win %", 0, 100, 50)
with col2:
    team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
    team2_form = st.slider(f"{team2} Recent Form (0–10)", 0, 10, 5)
    team2_win_pct = st.slider(f"{team2} Overall Win %", 0, 100, 50)

venue = st.selectbox("Select Venue", venues)

# ======================================================
# 5. WIN PROBABILITY (Dynamic Pie Chart)
# ======================================================
winner, win_probs = predict_match_winner(model, team_encoder, venue_encoder,
                                         team1, team2, venue,
                                         team1_form, team2_form,
                                         team1_win_pct, team2_win_pct)

fig = go.Figure(data=[go.Pie(
    labels=[team1, team2],
    values=[win_probs[team1], win_probs[team2]],
    hole=0.4,
    textinfo='label+percent',
    marker=dict(colors=['#1f77b4', '#ff4d4d'])
)])
fig.update_layout(title="Winning Probability", transition_duration=500)
st.plotly_chart(fig, use_container_width=True)

st.success(f"Predicted Winner: **{winner}**")

# ======================================================
# 6. HEAD-TO-HEAD STATS
# ======================================================
st.markdown("---")
st.subheader("📈 Head-to-Head Stats")

h2h = matches[
    ((matches["team1"] == team1) & (matches["team2"] == team2)) |
    ((matches["team1"] == team2) & (matches["team2"] == team1))
]
if len(h2h) == 0:
    st.info("No head-to-head records found.")
else:
    team1_wins = (h2h["winner"] == team1).sum()
    team2_wins = (h2h["winner"] == team2).sum()
    st.metric(f"{team1} Wins", team1_wins)
    st.metric(f"{team2} Wins", team2_wins)
    st.write(f"Total Matches Played: {len(h2h)}")

# ======================================================
# 7. TEAM AVERAGES AT VENUE
# ======================================================
st.markdown("---")
st.subheader(f"🏟️ Team Averages at {venue}")

# Merge deliveries + matches for venue-based analysis
merged = pd.merge(deliveries, matches[["match_id", "venue"]], on="match_id", how="left")

team1_venue = merged[(merged["inning_team"] == team1) & (merged["venue"] == venue)]
team2_venue = merged[(merged["inning_team"] == team2) & (merged["venue"] == venue)]

def team_avg(df):
    if df.empty:
        return 0, 0
    total_runs = df.groupby("match_id")["runs_scored"].sum()
    total_wickets = df.groupby("match_id")["is_wicket"].sum()
    return round(total_runs.mean(), 2), round(total_wickets.mean(), 2)

team1_avg_runs, team1_avg_wkts = team_avg(team1_venue)
team2_avg_runs, team2_avg_wkts = team_avg(team2_venue)

colA, colB = st.columns(2)
with colA:
    st.metric(f"{team1} Avg Score", f"{team1_avg_runs} runs")
    st.metric(f"{team1} Avg Wickets Lost", f"{team1_avg_wkts}")
with colB:
    st.metric(f"{team2} Avg Score", f"{team2_avg_runs} runs")
    st.metric(f"{team2} Avg Wickets Lost", f"{team2_avg_wkts}")

# ======================================================
# 8. TOP BATTERS & BOWLERS (vs Each Other)
# ======================================================
st.markdown("---")
st.subheader(f"🔥 Top Batters & Bowlers in {team1} vs {team2}")

vs_df = deliveries[
    ((deliveries["inning_team"] == team1) & (deliveries["bowler"].isin(deliveries[deliveries["inning_team"] == team2]["bowler"]))) |
    ((deliveries["inning_team"] == team2) & (deliveries["bowler"].isin(deliveries[deliveries["inning_team"] == team1]["bowler"])))
]

if not vs_df.empty:
    top_batters = vs_df.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_batters)

    top_bowlers = vs_df[vs_df["is_wicket"] == 1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_bowlers)
else:
    st.info("No detailed delivery data available for these teams.")

# ======================================================
# 9. SIMPLE CHATBOT USING GOOGLE SEARCH
# ======================================================
st.markdown("---")
st.subheader("🤖 Ask the IPL Chatbot")

query = st.text_input("Ask your IPL question here:")
if query:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        search_cx = os.getenv("GOOGLE_SEARCH_CX")

        if not all([api_key, search_cx]):
            st.warning("Please add your Google API keys in Streamlit Secrets.")
        else:
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=query, cx=search_cx, num=2).execute()
            results = res.get("items", [])
            if results:
                for r in results:
                    st.markdown(f"### [{r['title']}]({r['link']})")
                    st.write(r["snippet"])
            else:
                st.info("No relevant results found.")
    except Exception as e:
        st.error(f"Chatbot Error: {e}")
