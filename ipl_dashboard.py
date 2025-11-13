import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from googleapiclient.discovery import build
import os

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="🏏 IPL Match Predictor & Deep Dive Dashboard", layout="wide")
st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        return model, team_encoder, venue_encoder, toss_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

@st.cache_data
def load_data():
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    return matches, deliveries

model, team_encoder, venue_encoder, toss_encoder, winner_encoder = load_model()
matches, deliveries = load_data()

if model is None:
    st.stop()

# ------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------
def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    try:
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
        win_probs = {team1: proba[1]*100, team2: proba[0]*100}
        return winner, win_probs
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, {team1: 50, team2: 50}

# ------------------------------------------------------------------
# Google Search Chatbot
# ------------------------------------------------------------------
def google_search_query(query):
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_SEARCH_CX"), num=3).execute()
        results = res.get("items", [])
        if not results:
            return "No relevant results found."
        answer = "\n\n".join([f"🔹 {item['title']}\n{item['snippet']}" for item in results])
        return answer
    except Exception:
        return "Error fetching search results. Check your API key or CX ID."

# ------------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------------
st.sidebar.header("Match Setup")

teams = sorted(matches["team1"].unique())
venues = sorted(matches["venue"].unique())

team1 = st.sidebar.selectbox("Select Team 1", teams, key="team1_select")
team2 = st.sidebar.selectbox("Select Team 2", [t for t in teams if t != team1], key="team2_select")
venue = st.sidebar.selectbox("Select Venue", venues, key="venue_select")
toss_winner = st.sidebar.selectbox("Toss Winner", [team1, team2], key="toss_select")

col1, col2 = st.sidebar.columns(2)
team1_form = col1.slider(f"{team1} Recent Form (0–10)", 0, 10, 5, key="form1")
team2_form = col2.slider(f"{team2} Recent Form (0–10)", 0, 10, 5, key="form2")

col3, col4 = st.sidebar.columns(2)
team1_win_pct = col3.slider(f"{team1} Overall Win %", 0, 100, 50, key="win1")
team2_win_pct = col4.slider(f"{team2} Overall Win %", 0, 100, 50, key="win2")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.sidebar.button("Predict Winner"):
    winner, win_probs = predict_match_winner(
        model, team_encoder, venue_encoder, toss_encoder,
        team1, team2, venue, toss_winner, team1_form, team2_form, team1_win_pct, team2_win_pct
    )

    if winner:
        st.subheader(f"🏆 Predicted Winner: {winner}")
        fig = go.Figure(data=[go.Pie(
            labels=list(win_probs.keys()),
            values=list(win_probs.values()),
            hole=0.4,
            marker=dict(colors=["#1f77b4", "#ff7f0e"])
        )])
        fig.update_traces(textinfo='label+percent', pull=[0.05, 0])
        fig.update_layout(title_text="Win Probability", template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Head-to-Head Stats
# ------------------------------------------------------------------
st.header("📊 Head-to-Head Record")
h2h = matches[((matches["team1"] == team1) & (matches["team2"] == team2)) |
              ((matches["team1"] == team2) & (matches["team2"] == team1))]

if not h2h.empty:
    team1_wins = (h2h["winner"] == team1).sum()
    team2_wins = (h2h["winner"] == team2).sum()
    st.write(f"{team1} Wins: {team1_wins}")
    st.write(f"{team2} Wins: {team2_wins}")
else:
    st.write("No head-to-head data available.")

# ------------------------------------------------------------------
# Team Averages at Selected Venue
# ------------------------------------------------------------------
st.header(f"📍 Team Averages at {venue}")
venue_matches = matches[matches["venue"] == venue]
if not venue_matches.empty:
    merged = pd.merge(venue_matches, deliveries, on="match_id", how="left")
    team1_data = merged[merged["inning_team"] == team1]
    team2_data = merged[merged["inning_team"] == team2]
    if not team1_data.empty and not team2_data.empty:
        team1_avg_score = team1_data.groupby("match_id")["runs_scored"].sum().mean()
        team2_avg_score = team2_data.groupby("match_id")["runs_scored"].sum().mean()
        team1_avg_wkts = team1_data.groupby("match_id")["is_wicket"].sum().mean()
        team2_avg_wkts = team2_data.groupby("match_id")["is_wicket"].sum().mean()
        st.write(f"**{team1}** - Avg Score: {team1_avg_score:.1f}, Avg Wickets Lost: {team1_avg_wkts:.1f}")
        st.write(f"**{team2}** - Avg Score: {team2_avg_score:.1f}, Avg Wickets Lost: {team2_avg_wkts:.1f}")
    else:
        st.write("No team performance data found for this venue.")
else:
    st.write("No matches found at this venue.")

# ------------------------------------------------------------------
# Top Batters and Bowlers vs Opponent
# ------------------------------------------------------------------
st.header(f"🔥 Top Players: {team1} vs {team2}")
filtered = deliveries[(deliveries["inning_team"].isin([team1, team2]))]

if not filtered.empty:
    top_batters = filtered.groupby("batter")["runs_scored"].sum().nlargest(5)
    top_bowlers = filtered.groupby("bowler")["is_wicket"].sum().nlargest(5)

    col1, col2 = st.columns(2)
    col1.bar_chart(top_batters)
    col1.write("Top Batters (Runs)")
    col2.bar_chart(top_bowlers)
    col2.write("Top Bowlers (Wickets)")
else:
    st.write("No batting or bowling data available for selected teams.")

# ------------------------------------------------------------------
# Chatbot
# ------------------------------------------------------------------
st.header("🤖 IPL Assistant Chatbot")
query = st.text_input("Ask anything about IPL:")
if st.button("Ask"):
    if query.strip():
        st.info(google_search_query(query))
    else:
        st.warning("Please enter a question first.")

