import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from googleapiclient.discovery import build
import google.generativeai as genai
import os

st.set_page_config(page_title="IPL Match Predictor & AI Dashboard", layout="wide")

# ----------------------- CONFIG & STYLING -----------------------
st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        h1, h2, h3, h4 { color: #00FFFF; }
    </style>
""", unsafe_allow_html=True)

# ----------------------- CACHING -----------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    return matches, deliveries

@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        return model, team_encoder, venue_encoder, toss_encoder
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None, None, None

# ----------------------- LOAD -----------------------
matches, deliveries = load_data()
model, team_encoder, venue_encoder, toss_encoder = load_model_and_encoders()

if matches is None or deliveries is None or model is None:
    st.stop()

# ----------------------- PREDICTION FUNCTION -----------------------
def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    try:
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]
        toss_enc = toss_encoder.transform([toss_winner])[0]

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
        return None, None

# ----------------------- STATS FUNCTIONS -----------------------
def team_vs_team_stats(matches, team1, team2):
    h2h = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                  ((matches['team1'] == team2) & (matches['team2'] == team1))]
    t1_wins = (h2h['winner'] == team1).sum()
    t2_wins = (h2h['winner'] == team2).sum()
    return len(h2h), t1_wins, t2_wins

def avg_score_and_wickets(deliveries, matches, team, venue):
    merged = deliveries.merge(matches[['match_id', 'venue']], on='match_id', how='left')
    data = merged[(merged['venue'] == venue) & (merged['inning_team'] == team)]
    if data.empty:
        return 0, 0
    avg_score = data.groupby('match_id')['runs_scored'].sum().mean()
    avg_wickets = data.groupby('match_id')['is_wicket'].sum().mean()
    return round(avg_score, 1), round(avg_wickets, 1)

def top_players_against_team(deliveries, matches, team1, team2):
    merged = deliveries.merge(matches[['match_id', 'team1', 'team2']], on='match_id', how='left')
    mask = ((merged['inning_team'] == team1) & (merged['team2'] == team2)) | \
           ((merged['inning_team'] == team2) & (merged['team1'] == team1))
    data = merged[mask]
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    top_batters = data.groupby('batter')['runs_scored'].sum().nlargest(5).reset_index()
    top_bowlers = data.groupby('bowler')['is_wicket'].sum().nlargest(5).reset_index()
    return top_batters, top_bowlers

# ----------------------- CHATBOT -----------------------
def chatbot_response(query):
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        GOOGLE_SEARCH_KEY = st.secrets["GOOGLE_SEARCH_KEY"]
        GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]

        genai.configure(api_key=GOOGLE_API_KEY)
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_KEY)
        response = service.cse().list(q=query, cx=GOOGLE_SEARCH_CX, num=5).execute()
        items = response.get("items", [])
        snippets = "\n".join([f"{i['title']}: {i['snippet']}" for i in items])

        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Based on the following web results, answer the IPL-related query clearly:\n{snippets}\n\nQuery: {query}"
        result = model.generate_content(prompt)
        return result.text.strip() if result else "No relevant answer found."
    except Exception as e:
        return f"Error in chatbot: {e}"

# ----------------------- DASHBOARD -----------------------
st.title("🏏 IPL Match Predictor & AI Deep Dive Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    team1 = st.selectbox("Select Team 1", sorted(matches['team1'].unique()))
    team1_form = st.slider(f"{team1} Recent Form (0–10)", 0, 10, 5)
    team1_win_pct = st.slider(f"{team1} Overall Win %", 0, 100, 50)

with col2:
    team2 = st.selectbox("Select Team 2", sorted(matches['team2'].unique()))
    team2_form = st.slider(f"{team2} Recent Form (0–10)", 0, 10, 5)
    team2_win_pct = st.slider(f"{team2} Overall Win %", 0, 100, 50)

with col3:
    venue = st.selectbox("Select Venue", sorted(matches['venue'].unique()))
    toss_winner = st.selectbox("Toss Winner", [team1, team2])

if st.button("Predict Winner"):
    winner, win_probs = predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                                             team1, team2, venue, toss_winner,
                                             team1_form, team2_form, team1_win_pct, team2_win_pct)
    if winner:
        st.subheader(f"🏆 Predicted Winner: **{winner}**")

        # Animated pie chart using Plotly
        fig = go.Figure(data=[go.Pie(labels=list(win_probs.keys()),
                                     values=list(win_probs.values()),
                                     hole=0.4)])
        fig.update_traces(textinfo='label+percent', pull=[0.05, 0])
        fig.update_layout(title_text="Win Probability", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------- H2H + TEAM STATS -----------------------
st.markdown("---")
st.header("📊 Head-to-Head Statistics & Venue Insights")

matches_played, t1_wins, t2_wins = team_vs_team_stats(matches, team1, team2)
t1_avg_score, t1_avg_wkts = avg_score_and_wickets(deliveries, matches, team1, venue)
t2_avg_score, t2_avg_wkts = avg_score_and_wickets(deliveries, matches, team2, venue)

colA, colB, colC = st.columns(3)
colA.metric("Matches Played", matches_played)
colB.metric(f"{team1} Wins", t1_wins)
colC.metric(f"{team2} Wins", t2_wins)

st.subheader(f"Team Averages at {venue}")
st.write(f"**{team1}:** Avg Score = {t1_avg_score}, Avg Wickets Lost = {t1_avg_wkts}")
st.write(f"**{team2}:** Avg Score = {t2_avg_score}, Avg Wickets Lost = {t2_avg_wkts}")

# ----------------------- PLAYER STATS -----------------------
st.markdown("---")
st.header("🔥 Top Players in Head-to-Head Battles")

top_batters, top_bowlers = top_players_against_team(deliveries, matches, team1, team2)
if not top_batters.empty:
    st.subheader("Top Batters")
    st.bar_chart(top_batters.set_index("batter"))
if not top_bowlers.empty:
    st.subheader("Top Bowlers")
    st.bar_chart(top_bowlers.set_index("bowler"))

# ----------------------- CHATBOT -----------------------
st.markdown("---")
st.header("🤖 AI Chatbot: Ask about IPL or Teams")

user_query = st.text_input("Type your IPL-related question:")
if user_query:
    with st.spinner("Fetching AI insights..."):
        response = chatbot_response(user_query)
    st.write(response)
