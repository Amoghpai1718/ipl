import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from predict_winner import predict_match_winner
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(page_title="IPL AI Dashboard", layout="wide")
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stButton>button {background-color:#f63366; color:white;}
h1, h2, h3, h4 {color:white;}
</style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL & ENCODERS -------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        return model, team_encoder, toss_encoder, venue_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model/encoder files: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model_files()

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_matches():
    return pd.read_csv("all_matches.csv")

@st.cache_data
def load_deliveries():
    return pd.read_csv("all_deliveries.csv")

matches = load_matches()
deliveries = load_deliveries()

# ------------------- SIDEBAR INPUTS -------------------
st.title("🏏 IPL AI Dashboard & Winner Predictor")

all_teams = sorted(list(team_encoder.classes_)) if team_encoder else []
all_venues = sorted(list(venue_encoder.classes_)) if venue_encoder else []
all_toss = sorted(list(toss_encoder.classes_)) if toss_encoder else []

team1 = st.selectbox("Select Team 1", all_teams)
team2 = st.selectbox("Select Team 2", [t for t in all_teams if t != team1])
venue = st.selectbox("Select Venue", all_venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", all_toss)
team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)
team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)

# ------------------- PREDICTION -------------------
if st.button("Predict Winner"):
    if None in [model, team_encoder, toss_encoder, venue_encoder, winner_encoder]:
        st.error("Missing model or encoder files. Prediction cannot proceed.")
    else:
        try:
            pred, prob = predict_match_winner(
                model, team_encoder, toss_encoder, venue_encoder,
                team1, team2, venue, toss_winner, toss_decision,
                team1_form, team2_form
            )
            st.success(f"🏆 Predicted Winner: {pred}")
            st.write(f"{team1} Win Probability: {prob[team1]*100:.1f}%")
            st.write(f"{team2} Win Probability: {prob[team2]*100:.1f}%")
            
            # Pie chart
            fig = px.pie(
                names=[team1, team2],
                values=[prob[team1]*100, prob[team2]*100],
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig)
            
            # ------------------- HEAD-TO-HEAD -------------------
            st.subheader(f"Head-to-Head Stats: {team1} vs {team2}")
            h2h = matches[
                ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                ((matches["team1"] == team2) & (matches["team2"] == team1))
            ]
            total_matches = len(h2h)
            team1_wins = len(h2h[h2h["winner"] == team1])
            team2_wins = len(h2h[h2h["winner"] == team2])
            st.write(f"Total Matches: {total_matches}")
            st.write(f"{team1} Wins: {team1_wins}")
            st.write(f"{team2} Wins: {team2_wins}")
            
            # ------------------- TOP BATTERS & BOWLERS -------------------
            st.subheader("Top Batters & Bowlers")
            team1_del = deliveries[deliveries["inning_team"]==team1]
            team2_del = deliveries[deliveries["inning_team"]==team2]

            # Top batters
            top_batters1 = team1_del.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
            top_batters2 = team2_del.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
            st.write(f"Top Batters - {team1}")
            st.bar_chart(top_batters1)
            st.write(f"Top Batters - {team2}")
            st.bar_chart(top_batters2)
            
            # Top bowlers
            top_bowlers1 = team1_del.groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
            top_bowlers2 = team2_del.groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
            st.write(f"Top Bowlers - {team1}")
            st.bar_chart(top_bowlers1)
            st.write(f"Top Bowlers - {team2}")
            st.bar_chart(top_bowlers2)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ------------------- OPTIONAL CHATBOT -------------------
tab1, tab2 = st.tabs(["🏆 Predictor", "🤖 Chatbot"])

with tab2:
    st.header("AI IPL Chatbot")
    st.info("Ask questions about IPL history. Chatbot uses your API keys.")
    st.text("Chatbot functionality placeholder (Google Gemini/API integration)")

