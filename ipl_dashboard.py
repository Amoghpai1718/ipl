import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import plotly.express as px
from predict_winner import predict_match_winner

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="IPL AI Dashboard", layout="wide")
st.title("🏏 IPL Advanced Analytics & Winner Predictor")

# ------------------ LOAD MODEL & ENCODERS ------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        logging.info("All model and encoder files loaded successfully.")
        return model, team_encoder, toss_encoder, venue_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model_files()

# ------------------ LOAD MATCH DATA ------------------
@st.cache_data
def load_matches():
    try:
        matches = pd.read_csv("all_matches.csv")
        deliveries = pd.read_csv("all_deliveries.csv")
        return matches, deliveries
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return pd.DataFrame(), pd.DataFrame()

matches, deliveries = load_matches()

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["🏆 Predict Winner", "📊 Top Players Stats", "🤖 IPL Chatbot"])

# ------------------ TAB 1: PREDICT WINNER ------------------
with tab1:
    st.header("Predict Match Winner")

    if model is None:
        st.error("Model or encoders not loaded. Cannot predict.")
    else:
        all_teams = sorted(list(team_encoder.classes_))
        all_venues = sorted(list(venue_encoder.classes_))
        all_toss_decisions = sorted(list(toss_encoder.classes_))

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", all_teams, index=0)
            team2 = st.selectbox("Select Team 2", all_teams, index=1)
        with col2:
            venue = st.selectbox("Select Venue", all_venues)
            toss_winner = st.selectbox("Toss Winner", [team1, team2])
            toss_decision = st.selectbox("Toss Decision", all_toss_decisions)

        if st.button("Predict Winner"):
            if team1 == team2:
                st.error("Team 1 and Team 2 cannot be the same.")
            else:
                try:
                    team1_enc = team_encoder.transform([team1])[0]
                    team2_enc = team_encoder.transform([team2])[0]
                    venue_enc = venue_encoder.transform([venue])[0]
                    toss_enc = toss_encoder.transform([toss_decision])[0]
                    toss_winner_enc = team_encoder.transform([toss_winner])[0]

                    team1_form = 0.5
                    team2_form = 0.5

                    features = pd.DataFrame({
                        'team1_enc':[team1_enc],
                        'team2_enc':[team2_enc],
                        'venue_enc':[venue_enc],
                        'toss_enc':[toss_enc],
                        'toss_winner_enc':[toss_winner_enc],
                        'team1_form':[team1_form],
                        'team2_form':[team2_form]
                    })

                    pred, prob = predict_match_winner(model, features, team1, team2)
                    st.success(f"🏆 Predicted Winner: **{pred}**")
                    st.write(f"Winning Probability - {team1}: {prob[0]*100:.1f}%, {team2}: {prob[1]*100:.1f}%")

                    # Head-to-head stats
                    st.subheader("📊 Head-to-Head Stats")
                    h2h = matches[((matches["team1"]==team1) & (matches["team2"]==team2)) |
                                  ((matches["team1"]==team2) & (matches["team2"]==team1))]
                    if h2h.empty:
                        st.write("No H2H data available.")
                    else:
                        total = len(h2h)
                        wins = h2h["winner"].value_counts()
                        st.write(f"Total Matches: {total}")
                        st.write(f"{team1} Wins: {wins.get(team1,0)}")
                        st.write(f"{team2} Wins: {wins.get(team2,0)}")
                        st.bar_chart(wins)

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# ------------------ TAB 2: TOP PLAYERS ------------------
with tab2:
    st.header("Top Batters & Bowlers")

    if deliveries.empty:
        st.error("Deliveries CSV not loaded.")
    else:
        # Top batters
        top_batters = deliveries.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_bat = px.bar(top_batters, x="batter", y="runs_scored", title="Top 10 Run Scorers", text="runs_scored")
        st.plotly_chart(fig_bat, use_container_width=True)

        # Top bowlers
        top_bowlers = deliveries.groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_bowl = px.bar(top_bowlers, x="bowler", y="is_wicket", title="Top 10 Wicket Takers", text="is_wicket")
        st.plotly_chart(fig_bowl, use_container_width=True)

# ------------------ TAB 3: IPL CHATBOT ------------------
with tab3:
    st.header("🤖 IPL Chatbot")
    st.info("Ask questions about IPL history, players, matches, etc.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("AI is thinking..."):
            # Since OpenAI/Gemini integration might vary, we can leave this for user key
            st.info("Chatbot response would appear here if API is configured.")
