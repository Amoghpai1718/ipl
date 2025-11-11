import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
import logging

from sklearn.preprocessing import LabelEncoder

# --------------------- Logging ---------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------- App Config ---------------------
st.set_page_config(page_title="IPL AI Dashboard", layout="wide")
st.title("🏏 IPL Advanced Analytics & Match Winner Predictor")

# --------------------- Load Model & Encoders ---------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        return model, team_encoder, toss_encoder, venue_encoder, winner_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model_files()

# --------------------- Load Data ---------------------
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("all_matches.csv")
        deliveries = pd.read_csv("all_deliveries.csv")
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        return matches, deliveries
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}")
        return pd.DataFrame(), pd.DataFrame()

matches, deliveries = load_data()

# --------------------- Safe Encoding ---------------------
def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return 0  # default encoding for unseen labels

# --------------------- Google Gemini Chatbot ---------------------
api_key = st.secrets.get("GEMINI_API_KEY", None)
if api_key:
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    model_gemini = None

# --------------------- Tabs ---------------------
tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "🤖 IPL Chatbot"])

# --------------------- TAB 1: Match Prediction ---------------------
with tab1:
    st.header("Predict Match Winner")
    if model is None or matches.empty:
        st.error("Model or match data not loaded.")
    else:
        all_teams = sorted(team_encoder.classes_)
        all_venues = sorted(venue_encoder.classes_)
        all_toss = sorted(toss_encoder.classes_)

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", all_teams, index=0)
            team2 = st.selectbox("Select Team 2", all_teams, index=1)
        with col2:
            venue = st.selectbox("Select Venue", all_venues)
            toss_winner = st.selectbox("Toss Winner", [team1, team2])
            toss_decision = st.selectbox("Toss Decision", all_toss)

        team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)
        team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)

        # Historical win percentage
        def calc_win_pct(team):
            team_matches = matches[(matches['team1']==team)|(matches['team2']==team)]
            if team_matches.empty:
                return 0.5
            wins = team_matches['winner'].value_counts().get(team, 0)
            return wins / len(team_matches)

        team1_win_pct = calc_win_pct(team1)
        team2_win_pct = calc_win_pct(team2)

        if st.button("Predict Winner"):
            if team1 == team2:
                st.error("Team 1 and Team 2 cannot be the same!")
            else:
                try:
                    input_data = pd.DataFrame({
                        'team1_enc': [safe_encode(team_encoder, team1)],
                        'team2_enc': [safe_encode(team_encoder, team2)],
                        'venue_enc': [safe_encode(venue_encoder, venue)],
                        'toss_enc': [safe_encode(team_encoder, toss_winner)],
                        'toss_decision_enc': [safe_encode(toss_encoder, toss_decision)],
                        'team1_form': [team1_form],
                        'team2_form': [team2_form],
                        'team1_win_pct': [team1_win_pct],
                        'team2_win_pct': [team2_win_pct]
                    })

                    input_data = input_data[['team1_enc','team2_enc','venue_enc','toss_enc',
                                             'toss_decision_enc','team1_form','team2_form',
                                             'team1_win_pct','team2_win_pct']]

                    pred = model.predict(input_data)[0]
                    probs = model.predict_proba(input_data)[0]
                    winner = team1 if pred == 1 else team2
                    team1_prob = probs[1]*100
                    team2_prob = probs[0]*100

                    st.success(f"🏆 Predicted Winner: **{winner}**")
                    st.write(f"Confidence - {team1}: {team1_prob:.1f}%, {team2}: {team2_prob:.1f}%")

                    # Head-to-head
                    h2h = matches[((matches["team1"]==team1)&(matches["team2"]==team2)) |
                                  ((matches["team1"]==team2)&(matches["team2"]==team1))]
                    if h2h.empty:
                        st.info("No head-to-head data available")
                    else:
                        st.subheader("📊 Head-to-Head Stats")
                        st.write(f"Total Matches: {len(h2h)}")
                        wins = h2h['winner'].value_counts()
                        st.write(f"{team1} Wins: {wins.get(team1,0)}")
                        st.write(f"{team2} Wins: {wins.get(team2,0)}")
                        st.bar_chart(wins)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# --------------------- TAB 2: Chatbot ---------------------
with tab2:
    st.header("🤖 Ask IPL Chatbot")
    if not model_gemini:
        st.error("GEMINI_API_KEY not set. Chatbot disabled.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask IPL AI Analyst..."):
            st.session_state.messages.append({"role":"user","content":user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.spinner("AI is thinking..."):
                try:
                    chat_history_for_api = []
                    for msg in st.session_state.messages:
                        chat_history_for_api.append({"role":"user" if msg["role"]=="user" else "model",
                                                     "parts":[{"text": msg["content"]}]})
                    chat = model_gemini.start_chat(history=chat_history_for_api[:-1])
                    system_prompt = f"You are an expert IPL analyst. Answer based on historical data."
                    response = chat.send_message(f"{system_prompt}\n\nUser Question: {user_input}")
                    st.session_state.messages.append({"role":"assistant","content":response.text})
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Chatbot error: {e}")

