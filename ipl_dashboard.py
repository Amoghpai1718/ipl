import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from predict_winner import predict_match_winner
import google.generativeai as genai

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- APP CONFIG ----------------------
st.set_page_config(page_title="IPL Advanced Dashboard", layout="wide")
st.title("🏏 IPL Advanced Analytics & Match Predictor")

# ---------------------- LOAD MODELS & ENCODERS ----------------------
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
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model_files()

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("all_matches.csv")
        deliveries = pd.read_csv("all_deliveries.csv")
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        return matches, deliveries
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}")
        return pd.DataFrame(), pd.DataFrame()

matches, deliveries = load_data()

# ---------------------- GOOGLE GEMINI CHATBOT ----------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception:
    model_gemini = None
    st.sidebar.warning("GEMINI_API_KEY not found. Chatbot disabled.")

# ---------------------- STREAMLIT TABS ----------------------
tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "🤖 IPL Chatbot"])

# ---------------------- TAB 1: MATCH WINNER ----------------------
with tab1:
    st.header("Predict Match Winner & Advanced Stats")

    # Inputs
    all_teams = sorted(list(team_encoder.classes_))
    all_venues = sorted(list(venue_encoder.classes_))
    all_toss = sorted(list(toss_encoder.classes_))

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

    if st.button("Predict Winner"):
        if team1 == team2:
            st.error("Team 1 and Team 2 cannot be the same.")
        else:
            try:
                # ENCODE
                team1_enc = team_encoder.transform([team1])[0]
                team2_enc = team_encoder.transform([team2])[0]
                venue_enc = venue_encoder.transform([venue])[0]
                toss_winner_enc = team_encoder.transform([toss_winner])[0]
                toss_decision_enc = toss_encoder.transform([toss_decision])[0]

                # HEAD-TO-HEAD WIN %
                h2h = matches[
                    ((matches["team1"]==team1) & (matches["team2"]==team2)) |
                    ((matches["team1"]==team2) & (matches["team2"]==team1))
                ]
                wins = h2h["winner"].value_counts()
                total = len(h2h)
                team1_win_pct = wins.get(team1,0)/total if total>0 else 0.5
                team2_win_pct = wins.get(team2,0)/total if total>0 else 0.5

                # INPUT FEATURE DF
                features = pd.DataFrame({
                    "team1_enc":[team1_enc],
                    "team2_enc":[team2_enc],
                    "venue_enc":[venue_enc],
                    "toss_winner_enc":[toss_winner_enc],
                    "toss_decision_enc":[toss_decision_enc],
                    "team1_form":[team1_form],
                    "team2_form":[team2_form],
                    "team1_win_pct":[team1_win_pct],
                    "team2_win_pct":[team2_win_pct]
                })

                # PREDICT
                pred, prob = predict_match_winner(
                    model, features, team1, team2,
                    team1_form, team2_form, team1_win_pct, team2_win_pct
                )

                winner = team1 if pred==1 else team2
                team1_prob = prob[1]*100
                team2_prob = prob[0]*100

                st.success(f"🏆 Predicted Winner: {winner}")
                st.write(f"Winning Probability: {team1}: {team1_prob:.1f}% | {team2}: {team2_prob:.1f}%")

                # ---------------- HEAD-TO-HEAD STATS ----------------
                st.subheader(f"📊 Head-to-Head: {team1} vs {team2}")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Matches", total)
                col_b.metric(f"{team1} Wins", wins.get(team1,0))
                col_c.metric(f"{team2} Wins", wins.get(team2,0))
                st.bar_chart(wins)

                # ---------------- TOP BATTERS & BOWLERS ----------------
                st.subheader("🔥 Key Players Stats")
                # BATTERS
                team_deliv = deliveries[deliveries["inning_team"].isin([team1, team2])]
                top_batters = team_deliv.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
                st.write("Top Batters")
                st.bar_chart(top_batters)

                # BOWLERS
                top_bowlers = team_deliv.groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
                st.write("Top Bowlers")
                st.bar_chart(top_bowlers)

            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                st.error(f"Prediction Error: {e}")

# ---------------------- TAB 2: CHATBOT ----------------------
with tab2:
    st.header("🤖 IPL Chatbot")
    if not model_gemini:
        st.error("Chatbot disabled. Add GEMINI_API_KEY to secrets.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask IPL question..."):
            st.session_state.messages.append({"role":"user","content":user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.spinner("AI is thinking..."):
                try:
                    chat_history = [{"role":"user","parts":[{"text":m["content"]}]} for m in st.session_state.messages[:-1]]
                    chat = model_gemini.start_chat(history=chat_history)
                    system_prompt = f"You are an expert IPL analyst. Answer based on historical data."
                    response = chat.send_message(f"{system_prompt}\n\nUser Question: {user_input}")
                    text = response.text
                    st.session_state.messages.append({"role":"assistant","content":text})
                    with st.chat_message("assistant"):
                        st.markdown(text)
                except Exception as e:
                    st.error(f"Chatbot Error: {e}")
