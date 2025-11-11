import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai
import logging
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ App Config ------------------
st.set_page_config(page_title="IPL Advanced Dashboard", layout="wide")

# ------------------ Load Model & Encoders ------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("ipl_winner_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        logging.info("Model and encoders loaded successfully.")
        return model, team_encoder, toss_encoder, venue_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None

model, team_encoder, toss_encoder, venue_encoder = load_model_files()

# ------------------ Load Match Data ------------------
@st.cache_data
def load_matches():
    try:
        df = pd.read_csv("all_matches.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading all_matches.csv: {e}")
        return pd.DataFrame()

matches = load_matches()

@st.cache_data
def load_deliveries():
    try:
        df = pd.read_csv("all_deliveries.csv")
        return df
    except Exception as e:
        st.error(f"Error loading all_deliveries.csv: {e}")
        return pd.DataFrame()

deliveries = load_deliveries()

# ------------------ Google Gemini Chatbot ------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

if api_key:
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    model_gemini = None

# ------------------ Tabs ------------------
st.title("🏏 IPL Advanced Dashboard & Winner Predictor")

if model is None or matches.empty:
    st.error("Essential files missing. Cannot run the app.")
else:
    tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "📊 Player Stats & Visuals", "🤖 Ask IPL Chatbot"])

    # ------------------ TAB 1: Winner Prediction ------------------
    with tab1:
        st.header("Predict Match Winner")

        all_teams = sorted(list(team_encoder.classes_))
        all_venues = sorted(list(venue_encoder.classes_))
        all_toss = sorted(list(toss_encoder.classes_))

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", all_teams, index=0)
            team2 = st.selectbox("Select Team 2", all_teams, index=1)
            team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)
        with col2:
            venue = st.selectbox("Select Venue", all_venues)
            toss_winner = st.selectbox("Toss Winner", [team1, team2])
            toss_decision = st.selectbox("Toss Decision", all_toss)
            team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)

        if st.button("Predict Winner"):
            if team1 == team2:
                st.error("Team 1 and Team 2 cannot be the same.")
            else:
                try:
                    # Encode features
                    team1_enc = team_encoder.transform([team1])[0]
                    team2_enc = team_encoder.transform([team2])[0]
                    venue_enc = venue_encoder.transform([venue])[0]
                    toss_winner_enc = team_encoder.transform([toss_winner])[0]
                    toss_decision_enc = toss_encoder.transform([toss_decision])[0]

                    # Create input DataFrame with correct column order
                    input_data = pd.DataFrame({
                        'team1_enc':[team1_enc],
                        'team2_enc':[team2_enc],
                        'venue_enc':[venue_enc],
                        'toss_winner_enc':[toss_winner_enc],
                        'toss_decision_enc':[toss_decision_enc],
                        'team1_form':[team1_form],
                        'team2_form':[team2_form]
                    })

                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0]

                    winner = team1 if pred == 1 else team2
                    team1_prob = prob[1]*100
                    team2_prob = prob[0]*100

                    st.success(f"🏆 Predicted Winner: {winner}")
                    st.write(f"Confidence for {team1}: {team1_prob:.1f}%")
                    st.write(f"Confidence for {team2}: {team2_prob:.1f}%")

                    # Head-to-Head Analysis
                    st.subheader(f"📊 Head-to-Head: {team1} vs {team2}")
                    h2h = matches[((matches['team1']==team1)&(matches['team2']==team2))|
                                  ((matches['team1']==team2)&(matches['team2']==team1))].copy()
                    if h2h.empty:
                        st.write("No historical data.")
                    else:
                        total_matches = len(h2h)
                        wins = h2h['winner'].value_counts()
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Matches", total_matches)
                        col_b.metric(f"{team1} Wins", wins.get(team1,0))
                        col_c.metric(f"{team2} Wins", wins.get(team2,0))
                        st.bar_chart(wins)

                        # Venue-specific
                        venue_h2h = h2h[h2h['venue']==venue]
                        if not venue_h2h.empty:
                            st.write(f"Venue Stats ({venue}):")
                            venue_wins = venue_h2h['winner'].value_counts()
                            st.write(f"{team1} Wins: {venue_wins.get(team1,0)} | {team2} Wins: {venue_wins.get(team2,0)}")
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
                    logging.error(e)

    # ------------------ TAB 2: Player Stats & Visuals ------------------
    with tab2:
        st.header("📊 Top Batters & Bowlers")
        if deliveries.empty:
            st.warning("Deliveries data not loaded.")
        else:
            team_filter = st.selectbox("Select Team for Stats", sorted(deliveries['inning_team'].unique()))
            top_batters = deliveries[deliveries['inning_team']==team_filter].groupby('batter')['runs_scored'].sum().sort_values(ascending=False).head(10)
            top_bowlers = deliveries[deliveries['inning_team']==team_filter].groupby('bowler')['is_wicket'].sum().sort_values(ascending=False).head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Top Batters - {team_filter}")
                fig1 = px.bar(top_batters, x=top_batters.index, y=top_batters.values,
                              labels={'x':'Batter','y':'Runs'}, text=top_batters.values)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.subheader(f"Top Bowlers - {team_filter}")
                fig2 = px.bar(top_bowlers, x=top_bowlers.index, y=top_bowlers.values,
                              labels={'x':'Bowler','y':'Wickets'}, text=top_bowlers.values)
                st.plotly_chart(fig2, use_container_width=True)

    # ------------------ TAB 3: Chatbot ------------------
    with tab2 if not deliveries.empty else tab2:
        st.header("🤖 Ask IPL Chatbot")
        if not model_gemini:
            st.error("Chatbot disabled. Add GEMINI_API_KEY to secrets.")
        else:
            if "messages" not in st.session_state:
                st.session_state.messages=[]
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if user_input := st.chat_input("Ask anything IPL related"):
                st.session_state.messages.append({"role":"user","content":user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.spinner("AI Analyst is thinking..."):
                    try:
                        chat = model_gemini.start_chat()
                        system_prompt = f"""
                        You are an expert IPL analyst.
                        Answer user questions based on historical data and statistics.
                        """
                        response = chat.send_message(f"{system_prompt}\n\nQuestion: {user_input}")
                        st.session_state.messages.append({"role":"assistant","content":response.text})
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Chatbot error: {e}")
