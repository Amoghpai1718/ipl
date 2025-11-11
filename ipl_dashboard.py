import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# -----------------------------
# Load Models & Encoders
# -----------------------------
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")

# Load historical data
matches = pd.read_csv("all_matches.csv")
deliveries = pd.read_csv("all_deliveries.csv")

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Advanced IPL Predictor", layout="wide")

tabs = st.tabs(["Match Prediction", "IPL Chatbot"])

# -----------------------------
# Tab 1: Match Prediction
# -----------------------------
with tabs[0]:
    st.header("IPL Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1", sorted(matches['team1'].unique()))
        team2 = st.selectbox("Select Team 2", sorted(matches['team2'].unique()))
        venue = st.selectbox("Select Venue", sorted(matches['venue'].unique()))
        
    with col2:
        toss_winner = st.selectbox("Select Toss Winner", [team1, team2])
        form_team1 = st.slider(f"Recent Form of {team1} (0=poor, 1=excellent)", 0.0, 1.0, 0.5)
        form_team2 = st.slider(f"Recent Form of {team2} (0=poor, 1=excellent)", 0.0, 1.0, 0.5)
    
    if st.button("Predict Match Outcome"):
        # -----------------------------
        # Feature Engineering
        # -----------------------------
        # Head-to-head win %
        h2h_matches = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                              ((matches['team1'] == team2) & (matches['team2'] == team1))]
        if len(h2h_matches) == 0:
            h2h_win_team1 = 0.5
        else:
            h2h_win_team1 = np.mean(h2h_matches['winner'] == team1)
        
        # Encode categorical features
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]
        toss_enc = toss_encoder.transform([toss_winner])[0]
        
        # Prepare feature vector
        features = np.array([[team1_enc, team2_enc, venue_enc, toss_enc, form_team1, form_team2, h2h_win_team1]])
        
        # Predict probability
        pred_probs = model.predict_proba(features)[0]
        winner_idx = np.argmax(pred_probs)
        winner_team = team_encoder.inverse_transform([winner_idx])[0]
        
        st.subheader("Prediction Result")
        st.write(f"**Most Likely Winner:** {winner_team}")
        st.write(f"**Winning Probability:** {pred_probs[0]*100:.2f}% vs {pred_probs[1]*100:.2f}%")
        
        st.subheader("Detailed Analysis")
        st.write(f"- Head-to-Head win % of {team1} vs {team2}: {h2h_win_team1*100:.2f}%")
        st.write(f"- Recent Form: {team1}: {form_team1}, {team2}: {form_team2}")
        st.write(f"- Toss Winner: {toss_winner} may get advantage depending on historical stats")
        
# -----------------------------
# Tab 2: IPL Chatbot
# -----------------------------
with tabs[1]:
    st.header("IPL AI Chatbot")
    
    user_input = st.text_input("Ask anything about IPL, teams, or players:")
    
    if st.button("Send"):
        if user_input.strip() != "":
            # For demo purposes, we'll use a simple response. Replace this with your AI API call
            # Example: Google Generative AI / OpenAI API call
            response = f"🤖 Chatbot Response: Sorry, I can't process '{user_input}' yet. Integrate AI API here."
            st.write(response)
