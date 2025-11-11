import streamlit as st
import pandas as pd
import joblib
from predict_winner import predict_match_winner

# ------------------- Load models & encoders -------------------
model = joblib.load("ipl_model.pkl")  # your trained XGBoost model
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")

# ------------------- Load match data -------------------
matches = pd.read_csv("all_matches.csv")
deliveries = pd.read_csv("all_deliveries.csv")

st.title("IPL Match Outcome Predictor")
st.subheader("Advanced Match Analysis & Prediction")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("Match Inputs")
team1 = st.sidebar.selectbox("Select Team 1", matches['team1'].unique(), key="team1")
team2 = st.sidebar.selectbox("Select Team 2", matches['team2'].unique(), key="team2")
venue = st.sidebar.selectbox("Select Venue", matches['venue'].unique(), key="venue")
toss_winner = st.sidebar.selectbox("Toss Winner", [team1, team2], key="toss")
team1_form = st.sidebar.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="team1_form")
team2_form = st.sidebar.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="team2_form")

# ------------------- Encode Inputs -------------------
team1_enc = team_encoder.transform([team1])[0]
team2_enc = team_encoder.transform([team2])[0]
venue_enc = venue_encoder.transform([venue])[0]
toss_enc = toss_encoder.transform([toss_winner])[0]

# ------------------- Compute Head-to-Head -------------------
h2h_matches = matches[((matches['team1']==team1) & (matches['team2']==team2)) |
                      ((matches['team1']==team2) & (matches['team2']==team1))]

team1_wins = len(h2h_matches[h2h_matches['winner']==team1])
team2_wins = len(h2h_matches[h2h_matches['winner']==team2])
total_h2h = team1_wins + team2_wins

team1_win_pct = round(team1_wins/total_h2h,2) if total_h2h>0 else 0
team2_win_pct = round(team2_wins/total_h2h,2) if total_h2h>0 else 0

# ------------------- Prediction -------------------
if st.sidebar.button("Predict Winner"):
    features = pd.DataFrame([[
        team1_enc, team2_enc, venue_enc, toss_enc, team1_win_pct, team2_win_pct, team1_form, team2_form
    ]], columns=[
        'team1_enc', 'team2_enc', 'venue_enc', 'toss_enc', 
        'team1_win_pct', 'team2_win_pct', 'team1_form', 'team2_form'
    ])
    
    try:
        pred_probs = model.predict_proba(features)[0]
        winner_idx = pred_probs.argmax()
        winner_name = [team1, team2][winner_idx]
        st.success(f"Predicted Winner: {winner_name}")
        st.info(f"{team1}: {round(pred_probs[0]*100,2)}% | {team2}: {round(pred_probs[1]*100,2)}%")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# ------------------- Head-to-Head Analysis -------------------
st.subheader("Head-to-Head Analysis")
st.write(f"Total Matches Played: {total_h2h}")
st.write(f"{team1} Wins: {team1_wins}")
st.write(f"{team2} Wins: {team2_wins}")

# ------------------- Optional Chatbot Tab -------------------
st.subheader("Ask About Teams / Matches")
user_query = st.text_input("Enter your question here:", key="chat_input")
if st.button("Get Answer", key="chat_button"):
    # Placeholder chatbot response (can integrate AI later)
    st.write(f"Answer to '{user_query}': Currently this feature is in development.")
