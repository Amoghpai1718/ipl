import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load datasets
# -------------------------------
matches = pd.read_csv("all_matches.csv")
deliveries = pd.read_csv("all_deliveries.csv")

# -------------------------------
# Load saved models & encoders
# -------------------------------
model = joblib.load("ipl_winner_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")

# -------------------------------
# Helper Functions
# -------------------------------

def calculate_form(team, matches_df, last_n=5):
    """Return normalized recent form of a team"""
    recent_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)].sort_values(by='date', ascending=False).head(last_n)
    wins = (recent_matches['winner'] == team).sum()
    return wins / last_n if last_n > 0 else 0.5

def calculate_h2h(team1, team2, matches_df):
    """Return head-to-head winning percentages"""
    h2h_matches = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
                             ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
    total = len(h2h_matches)
    if total == 0:
        return 0.5, 0.5
    team1_wins = (h2h_matches['winner'] == team1).sum()
    team2_wins = (h2h_matches['winner'] == team2).sum()
    return team1_wins/total, team2_wins/total

def prepare_features(team1, team2, venue, toss_winner):
    """Prepare features for prediction"""
    t1_form = calculate_form(team1, matches)
    t2_form = calculate_form(team2, matches)
    t1_h2h, t2_h2h = calculate_h2h(team1, team2, matches)

    # Encode categorical features
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = 0 if toss_winner == team1 else 1

    features = np.array([[team1_enc, team2_enc, venue_enc, toss_enc, t1_form, t2_form, t1_h2h, t2_h2h]])
    return features, t1_form, t2_form, t1_h2h, t2_h2h

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("IPL Match Winner Prediction")

teams = sorted(matches['team1'].unique())
venues = sorted(matches['venue'].unique())

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])

if st.button("Predict Winner"):
    features, t1_form, t2_form, t1_h2h, t2_h2h = prepare_features(team1, team2, venue, toss_winner)
    pred_probs = model.predict_proba(features)[0]
    winner_index = np.argmax(pred_probs)
    winner_team = team_encoder.inverse_transform([winner_index])[0]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Winner:** {winner_team}")
    st.write(f"{team1} Win Probability: {pred_probs[0]*100:.2f}%")
    st.write(f"{team2} Win Probability: {pred_probs[1]*100:.2f}%")
    st.write("---")
    st.write(f"**Additional Features:**")
    st.write(f"{team1} Form (0-1): {t1_form:.2f}")
    st.write(f"{team2} Form (0-1): {t2_form:.2f}")
    st.write(f"Head-to-Head {team1} Win %: {t1_h2h:.2f}")
    st.write(f"Head-to-Head {team2} Win %: {t2_h2h:.2f}")

