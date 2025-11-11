import streamlit as st
import pandas as pd
import joblib
from predict_winner import predict_match_winner
import openai  # Make sure openai is in requirements.txt

# -----------------------------
# Load trained model and encoders
# -----------------------------
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")

# -----------------------------
# Load match & delivery data for analysis
# -----------------------------
matches = pd.read_csv("all_matches.csv")
deliveries = pd.read_csv("all_deliveries.csv")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")
st.title("Advanced IPL Match Predictor")

# Tabs
tab1, tab2 = st.tabs(["Predictor", "AI Assistant"])

# -----------------------------
# Predictor Tab
# -----------------------------
with tab1:
    st.header("Predict Match Winner")

    team1 = st.selectbox("Select Team 1", sorted(team_encoder.classes_), key="team1")
    team2 = st.selectbox("Select Team 2", sorted(team_encoder.classes_), key="team2")
    venue = st.selectbox("Select Venue", sorted(venue_encoder.classes_), key="venue")
    toss_winner = st.selectbox("Toss Winner", sorted(team_encoder.classes_), key="toss")

    team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="team1_form")
    team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="team2_form")

    # Head-to-head calculation
    def get_win_percentage(team_a, team_b):
        matches_between = matches[((matches['team1']==team_a) & (matches['team2']==team_b)) | 
                                  ((matches['team1']==team_b) & (matches['team2']==team_a))]
        if len(matches_between) == 0:
            return 0.5, 0.5
        team_a_wins = matches_between[matches_between['winner']==team_a].shape[0]
        team_b_wins = matches_between[matches_between['winner']==team_b].shape[0]
        total = team_a_wins + team_b_wins
        if total == 0:
            return 0.5, 0.5
        return team_a_wins/total, team_b_wins/total

    team1_win_pct, team2_win_pct = get_win_percentage(team1, team2)

    if st.button("Predict Winner"):
        try:
            pred, pred_probs = predict_match_winner(
                model, team_encoder, venue_encoder, toss_encoder,
                team1, team2, venue, toss_winner,
                team1_win_pct, team2_win_pct, team1_form, team2_form
            )

            st.subheader("Match Prediction")
            st.write(f"**Predicted Winner:** {pred}")
            st.write(f"**Winning Probability:** {pred_probs[pred]:.2%}")

            # Head-to-head summary
            st.subheader("Head-to-Head Analysis")
            st.write(f"**{team1} win % vs {team2}:** {team1_win_pct:.2%}")
            st.write(f"**{team2} win % vs {team1}:** {team2_win_pct:.2%}")
            st.write(f"**Recent Form:** {team1}: {team1_form}, {team2}: {team2_form}")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

 # ---------------- CHATBOT SECTION ----------------
import os
from googleapiclient.discovery import build

def ask_chatbot(query):
    """
    Uses Google Custom Search API to get a response for the query.
    """
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
    res = service.cse().list(
        q=query,
        cx=os.getenv("GOOGLE_SEARCH_CX"),
        num=1,
    ).execute()

    if "items" in res:
        snippet = res['items'][0]['snippet']
        return snippet
    else:
        return "No results found. Try another query."

# Streamlit UI
with st.expander("Powered Chatbot"):
    user_query = st.text_input("Ask about IPL, Teams, Stats...")
    if st.button("Send"):
        if user_query:
            response = ask_chatbot(user_query)
            st.write(response)
