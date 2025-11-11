import streamlit as st
import pandas as pd
import joblib
import requests

from predict_winner import predict_match_winner

# -----------------------------
# Load model and encoders
# -----------------------------
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")

# Load match data for head-to-head stats
matches = pd.read_csv("all_matches.csv")

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")
st.title("IPL Match Predictor & AI Chatbot")

# Tabs for Prediction and Chatbot
tab1, tab2 = st.tabs(["Predict Match Winner", "AI Chatbot"])

# -----------------------------
# Tab 1: Match Prediction
# -----------------------------
with tab1:
    st.header("Predict Match Winner")

    # Select teams
    teams = matches['team1'].unique().tolist()
    team1 = st.selectbox("Select Team 1", teams, key="team1_select")
    team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1], key="team2_select")

    # Select venue
    venues = matches['venue'].unique().tolist()
    venue = st.selectbox("Select Venue", venues, key="venue_select")

    # Toss winner
    toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss_select")

    # Recent form sliders
    team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="form1")
    team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="form2")

    # Past head-to-head stats
    team1_matches = matches[(matches['team1'] == team1) | (matches['team2'] == team1)]
    team2_matches = matches[(matches['team1'] == team2) | (matches['team2'] == team2)]
    h2h_matches = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                          ((matches['team1'] == team2) & (matches['team2'] == team1))]

    team1_wins = ((h2h_matches['winner'] == team1).sum())
    team2_wins = ((h2h_matches['winner'] == team2).sum())
    total_h2h = len(h2h_matches)

    team1_win_pct = team1_wins / total_h2h if total_h2h > 0 else 0.5
    team2_win_pct = team2_wins / total_h2h if total_h2h > 0 else 0.5

    if st.button("Predict Winner"):
        try:
            winner, win_prob = predict_match_winner(
                team1, team2, venue, toss_winner, 
                team1_form, team2_form, 
                team1_win_pct, team2_win_pct
            )

            st.success(f"Predicted Winner: {winner}")
            st.info(f"Winning Probability: {win_prob*100:.2f}%")

            st.subheader("Head-to-Head Stats")
            st.write(f"Total Matches Between {team1} & {team2}: {total_h2h}")
            st.write(f"{team1} Wins: {team1_wins}")
            st.write(f"{team2} Wins: {team2_wins}")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# -----------------------------
# Tab 2: AI Chatbot
# -----------------------------
with tab2:
    st.header("Ask IPL Questions")

    user_query = st.text_input("Ask your question about IPL teams, players, or stats:")

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    GOOGLE_SEARCH_CX = st.secrets.get("GOOGLE_SEARCH_CX", "")

    if st.button("Get Answer"):
        if not user_query.strip():
            st.warning("Please type a question.")
        elif not GOOGLE_API_KEY or not GOOGLE_SEARCH_CX:
            st.error("Google API keys not configured.")
        else:
            try:
                params = {
                    "q": user_query,
                    "cx": GOOGLE_SEARCH_CX,
                    "key": GOOGLE_API_KEY,
                    "num": 3
                }
                response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
                data = response.json()

                answers = []
                for item in data.get("items", []):
                    title = item.get("title")
                    snippet = item.get("snippet")
                    link = item.get("link")
                    answers.append(f"**{title}**\n{snippet}\n<{link}>\n")

                if answers:
                    st.markdown("\n---\n".join(answers))
                else:
                    st.info("No results found.")

            except Exception as e:
                st.error(f"Chatbot Error: {e}")
