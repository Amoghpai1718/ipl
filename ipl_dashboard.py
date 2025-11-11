import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px

from predict_winner import predict_match_winner

# -----------------------------
# Load model and encoders
# -----------------------------
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")

# Load match & deliveries data
matches = pd.read_csv("all_matches.csv")
deliveries = pd.read_csv("all_deliveries.csv")

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")
st.title("IPL Match Predictor & AI Chatbot")

# Tabs
tab1, tab2 = st.tabs(["Predict Match Winner", "AI Chatbot"])

# -----------------------------
# Tab 1: Match Prediction & Advanced Stats
# -----------------------------
with tab1:
    st.header("Predict Match Winner")

    # Select teams and venue
    teams = matches['team1'].unique().tolist()
    team1 = st.selectbox("Select Team 1", teams, key="team1_select")
    team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1], key="team2_select")
    venues = matches['venue'].unique().tolist()
    venue = st.selectbox("Select Venue", venues, key="venue_select")
    toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss_select")
    team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="form1")
    team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="form2")

    # H2H stats
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

            # -----------------------------
            # Head-to-Head Charts
            # -----------------------------
            st.subheader("Head-to-Head Stats")
            h2h_df = pd.DataFrame({
                "Team": [team1, team2],
                "Wins": [team1_wins, team2_wins]
            })
            fig_h2h = px.bar(h2h_df, x="Team", y="Wins", color="Team", text="Wins",
                              title=f"H2H Wins: {team1} vs {team2}")
            st.plotly_chart(fig_h2h, use_container_width=True)

            # -----------------------------
            # Top Players Visualization
            # -----------------------------
            st.subheader("Top Batsmen & Bowlers")
            relevant_deliveries = deliveries[(deliveries['batting_team'].isin([team1, team2])) &
                                             (deliveries['bowling_team'].isin([team1, team2]))]

            # Top batsmen
            top_batsmen = relevant_deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(5).reset_index()
            fig_bat = px.bar(top_batsmen, x='batsman', y='batsman_runs', text='batsman_runs', title="Top 5 Batsmen")
            st.plotly_chart(fig_bat, use_container_width=True)

            # Top bowlers
            top_bowlers = relevant_deliveries.groupby('bowler')['total_runs'].sum().sort_values().head(5).reset_index()
            fig_bowl = px.bar(top_bowlers, x='bowler', y='total_runs', text='total_runs', title="Top 5 Bowlers (Least Runs Conceded)")
            st.plotly_chart(fig_bowl, use_container_width=True)

            # Key Players
            st.subheader("Key Players Summary")
            key_players = pd.DataFrame({
                "Player": list(top_batsmen['batsman']) + list(top_bowlers['bowler']),
                "Type": ["Batsman"]*5 + ["Bowler"]*5
            })
            st.table(key_players)

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
