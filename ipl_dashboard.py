import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import requests
from predict_winner import predict_match_winner
import os

# ==========================================================
# 1. APP CONFIG
# ==========================================================
st.set_page_config(page_title="IPL AI Assistant", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: white; }
    .stButton>button { background-color: #FF6F00; color: white; border-radius:10px; }
    .stMetric { background-color: #1E1E1E; color:white; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# 2. LOAD MODEL & ENCODERS
# ==========================================================
@st.cache_resource
def load_model_files():
    try:
        if not all(os.path.exists(f) for f in [
            "ipl_model.pkl", "team_encoder.pkl", "toss_encoder.pkl",
            "venue_encoder.pkl", "winner_encoder.pkl"
        ]):
            raise FileNotFoundError("Model or encoders missing in directory.")

        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")

        return model, team_encoder, toss_encoder, venue_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model_files()

if not model:
    st.error("Model or encoders are missing. Upload .pkl files and restart.")
    st.stop()

# ==========================================================
# 3. LOAD CSV DATA
# ==========================================================
@st.cache_data
def load_match_data():
    return pd.read_csv("all_matches.csv")

@st.cache_data
def load_delivery_data():
    return pd.read_csv("all_deliveries.csv")

matches = load_match_data()
deliveries = load_delivery_data()

# ==========================================================
# 4. STREAMLIT TABS
# ==========================================================
tab1, tab2 = st.tabs(["🏏 Match Predictor & Analysis", "🤖 IPL Chatbot"])

# ==========================================================
# TAB 1 — MATCH PREDICTOR
# ==========================================================
with tab1:
    st.header("🏏 IPL Match Predictor & Deep Dive Dashboard")

    all_teams = sorted(team_encoder.classes_)
    all_venues = sorted(venue_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=0)
        team2 = st.selectbox("Select Team 2", all_teams, index=1)
    with col2:
        venue = st.selectbox("Select Venue", all_venues)
        toss_winner = st.selectbox("Select Toss Winner", [team1, team2])

    st.markdown("### Team Form Inputs (0 = poor, 1 = excellent)")
    colf1, colf2 = st.columns(2)
    with colf1:
        team1_form = st.slider(f"{team1} Recent Form", 0.0, 1.0, 0.5, 0.01)
    with colf2:
        team2_form = st.slider(f"{team2} Recent Form", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict Winner"):
        if team1 == team2:
            st.error("Team 1 and Team 2 cannot be the same.")
        else:
            try:
                # Head-to-Head stats
                h2h = matches[
                    ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                    ((matches["team1"] == team2) & (matches["team2"] == team1))
                ]
                team1_wins = (h2h["winner"] == team1).sum()
                team2_wins = (h2h["winner"] == team2).sum()
                total_h2h = len(h2h)
                team1_win_pct = team1_wins / max(total_h2h, 1)
                team2_win_pct = team2_wins / max(total_h2h, 1)

                # Predict
                winner, win_probs = predict_match_winner(
                    model, team_encoder, venue_encoder, toss_encoder,
                    team1, team2, venue, toss_winner,
                    team1_form, team2_form, team1_win_pct, team2_win_pct
                )

                # --- Display Prediction
                st.subheader(f"🏆 Predicted Winner: {winner}")

                # Smooth Pie chart using Plotly
                fig = go.Figure(data=[
                    go.Pie(
                        labels=[team1, team2],
                        values=[win_probs[team1], win_probs[team2]],
                        hole=0.4,
                        marker_colors=["#FF6F00", "#1E90FF"],
                    )
                ])
                fig.update_traces(textinfo="label+percent", pull=[0.05, 0])
                fig.update_layout(
                    showlegend=False, title="Win Probability", height=400,
                    paper_bgcolor="#0E1117", font_color="white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Head-to-Head Display
                st.markdown("### 📊 Head-to-Head Record")
                st.write(f"Total Matches: {total_h2h}")
                st.write(f"{team1} Wins: {team1_wins}")
                st.write(f"{team2} Wins: {team2_wins}")

                # ======================================================
                # 5. TEAM AVERAGES AND PLAYER STATS (Fixed)
                # ======================================================
                st.markdown("---")
                st.subheader("📊 Team Averages & Key Player Stats")

                # Identify correct batting and bowling column names safely
                bat_col = None
                for c in ["batting_team", "bat_team", "innings_team"]:
                    if c in deliveries.columns:
                        bat_col = c
                        break

                bowl_col = None
                for c in ["bowling_team", "bowl_team", "opponent_team"]:
                    if c in deliveries.columns:
                        bowl_col = c
                        break

                if not bat_col or not bowl_col:
                    st.warning("Dataset missing batting or bowling team columns.")
                else:
                    # Calculate team averages at venue
                    team1_venue = deliveries[
                        (deliveries[bat_col] == team1) &
                        (deliveries["venue"] == venue)
                    ]
                    team2_venue = deliveries[
                        (deliveries[bat_col] == team2) &
                        (deliveries["venue"] == venue)
                    ]

                    def team_stats(df):
                        if df.empty:
                            return 0, 0
                        total_runs = df.groupby("match_id")["runs_scored"].sum()
                        total_wkts = df.groupby("match_id")["is_wicket"].sum()
                        return round(total_runs.mean(), 2), round(total_wkts.mean(), 2)

                    team1_avg_runs, team1_avg_wkts = team_stats(team1_venue)
                    team2_avg_runs, team2_avg_wkts = team_stats(team2_venue)

                    colA, colB = st.columns(2)
                    with colA:
                        st.metric(f"{team1} Avg Score @ {venue}", f"{team1_avg_runs} runs")
                        st.metric(f"{team1} Avg Wickets Lost", f"{team1_avg_wkts}")
                    with colB:
                        st.metric(f"{team2} Avg Score @ {venue}", f"{team2_avg_runs} runs")
                        st.metric(f"{team2} Avg Wickets Lost", f"{team2_avg_wkts}")

                    # Filter only matches between selected teams
                    vs_df = deliveries[
                        ((deliveries[bat_col] == team1) & (deliveries[bowl_col] == team2)) |
                        ((deliveries[bat_col] == team2) & (deliveries[bowl_col] == team1))
                    ]

                    # Top Batters
                    st.markdown("### 🔥 Top Batters in Head-to-Head Matches")
                    top_batters = (
                        vs_df.groupby("batter")["runs_scored"]
                        .sum().sort_values(ascending=False).head(5)
                    )
                    st.bar_chart(top_batters)

                    # Top Bowlers
                    st.markdown("### 🎯 Top Bowlers in Head-to-Head Matches")
                    top_bowlers = (
                        vs_df[vs_df["is_wicket"] == 1]
                        .groupby("bowler")["is_wicket"].sum()
                        .sort_values(ascending=False).head(5)
                    )
                    st.bar_chart(top_bowlers)

            except Exception as e:
                st.error(f"Error in prediction: {e}")

# ==========================================================
# TAB 2 — CHATBOT
# ==========================================================
with tab2:
    st.header("🤖 IPL Chatbot Assistant")

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    GOOGLE_SEARCH_KEY = st.secrets.get("GOOGLE_SEARCH_KEY")
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX")

    if not (GOOGLE_API_KEY and GOOGLE_SEARCH_KEY and GOOGLE_CX):
        st.warning("Google API keys missing. Add them in Streamlit Secrets.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role']}:** {msg['content']}")

        user_query = st.text_input("Ask your IPL question:")
        if st.button("Send") and user_query:
            try:
                # Use Google Custom Search API
                url = (
                    f"https://www.googleapis.com/customsearch/v1?"
                    f"q={user_query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num=3"
                )
                response = requests.get(url).json()

                if "items" in response:
                    answer = "\n\n".join(
                        [f"**{item['title']}**\n{item['snippet']}" for item in response["items"]]
                    )
                else:
                    answer = "No relevant results found."

                st.session_state.chat_history.append({"role": "User", "content": user_query})
                st.session_state.chat_history.append({"role": "AI", "content": answer})

                st.markdown(f"**AI:** {answer}")
            except Exception as e:
                st.error(f"Chatbot error: {e}")
