import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import requests
from predict_winner import predict_match_winner

# ------------------------------------------------
# 1. PAGE CONFIG & STYLE
# ------------------------------------------------
st.set_page_config(page_title="IPL AI Dashboard", layout="wide")
st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .stButton>button { background-color: #FF6F00; color: white; border-radius: 8px; }
    .stSelectbox, .stSlider { color: black; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 2. LOAD MODEL & ENCODERS
# ------------------------------------------------
@st.cache_resource
def load_model():
    """Safely load model and encoders"""
    try:
        required_files = [
            "ipl_model.pkl",
            "team_encoder.pkl",
            "venue_encoder.pkl",
            "toss_encoder.pkl",
            "winner_encoder.pkl"
        ]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            st.error(f"Missing files: {', '.join(missing)}")
            return None, None, None, None, None

        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        return model, team_encoder, venue_encoder, toss_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None, None

model, team_encoder, venue_encoder, toss_encoder, winner_encoder = load_model()

# ------------------------------------------------
# 3. LOAD MATCH & DELIVERY DATA
# ------------------------------------------------
@st.cache_data
def load_csv_data():
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    return matches, deliveries

matches, deliveries = load_csv_data()

# ------------------------------------------------
# 4. MAIN TABS
# ------------------------------------------------
tab1, tab2 = st.tabs(["🏏 Match Predictor", "🤖 IPL Chatbot"])

# ------------------------------------------------
# TAB 1: MATCH PREDICTOR
# ------------------------------------------------
with tab1:
    st.header("🏏 IPL Match Predictor & Deep Dive Dashboard")

    if model is None:
        st.warning("Model or encoders are missing. Please ensure all .pkl files exist.")
        st.stop()

    all_teams = sorted(team_encoder.classes_)
    all_venues = sorted(venue_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=0)
        team2 = st.selectbox("Select Team 2", [t for t in all_teams if t != team1], index=0)
    with col2:
        venue = st.selectbox("Select Venue", all_venues)
        toss_winner = st.selectbox("Select Toss Winner", [team1, team2])

    team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)
    team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict Winner"):
        try:
            # Head-to-Head Stats
            h2h = matches[
                ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                ((matches["team1"] == team2) & (matches["team2"] == team1))
            ]
            team1_wins = (h2h["winner"] == team1).sum()
            team2_wins = (h2h["winner"] == team2).sum()
            total_h2h = len(h2h)
            team1_win_pct = team1_wins / total_h2h if total_h2h > 0 else 0.5
            team2_win_pct = team2_wins / total_h2h if total_h2h > 0 else 0.5

            # Prediction
            winner, win_probs = predict_match_winner(
                model, team_encoder, venue_encoder, toss_encoder,
                team1, team2, venue, toss_winner,
                team1_form, team2_form, team1_win_pct, team2_win_pct
            )

            # Display Results
            st.subheader(f"🏆 Predicted Winner: {winner}")
            fig, ax = plt.subplots()
            ax.pie(
                [win_probs[team1], win_probs[team2]],
                labels=[team1, team2],
                autopct="%1.1f%%",
                colors=["#FF6F00", "#1E90FF"],
                startangle=90
            )
            ax.axis("equal")
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("📊 Head-to-Head Stats")
            st.write(f"Total Matches: {total_h2h}")
            st.write(f"{team1} Wins: {team1_wins}")
            st.write(f"{team2} Wins: {team2_wins}")

            # Venue-based averages
            st.markdown("---")
            st.subheader("🏟️ Venue-Based Team Averages")
            venue_matches = matches[matches["venue"] == venue]
            team1_venue = venue_matches[
                (venue_matches["team1"] == team1) | (venue_matches["team2"] == team1)
            ]
            team2_venue = venue_matches[
                (venue_matches["team1"] == team2) | (venue_matches["team2"] == team2)
            ]

            def calc_avg(df, team):
                team_innings = deliveries[
                    (deliveries["match_id"].isin(df["id"])) &
                    (deliveries["batting_team"] == team)
                ]
                avg_runs = team_innings.groupby("match_id")["runs_scored"].sum().mean()
                avg_wkts = team_innings.groupby("match_id")["is_wicket"].sum().mean()
                return round(avg_runs, 1), round(avg_wkts, 1)

            t1_runs, t1_wkts = calc_avg(team1_venue, team1)
            t2_runs, t2_wkts = calc_avg(team2_venue, team2)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{team1} Avg Score @ {venue}", f"{t1_runs} / {t1_wkts}")
            with col2:
                st.metric(f"{team2} Avg Score @ {venue}", f"{t2_runs} / {t2_wkts}")

            # Team-specific Top Batters & Bowlers
            st.markdown("---")
            st.subheader("🔥 Key Player Highlights (vs Selected Team)")
            vs_team1 = deliveries[
                (deliveries["batting_team"] == team1) & (deliveries["bowling_team"] == team2)
            ]
            vs_team2 = deliveries[
                (deliveries["batting_team"] == team2) & (deliveries["bowling_team"] == team1)
            ]

            st.write(f"**Top 5 Batters ({team1} vs {team2})**")
            top_batters1 = vs_team1.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(top_batters1)

            st.write(f"**Top 5 Batters ({team2} vs {team1})**")
            top_batters2 = vs_team2.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(top_batters2)

            st.write(f"**Top 5 Bowlers ({team1} vs {team2})**")
            top_bowlers1 = vs_team1[vs_team1["is_wicket"] == 1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(top_bowlers1)

            st.write(f"**Top 5 Bowlers ({team2} vs {team1})**")
            top_bowlers2 = vs_team2[vs_team2["is_wicket"] == 1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(top_bowlers2)

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ------------------------------------------------
# TAB 2: IPL AI CHATBOT
# ------------------------------------------------
with tab2:
    st.header("🤖 IPL Chatbot Assistant")

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_SEARCH_KEY")
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX")

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        st.warning("Google API keys missing in Streamlit secrets. Chatbot disabled.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role']}:** {msg['content']}")

        query = st.text_input("Ask anything about IPL...")
        if st.button("Send") and query:
            try:
                url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CX}&key={GOOGLE_API_KEY}&num=3"
                res = requests.get(url).json()

                answer = ""
                if "items" in res:
                    for item in res["items"]:
                        answer += f"**{item['title']}**\n{item['snippet']}\n\n"
                else:
                    answer = "No relevant information found."

                st.session_state.chat_history.append({"role": "User", "content": query})
                st.session_state.chat_history.append({"role": "AI", "content": answer})

                st.markdown(f"**AI:** {answer}")
            except Exception as e:
                st.error(f"Chatbot error: {e}")
