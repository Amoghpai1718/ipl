import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import requests
from predict_winner import predict_match_winner

st.set_page_config(page_title="IPL Match Predictor & Deep Dive Dashboard", layout="wide")

st.markdown("""
    <style>
    body {background-color: #0E1117; color: white;}
    .stButton>button {background-color: #FF6F00; color: white; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load model & encoders
# -------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        return model, team_encoder, toss_encoder, venue_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model()

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

if model is None:
    st.stop()

st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

tab1, tab2 = st.tabs(["📊 Prediction & Insights", "💬 IPL Chatbot"])

# ======================================================
# TAB 1: Prediction and Deep Dive
# ======================================================
with tab1:
    all_teams = sorted(team_encoder.classes_)
    all_venues = sorted(venue_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=0)
        team2 = st.selectbox("Select Team 2", all_teams, index=1)
    with col2:
        venue = st.selectbox("Select Venue", all_venues)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])

    st.subheader("Team Form & Win %")
    col3, col4 = st.columns(2)
    with col3:
        team1_form = st.slider(f"{team1} Recent Form (0–10)", 0.0, 10.0, 5.0)
        team1_win_pct = st.slider(f"{team1} Overall Win %", 0.0, 100.0, 50.0)
    with col4:
        team2_form = st.slider(f"{team2} Recent Form (0–10)", 0.0, 10.0, 5.0)
        team2_win_pct = st.slider(f"{team2} Overall Win %", 0.0, 100.0, 50.0)

    if st.button("Predict Winner"):
        try:
            winner, win_probs = predict_match_winner(
                model, team_encoder, venue_encoder, toss_encoder,
                team1, team2, venue, toss_winner,
                team1_form, team2_form, team1_win_pct, team2_win_pct
            )

            st.subheader(f"🏆 Predicted Winner: {winner}")

            # Animated Pie Chart (Plotly)
            fig = go.Figure(data=[
                go.Pie(
                    labels=[team1, team2],
                    values=[win_probs[team1], win_probs[team2]],
                    hole=0.3,
                    marker_colors=["#FF6F00", "#1E90FF"]
                )
            ])
            fig.update_traces(textinfo='label+percent')
            st.plotly_chart(fig, use_container_width=True)

            # Head-to-Head
            h2h = matches[
                ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                ((matches["team1"] == team2) & (matches["team2"] == team1))
            ]
            st.subheader("📈 Head-to-Head Statistics")
            st.write(f"Total Matches Played: {len(h2h)}")
            st.write(f"{team1} Wins: {(h2h['winner'] == team1).sum()}")
            st.write(f"{team2} Wins: {(h2h['winner'] == team2).sum()}")

            # Venue averages
            venue_matches = matches[matches["venue"] == venue]
            team1_ids = venue_matches[
                (venue_matches["team1"] == team1) | (venue_matches["team2"] == team1)
            ]["match_id"]
            team2_ids = venue_matches[
                (venue_matches["team1"] == team2) | (venue_matches["team2"] == team2)
            ]["match_id"]

            team1_batting = deliveries[(deliveries["inning_team"] == team1) & (deliveries["match_id"].isin(team1_ids))]
            team2_batting = deliveries[(deliveries["inning_team"] == team2) & (deliveries["match_id"].isin(team2_ids))]

            avg_runs_team1 = team1_batting.groupby("match_id")["runs_scored"].sum().mean()
            avg_runs_team2 = team2_batting.groupby("match_id")["runs_scored"].sum().mean()
            avg_wickets_team1 = team1_batting.groupby("match_id")["is_wicket"].sum().mean()
            avg_wickets_team2 = team2_batting.groupby("match_id")["is_wicket"].sum().mean()

            st.subheader(f"🏟️ Venue Averages at {venue}")
            st.write(f"{team1} - Avg Runs: {avg_runs_team1:.1f}, Avg Wickets Lost: {avg_wickets_team1:.1f}")
            st.write(f"{team2} - Avg Runs: {avg_runs_team2:.1f}, Avg Wickets Lost: {avg_wickets_team2:.1f}")

            # Top Batters and Bowlers
            st.subheader("🔥 Top Performers (vs Opponent)")
            vs_data = deliveries[
                (deliveries["inning_team"].isin([team1, team2])) &
                (deliveries["match_id"].isin(h2h["match_id"]))
            ]
            top_batters = vs_data.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
            top_bowlers = vs_data.groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)

            col5, col6 = st.columns(2)
            with col5:
                st.bar_chart(top_batters)
            with col6:
                st.bar_chart(top_bowlers)

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ======================================================
# TAB 2: Simple AI Chatbot
# ======================================================
with tab2:
    st.subheader("💬 Ask the IPL Chatbot")

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX")

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        st.warning("Chatbot disabled. Add GOOGLE_API_KEY and GOOGLE_SEARCH_CX in Streamlit Secrets.")
    else:
        user_query = st.text_input("Ask anything about IPL...")
        if st.button("Ask"):
            try:
                url = f"https://www.googleapis.com/customsearch/v1?q={user_query}&cx={GOOGLE_CX}&key={GOOGLE_API_KEY}"
                resp = requests.get(url).json()
                if "items" in resp:
                    for item in resp["items"][:3]:
                        st.write(f"**{item['title']}** - {item['snippet']}")
                else:
                    st.write("No relevant results found.")
            except Exception as e:
                st.error(f"Chatbot error: {e}")
