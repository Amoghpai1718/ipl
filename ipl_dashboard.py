import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================
# Load Model and Encoders
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        return model, team_encoder, venue_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    return matches, deliveries

# =========================
# Predict Function
# =========================
def predict_winner(model, team_encoder, venue_encoder,
                   team1, team2, venue, team1_form, team2_form,
                   team1_win_pct, team2_win_pct):
    try:
        # Encode inputs
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]

        # Model input — matches training feature order (no toss_enc)
        input_df = pd.DataFrame({
            "team1_enc": [team1_enc],
            "team2_enc": [team2_enc],
            "venue_enc": [venue_enc],
            "team1_form": [team1_form],
            "team2_form": [team2_form],
            "team1_win_pct": [team1_win_pct],
            "team2_win_pct": [team2_win_pct]
        })

        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        winner = team1 if pred == 1 else team2
        win_probs = {team1: proba[1] * 100, team2: proba[0] * 100}
        return winner, win_probs

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# =========================
# Team Stats at Venue
# =========================
def team_avg_stats_at_venue(deliveries, matches, team, venue):
    try:
        match_ids = matches[matches['venue'] == venue]['match_id']
        venue_data = deliveries[deliveries['match_id'].isin(match_ids)]
        team_data = venue_data[venue_data['inning_team'] == team]
        if team_data.empty:
            return 0, 0
        avg_runs = team_data.groupby('match_id')['runs_scored'].sum().mean()
        avg_wkts = team_data.groupby('match_id')['is_wicket'].sum().mean()
        return round(avg_runs, 1), round(avg_wkts, 1)
    except Exception:
        return 0, 0

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="IPL Match Predictor", layout="wide")
st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

model, team_encoder, venue_encoder = load_model()
matches, deliveries = load_data()

if model is None or team_encoder is None or venue_encoder is None:
    st.stop()

teams = sorted(matches['team1'].unique())
venues = sorted(matches['venue'].unique())

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", teams)
    team1_form = st.slider(f"{team1} Recent Form (0–10)", 0, 10, 5)
    team1_win_pct = st.slider(f"{team1} Overall Win %", 0, 100, 50)

with col2:
    team2 = st.selectbox("Select Team 2", teams, index=1)
    team2_form = st.slider(f"{team2} Recent Form (0–10)", 0, 10, 5)
    team2_win_pct = st.slider(f"{team2} Overall Win %", 0, 100, 50)

venue = st.selectbox("Select Venue", venues)

# =========================
# Prediction Section
# =========================
if st.button("Predict Winner"):
    winner, win_probs = predict_winner(
        model, team_encoder, venue_encoder,
        team1, team2, venue, team1_form, team2_form,
        team1_win_pct, team2_win_pct
    )

    if winner:
        st.subheader(f"🏆 Predicted Winner: {winner}")

        # Plotly animated pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=list(win_probs.keys()),
                values=list(win_probs.values()),
                hole=0.4,
                textinfo='label+percent',
                marker=dict(line=dict(color='#000000', width=1))
            )
        ])
        fig.update_traces(hoverinfo='label+percent', textfont_size=16)
        fig.update_layout(title_text="Winning Probability", title_x=0.4)
        st.plotly_chart(fig, use_container_width=True)

        # Venue stats
        st.markdown("### 📊 Team Averages at Venue")
        t1_avg_runs, t1_avg_wkts = team_avg_stats_at_venue(deliveries, matches, team1, venue)
        t2_avg_runs, t2_avg_wkts = team_avg_stats_at_venue(deliveries, matches, team2, venue)
        st.write(f"**{team1}** – Avg Runs: {t1_avg_runs}, Avg Wickets Lost: {t1_avg_wkts}")
        st.write(f"**{team2}** – Avg Runs: {t2_avg_runs}, Avg Wickets Lost: {t2_avg_wkts}")

# =========================
# Simple Chatbot Placeholder
# =========================
st.markdown("---")
st.subheader("💬 Cricket Chatbot")
user_query = st.text_input("Ask about teams, venues, or players:")
if st.button("Ask"):
    if user_query.strip():
        st.write(f"Bot: Searching for '{user_query}' (requires Google API setup)")
    else:
        st.info("Please enter a question first.")
