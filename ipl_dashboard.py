import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import requests
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="🏏 IPL Match Predictor & Chat Assistant",
    layout="wide",
    page_icon="🏏"
)

st.title("🏏 IPL Match Predictor & Chat Assistant")

# ------------------------------
# MODEL LOADING
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ipl_winner_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        return model, team_encoder, venue_encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, team_encoder, venue_encoder = load_model()

# ------------------------------
# DATASET (for teams & venues)
# ------------------------------
matches = pd.read_csv("all_matches.csv")

if not {"team1", "team2", "venue"}.issubset(matches.columns):
    st.error("Dataset missing required columns: team1, team2, or venue.")
    st.stop()

teams = sorted(set(matches["team1"]).union(set(matches["team2"])))
venues = sorted(matches["venue"].unique())

# ------------------------------
# INPUT SECTION
# ------------------------------
st.header("Match Setup")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams, key="team1_select")
    team1_form = st.slider(f"{team1} Recent Form (0–10)", 0, 10, 5, key="team1_form_slider")
    team1_win_pct = st.slider(f"{team1} Overall Win %", 0, 100, 50, key="team1_win_slider")

with col2:
    team2 = st.selectbox("Select Team 2", teams, key="team2_select")
    team2_form = st.slider(f"{team2} Recent Form (0–10)", 0, 10, 5, key="team2_form_slider")
    team2_win_pct = st.slider(f"{team2} Overall Win %", 0, 100, 50, key="team2_win_slider")

venue = st.selectbox("Select Venue", venues, key="venue_select")
toss_winner = st.selectbox("Select Toss Winner", [team1, team2], key="toss_select")

# ------------------------------
# PREDICTION
# ------------------------------
st.header("Prediction Result")

if st.button("Predict Match Winner", key="predict_button"):
    try:
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]
        toss_enc = team_encoder.transform([toss_winner])[0]

        input_df = pd.DataFrame([{
            "team1_enc": team1_enc,
            "team2_enc": team2_enc,
            "venue_enc": venue_enc,
            "toss_enc": toss_enc,
            "team1_form": team1_form,
            "team2_form": team2_form,
            "team1_win_pct": team1_win_pct,
            "team2_win_pct": team2_win_pct
        }])

        preds = model.predict_proba(input_df)[0]
        team1_prob, team2_prob = preds[0] * 100, preds[1] * 100
        winner = team1 if team1_prob > team2_prob else team2

        colp1, colp2 = st.columns(2)

        with colp1:
            st.subheader(f"Predicted Winner: {winner}")
            st.write(f"{team1}: **{team1_prob:.2f}%**")
            st.write(f"{team2}: **{team2_prob:.2f}%**")

        with colp2:
            fig = go.Figure(data=[go.Pie(
                labels=[team1, team2],
                values=[team1_prob, team2_prob],
                hole=0.4
            )])
            fig.update_traces(textinfo="label+percent", pull=[0.05, 0])
            fig.update_layout(title_text="Winning Probability", transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in prediction: {e}")

# ------------------------------
# AI CHATBOT SECTION
# ------------------------------
st.markdown("---")
st.header("💬 Cricket Knowledge Assistant")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

query = st.text_input("Ask anything about IPL, teams, or cricket:", key="chat_query")

def google_search(query):
    if not all([GOOGLE_SEARCH_KEY, GOOGLE_SEARCH_CX]):
        return "Google API credentials missing. Please set them in your environment."

    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_SEARCH_KEY}&cx={GOOGLE_SEARCH_CX}"
    try:
        res = requests.get(url)
        data = res.json()
        if "items" not in data:
            return "No relevant results found."
        top = data["items"][0]
        return f"**{top['title']}**\n\n{top['snippet']}\n\n[Read more]({top['link']})"
    except Exception as e:
        return f"Search error: {e}"

if st.button("Ask Chatbot", key="chat_button"):
    if query.strip():
        response = google_search(query)
        st.markdown(response)
    else:
        st.warning("Please enter a question.")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Developed by Amogh Pai | Data Science Project 2025")
