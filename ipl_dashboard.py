import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import requests
from predict_winner import predict_match_winner
import logging

# -------------------------------
# 1. APP CONFIG
# -------------------------------
st.set_page_config(page_title="IPL AI Assistant", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: white; }
    .stButton>button { background-color: #FF6F00; color: white; }
    .stSelectbox, .stSlider { color: black; }
    </style>
    """, unsafe_allow_html=True
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# 2. LOAD MODEL & ENCODERS
# -------------------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        winner_encoder = joblib.load("winner_encoder.pkl")
        return model, team_encoder, toss_encoder, venue_encoder, winner_encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

model, team_encoder, toss_encoder, venue_encoder, winner_encoder = load_model_files()

# -------------------------------
# 3. LOAD CSV DATA
# -------------------------------
@st.cache_data
def load_match_data():
    return pd.read_csv("all_matches.csv")

@st.cache_data
def load_delivery_data():
    return pd.read_csv("all_deliveries.csv")

matches = load_match_data()
deliveries = load_delivery_data()

# -------------------------------
# 4. STREAMLIT TABS
# -------------------------------
tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "🤖 IPL AI Chatbot"])

# -------------------------------
# TAB 1: Predict Winner
# -------------------------------
with tab1:
    st.header("Match Winner Predictor")
    
    all_teams = sorted(team_encoder.classes_)
    all_venues = sorted(venue_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=0)
        team2 = st.selectbox("Select Team 2", all_teams, index=1)
    with col2:
        venue = st.selectbox("Select Venue", all_venues)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    
    team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)
    team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict Winner"):
        if team1 == team2:
            st.error("Team 1 and Team 2 cannot be the same.")
        else:
            try:
                # --- Head-to-Head Win % ---
                h2h = matches[
                    ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                    ((matches["team1"] == team2) & (matches["team2"] == team1))
                ]
                team1_win_pct = (h2h["winner"] == team1).sum() / max(len(h2h), 1)
                team2_win_pct = (h2h["winner"] == team2).sum() / max(len(h2h), 1)

                # --- Predict Winner ---
                winner, win_probs = predict_match_winner(
                    model, team_encoder, venue_encoder, toss_encoder,
                    team1, team2, venue, toss_winner,
                    team1_form, team2_form, team1_win_pct, team2_win_pct
                )
                
                st.subheader(f"🏆 Predicted Winner: {winner}")
                
                # --- Pie Chart for Win Probabilities ---
                fig, ax = plt.subplots()
                ax.pie([win_probs[team1], win_probs[team2]], labels=[team1, team2],
                       autopct="%1.1f%%", colors=["#FF6F00","#1E90FF"], startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # --- Head-to-Head Stats ---
                st.subheader("📊 Head-to-Head Stats")
                st.write(f"Total Matches Played: {len(h2h)}")
                st.write(f"{team1} Wins: {(h2h['winner']==team1).sum()}")
                st.write(f"{team2} Wins: {(h2h['winner']==team2).sum()}")

                # --- Top Batters & Bowlers ---
                st.subheader("🔥 Key Player Stats")
                
                top_batters = deliveries.groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(5)
                st.bar_chart(top_batters)
                
                top_bowlers = deliveries[deliveries["is_wicket"]==1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5)
                st.bar_chart(top_bowlers)

            except Exception as e:
                st.error(f"Error in prediction: {e}")

# -------------------------------
# TAB 2: IPL AI Chatbot
# -------------------------------
with tab2:
    st.header("🤖 Ask IPL AI Chatbot")
    
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_SEARCH_KEY")
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX")
    
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        st.warning("Google API keys missing in Streamlit secrets. Chatbot disabled.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display previous chat
        for msg in st.session_state.chat_history:
            role = msg["role"]
            st.markdown(f"**{role}:** {msg['content']}")
        
        # User input
        user_query = st.text_input("Ask anything about IPL...", key="chat_input")
        if st.button("Send", key="send_button") and user_query:
            try:
                search_url = f"https://www.googleapis.com/customsearch/v1?q={user_query}&cx={GOOGLE_CX}&key={GOOGLE_API_KEY}&num=3"
                response = requests.get(search_url).json()
                
                answer = ""
                if "items" in response:
                    for item in response["items"]:
                        answer += f"**{item['title']}**\n{item['snippet']}\n\n"
                else:
                    answer = "No relevant results found."
                
                st.session_state.chat_history.append({"role":"User","content":user_query})
                st.session_state.chat_history.append({"role":"AI","content":answer})
                
                st.markdown(f"**AI:** {answer}")
            except Exception as e:
                st.error(f"Chatbot error: {e}")
