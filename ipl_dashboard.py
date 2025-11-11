import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# Load model and encoders
# -------------------------
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")

# Load historical data
matches = pd.read_csv("all_matches.csv")
deliveries = pd.read_csv("all_deliveries.csv")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")

st.title("IPL Match Winner Prediction & Analysis")

tab1, tab2 = st.tabs(["Prediction", "Chatbot"])

# -------------------------
# Tab 1: Prediction
# -------------------------
with tab1:
    st.header("Predict the Match Winner")
    
    teams = matches['team1'].unique().tolist()
    venues = matches['venue'].unique().tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1", teams)
        team2 = st.selectbox("Select Team 2", teams)
    with col2:
        venue = st.selectbox("Select Venue", venues)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    
    col3, col4 = st.columns(2)
    with col3:
        team1_form = st.number_input(f"Recent Form of {team1} (0-1)", 0.0, 1.0, 0.5)
    with col4:
        team2_form = st.number_input(f"Recent Form of {team2} (0-1)", 0.0, 1.0, 0.5)

    if st.button("Predict Winner"):
        # Prepare features
        features = pd.DataFrame({
            'team1': [team_encoder.transform([team1])[0]],
            'team2': [team_encoder.transform([team2])[0]],
            'venue': [venue_encoder.transform([venue])[0]],
            'toss_winner': [toss_encoder.transform([toss_winner])[0]],
            'team1_form': [team1_form],
            'team2_form': [team2_form]
        })
        
        # Predict
        pred_probs = model.predict_proba(features)[0]
        predicted_winner_idx = np.argmax(pred_probs)
        predicted_winner = winner_encoder.inverse_transform([predicted_winner_idx])[0]
        probability = pred_probs[predicted_winner_idx] * 100
        
        st.success(f"Predicted Winner: {predicted_winner} ({probability:.2f}%)")
        
        # -------------------------
        # Head-to-head analysis
        # -------------------------
        st.subheader("Head-to-Head Analysis")
        h2h = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                      ((matches['team1'] == team2) & (matches['team2'] == team1))]
        if h2h.empty:
            st.info("No historical matches found between these two teams.")
        else:
            h2h_summary = h2h['winner'].value_counts(normalize=True) * 100
            st.write("Win percentage in previous encounters:")
            st.dataframe(h2h_summary.reset_index().rename(columns={'index': 'Team', 'winner': 'Win %'}))

# -------------------------
# Tab 2: Chatbot
# -------------------------
with tab2:
    st.header("IPL Data Chatbot")
    st.write("Ask anything about teams, matches, or player stats.")

    user_question = st.text_input("Your Question:")
    
    if st.button("Ask"):
        if user_question.strip() != "":
            # Basic logic: search in datasets for relevant info
            lower_q = user_question.lower()
            response = "Sorry, I could not find relevant info."
            
            if "team" in lower_q and "win" in lower_q:
                team_win_counts = matches['winner'].value_counts()
                response = "Team win counts:\n" + team_win_counts.to_string()
            elif "matches" in lower_q or "match" in lower_q:
                response = f"Total matches in dataset: {matches.shape[0]}"
            elif "player" in lower_q:
                response = "You can query player stats from deliveries dataset."
            
            st.write(response)
