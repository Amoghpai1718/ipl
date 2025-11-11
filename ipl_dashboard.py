import streamlit as st
import pandas as pd
import joblib
from predict_winner import predict_match_winner

# ------------------ Load Encoders and Model ------------------
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")
model = joblib.load("ipl_model.pkl")

# ------------------ Load Datasets ------------------
matches_df = pd.read_csv("all_matches.csv")
deliveries_df = pd.read_csv("all_deliveries.csv")

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")
st.title("Advanced IPL Match Predictor & Analytics")

tabs = st.tabs(["Predict Match Winner", "Head-to-Head Analysis"])

# ------------------ Tab 1: Prediction ------------------
with tabs[0]:
    st.header("Predict Match Winner")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", sorted(matches_df['team1'].unique()))
        team2 = st.selectbox("Select Team 2", sorted(matches_df['team2'].unique()))
        venue = st.selectbox("Select Venue", sorted(matches_df['venue'].unique()))
    with col2:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)
        team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict"):
        try:
            # Encode categorical inputs
            team1_enc = team_encoder.transform([team1])[0]
            team2_enc = team_encoder.transform([team2])[0]
            venue_enc = venue_encoder.transform([venue])[0]
            toss_enc = toss_encoder.transform([toss_winner])[0]

            # Create input DataFrame in exact order used during training
            input_df = pd.DataFrame([{
                "team1": team1_enc,
                "team2": team2_enc,
                "venue": venue_enc,
                "toss_winner": toss_enc,
                "team1_form": team1_form,
                "team2_form": team2_form
            }])

            # Align features with model
            expected_features = model.get_booster().feature_names
            input_df = input_df[expected_features]

            # Predict probabilities
            pred_probs = model.predict_proba(input_df)[0]

            team1_prob = round(pred_probs[0]*100, 2)
            team2_prob = round(pred_probs[1]*100, 2)

            winner = team1 if team1_prob > team2_prob else team2

            st.success(f"Predicted Winner: {winner}")
            st.info(f"{team1}: {team1_prob}% chance | {team2}: {team2_prob}% chance")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ------------------ Tab 2: Head-to-Head ------------------
with tabs[1]:
    st.header("Head-to-Head Analysis")

    h2h_team1 = st.selectbox("Select Team 1 for H2H", sorted(matches_df['team1'].unique()), key="h2h1")
    h2h_team2 = st.selectbox("Select Team 2 for H2H", sorted(matches_df['team2'].unique()), key="h2h2")

    h2h_matches = matches_df[((matches_df['team1']==h2h_team1) & (matches_df['team2']==h2h_team2)) |
                              ((matches_df['team1']==h2h_team2) & (matches_df['team2']==h2h_team1))]

    if h2h_matches.empty:
        st.warning("No previous matches found between these teams.")
    else:
        st.write(f"Total Matches Played: {len(h2h_matches)}")
        team1_wins = (h2h_matches['winner']==h2h_team1).sum()
        team2_wins = (h2h_matches['winner']==h2h_team2).sum()
        st.write(f"{h2h_team1} Wins: {team1_wins}")
        st.write(f"{h2h_team2} Wins: {team2_wins}")
        st.write("Match-wise Details:")
        st.dataframe(h2h_matches[['date','team1','team2','winner','venue']].sort_values(by='date', ascending=False))

# ------------------ Optional: Add Chatbot ------------------
st.sidebar.header("Powered AI Chatbot")
user_query = st.sidebar.text_input("Ask something about IPL or predictions")
if st.sidebar.button("Send"):
    if user_query:
        st.sidebar.write("Chatbot: Sorry, chatbot integration not implemented yet.")
