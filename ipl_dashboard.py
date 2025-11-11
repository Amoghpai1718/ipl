import streamlit as st
import pandas as pd
from predict_winner import predict_match_winner
from googleapiclient.discovery import build

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="IPL Match Predictor & Chatbot", layout="wide")
st.title("IPL Match Predictor & Chatbot")

tab1, tab2 = st.tabs(["Predict Winner", "AI Chat"])

# ----------------- Tab 1: Match Prediction -----------------
with tab1:
    st.header("Predict Match Winner")
    matches_df = pd.read_csv("all_matches.csv")
    teams = sorted(matches_df['team1'].unique())
    venues = sorted(matches_df['venue'].unique())

    team1 = st.selectbox("Select Team 1", teams, key="team1")
    team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1], key="team2")
    venue = st.selectbox("Select Venue", venues)
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    
    # Recent form sliders
    team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="form1")
    team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="form2")

    if st.button("Predict Winner"):
        try:
            predicted_team, prob1, prob2, h2h_stats = predict_match_winner(
                team1, team2, venue, toss_winner, team1_form, team2_form
            )
            st.subheader(f"Predicted Winner: {predicted_team}")
            st.write(f"{team1}: {prob1*100:.2f}% chance")
            st.write(f"{team2}: {prob2*100:.2f}% chance")
            st.write("### Head-to-Head Stats")
            st.write(h2h_stats)
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ----------------- Tab 2: Google Chatbot -----------------
with tab2:
    st.header("Ask IPL Chatbot")
    query = st.text_input("Ask anything about IPL:")

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    GOOGLE_SEARCH_CX = st.secrets.get("GOOGLE_SEARCH_CX")
    GOOGLE_SEARCH_KEY = st.secrets.get("GOOGLE_SEARCH_KEY")

    if st.button("Ask") and query:
        try:
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            res = service.cse().list(q=query, cx=GOOGLE_SEARCH_CX, num=3).execute()
            results = res.get('items', [])
            if results:
                for i, r in enumerate(results, 1):
                    st.markdown(f"**Result {i}:** [{r['title']}]({r['link']})")
                    st.write(r.get('snippet', ''))
            else:
                st.write("No results found.")
        except Exception as e:
            st.error(f"Chatbot Error: {e}")
