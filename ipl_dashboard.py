import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import plotly.express as px

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ App Config ------------------
st.set_page_config(page_title="IPL Advanced Dashboard", layout="wide")

# ------------------ Load Model & Encoders ------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("ipl_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        toss_encoder = joblib.load("toss_encoder.pkl")
        venue_encoder = joblib.load("venue_encoder.pkl")
        logging.info("Model and encoders loaded successfully.")
        return model, team_encoder, toss_encoder, venue_encoder
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None

model, team_encoder, toss_encoder, venue_encoder = load_model_files()

# ------------------ Load CSVs ------------------
@st.cache_data
def load_matches():
    try:
        df = pd.read_csv("all_matches.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading all_matches.csv: {e}")
        return pd.DataFrame()

matches = load_matches()

@st.cache_data
def load_deliveries():
    try:
        df = pd.read_csv("all_deliveries.csv")
        return df
    except Exception as e:
        st.error(f"Error loading all_deliveries.csv: {e}")
        return pd.DataFrame()

deliveries = load_deliveries()

# ------------------ Tabs ------------------
st.title("🏏 IPL Advanced Dashboard & Winner Predictor")

if model is None or matches.empty:
    st.error("Essential files missing. Cannot run the app.")
else:
    tab1, tab2, tab3 = st.tabs(["🏆 Predict Match Winner", "📊 Player Stats", "🤖 IPL Chatbot"])

    # ------------------ TAB 1: Winner Prediction ------------------
    with tab1:
        st.header("Predict Match Winner")
        all_teams = sorted(list(team_encoder.classes_))
        all_venues = sorted(list(venue_encoder.classes_))
        all_toss = sorted(list(toss_encoder.classes_))

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", all_teams, key="team1")
            team2 = st.selectbox("Select Team 2", all_teams, key="team2")
            team1_form = st.slider(f"{team1} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="team1_form")
        with col2:
            venue = st.selectbox("Select Venue", all_venues, key="venue")
            toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss_winner")
            toss_decision = st.selectbox("Toss Decision", all_toss, key="toss_decision")
            team2_form = st.slider(f"{team2} Recent Form (0-1)", 0.0, 1.0, 0.5, 0.01, key="team2_form")

        if st.button("Predict Winner"):
            if team1 == team2:
                st.error("Team 1 and Team 2 cannot be the same.")
            else:
                try:
                    team1_enc = team_encoder.transform([team1])[0]
                    team2_enc = team_encoder.transform([team2])[0]
                    venue_enc = venue_encoder.transform([venue])[0]
                    toss_enc = toss_encoder.transform([toss_winner])[0]

                    # Example: Using 8 features in correct order
                    input_df = pd.DataFrame({
                        'team1_enc':[team1_enc],
                        'team2_enc':[team2_enc],
                        'venue_enc':[venue_enc],
                        'toss_enc':[toss_enc],
                        'team1_form':[team1_form],
                        'team2_form':[team2_form],
                        'team1_win_pct':[0.5],  # default; can enhance later
                        'team2_win_pct':[0.5]   # default
                    })

                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0]

                    winner = team1 if pred==1 else team2
                    st.success(f"🏆 Predicted Winner: {winner}")
                    st.write(f"Confidence {team1}: {prob[1]*100:.1f}%")
                    st.write(f"Confidence {team2}: {prob[0]*100:.1f}%")

                    # Head-to-Head
                    st.subheader(f"📊 Head-to-Head: {team1} vs {team2}")
                    h2h = matches[((matches['team1']==team1)&(matches['team2']==team2))|
                                  ((matches['team1']==team2)&(matches['team2']==team1))]
                    if h2h.empty:
                        st.write("No historical matches.")
                    else:
                        total = len(h2h)
                        wins = h2h['winner'].value_counts()
                        st.metric("Total Matches", total)
                        st.metric(f"{team1} Wins", wins.get(team1,0))
                        st.metric(f"{team2} Wins", wins.get(team2,0))
                        st.bar_chart(wins)

    # ------------------ TAB 2: Top Players ------------------
    with tab2:
        st.header("📊 Top Batters & Bowlers")
        if deliveries.empty:
            st.warning("No deliveries data.")
        else:
            team_filter = st.selectbox("Select Team for Stats", sorted(deliveries['inning_team'].unique()), key="team_stats")
            top_batters = deliveries[deliveries['inning_team']==team_filter].groupby('batter')['runs_scored'].sum().sort_values(ascending=False).head(10)
            top_bowlers = deliveries[deliveries['inning_team']==team_filter].groupby('bowler')['is_wicket'].sum().sort_values(ascending=False).head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Top Batters - {team_filter}")
                fig1 = px.bar(top_batters, x=top_batters.index, y=top_batters.values, labels={'x':'Batter','y':'Runs'}, text=top_batters.values)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.subheader(f"Top Bowlers - {team_filter}")
                fig2 = px.bar(top_bowlers, x=top_bowlers.index, y=top_bowlers.values, labels={'x':'Bowler','y':'Wickets'}, text=top_bowlers.values)
                st.plotly_chart(fig2, use_container_width=True)

    # ------------------ TAB 3: IPL Chatbot ------------------
    with tab3:
        st.header("🤖 Ask IPL Chatbot")
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Add GEMINI_API_KEY in Streamlit secrets to enable chatbot.")
        else:
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")

            if "messages" not in st.session_state:
                st.session_state.messages=[]
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if user_input := st.chat_input("Ask anything IPL related"):
                st.session_state.messages.append({"role":"user","content":user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.spinner("AI is thinking..."):
                    try:
                        chat = model_gemini.start_chat()
                        system_prompt = "You are an IPL expert analyst. Answer based on IPL history."
                        response = chat.send_message(f"{system_prompt}\n\nQuestion: {user_input}")
                        st.session_state.messages.append({"role":"assistant","content":response.text})
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Chatbot error: {e}")
