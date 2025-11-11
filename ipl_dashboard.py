import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import requests

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="IPL Dashboard with AI Chatbot", layout="wide")
st.title("🏏 IPL Match Prediction & AI Chat Assistant")

# --------------------------------------------------
# 2. LOAD MODEL & DATA
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("ipl_winner_model.pkl")
    team_encoder = joblib.load("team_encoder.pkl")
    matches = pd.read_csv("all_matches.csv")
    return model, team_encoder, matches

model, team_encoder, matches = load_resources()

# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏆 Match Prediction", "🤖 AI Chatbot"])

# --------------------------------------------------
# 4. MATCH PREDICTION PAGE
# --------------------------------------------------
if page == "🏆 Match Prediction":
    st.header("📊 IPL Match Predictor")

    teams = sorted(matches["team1"].dropna().unique().tolist())
    venues = sorted(matches["venue"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        team1 = st.selectbox("Select Team 1", teams)
    with col2:
        team2 = st.selectbox("Select Team 2", teams)
    with col3:
        venue = st.selectbox("Select Venue", venues)

    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    if st.button("Predict Winner"):
        if team1 == team2:
            st.error("Please select two different teams.")
        else:
            try:
                t1 = team_encoder.transform([team1])[0]
                t2 = team_encoder.transform([team2])[0]
                toss_win = team_encoder.transform([toss_winner])[0]
                toss_dec = 1 if toss_decision == "bat" else 0

                X_test = [[t1, t2, toss_win, toss_dec]]
                pred = model.predict(X_test)
                winner = team_encoder.inverse_transform(pred)[0]

                st.success(f"🏆 Predicted Winner: **{winner}**")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # Head-to-head and venue insights
    st.header("📈 Head-to-Head & Venue Insights")
    if st.button("Show Insights"):
        h2h = matches[((matches["team1"] == team1) & (matches["team2"] == team2)) |
                      ((matches["team1"] == team2) & (matches["team2"] == team1))]
        if not h2h.empty:
            wins = h2h["winner"].value_counts()
            st.subheader(f"Head-to-Head ({team1} vs {team2})")
            st.bar_chart(wins)
        else:
            st.info("No direct matches found between these teams.")

        venue_data = matches[matches["venue"] == venue]["winner"].value_counts()
        st.subheader(f"Top Winners at {venue}")
        st.bar_chart(venue_data)

# --------------------------------------------------
# 5. AI CHATBOT PAGE
# --------------------------------------------------
elif page == "🤖 AI Chatbot":
    st.header("🤖 IPL AI Chatbot (Gemini + Google Search)")

    # API Configuration
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    GOOGLE_SEARCH_KEY = st.secrets.get("GOOGLE_SEARCH_KEY")
    GOOGLE_SEARCH_CX = st.secrets.get("GOOGLE_SEARCH_CX")

    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    else:
        st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
        st.stop()

    # Google Search Helper
    def google_search(query):
        try:
            url = (
                f"https://www.googleapis.com/customsearch/v1"
                f"?key={GOOGLE_SEARCH_KEY}"
                f"&cx={GOOGLE_SEARCH_CX}"
                f"&q={query}"
            )
            res = requests.get(url)
            data = res.json()
            if "items" in data:
                results = "\n".join(
                    [f"- {item['title']}: {item['link']}" for item in data["items"][:3]]
                )
                return results
            return "No recent IPL updates found."
        except Exception as e:
            return f"Search error: {e}"

    # Chat Interface
    st.markdown("### Ask anything about IPL teams, players, or matches!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Enter your question:", placeholder="e.g., Who was the top scorer in IPL 2023?")

    if st.button("Ask AI"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing..."):
                search_data = google_search(user_input)
                prompt = f"""
                You are an IPL analyst. Use the data below and your reasoning to answer factually.

                Question: {user_input}

                Latest Google Search Results:
                {search_data}
                """

                try:
                    response = gemini_model.generate_content(prompt)
                    answer = response.text.strip()

                    st.session_state.chat_history.append(("User", user_input))
                    st.session_state.chat_history.append(("AI", answer))
                except Exception as e:
                    st.session_state.chat_history.append(("AI", f"AI error: {e}"))

    # Display Chat History
    for role, msg in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")
