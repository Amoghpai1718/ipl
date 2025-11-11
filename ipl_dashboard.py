# ipl_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="IPL AI Assistant", layout="wide")

# dark theme CSS
st.markdown(
    """
    <style>
      .reportview-container { background: #0E1117; color: #E6EEF3; }
      .stApp { background: #0E1117; color: #E6EEF3; }
      .stButton>button { background-color: #FF6F00; color: white; border: none; }
      .stMetric { color: #E6EEF3; }
      .stSelectbox, .stTextInput, .stSlider { color: black; }
      .css-1v3fvcr { color: #E6EEF3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers: safe encoder transform
# ----------------------------
def safe_label_encode(encoder, value):
    """
    Returns integer encoding for value using LabelEncoder-like object `encoder`.
    If value not seen during training, return a new index = len(encoder.classes_).
    This prevents transform() raising error for unseen labels.
    (Note: model may not generalize to unseen categories.)
    """
    try:
        classes = list(encoder.classes_)
    except Exception:
        # if not a sklearn LabelEncoder, try to use transform directly
        try:
            return int(encoder.transform([value])[0])
        except Exception:
            return -1

    if value in classes:
        return int(np.where(np.array(classes) == value)[0][0])
    else:
        return len(classes)  # new unseen index

# ----------------------------
# Load model & encoders
# ----------------------------
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("ipl_model.pkl")
        team_enc = joblib.load("team_encoder.pkl")
        toss_enc = joblib.load("toss_encoder.pkl")
        venue_enc = joblib.load("venue_encoder.pkl")
        winner_enc = joblib.load("winner_encoder.pkl") if st.runtime.exists("winner_encoder.pkl") else None
        return model, team_enc, toss_enc, venue_enc, winner_enc
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None, None, None, None

model, team_enc, toss_enc, venue_enc, winner_enc = load_model_and_encoders()

# ----------------------------
# Load CSVs
# ----------------------------
@st.cache_data
def load_matches():
    return pd.read_csv("all_matches.csv")

@st.cache_data
def load_deliveries():
    return pd.read_csv("all_deliveries.csv")

matches = load_matches()
deliveries = load_deliveries()

# ----------------------------
# Import predict function (fallback if missing)
# ----------------------------
try:
    from predict_winner import predict_match_winner
except Exception:
    # Fallback implementation - follows the fixed 8-feature contract
    def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                             team1, team2, venue, toss_winner,
                             team1_form, team2_form, team1_win_pct, team2_win_pct) -> Tuple[str, Dict[str, float]]:
        # encode safely
        t1 = safe_label_encode(team_encoder, team1)
        t2 = safe_label_encode(team_encoder, team2)
        ven = safe_label_encode(venue_encoder, venue)
        toss_enc_val = safe_label_encode(team_encoder, toss_winner)

        features = pd.DataFrame({
            "team1_enc": [t1],
            "team2_enc": [t2],
            "venue_enc": [ven],
            "toss_enc": [toss_enc_val],
            "team1_form": [float(team1_form)],
            "team2_form": [float(team2_form)],
            "team1_win_pct": [float(team1_win_pct)],
            "team2_win_pct": [float(team2_win_pct)],
        })

        # Ensure columns order matches model.feature_names_in_ if present
        feature_order = getattr(model, "feature_names_in_", None)
        if feature_order is not None:
            features = features[[c for c in feature_order if c in features.columns]]

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        winner = team1 if int(pred) == 1 else team2
        # Assumes model's proba index 1 corresponds to team1 winning (training contract)
        win_probs = {team1: float(proba[1] * 100), team2: float(proba[0] * 100)}
        return winner, win_probs

# ----------------------------
# UI: Tabs and controls
# ----------------------------
st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

if model is None or team_enc is None or venue_enc is None or toss_enc is None:
    st.error("Model or encoders are missing. Make sure ipl_model.pkl and encoder .pkl files are present.")
    st.stop()

teams = sorted(list(team_enc.classes_))
venues = sorted(list(venue_enc.classes_))

tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "🤖 IPL Chatbot"])

with tab1:
    st.header("Predict Match Winner")
    c1, c2 = st.columns([2, 1])
    with c1:
        team1 = st.selectbox("Team 1", teams, key="t1")
        team2 = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, key="t2")
        venue = st.selectbox("Venue", venues, key="venue")
        toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss")
    with c2:
        st.write("Adjust recent form (0 = poor, 1 = excellent)")
        team1_form = st.slider(f"{team1} Recent Form", 0.0, 1.0, 0.5, 0.01, key="form1")
        team2_form = st.slider(f"{team2} Recent Form", 0.0, 1.0, 0.5, 0.01, key="form2")

    # Compute head-to-head history between these two teams
    h2h_matches = matches[
        ((matches["team1"] == team1) & (matches["team2"] == team2)) |
        ((matches["team1"] == team2) & (matches["team2"] == team1))
    ]
    total_h2h = len(h2h_matches)
    team1_h2h_wins = int((h2h_matches["winner"] == team1).sum())
    team2_h2h_wins = int((h2h_matches["winner"] == team2).sum())

    # compute win pct (guard divide by zero)
    team1_win_pct = (team1_h2h_wins / total_h2h) if total_h2h > 0 else 0.5
    team2_win_pct = (team2_h2h_wins / total_h2h) if total_h2h > 0 else 0.5

    # Button to predict
    if st.button("Predict", key="predict_btn"):
        if team1 == team2:
            st.error("Choose two different teams.")
        else:
            try:
                # call predict function (ensures 8-feature contract)
                winner, win_probs = predict_match_winner(
                    model, team_enc, venue_enc, toss_enc,
                    team1, team2, venue, toss_winner,
                    team1_form, team2_form, team1_win_pct, team2_win_pct
                )

                # show prediction
                st.markdown(f"### 🏆 Predicted Winner: **{winner}**")
                st.write(f"{team1}: {win_probs.get(team1,0):.2f}%  —  {team2}: {win_probs.get(team2,0):.2f}%")

                # Pie chart
                fig1, ax1 = plt.subplots(figsize=(4,4))
                ax1.pie([win_probs.get(team1,0), win_probs.get(team2,0)],
                        labels=[team1, team2], autopct="%1.1f%%", startangle=90)
                ax1.axis("equal")
                st.pyplot(fig1)

                # H2H stats
                st.subheader("📊 Head-to-Head")
                st.metric("Total H2H Matches", total_h2h)
                colA, colB, colC = st.columns(3)
                colA.metric(f"{team1} Wins", team1_h2h_wins)
                colB.metric(f"{team2} Wins", team2_h2h_wins)
                colC.metric("Draw/No result", int(total_h2h - (team1_h2h_wins + team2_h2h_wins)))

                # Venue-specific average score & average wickets lost per innings for each team
                st.subheader(f"🏟 Venue performance at {venue}")
                # find matches at the venue where selected team played
                matches_at_venue = matches[matches["venue"] == venue]
                def team_avg_at_venue(team):
                    # matches at venue where team was either team1 or team2
                    m_ids = matches_at_venue[(matches_at_venue["team1"]==team) | (matches_at_venue["team2"]==team)]["match_id"].unique().tolist()
                    if not m_ids:
                        return None
                    # deliveries for those matches where inning_team == team
                    df = deliveries[deliveries["match_id"].isin(m_ids) & (deliveries["inning_team"] == team)]
                    if df.empty:
                        return None
                    # compute total runs per match and wickets per match per inning
                    grouped = df.groupby("match_id").agg(total_runs=("runs_scored","sum"), total_wickets=("is_wicket","sum"))
                    avg_runs = float(grouped["total_runs"].mean())
                    avg_wickets = float(grouped["total_wickets"].mean())
                    return {"avg_runs": avg_runs, "avg_wickets": avg_wickets, "matches": len(grouped)}
                a1 = team_avg_at_venue(team1)
                a2 = team_avg_at_venue(team2)

                col1v, col2v = st.columns(2)
                if a1:
                    col1v.metric(f"{team1} Avg Runs @ {venue}", f"{a1['avg_runs']:.1f}")
                    col1v.metric(f"{team1} Avg Wickets Lost", f"{a1['avg_wickets']:.1f}")
                    col1v.write(f"Matches counted: {a1['matches']}")
                else:
                    col1v.write(f"No innings data for {team1} at {venue}")

                if a2:
                    col2v.metric(f"{team2} Avg Runs @ {venue}", f"{a2['avg_runs']:.1f}")
                    col2v.metric(f"{team2} Avg Wickets Lost", f"{a2['avg_wickets']:.1f}")
                    col2v.write(f"Matches counted: {a2['matches']}")
                else:
                    col2v.write(f"No innings data for {team2} at {venue}")

                # Key players: top batters & top bowlers **against the selected opponent** (H2H deliveries)
                st.subheader("🔥 Key Players (in matches between these two teams)")
                # get match_ids where both teams faced each other
                matchup_ids = h2h_matches["match_id"].unique().tolist()
                if len(matchup_ids) == 0:
                    st.write("No head-to-head delivery data available for this pairing.")
                else:
                    h2h_del = deliveries[deliveries["match_id"].isin(matchup_ids)]
                    # top batters for team1 vs team2
                    t1_bat = h2h_del[h2h_del["inning_team"]==team1].groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(8)
                    t2_bat = h2h_del[h2h_del["inning_team"]==team2].groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(8)
                    t1_bowl = h2h_del[h2h_del["inning_team"]==team2].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(8)  # bowlers who bowled to team1
                    t2_bowl = h2h_del[h2h_del["inning_team"]==team1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(8)

                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        st.write(f"Top {team1} batters vs {team2}")
                        if not t1_bat.empty:
                            st.bar_chart(t1_bat)
                        else:
                            st.write("No data")
                        st.write(f"Top {team1} bowlers vs {team2}")
                        if not t1_bowl.empty:
                            st.bar_chart(t1_bowl)
                        else:
                            st.write("No data")
                    with bcol2:
                        st.write(f"Top {team2} batters vs {team1}")
                        if not t2_bat.empty:
                            st.bar_chart(t2_bat)
                        else:
                            st.write("No data")
                        st.write(f"Top {team2} bowlers vs {team1}")
                        if not t2_bowl.empty:
                            st.bar_chart(t2_bowl)
                        else:
                            st.write("No data")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

with tab2:
    st.header("🤖 IPL Chatbot (Google Custom Search)")
    GOOGLE_KEY = st.secrets.get("GOOGLE_SEARCH_KEY") if isinstance(st.secrets, dict) or hasattr(st.secrets, "get") else None
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX") if isinstance(st.secrets, dict) or hasattr(st.secrets, "get") else None

    if not GOOGLE_KEY or not GOOGLE_CX:
        st.warning("Google API keys not configured in Streamlit secrets. Add GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_CX to enable chatbot.")
        st.info("You can ask simple factual questions; when keys are present the chatbot returns top search snippets.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_q = st.text_input("Ask about IPL history, players, results...", key="chat_input2")
        if st.button("Send", key="send_chat"):
            if not user_q:
                st.warning("Please type a question.")
            else:
                try:
                    params = {
                        "q": user_q,
                        "cx": GOOGLE_CX,
                        "key": GOOGLE_KEY,
                        "num": 3,
                    }
                    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
                    r.raise_for_status()
                    res = r.json()
                    answer = ""
                    if "items" in res:
                        for item in res["items"]:
                            title = item.get("title", "")
                            snippet = item.get("snippet", "")
                            link = item.get("link", "")
                            answer += f"**{title}**\n{snippet}\n<{link}>\n\n"
                    else:
                        answer = "No search results found."
                    st.session_state.chat_history.append(("User", user_q))
                    st.session_state.chat_history.append(("AI", answer))
                except Exception as e:
                    st.error(f"Chatbot Error: {e}")

        # display chat history
        for role, content in st.session_state.get("chat_history", []):
            if role == "User":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**AI:** {content}")

st.caption("App: uses preprocessed CSVs (all_matches.csv, all_deliveries.csv) and model files. Make sure your repository includes ipl_model.pkl, team_encoder.pkl, toss_encoder.pkl, venue_encoder.pkl before deploying to Streamlit Cloud.")
