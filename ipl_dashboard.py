# ipl_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from predict_winner import predict_match_winner
import requests
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="IPL Match Predictor & Deep Dive", layout="wide")

# ---------- CSS dark theme ----------
st.markdown(
    """
    <style>
    .reportview-container { background: #0b1020; color: #e6eef8; }
    .stApp { background: #0b1020; color: #e6eef8; }
    .css-1offfwp { color: #e6eef8; }
    .stButton>button { background-color: #FF6F00; color: #ffffff; }
    .stMetric > div { color: #e6eef8; }
    </style>
    """, unsafe_allow_html=True
)

# ---------- Helpers to load model & encoders ----------
@st.cache_resource
def load_model_and_encoders():
    needed = {
        "model": "ipl_model.pkl",
        "team_encoder": "team_encoder.pkl",
        "toss_encoder": "toss_encoder.pkl",
        "venue_encoder": "venue_encoder.pkl",
        "winner_encoder": "winner_encoder.pkl"
    }
    loaded = {}
    missing = []
    for name, fname in needed.items():
        p = Path(fname)
        if not p.exists():
            missing.append(fname)
        else:
            try:
                loaded[name] = joblib.load(fname)
            except Exception as e:
                st.error(f"Failed to load {fname}: {e}")
                loaded[name] = None
    if missing:
        st.error(f"Missing files: {', '.join(missing)}. Please add them to the repo root.")
    return loaded

# ---------- Load data ----------
@st.cache_data
def load_matches():
    p = Path("all_matches.csv")
    if not p.exists():
        st.error("all_matches.csv not found in repo root.")
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data
def load_deliveries():
    p = Path("all_deliveries.csv")
    if not p.exists():
        st.error("all_deliveries.csv not found in repo root.")
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df

loaded = load_model_and_encoders()
model = loaded.get("model")
team_encoder = loaded.get("team_encoder")
toss_encoder = loaded.get("toss_encoder")
venue_encoder = loaded.get("venue_encoder")
winner_encoder = loaded.get("winner_encoder")

matches = load_matches()
deliveries = load_deliveries()

# Quick safety: if any are None, show message but allow partial UI
if model is None or team_encoder is None or toss_encoder is None or venue_encoder is None:
    st.warning("Model or encoders not fully loaded. Prediction will not work until all files are present and valid.")

# ---------- UI ----------
st.title("🏏 IPL Match Predictor & Deep Dive Dashboard")

tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "🤖 IPL Chatbot"])

with tab1:
    st.header("Predict Match Winner")

    # Teams and venues from encoders or match file fallback
    if team_encoder is not None and hasattr(team_encoder, "classes_"):
        teams = sorted(list(team_encoder.classes_))
    else:
        teams = sorted(pd.unique(pd.concat([matches["team1"].dropna(), matches["team2"].dropna()]))) if not matches.empty else []

    if venue_encoder is not None and hasattr(venue_encoder, "classes_"):
        venues = sorted(list(venue_encoder.classes_))
    else:
        venues = sorted(matches["venue"].dropna().unique()) if not matches.empty else []

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        team1 = st.selectbox("Select Team 1", teams, index=0 if teams else 0, key="team1")
        team1_form = st.slider(f"{team1} Recent Form (0–10)", 0, 10, 5, key="team1_form")
        team1_form = float(team1_form) / 10.0  # normalize 0-1
    with col2:
        team2 = st.selectbox("Select Team 2", teams, index=1 if len(teams) > 1 else 0, key="team2")
        team2_form = st.slider(f"{team2} Recent Form (0–10)", 0, 10, 5, key="team2_form")
        team2_form = float(team2_form) / 10.0  # normalize 0-1
    with col3:
        venue = st.selectbox("Select Venue", venues, index=0 if venues else 0)
        toss_winner = st.selectbox("Toss Winner", [team1, team2], index=0)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"], index=1)

    st.write("")  # spacing

    # Compute H2H win percentages
    if not matches.empty:
        h2h_matches = matches[
            ((matches["team1"] == team1) & (matches["team2"] == team2)) |
            ((matches["team1"] == team2) & (matches["team2"] == team1))
        ]
        total_h2h = len(h2h_matches)
        team1_wins = int((h2h_matches["winner"] == team1).sum()) if total_h2h > 0 else 0
        team2_wins = int((h2h_matches["winner"] == team2).sum()) if total_h2h > 0 else 0
        team1_win_pct = (team1_wins / total_h2h) if total_h2h > 0 else 0.0
        team2_win_pct = (team2_wins / total_h2h) if total_h2h > 0 else 0.0
    else:
        total_h2h = 0
        team1_wins = team2_wins = 0
        team1_win_pct = team2_win_pct = 0.0

    # Predict button
    if st.button("Predict Winner"):
        if team1 == team2:
            st.error("Team 1 and Team 2 cannot be the same.")
        else:
            if model is None:
                st.error("Model not loaded. Cannot run prediction.")
            else:
                # Build features dict
                features = {
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "team1_form": team1_form,
                    "team2_form": team2_form,
                    "team1_win_pct": team1_win_pct,
                    "team2_win_pct": team2_win_pct
                }
                try:
                    winner, probs = predict_match_winner(model, team_encoder, venue_encoder, toss_encoder, features)
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                else:
                    st.subheader(f"🏆 Predicted Winner: {winner}")
                    # Probabilities dict -> Plotly pie
                    labels = [team1, team2]
                    vals = [probs.get(team1, 0.0), probs.get(team2, 0.0)]
                    fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0.35,
                                                 marker=dict(colors=["#FF6F00","#1E90FF"]),
                                                 sort=False)])
                    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="#0b1020",
                                      font=dict(color="#e6eef8"))
                    st.plotly_chart(fig, use_container_width=True)

                    # Display numeric probs
                    colp1, colp2 = st.columns(2)
                    colp1.metric(f"{team1} Win %", f"{vals[0]:.2f}%")
                    colp2.metric(f"{team2} Win %", f"{vals[1]:.2f}%")

                    # H2H summary
                    st.markdown("### 📊 Head-to-Head")
                    st.write(f"Total Matches Between {team1} & {team2}: {total_h2h}")
                    st.write(f"{team1} Wins: {team1_wins}")
                    st.write(f"{team2} Wins: {team2_wins}")

                    # Venue-specific averages: compute average runs scored by each team at selected venue,
                    # and average wickets lost (approx by counting is_wicket per match)
                    if not matches.empty and not deliveries.empty:
                        # find match_ids where either team played at that venue
                        venue_matches = matches[(matches["venue"] == venue) & ((matches["team1"].isin([team1, team2])) | (matches["team2"].isin([team1, team2])))]
                        vm_ids = venue_matches["match_id"].unique().tolist()
                        if len(vm_ids) > 0:
                            dv = deliveries[deliveries["match_id"].isin(vm_ids)]
                            # team1 batting totals at venue (deliveries has inning_team field)
                            t1_del = dv[dv["inning_team"] == team1]
                            t2_del = dv[dv["inning_team"] == team2]
                            # approximate runs per match
                            if len(venue_matches) > 0:
                                team1_avg = t1_del["runs_scored"].sum() / max(1, len(venue_matches))
                                team2_avg = t2_del["runs_scored"].sum() / max(1, len(venue_matches))
                                team1_wkts = t1_del["is_wicket"].sum() / max(1, len(venue_matches))
                                team2_wkts = t2_del["is_wicket"].sum() / max(1, len(venue_matches))
                                st.write(f"**{team1} avg runs at {venue}:** {team1_avg:.1f} (avg wickets lost: {team1_wkts:.1f})")
                                st.write(f"**{team2} avg runs at {venue}:** {team2_avg:.1f} (avg wickets lost: {team2_wkts:.1f})")
                            else:
                                st.write("No venue historical data.")
                        else:
                            st.write("No historical matches at this venue between these teams or data not present.")
                    else:
                        st.write("Delivery or match data missing for venue analysis.")

    # ---------------- Key player charts for only selected teams ----------------
    st.markdown("---")
    st.header("🔥 Key Players & Top Performers (filtered for selected teams)")

    if deliveries.empty:
        st.warning("Delivery data not loaded; top player charts cannot be shown.")
    else:
        # Filter deliveries to matches where selected teams played each other OR in general include all matches where inning_team = team
        # The requirement: show players stats only against the selected opponent. So first try H2H match_ids, else show team overall.
        if not matches.empty:
            # find match_ids where both teams played (any order)
            h2h_ids = matches[
                ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                ((matches["team1"] == team2) & (matches["team2"] == team1))
            ]["match_id"].unique().tolist()
        else:
            h2h_ids = []

        # For batting: we want top run scorers for each team *against the opponent* (use h2h_ids).
        if len(h2h_ids) > 0:
            dv_h2h = deliveries[deliveries["match_id"].isin(h2h_ids)]
            t1_bat = dv_h2h[dv_h2h["inning_team"] == team1].groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(8)
            t2_bat = dv_h2h[dv_h2h["inning_team"] == team2].groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(8)
            t1_bowl = dv_h2h[dv_h2h["inning_team"] == team2].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(8)
            t2_bowl = dv_h2h[dv_h2h["inning_team"] == team1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(8)
            note = f"_Stats against selected opponent (head-to-head: {len(h2h_ids)} matches)_"
        else:
            # no H2H data: fallback to overall team stats (all matches)
            dv_h2h = deliveries
            t1_bat = dv_h2h[dv_h2h["inning_team"] == team1].groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(8)
            t2_bat = dv_h2h[dv_h2h["inning_team"] == team2].groupby("batter")["runs_scored"].sum().sort_values(ascending=False).head(8)
            t1_bowl = dv_h2h[dv_h2h["inning_team"] == team2].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(8)
            t2_bowl = dv_h2h[dv_h2h["inning_team"] == team1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(8)
            note = "_No head-to-head deliveries found — showing overall team stats._"

        st.write(note)
        # Plot side-by-side
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.subheader(f"Top Batters: {team1}")
            if not t1_bat.empty:
                fig = px.bar(x=t1_bat.values, y=t1_bat.index, orientation='h', labels={'x': 'Runs', 'y': 'Batter'})
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="#0b1020", font_color="#e6eef8")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No batting data.")

        with bcol2:
            st.subheader(f"Top Batters: {team2}")
            if not t2_bat.empty:
                fig = px.bar(x=t2_bat.values, y=t2_bat.index, orientation='h', labels={'x': 'Runs', 'y': 'Batter'})
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="#0b1020", font_color="#e6eef8")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No batting data.")

        # Bowlers
        bcol3, bcol4 = st.columns(2)
        with bcol3:
            st.subheader(f"Top Wicket-takers vs {team2}")
            if not t1_bowl.empty:
                fig = px.bar(x=t1_bowl.values, y=t1_bowl.index, orientation='h', labels={'x': 'Wickets', 'y': 'Bowler'})
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="#0b1020", font_color="#e6eef8")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No bowling data.")
        with bcol4:
            st.subheader(f"Top Wicket-takers vs {team1}")
            if not t2_bowl.empty:
                fig = px.bar(x=t2_bowl.values, y=t2_bowl.index, orientation='h', labels={'x': 'Wickets', 'y': 'Bowler'})
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="#0b1020", font_color="#e6eef8")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No bowling data.")

with tab2:
    st.header("🤖 IPL Chatbot (Google Custom Search)")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_SEARCH_KEY") if hasattr(st, "secrets") else None
    GOOGLE_CX = st.secrets.get("GOOGLE_SEARCH_CX") if hasattr(st, "secrets") else None

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        st.warning("Google Custom Search keys are missing in Streamlit secrets. The chatbot will be disabled.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask anything about IPL (the bot will search the web):", key="gsearch_input")
        if st.button("Search"):
            if not query.strip():
                st.warning("Type a question first.")
            else:
                try:
                    search_url = "https://www.googleapis.com/customsearch/v1"
                    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query, "num": 3}
                    resp = requests.get(search_url, params=params, timeout=10)
                    data = resp.json()
                    if resp.status_code != 200:
                        st.error(f"Search API error: {data.get('error', data)}")
                    else:
                        items = data.get("items", [])
                        if not items:
                            st.info("No results found.")
                        else:
                            answer = ""
                            for it in items:
                                title = it.get("title")
                                snippet = it.get("snippet")
                                link = it.get("link")
                                answer += f"**{title}**\n{snippet}\n<{link}>\n\n"
                            st.markdown(answer)
                            st.session_state.chat_history.append({"q": query, "a": answer})
                except Exception as e:
                    st.error(f"Chat error: {e}")

# ----------------- End -----------------
