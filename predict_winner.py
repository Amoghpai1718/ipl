import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")

# Load historical match data
matches_df = pd.read_csv("all_matches.csv")

def predict_match_winner(team1, team2, venue, toss_winner, team1_form, team2_form):
    """
    Predicts the winner and returns:
    - Predicted team
    - Probability for team1 and team2
    - Head-to-head stats
    """
    # Encode categorical features
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = toss_encoder.transform([toss_winner])[0]

    # Compute historical win % for both teams
    team1_wins = matches_df[(matches_df['team1'] == team1) & (matches_df['winner'] == team1)].shape[0] + \
                 matches_df[(matches_df['team2'] == team1) & (matches_df['winner'] == team1)].shape[0]
    team1_total = matches_df[(matches_df['team1'] == team1) | (matches_df['team2'] == team1)].shape[0]
    team1_win_pct = team1_wins / team1_total if team1_total > 0 else 0.5

    team2_wins = matches_df[(matches_df['team1'] == team2) & (matches_df['winner'] == team2)].shape[0] + \
                 matches_df[(matches_df['team2'] == team2) & (matches_df['winner'] == team2)].shape[0]
    team2_total = matches_df[(matches_df['team1'] == team2) | (matches_df['team2'] == team2)].shape[0]
    team2_win_pct = team2_wins / team2_total if team2_total > 0 else 0.5

    # Prepare feature vector in exact order
    feature_vector = pd.DataFrame([[
        team1_enc, team2_enc, venue_enc, toss_enc,
        team1_win_pct, team2_win_pct, team1_form, team2_form
    ]], columns=[
        'team1_enc', 'team2_enc', 'venue_enc', 'toss_enc',
        'team1_win_pct', 'team2_win_pct', 'team1_form', 'team2_form'
    ])

    # Predict probabilities
    pred_probs = model.predict_proba(feature_vector)[0]
    predicted_team = team1 if pred_probs[0] > pred_probs[1] else team2

    # Head-to-head stats
    h2h_df = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
                        ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
    h2h_total = len(h2h_df)
    if h2h_total > 0:
        h2h_team1_wins = h2h_df[h2h_df['winner'] == team1].shape[0]
        h2h_team2_wins = h2h_df[h2h_df['winner'] == team2].shape[0]
        h2h_stats = f"{team1}: {h2h_team1_wins} wins, {team2}: {h2h_team2_wins} wins out of {h2h_total} matches"
    else:
        h2h_stats = "No previous head-to-head records."

    return predicted_team, pred_probs[0], pred_probs[1], h2h_stats
