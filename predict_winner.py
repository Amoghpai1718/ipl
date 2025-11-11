import pandas as pd
import joblib

# Load the model and encoders
model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")

def predict_match_winner(team1, team2, venue, toss_winner, team1_form, team2_form, team1_win_pct, team2_win_pct):
    # Encode categorical features
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = toss_encoder.transform([toss_winner])[0]

    # Create features in **exact order** expected by model
    features = pd.DataFrame([{
        "team1_enc": team1_enc,
        "team2_enc": team2_enc,
        "venue_enc": venue_enc,
        "toss_enc": toss_enc,
        "team1_form": team1_form,
        "team2_form": team2_form,
        "team1_win_pct": team1_win_pct,
        "team2_win_pct": team2_win_pct
    }])

    # Explicitly reorder columns
    features = features[['team1_enc', 'team2_enc', 'venue_enc', 'toss_enc', 
                         'team1_form', 'team2_form', 'team1_win_pct', 'team2_win_pct']]

    # Predict probability
    pred_probs = model.predict_proba(features)[0]

    winner = team1 if pred_probs[0] > pred_probs[1] else team2
    win_pct = max(pred_probs)

    return winner, win_pct
