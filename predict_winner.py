import pandas as pd
import numpy as np

def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    """
    Predicts the winner between two IPL teams, including form and H2H stats.
    Returns:
        winner_name (str)
        win_probabilities (dict)
    """
    # Encode categorical inputs
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]  # model uses toss winner encoding

    # Build input DataFrame with exact 8 features
    input_df = pd.DataFrame({
        "team1_enc": [team1_enc],
        "team2_enc": [team2_enc],
        "venue_enc": [venue_enc],
        "toss_enc": [toss_enc],
        "team1_form": [team1_form],
        "team2_form": [team2_form],
        "team1_win_pct": [team1_win_pct],
        "team2_win_pct": [team2_win_pct]
    })

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Decode prediction
    winner = team1 if pred == 1 else team2
    win_probs = {team1: proba[1]*100, team2: proba[0]*100}

    return winner, win_probs
