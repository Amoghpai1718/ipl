import pandas as pd
import numpy as np

def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    """
    Predicts the winner between two IPL teams using the trained model.
    Includes team form and win percentage as inputs.
    Returns:
        winner_name (str)
        win_probabilities (dict)
    """

    # Encode categorical inputs
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]

    # Input DataFrame (must match model’s 8 feature structure)
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

    # Predict probabilities
    proba = model.predict_proba(input_df)[0]

    # Get predicted class
    pred = model.predict(input_df)[0]

    # If the model encodes classes using LabelEncoder for winner
    classes = model.classes_

    # Map predicted class back to team name
    try:
        winner = team_encoder.inverse_transform([pred])[0]
    except Exception:
        # fallback if model directly outputs 0/1
        winner = team1 if pred == 1 else team2

    # Probabilities mapped to teams
    win_probs = {
        team1: float(proba[1] * 100) if winner == team1 else float(proba[0] * 100),
        team2: float(proba[0] * 100) if winner == team1 else float(proba[1] * 100)
    }

    return winner, win_probs
