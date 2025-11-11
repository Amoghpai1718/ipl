import pandas as pd
import numpy as np

def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form):
    """
    Predict the match winner with probabilities.
    Only takes 10 arguments now to avoid mismatch.
    """
    # Encode categorical features
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]

    # Construct feature DataFrame in correct order
    X = pd.DataFrame({
        'team1_enc': [team1_enc],
        'team2_enc': [team2_enc],
        'venue_enc': [venue_enc],
        'toss_enc': [toss_enc],
        'team1_form': [team1_form],
        'team2_form': [team2_form]
    })

    # Predict
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    # Map prediction to team
    winner = team1 if pred == 1 else team2
    win_probs = {team1: probs[1]*100, team2: probs[0]*100}

    return winner, win_probs
