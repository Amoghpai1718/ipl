import pandas as pd

def predict_match_winner(model, team_encoder, toss_encoder, venue_encoder,
                         team1, team2, venue, toss_winner, toss_decision,
                         team1_form, team2_form):
    """
    Predict IPL match winner with probabilities.

    Returns:
        winner_name: str
        win_probs: dict {team1: prob, team2: prob}
    """
    # Encode categorical features
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]
    toss_decision_enc = toss_encoder.transform([toss_decision])[0]

    # Align features exactly as used in training
    features = pd.DataFrame({
        "team1_enc": [team1_enc],
        "team2_enc": [team2_enc],
        "venue_enc": [venue_enc],
        "toss_enc": [toss_enc],
        "toss_decision_enc": [toss_decision_enc],
        "team1_form": [team1_form],
        "team2_form": [team2_form]
    })

    # Predict
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    # Map prediction to team names
    winner_name = team1 if prediction == 1 else team2
    win_probs = {team1: proba[1], team2: proba[0]}

    return winner_name, win_probs
