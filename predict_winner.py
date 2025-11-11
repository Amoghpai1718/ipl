import pandas as pd

def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder,
                         team1, team2, venue, toss_winner,
                         team1_form, team2_form, team1_win_pct, team2_win_pct):
    """
    Predict the match winner and return probability for each team.
    This function aligns with ipl_model.pkl feature order.
    """

    # Encode categorical variables
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]  # toss winner encoded as team

    # Create input dataframe in correct feature order
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

    # Predict winner (model predicts 1 if team1 wins, 0 if team2 wins)
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    winner = team1 if pred == 1 else team2
    win_probs = {
        team1: proba[1],
        team2: proba[0]
    }

    return winner, win_probs
