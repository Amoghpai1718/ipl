import joblib
import pandas as pd

# Load the trained model
model = joblib.load("ipl_model.pkl")  # Use the correct file

def predict_match_winner(team1_enc, team2_enc, venue_enc, toss_enc, team1_form, team2_form):
    """
    Predicts the winner probability given encoded features and form values.
    """
    input_df = pd.DataFrame([{
        "team1": team1_enc,
        "team2": team2_enc,
        "venue": venue_enc,
        "toss_winner": toss_enc,
        "team1_form": team1_form,
        "team2_form": team2_form
    }])

    # Align features exactly with training
    expected_features = model.get_booster().feature_names
    input_df = input_df[expected_features]

    # Predict probabilities
    pred_probs = model.predict_proba(input_df)[0]
    return pred_probs
