import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("ipl_winner_model.pkl")
encoder = joblib.load("team_encoder.pkl")

# Input data
team1 = input("Enter Team 1 name: ")
team2 = input("Enter Team 2 name: ")
toss_winner = input("Enter Toss Winner: ")
toss_decision = input("Enter Toss Decision (bat/field): ")

# Encode inputs
data = pd.DataFrame({
    'team1': [team1],
    'team2': [team2],
    'toss_winner': [toss_winner],
    'toss_decision': [toss_decision]
})

for col in data.columns:
    data[col] = encoder.transform(data[col])

# Predict winner
pred = model.predict(data)
winner = encoder.inverse_transform(pred)
print(f"Predicted Winner: {winner[0]}")
