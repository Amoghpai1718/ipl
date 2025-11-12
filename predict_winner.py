# predict_winner.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def _safe_encode(encoder, value, name="encoder"):
    """
    Safely encode a value with a LabelEncoder-like object that has .classes_.
    If value unseen, append it to classes_ and return a new index.
    """
    if value is None:
        return np.nan
    if not hasattr(encoder, "classes_"):
        raise ValueError(f"{name} has no classes_ attribute")
    classes = list(encoder.classes_)
    if value in classes:
        return int(np.where(encoder.classes_ == value)[0][0])
    else:
        # Append unseen label (monkeypatch encoder.classes_) so transform won't fail next time.
        # This preserves runtime behavior (prevents crash), though model semantics for new label are uncertain.
        logging.warning(f"Value '{value}' not found in {name}. Adding as new class (fallback).")
        new_classes = np.append(encoder.classes_, value)
        # Monkeypatch encoder.classes_
        try:
            encoder.classes_ = new_classes
        except Exception:
            # If cannot set, just return next index
            return len(classes)
        return len(classes)

def build_input_df_for_model(model, team_encoder, venue_encoder, toss_encoder, features_dict):
    """
    Build a pandas DataFrame with columns in the exact order expected by the model.
    features_dict must contain keys: team1, team2, venue, toss_winner, toss_decision,
    team1_form, team2_form, team1_win_pct, team2_win_pct (some may be unused depending on model).
    """
    # Determine expected feature names from model
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        expected = list(model.feature_names_in_)
    else:
        # Try booster feature names (XGBoost)
        try:
            expected = list(model.get_booster().feature_names)
        except Exception:
            # fallback to a common 8-feature set
            expected = [
                "team1_enc", "team2_enc", "venue_enc", "toss_enc",
                "team1_form", "team2_form", "team1_win_pct", "team2_win_pct"
            ]
    # Build mapping values
    team1 = features_dict.get("team1")
    team2 = features_dict.get("team2")
    venue = features_dict.get("venue")
    toss_winner = features_dict.get("toss_winner")
    toss_decision = features_dict.get("toss_decision")
    team1_form = float(features_dict.get("team1_form", 0.5))
    team2_form = float(features_dict.get("team2_form", 0.5))
    team1_win_pct = float(features_dict.get("team1_win_pct", 0.0))
    team2_win_pct = float(features_dict.get("team2_win_pct", 0.0))

    # Encode with safe fallback
    # Some models used 'toss_enc' as encoded toss winner; some used 'toss_decision_enc' for bat/field.
    enc_values = {}
    if "team1_enc" in expected:
        enc_values["team1_enc"] = _safe_encode(team_encoder, team1, "team_encoder")
    if "team2_enc" in expected:
        enc_values["team2_enc"] = _safe_encode(team_encoder, team2, "team_encoder")
    if "venue_enc" in expected:
        enc_values["venue_enc"] = _safe_encode(venue_encoder, venue, "venue_encoder")
    if "toss_enc" in expected:
        # encode toss winner using team encoder (some training used that)
        enc_values["toss_enc"] = _safe_encode(team_encoder, toss_winner, "team_encoder")
    if "toss_winner_enc" in expected:
        enc_values["toss_winner_enc"] = _safe_encode(team_encoder, toss_winner, "team_encoder")
    if "toss_decision_enc" in expected:
        enc_values["toss_decision_enc"] = _safe_encode(toss_encoder, toss_decision, "toss_encoder")

    # numeric features
    num_values = {
        "team1_form": team1_form,
        "team2_form": team2_form,
        "team1_win_pct": team1_win_pct,
        "team2_win_pct": team2_win_pct
    }

    # Build row following expected column order
    row = []
    for col in expected:
        if col in enc_values:
            row.append(enc_values[col])
        elif col in num_values:
            row.append(num_values[col])
        else:
            # If model expects some other column, attempt to supply a reasonable default
            # Try common patterns
            if col == "toss_winner" and toss_winner is not None:
                row.append(toss_winner)
            else:
                # default numeric 0
                row.append(0)
    df = pd.DataFrame([row], columns=expected)
    return df

def predict_match_winner(model, team_encoder, venue_encoder, toss_encoder, features_dict):
    """
    Top-level helper: returns (winner_name, win_probabilities_dict).
    features_dict should include values as described in build_input_df_for_model docstring.
    """
    # Build input
    input_df = build_input_df_for_model(model, team_encoder, venue_encoder, toss_encoder, features_dict)

    # Predict using the model
    # Protect against models that raise on unseen feature sets
    try:
        proba = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]
    except Exception as e:
        # Try forcing column order to model.feature_names_in_ if available
        logging.warning("Predict failed first attempt, retrying with input columns as model.feature_names_in_. Error: %s", e)
        try:
            if hasattr(model, "feature_names_in_"):
                cols = list(model.feature_names_in_)
                input_df = input_df.reindex(columns=cols, fill_value=0)
            proba = model.predict_proba(input_df)[0]
            pred = model.predict(input_df)[0]
        except Exception as e2:
            logging.error("Predict failed on retry: %s", e2)
            raise

    # Interpret result: many training scripts set label 1 for team1 win, 0 for team2.
    # But we can't be 100% sure — we will assume that mapping: pred==1 => team1, else team2.
    team1 = features_dict.get("team1")
    team2 = features_dict.get("team2")

    winner = team1 if int(pred) == 1 else team2

    # Build probability dict: try to map proba[1] to team1 and proba[0] to team2.
    # If model.classes_ available and not {0,1}, attempt to map correctly.
    team_probs = {}
    try:
        classes = list(model.classes_)
        # attempt to find index of label that corresponds to team1_win (common encoding: 1)
        if 1 in classes and 0 in classes:
            idx_team1 = classes.index(1)
            idx_team2 = classes.index(0)
            team_probs[team1] = float(proba[idx_team1]) * 100.0
            team_probs[team2] = float(proba[idx_team2]) * 100.0
        else:
            # fallback: assume proba[1] -> team1, proba[0] -> team2
            if len(proba) >= 2:
                team_probs[team1] = float(proba[1]) * 100.0
                team_probs[team2] = float(proba[0]) * 100.0
            else:
                team_probs[team1] = float(proba[0]) * 100.0
                team_probs[team2] = float(100.0 - team_probs[team1])
    except Exception:
        # fallback simple mapping
        if len(proba) >= 2:
            team_probs[team1] = float(proba[1]) * 100.0
            team_probs[team2] = float(proba[0]) * 100.0
        else:
            team_probs[team1] = float(proba[0]) * 100.0
            team_probs[team2] = float(100.0 - team_probs[team1])

    return winner, team_probs
