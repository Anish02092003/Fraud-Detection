import pickle
import pandas as pd


model = pickle.load(open("models/fraud_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
threshold = pickle.load(open("models/threshold.pkl", "rb"))


FEATURE_ORDER = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def predict_transaction(transaction_dict):
    """
    Input: dict with transaction features
    Output: fraud probability + decision
    """

    # Create DataFrame with fixed column order
    df = pd.DataFrame([[transaction_dict[col] for col in FEATURE_ORDER]],
                      columns=FEATURE_ORDER)

    # Scale required features
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    # Predict
    prob = model.predict_proba(df)[0][1]
    decision = "FRAUD 🚨" if prob >= threshold else "LEGIT ✅"

    return {
        "fraud_probability": round(prob, 4),
        "decision": decision
    }

