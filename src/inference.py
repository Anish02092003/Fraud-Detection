import pickle
import pandas as pd


# -----------------------------
# Load artifacts
# -----------------------------
model = pickle.load(open("models/fraud_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
threshold = pickle.load(open("models/threshold.pkl", "rb"))


def predict_transaction(transaction_dict):
    """
    Input: dict with same keys as dataset features
    Output: fraud probability + decision
    """

    df = pd.DataFrame([transaction_dict])

    # Scale required features
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    prob = model.predict_proba(df)[0][1]
    decision = "FRAUD 🚨" if prob >= threshold else "LEGIT ✅"

    return {
        "fraud_probability": round(prob, 4),
        "decision": decision
    }


# -----------------------------
# Local test
# -----------------------------
if __name__ == "__main__":

    sample_transaction = {
        "Time": 100000,
        "Amount": 150.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

    result = predict_transaction(sample_transaction)
    print(result)
