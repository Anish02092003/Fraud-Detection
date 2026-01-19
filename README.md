💳 Credit Card Fraud Detection System (Deployed ML API)

An end-to-end machine learning fraud detection system built on highly imbalanced financial transaction data and deployed as a production-ready REST API using Flask and Render.

The system predicts the probability of fraud for a transaction and returns a risk-based decision using a tuned threshold.

🚀 Live Deployment

Fraud Detection API (Render):
👉 https://fraud-detection-o8gr.onrender.com

Prediction Endpoint:

POST /predict

🧠 Problem Statement

Credit card fraud is a rare but high-impact event.
This project addresses the challenge of detecting fraudulent transactions where:

Fraud rate is ~0.17%

Accuracy is misleading

Recall and precision are critical business metrics

🎯 Project Objectives

Build a robust fraud detection model on imbalanced data

Optimize recall while controlling false positives

Implement threshold-based decision logic

Deploy a real-time inference API

Follow production ML best practices

📊 Dataset

Source: Kaggle – Credit Card Fraud Dataset

Records: 284,807 transactions

Fraud Rate: ~0.17%

Features:

V1–V28: PCA-transformed features

Time, Amount: Raw numerical features

Class: Target (1 = Fraud, 0 = Legit)

📌 Raw dataset is not committed to the repository to avoid data leakage and size issues.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Flask

Render (Deployment)

Postman (API Testing)

🧠 Machine Learning Approach
1️⃣ Exploratory Data Analysis (EDA)

Identified extreme class imbalance

Analyzed fraud vs legitimate transaction patterns

2️⃣ Preprocessing

Stratified train-test split

Feature scaling applied only to Time and Amount

Prevented data leakage

3️⃣ Baseline Models

Logistic Regression

Decision Tree

4️⃣ Class Imbalance Handling

Compared:

Class-weighted Logistic Regression

SMOTE + Logistic Regression

Evaluated trade-offs between recall and precision

5️⃣ Threshold Tuning

Tuned decision threshold between 0.2 – 0.3

Selected 0.25 based on business risk tolerance

Demonstrated that models output probabilities, not decisions

6️⃣ Production Inference Pipeline

Saved model, scaler, and threshold

Enforced strict feature ordering during inference

Returned probabilistic predictions with decisions

🔌 API Usage
🔹 Endpoint
POST /predict

🔹 Headers
Content-Type: application/json

🔹 Sample Request
{
  "Time": 100000,
  "Amount": 150.75,
  "V1": 0.0,
  "V2": 0.0,
  "V3": 0.0,
  "V4": 0.0,
  "V5": 0.0,
  "V6": 0.0,
  "V7": 0.0,
  "V8": 0.0,
  "V9": 0.0,
  "V10": 0.0,
  "V11": 0.0,
  "V12": 0.0,
  "V13": 0.0,
  "V14": 0.0,
  "V15": 0.0,
  "V16": 0.0,
  "V17": 0.0,
  "V18": 0.0,
  "V19": 0.0,
  "V20": 0.0,
  "V21": 0.0,
  "V22": 0.0,
  "V23": 0.0,
  "V24": 0.0,
  "V25": 0.0,
  "V26": 0.0,
  "V27": 0.0,
  "V28": 0.0
}

🔹 Sample Response
{
  "fraud_probability": 0.18,
  "decision": "LEGIT ✅"
}

📁 Project Structure
fraud-detection/
│
├── app.py                  # Flask API
├── requirements.txt
├── render.yaml
│
├── src/
│   ├── train_model.py
│   └── inference.py
│
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
│
└── notebooks/
    ├── EDA.ipynb
    ├── baseline_models.ipynb
    ├── imbalance_handling.ipynb
    └── threshold_tuning.ipynb

🧠 Key Learnings

Accuracy is misleading for rare-event detection

Recall is critical in fraud systems

Threshold tuning is a business decision

Feature schema consistency is essential in production

Deployment bugs are part of real ML systems

🧪 Testing

API tested using Postman

Deployed on Render Free Tier

Cold-start latency handled gracefully

📌 Future Improvements

Cost-sensitive loss optimization

Dynamic thresholds based on transaction amount

Model monitoring and drift detection

Batch prediction support

👨‍💻 Author

Pritish Kumar Lenka
Electronics & Communication Engineering | AI / ML
