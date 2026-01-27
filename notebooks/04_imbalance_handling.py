import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE


df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test_scaled[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

lr_weighted = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

lr_weighted.fit(X_train_scaled, y_train)

y_pred_w = lr_weighted.predict(X_test_scaled)
y_prob_w = lr_weighted.predict_proba(X_test_scaled)[:, 1]

print("\n===== Logistic Regression (Class Weights) =====")
print(classification_report(y_test, y_pred_w))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_w))


smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled, y_train
)

lr_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_smote.fit(X_train_smote, y_train_smote)

y_pred_s = lr_smote.predict(X_test_scaled)
y_prob_s = lr_smote.predict_proba(X_test_scaled)[:, 1]

print("\n===== Logistic Regression (SMOTE) =====")
print(classification_report(y_test, y_pred_s))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_s))

print("\nImbalance handling comparison completed ✅")

