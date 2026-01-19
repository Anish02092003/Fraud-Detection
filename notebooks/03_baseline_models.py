import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# 2. Split features & target
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 3. Scale Time & Amount
# -----------------------------
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test_scaled[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# -----------------------------
# 4. Logistic Regression
# -----------------------------
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n===== Logistic Regression =====")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# -----------------------------
# 5. Decision Tree
# -----------------------------
tree = DecisionTreeClassifier(
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

tree.fit(X_train_scaled, y_train)

y_pred_tree = tree.predict(X_test_scaled)
y_prob_tree = tree.predict_proba(X_test_scaled)[:, 1]

print("\n===== Decision Tree =====")
print(classification_report(y_test, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_tree))

print("\nBaseline modeling completed ✅")
