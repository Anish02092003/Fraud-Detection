import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", df.shape)

# -----------------------------
# 2. Separate features & target
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nTarget distribution:")
print(y.value_counts(normalize=True) * 100)

# -----------------------------
# 3. Train-Test Split (STRATIFIED)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# -----------------------------
# 4. Feature Scaling (ONLY Time & Amount)
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
# 5. Sanity Check
# -----------------------------
print("\nScaled feature statistics (TRAIN):")
print("Mean:\n", X_train_scaled[["Time", "Amount"]].mean())
print("\nStd Dev:\n", X_train_scaled[["Time", "Amount"]].std())

print("\nPreprocessing completed successfully ✅")
