import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print(df.head())

# Target distribution
print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nClass percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# Plot imbalance
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Legit Transactions")
plt.show()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum().max())

# Amount analysis
print("\nTransaction Amount stats by Class:")
print(df.groupby("Class")["Amount"].describe())
