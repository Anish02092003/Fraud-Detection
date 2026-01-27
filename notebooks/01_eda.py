import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print(df.head())

print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nClass percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

sns.countplot(x="Class", data=df)
plt.title("Fraud vs Legit Transactions")
plt.show()

print("\nMissing values:")
print(df.isnull().sum().max())

print("\nTransaction Amount stats by Class:")
print(df.groupby("Class")["Amount"].describe())

