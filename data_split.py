import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Loading dataset
df = pd.read_csv("./datasets/raw/RT_IOT2022.csv")

print(df["Attack_type"].value_counts())

os.makedirs('./datasets/split', exist_ok=True)

# Removing index
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Features and target
X = df.drop("Attack_type", axis=1)
y = df["Attack_type"]

# Dataset info
print(f"\nNumber of features: {X.shape[1]}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nNumber of samples: {X.shape[0]}")
print(f"\nNumber of classes: {len(y.unique())}")
print(f"\nClasses: {sorted(y.unique())}")
print(f"\nData types: {df.dtypes}")
print(f"\nMissing values: {df.isnull().sum()}")

# Class imbalance
print(f"\nClass distribution (count): {y.value_counts()}")
print(f"\nClass distribution (percent): {y.value_counts(normalize=True)}")

# Train, test, validation split
# 60,20,20
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Verifying class imbalance matches
print(f"\nTraining class distribution: {y_train.value_counts(normalize=True)}")
print(f"\nTest class distribution: {y_test.value_counts(normalize=True)}")
print(f"\nValidation class distribution: {y_val.value_counts(normalize=True)}")

# Saving datasets
X_train.to_csv("./datasets/split/X_train.csv", index=False)
X_val.to_csv("./datasets/split/X_val.csv", index=False)
X_test.to_csv("./datasets/split/X_test.csv", index=False)
y_train.to_csv("./datasets/split/y_train.csv", index=False)
y_val.to_csv("./datasets/split/y_val.csv", index=False)
y_test.to_csv("./datasets/split/y_test.csv", index=False)

print('\nAll data saved sucessfully!')
