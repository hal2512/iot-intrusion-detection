import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load preprocessed datasets
X_train = np.load("./datasets/preprocessed/X_train.npy")
X_val = np.load("./datasets/preprocessed/X_val.npy")
X_test = np.load("./datasets/preprocessed/X_test.npy")

y_train = np.load("./datasets/preprocessed/y_mc_train.npy")
y_val = np.load("./datasets/preprocessed/y_mc_val.npy")
y_test = np.load("./datasets/preprocessed/y_mc_test.npy")

# Check class distribution
print("\nClass distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} ({count/len(y_train)*100:.2f}%)")

# Random forest, with class_weight='balanced' to handle imbalance
rf_model = RandomForestClassifier(
    class_weight='balanced',    
    random_state=42,
    n_jobs=-1,                  
    verbose=1                   
)

rf_model.fit(X_train, y_train)

# Evaluating on validation set
y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Evaluation on test set
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

# Feature importance
with open('./datasets/feature_names.json') as f:
    feature_names = json.load(f)
importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

# Print
print(f"\nTop 20 most important features:")
print(importances.head(20).to_string())

# Plot
plt.figure(figsize=(10, 8))
importances.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 20 most important features (Random Forest Multi-class)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
