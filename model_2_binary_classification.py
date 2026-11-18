import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import OneHotEncoder
import json


   

def binary_classification():

    # Load preprocessed data

    X_train = np.load("./datasets/preprocessed/X_train.npy")
    X_val = np.load("./datasets/preprocessed/X_val.npy")
    X_test = np.load("./datasets/preprocessed/X_test.npy")
    
    y_train_bin = np.load("./datasets/preprocessed/y_bin_train.npy")
    y_test_bin  = np.load("./datasets/preprocessed//y_bin_test.npy")
    y_val_bin   = np.load("./datasets/preprocessed//y_bin_val.npy")
    
          
    
    unique, counts = np.unique(y_train_bin, return_counts=True)
    print(dict(zip(unique, counts)))


    # Random Forest training
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose = 1
        
    )
    model.fit(X_train, y_train_bin)
    print("\nModel trained!")

    # Validate on validation set
    y_pred_val = model.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val_bin, y_pred_val):.4f}")
    print(f"Validation F1 (macro): {f1_score(y_val_bin, y_pred_val, average='macro'):.4f}")

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    print("\n=== Binary Classification — Test Results ===")
    print(f"Accuracy      : {accuracy_score(y_test_bin, y_pred_test):.4f}")
    print(f"F1 (weighted) : {f1_score(y_test_bin, y_pred_test, average='weighted'):.4f}")
    print(f"F1 (macro)    : {f1_score(y_test_bin, y_pred_test, average='macro'):.4f}")
    print(f"ROC-AUC       : {roc_auc_score(y_test_bin, y_prob_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_bin, y_pred_test, target_names=['Normal', 'Attack']))
 

  # Feature importance
    with open('./datasets/feature_names.json') as f:
        feature_names = json.load(f)

    print("Total feature names:", len(feature_names))
    print("Actual array columns:", X_train.shape[1])  
    feature_importance = pd.Series(model.feature_importances_).sort_values(ascending=False)

    feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    plt.figure(figsize=(10, 6))
    feature_importance.head(20).plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title('Top 20 Feature Importances — Binary Classification')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_binary.png', dpi=150)
    plt.show()

    

    return model, feature_importance




if __name__ == "__main__":
    binary_classification()