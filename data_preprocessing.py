import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# Loading split datasets
X_train = pd.read_csv("./datasets/split/X_train.csv")
X_val = pd.read_csv("./datasets/split/X_val.csv")
X_test = pd.read_csv("./datasets/split/X_test.csv")
y_train = pd.read_csv("./datasets/split/y_train.csv").values.ravel()
y_val = pd.read_csv("./datasets/split/y_val.csv").values.ravel()
y_test = pd.read_csv("./datasets/split/y_test.csv").values.ravel()

os.makedirs('./datasets/preprocessed', exist_ok=True)

# Binary labels
NORMAL_CLASSES = [
    'MQTT_Publish',
    'Thing_Speak',
    'Wipro_bulb',
]

# 1 = Attack, 0 = Normal
y_bin_train = (~np.isin(y_train, NORMAL_CLASSES)).astype(int)
y_bin_val = (~np.isin(y_val, NORMAL_CLASSES)).astype(int)
y_bin_test = (~np.isin(y_test, NORMAL_CLASSES)).astype(int)

# Binary label distribution
print(f"\nBinary label distribution:")
print(f"Train - Attack: {y_bin_train.sum():,} | Normal: {len(y_bin_train)-y_bin_train.sum():,}")
print(f"Val   - Attack: {y_bin_val.sum():,} | Normal: {len(y_bin_val)-y_bin_val.sum():,}")
print(f"Test  - Attack: {y_bin_test.sum():,} | Normal: {len(y_bin_test)-y_bin_test.sum():,}")

# Preprocessing on training set only, transforming test and validation sets

# One-hot encoding features
one_hot_encoder = OneHotEncoder(sparse_output=False)
categorical_columns = X_train.select_dtypes(include=["str", "object"]).columns.tolist()
one_hot_encoder.fit(X_train[categorical_columns])
X_train_encoded = one_hot_encoder.transform(X_train[categorical_columns])
X_val_encoded = one_hot_encoder.transform(X_val[categorical_columns])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_columns])

# Creating DataFrames from encoded arrays
encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names,
    index=X_train.index
)

X_val_encoded_df = pd.DataFrame(
    X_val_encoded,
    columns=encoded_feature_names,
    index=X_val.index
)

X_test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=encoded_feature_names,
    index=X_test.index
)

# Adding encoded categorical columns
X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
X_val = X_val.drop(columns=categorical_columns).reset_index(drop=True)
X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)
X_train = pd.concat([X_train, X_train_encoded_df.reset_index(drop=True)], axis=1)
X_val = pd.concat([X_val, X_val_encoded_df.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test, X_test_encoded_df.reset_index(drop=True)], axis=1)

# Encoding check
print(f"X_train shape after one-hot: {X_train.shape}")

# Label encoding target for multi-class
label_encoder_y = LabelEncoder()
y_mc_train = label_encoder_y.fit_transform(y_train)
y_mc_val = label_encoder_y.transform(y_val)
y_mc_test = label_encoder_y.transform(y_test)

label_mapping = {i: name for i, name in enumerate(label_encoder_y.classes_)}

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Verification
print(f"One-hot encoding created {len(encoded_feature_names)} new features")
print(f"Feature means: {X_train_scaled.mean()}")
print(f"Feature std: {X_train_scaled.std():.3f}")

# Class weights
## Multi-class
mc_classes = np.unique(y_mc_train)
mc_class_weights = compute_class_weight('balanced', classes=mc_classes, y=y_mc_train)
mc_class_weight_dict = {int(k): float(v) for k, v in zip(mc_classes, mc_class_weights)}

## Binary
bin_classes = np.unique(y_bin_train)
bin_class_weights = compute_class_weight('balanced', classes=bin_classes, y=y_bin_train)
bin_class_weight_dict = {0: float(bin_class_weights[0]), 1: float(bin_class_weights[1])}

# SMOTE
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_train_smote,     y_mc_train_smote  = smote.fit_resample(X_train_scaled, y_mc_train)

smote_bin = SMOTE(random_state=RANDOM_STATE)
X_train_smote_bin, y_bin_train_smote = smote_bin.fit_resample(X_train_scaled, y_bin_train)

print(f'SMOTE multiclass: {len(y_mc_train_smote):,} samples')
print(f'SMOTE binary:     {len(y_bin_train_smote):,} samples')

# Saving preprocessed datasets
np.save('./datasets/preprocessed/X_train.npy',           X_train_scaled)
np.save('./datasets/preprocessed/X_val.npy',             X_val_scaled)
np.save('./datasets/preprocessed/X_test.npy',            X_test_scaled)
np.save('./datasets/preprocessed/y_mc_train.npy',        y_mc_train)
np.save('./datasets/preprocessed/y_mc_val.npy',          y_mc_val)
np.save('./datasets/preprocessed/y_mc_test.npy',         y_mc_test)
np.save('./datasets/preprocessed/y_bin_train.npy',       y_bin_train)
np.save('./datasets/preprocessed/y_bin_val.npy',         y_bin_val)
np.save('./datasets/preprocessed/y_bin_test.npy',        y_bin_test)
np.save('./datasets/preprocessed/X_train_smote.npy',     X_train_smote)
np.save('./datasets/preprocessed/y_mc_train_smote.npy',  y_mc_train_smote)
np.save('./datasets/preprocessed/X_train_smote_bin.npy', X_train_smote_bin)
np.save('./datasets/preprocessed/y_bin_train_smote.npy', y_bin_train_smote)

# Saving label mapping and class weights
with open('./datasets/label_mapping.json', 'w') as f:
    json.dump({str(k): v for k, v in label_mapping.items()}, f, indent=2)
with open('./datasets/class_weights.json', 'w') as f:
    json.dump(mc_class_weight_dict, f, indent=2)
with open('./datasets/bin_class_weights.json', 'w') as f:
    json.dump(bin_class_weight_dict, f, indent=2)

# Saving features for feature importance
feature_names = X_train.columns.tolist()
with open('./datasets/feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

print('\nAll data saved successfully!')
