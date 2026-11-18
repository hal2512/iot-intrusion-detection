# IoT Network Intrusion Detection

A from-scratch PyTorch MLP that classifies IoT network flows as normal traffic or one of 9 attack types. It was trained on the RT-IoT2022 dataset, which has a 3,380:1 class imbalance, using a leakage-free preprocessing pipeline.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)

## Results

All numbers are on a held-out test set (24,624 flows) that the model never saw during training or tuning.

| Task | Accuracy | Macro F1 |
|---|---|---|
| Binary: attack vs. normal | **99.70%** | **0.9917** |
| 12 classes: 9 attack types + 3 normal | **99.52%** | **0.9051** |

- **Binary:** attack class 1.00 precision / 1.00 recall; normal class 0.98 precision / 0.99 recall.
- **12 classes:** weighted F1 is 1.00 and macro F1 is 0.9051. The large classes are classified almost perfectly. The gap comes from the rarest attacks, which have only 6–7 test samples each.

Macro F1 is the main metric here. On this dataset, a model that always predicts the majority class already gets 77% accuracy. Macro F1 gives every class equal weight, so the rare attacks count as much as the common ones.

## Problem

IoT devices (smart bulbs, MQTT sensors, cloud-connected platforms) have little compute, weak security, and are common targets for network attacks. No one can inspect the traffic of thousands of devices by hand. The goal was to label each network flow automatically:

- **Binary:** attack or normal? This is a fast first-line filter.
- **12-class:** which of 9 attacks or 3 normal device behaviours is it? This identifies the specific threat.

## Dataset and class imbalance

[RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022) contains **123,117 real network flows** captured in an IoT testbed. Each flow has 83 features: 2 categorical (`proto`, `service`) and 81 numerical (packet timing, payload sizes, TCP flag counts, throughput).

| Type | Classes |
|---|---|
| Attack (9) | DOS_SYN_Hping, ARP_poisioning, NMAP_UDP_SCAN, NMAP_XMAS_TREE_SCAN, NMAP_OS_DETECTION, NMAP_TCP_scan, DDOS_Slowloris, Metasploit_Brute_Force_SSH, NMAP_FIN_SCAN |
| Normal (3) | Thing_Speak, MQTT_Publish, Wipro_bulb |

The imbalance is extreme. `DOS_SYN_Hping` (SYN flood) accounts for **76.9%** of all flows (94,659), while `NMAP_FIN_SCAN` has just **28 samples**, a **3,380:1** ratio.

The data is not included in this repo. See [`datasets/README.md`](datasets/README.md) for download instructions.

## Approach

### Leakage-free preprocessing pipeline

```
Raw CSV (123,117 rows)
        │
        ▼
1. Stratified 60/20/20 split          ← split FIRST: val/test stay untouched
        │
        ▼
2. One-hot encode proto + service     ← fit on train only  (~94 features)
        │
        ▼
3. StandardScaler                     ← fit on train only
        │
        ▼
4. SMOTE oversampling                 ← train only (k_neighbors=3 for 12-class)
        │
        ▼
Train (balanced) │ Validation (original mix) │ Test (original mix)
```

- **Split first, then fit.** The encoder, scaler and SMOTE are fitted only on the training split and then applied to validation and test. No statistics or synthetic samples from the evaluation data reach training.
- **SMOTE with `k_neighbors=3`.** `NMAP_FIN_SCAN` has only about 17 samples in the training split. Lowering k from the default 5 to 3 lets SMOTE create reliable synthetic neighbours for the rarest class.
- **Stratification** keeps the class proportions identical across all three splits, so even the 28-sample class appears in validation and test.

### Model: custom MLP with a hand-written training loop

```
Input (~94) → [Linear 128 → BatchNorm → ReLU → Dropout 0.3]
            → [Linear 256 → BatchNorm → ReLU → Dropout 0.3]
            → [Linear 128 → BatchNorm → ReLU → Dropout 0.2]
            → [Linear 64  → BatchNorm → ReLU → Dropout 0.2]
            → Linear → 12 logits (multi-class) | 1 logit (binary)
```

The model has about 88k trainable parameters. Both tasks use the same architecture; only the output layer changes. It is built from core PyTorch modules with no Keras or Lightning wrappers. The training loop, evaluation, early stopping and checkpointing are all written by hand.

| Choice | Why |
|---|---|
| MLP (not CNN/RNN) | Tabular data with one fixed-length vector per flow has no spatial or sequential structure to exploit |
| BatchNorm | Features have very different scales (byte counts, durations, flag counts); BatchNorm keeps training stable |
| Kaiming (He) init | Matches the ReLU activations and keeps signal variance stable across layers |
| Dropout 0.3 → 0.2 | Stronger regularisation in the wide early layers, lighter in the narrow later ones |
| Unweighted CrossEntropy / BCEWithLogits | SMOTE already balances the training set, so class weights would correct the imbalance twice |
| Adam (lr 1e-3, weight decay 1e-4) | Reliable with little tuning; L2 regularisation on top of dropout |
| ReduceLROnPlateau (×0.5, patience 5) | Finer convergence once validation loss stalls |
| Early stopping (patience 10) | Stops on validation loss and restores the best checkpoint before testing |
| `drop_last=True` on the train loader | Prevents a final batch of size 1, which breaks `BatchNorm1d` |

Seeds are fixed at 42. The device (CUDA, Apple Silicon MPS or CPU) is selected automatically.

### Baseline comparison

A teammate's Random Forest baseline (`class_weight='balanced'`, trained without SMOTE) scored higher macro F1: 0.998 for binary and about 0.95 for 12-class. This fits the common finding that tree ensembles are very strong on tabular data. It also suggests that SMOTE's synthetic samples for the 6–7-sample classes add noise and cost the MLP some precision on those classes.

## My role

This was a team project of 3 students in the *Deep Learning & Big Data* course at TH Deggendorf (03/2026 – 07/2026). I was responsible for:

- **Data pipeline:** the stratified 60/20/20 split (`data_split.py`) and preprocessing (`data_preprocessing.py`): one-hot encoding, scaling, label encoding and SMOTE applied to the training split only.
- **Model 1:** I designed and trained the custom MLP (`model_1.py`), including the Dataset/DataLoader setup, the hand-written training and evaluation loop, BatchNorm, dropout, Kaiming initialisation, early stopping and the ReduceLROnPlateau schedule.

The Random Forest baseline (`model_2_*.py`) and the exploratory plotting script (`data_visualization.py`) came from my teammates and are included here so the repo is complete. The original team repository is [aadil0405/Big-Data](https://github.com/aadil0405/Big-Data).

## How to run

```bash
git clone https://github.com/hal2512/iot-intrusion-detection.git
cd iot-intrusion-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put RT_IOT2022.csv in datasets/raw/ (see datasets/README.md), then run from the repo root:
python data_split.py            # stratified 60/20/20 split → datasets/split/
python data_preprocessing.py    # encode, scale, SMOTE      → datasets/preprocessed/
python model_1.py               # train + evaluate the MLP  → models/, plots/, results/

# optional
python model_2_binary_classification.py   # Random Forest baseline (binary)
python model_2_multiclass.py              # Random Forest baseline (12-class)
python data_visualization.py              # class distribution and flow-duration plots
```

`model_1.py` saves the best checkpoints to `models/`, the learning curves and confusion matrices to `plots/`, and the test metrics to `results/model1_results.json`.

## Project structure

```
.
├── data_split.py                       # stratified 60/20/20 train/val/test split
├── data_preprocessing.py               # one-hot, scaling, label encoding, SMOTE (train only)
├── model_1.py                          # custom PyTorch MLP: training loop, early stopping, evaluation
├── model_2_binary_classification.py    # Random Forest baseline, binary (teammate)
├── model_2_multiclass.py               # Random Forest baseline, 12-class (teammate)
├── data_visualization.py               # exploratory plots (teammate)
├── datasets/
│   └── README.md                       # where to download RT-IoT2022
└── requirements.txt
```

## Limitations and next steps

- **Focal loss** instead of plain cross-entropy, to concentrate on the rare classes the model still gets wrong.
- **Gentler per-class SMOTE ratios** instead of full balancing, to add less synthetic noise for classes with only a handful of real samples.
- **Drop the port columns** (`id.orig_p`, `id.resp_p`). In a fixed testbed they can encode the attack setup rather than general traffic behaviour.
- **Experiment tracking** (MLflow or Weights & Biases) instead of hand-saved JSON results.

## Tech stack

Python · PyTorch · scikit-learn · imbalanced-learn (SMOTE) · pandas · NumPy · matplotlib · seaborn

---

More projects: [hal2512.github.io](https://hal2512.github.io/)
