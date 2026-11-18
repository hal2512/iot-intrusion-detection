import numpy as np
import json, os, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

print(f'PyTorch {torch.__version__}  |  Device: {DEVICE}')

os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots',   exist_ok=True)

# Load data
DATA_DIR = './datasets/preprocessed'
DATA_DIR_2 = './datasets'

X_train     = np.load(f'{DATA_DIR}/X_train_smote.npy').astype(np.float32)
X_train_bin = np.load(f'{DATA_DIR}/X_train_smote_bin.npy').astype(np.float32)
X_val       = np.load(f'{DATA_DIR}/X_val.npy').astype(np.float32)
X_test      = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)

y_mc_train  = np.load(f'{DATA_DIR}/y_mc_train_smote.npy').astype(np.int64)
y_mc_val    = np.load(f'{DATA_DIR}/y_mc_val.npy').astype(np.int64)
y_mc_test   = np.load(f'{DATA_DIR}/y_mc_test.npy').astype(np.int64)

y_bin_train = np.load(f'{DATA_DIR}/y_bin_train_smote.npy').astype(np.float32)
y_bin_val   = np.load(f'{DATA_DIR}/y_bin_val.npy').astype(np.float32)
y_bin_test  = np.load(f'{DATA_DIR}/y_bin_test.npy').astype(np.float32)

with open(f'{DATA_DIR_2}/label_mapping.json') as f:
    label_mapping = {int(k): v for k, v in json.load(f).items()}

NUM_FEATURES = X_train.shape[1]
NUM_CLASSES  = len(label_mapping)
print(f'Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}')
print(f'Features: {NUM_FEATURES} | Classes: {NUM_CLASSES}')

# Dataset & DataLoader
class IoTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


BATCH_SIZE = 512

def make_loaders(X_tr, y_tr, X_v, y_v, X_te, y_te):
    _pin = (DEVICE.type == 'cuda')
    return (
        # drop_last on the training loader guards BatchNorm1d against a final batch of size 1
        DataLoader(IoTDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=_pin, drop_last=True),
        DataLoader(IoTDataset(X_v,  y_v),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin),
        DataLoader(IoTDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin),
    )


mc_train_loader,  mc_val_loader,  mc_test_loader  = make_loaders(X_train,     y_mc_train,  X_val, y_mc_val,  X_test, y_mc_test)
bin_train_loader, bin_val_loader, bin_test_loader = make_loaders(X_train_bin, y_bin_train, X_val, y_bin_val, X_test, y_bin_test)

# MLP Architecture
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes,
                 hidden_dims=[128, 256, 128, 64],
                 dropout_p=[0.3, 0.3, 0.2, 0.2]):
        super(MLP, self).__init__()
        layers, in_dim = [], input_dim
        for out_dim, drop in zip(hidden_dims, dropout_p):
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(p=drop)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


HIDDEN_DIMS   = [128, 256, 128, 64]
DROPOUT_PROBS = [0.3,  0.3,  0.2,  0.2]

model_mc  = MLP(NUM_FEATURES, NUM_CLASSES, HIDDEN_DIMS, DROPOUT_PROBS).to(DEVICE)
model_bin = MLP(NUM_FEATURES, 1,           HIDDEN_DIMS, DROPOUT_PROBS).to(DEVICE)

total_params = sum(p.numel() for p in model_mc.parameters() if p.requires_grad)
print(f'Trainable parameters: {total_params:,}')

# Loss functions
# Class imbalance is handled upstream by SMOTE oversampling
criterion_mc  = nn.CrossEntropyLoss()
criterion_bin = nn.BCEWithLogitsLoss()

# Optimisers & schedulers
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

optimizer_mc  = optim.Adam(model_mc.parameters(),  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer_bin = optim.Adam(model_bin.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler_mc  = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mc,  mode='min', factor=0.5, patience=5)
scheduler_bin = optim.lr_scheduler.ReduceLROnPlateau(optimizer_bin, mode='min', factor=0.5, patience=5)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4, path='best_model.pt'):
        self.patience  = patience
        self.delta     = delta
        self.path      = path
        self.best_loss = np.inf
        self.counter   = 0
        self.triggered = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path, map_location=DEVICE))
        print(f'Loaded best model from {self.path}')

# Training & evaluation functions
def train_one_epoch(model, loader, criterion, optimizer, task='mc'):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        if task == 'mc':
            loss  = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)
        else:
            loss  = criterion(logits.squeeze(), y_batch)
            preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (preds == y_batch.long()).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, task='mc'):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        if task == 'mc':
            loss  = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)
        else:
            loss  = criterion(logits.squeeze(), y_batch)
            preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long()
        total_loss += loss.item() * len(y_batch)
        correct    += (preds == y_batch.long()).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, task='mc'):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(DEVICE))
        if task == 'mc':
            preds = logits.argmax(dim=1).cpu().numpy()
        else:
            preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_history(history, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train', lw=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   lw=2, linestyle='--')
    axes[0].set_title(f'{title_prefix} — Loss'); axes[0].legend()
    axes[1].plot(epochs, history['train_acc'], label='Train', lw=2)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   lw=2, linestyle='--')
    axes[1].set_title(f'{title_prefix} — Accuracy'); axes[1].set_ylim(0, 1.05); axes[1].legend()
    plt.tight_layout()
    fname = f"plots/{title_prefix.lower().replace(' ', '_')}_curves.png"
    plt.savefig(fname, bbox_inches='tight'); plt.close()
    print(f'Saved: {fname}')


def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 task, model_path, num_epochs=100, patience=10):
    early_stop = EarlyStopping(patience=patience, path=model_path)
    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    print(f'\nTraining ({task}) for up to {num_epochs} epochs...')
    print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Val Loss":>10} | {"Train Acc":>10} | {"Val Acc":>10}')
    print('-' * 62)
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, task=task)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, task=task)
        scheduler.step(vl_loss)
        early_stop(vl_loss, model)
        history['train_loss'].append(tr_loss); history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc);   history['val_acc'].append(vl_acc)
        if epoch % 5 == 0 or epoch == 1:
            print(f'{epoch:>6} | {tr_loss:>10.4f} | {vl_loss:>10.4f} | {tr_acc:>10.4f} | {vl_acc:>10.4f}')
        if early_stop.triggered:
            print(f'Early stopping at epoch {epoch}.')
            break
    early_stop.load_best(model)
    print(f'Done in {time.time()-t0:.1f}s | Best val loss: {early_stop.best_loss:.4f}')
    return history

# Train multi-class
history_mc = run_training(
    model_mc, mc_train_loader, mc_val_loader,
    criterion_mc, optimizer_mc, scheduler_mc,
    task='mc', model_path='models/best_model_mc.pt'
)
plot_history(history_mc, 'Multi-class MLP')

# Evaluate multi-class
mc_preds, mc_true = get_predictions(model_mc, mc_test_loader, task='mc')
mc_acc = accuracy_score(mc_true, mc_preds)
mc_f1  = f1_score(mc_true, mc_preds, average='macro')
class_names = [label_mapping[i] for i in range(NUM_CLASSES)]

print(f'\n=== Multi-class Test Results ===')
print(f'Accuracy: {mc_acc:.4f} | F1 (macro): {mc_f1:.4f}')
print(classification_report(mc_true, mc_preds, target_names=class_names))

cm = confusion_matrix(mc_true, mc_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(13, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names,
            yticklabels=class_names, cmap='Blues', vmin=0, vmax=1)
plt.title('Confusion Matrix — Multi-class MLP')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('plots/confusion_matrix_mc.png', bbox_inches='tight'); plt.close()

# Train binary
history_bin = run_training(
    model_bin, bin_train_loader, bin_val_loader,
    criterion_bin, optimizer_bin, scheduler_bin,
    task='bin', model_path='models/best_model_bin.pt'
)
plot_history(history_bin, 'Binary MLP')

# Evaluate binary
bin_preds, bin_true = get_predictions(model_bin, bin_test_loader, task='bin')
bin_acc = accuracy_score(bin_true, bin_preds)
bin_f1  = f1_score(bin_true, bin_preds, average='macro')

print(f'\n=== Binary Test Results ===')
print(f'Accuracy: {bin_acc:.4f} | F1 (macro): {bin_f1:.4f}')
print(classification_report(bin_true, bin_preds, target_names=['Normal', 'Attack']))

cm_bin = confusion_matrix(bin_true, bin_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix — Binary MLP')
plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
plt.savefig('plots/confusion_matrix_bin.png', bbox_inches='tight'); plt.close()

# Save results
results = {
    'model1_mlp': {
        'multiclass': {'accuracy': round(float(mc_acc), 4),  'f1_macro': round(float(mc_f1), 4),
                       'epochs_trained': len(history_mc['train_loss'])},
        'binary':     {'accuracy': round(float(bin_acc), 4), 'f1_macro': round(float(bin_f1), 4),
                       'epochs_trained': len(history_bin['train_loss'])}
    }
}
with open('results/model1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nResults saved to results/model1_results.json')
print(json.dumps(results, indent=2))
