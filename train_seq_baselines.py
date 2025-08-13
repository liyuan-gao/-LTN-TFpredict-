import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

from utils_data import load_filtered_five_motifs
from models_seq import BiLSTMBackbone, CNNBiLSTMBackbone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
X, Y, motifs = load_filtered_five_motifs()
# Normalize inputs
X = (X - X.mean()) / (X.std() + 1e-8)

X_t = torch.tensor(X, dtype=torch.float32).to(device).unsqueeze(-1)  # (N, L, 1)
Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

# Datasets
dataset = TensorDataset(X_t, Y_t)
train_size = int(0.8 * len(Y))
train_ds, test_ds = random_split(dataset, [train_size, len(Y)-train_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Choose backbone via env var
BACKBONE = os.environ.get('BACKBONE', 'bilstm').lower()
if BACKBONE == 'bilstm':
    model = BiLSTMBackbone(input_features=1).to(device)
elif BACKBONE == 'cnn_bilstm':
    model = CNNBiLSTMBackbone(seq_len=X.shape[1]).to(device)
else:
    raise ValueError(f'Unknown backbone: {BACKBONE}')

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)


def evaluate(loader, model):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
    
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb).detach().cpu().numpy().flatten()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_probs.append(probs)
            all_true.append(yb.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)
    preds = (all_probs >= 0.5).astype(np.float32)
    
    # Calculate all metrics
    accuracy = accuracy_score(all_true, preds)
    precision = precision_score(all_true, preds, zero_division=0)
    recall = recall_score(all_true, preds, zero_division=0)
    f1 = f1_score(all_true, preds, zero_division=0)
    balanced_acc = balanced_accuracy_score(all_true, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    }

num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0:
        train_metrics = evaluate(train_loader, model)
        test_metrics = evaluate(test_loader, model)
        print(f"Epoch {epoch:03d} | Loss {total_loss/len(train_loader):.4f} | TrainAcc {train_metrics['accuracy']:.3f} | TestAcc {test_metrics['accuracy']:.3f} | TestPrec {test_metrics['precision']:.3f} | TestRec {test_metrics['recall']:.3f} | TestF1 {test_metrics['f1_score']:.3f} | TestBAcc {test_metrics['balanced_accuracy']:.3f}")

# Ensure model dir
os.makedirs('model', exist_ok=True)
# Save
pickle.dump({'backbone': BACKBONE}, open('model/baseline_backbone_meta.pkl','wb'))
torch.save(model.state_dict(), f'model/baseline_{BACKBONE}.pth') 