import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from Bio import SeqIO
import pathlib
import pickle
from fast_dataset_loader import load_dataset_fast

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset using the optimized loader
X, Y, Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR, Y_TB, Y_WHH = load_dataset_fast()

# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)
Y_ZF = np.array(Y_ZF)
Y_LZ = np.array(Y_LZ)
Y_HH = np.array(Y_HH)
Y_BHH = np.array(Y_BHH)
Y_HB = np.array(Y_HB)
Y_FH = np.array(Y_FH)
Y_POU = np.array(Y_POU)
Y_NR = np.array(Y_NR)
Y_TB = np.array(Y_TB)
Y_WHH = np.array(Y_WHH)

# Keep only TF samples that contain at least one of the five key motifs: ZF, LZ, BHLH, FH, WHH
five_motif_any = (
    Y_ZF.flatten() +
    Y_LZ.flatten() +
    Y_BHH.flatten() +
    Y_FH.flatten() +
    Y_WHH.flatten()
) > 0
keep_mask = (Y == 0) | ((Y == 1) & five_motif_any)

# Apply mask to all arrays
X = X[keep_mask]
Y = Y[keep_mask]
Y_ZF = Y_ZF[keep_mask]
Y_LZ = Y_LZ[keep_mask]
Y_HH = Y_HH[keep_mask]
Y_BHH = Y_BHH[keep_mask]
Y_HB = Y_HB[keep_mask]
Y_FH = Y_FH[keep_mask]
Y_POU = Y_POU[keep_mask]
Y_NR = Y_NR[keep_mask]
Y_TB = Y_TB[keep_mask]
Y_WHH = Y_WHH[keep_mask]

print("y_ZF:", np.count_nonzero(Y_ZF == 1))
print("y_HH:", np.count_nonzero(Y_HH == 1))
print("y_LZ:", np.count_nonzero(Y_LZ == 1))
print("Y_BHH:", np.count_nonzero(Y_BHH == 1))
print("Y_HB:", np.count_nonzero(Y_HB == 1))
print("Y_FH:", np.count_nonzero(Y_FH == 1))
print("Y_POU:", np.count_nonzero(Y_POU == 1))
print("Y_NR:", np.count_nonzero(Y_NR == 1))
print("Y_TB:", np.count_nonzero(Y_TB == 1))
print("Y_WHH:", np.count_nonzero(Y_WHH == 1))

print("NTF: ", np.count_nonzero(Y == 0))
print("TF: ", np.count_nonzero(Y == 1))


sequences = torch.tensor(X).float().to(device)
labels = torch.tensor(Y).long().to(device)
labels_Y_ZF = torch.tensor(Y_ZF).long().to(device)
labels_Y_LZ = torch.tensor(Y_LZ).long().to(device)
labels_Y_HH = torch.tensor(Y_HH).long().to(device)
labels_Y_BHH = torch.tensor(Y_BHH).long().to(device)
labels_Y_HB = torch.tensor(Y_HB).long().to(device)
labels_Y_FH = torch.tensor(Y_FH).long().to(device)
labels_Y_POU = torch.tensor(Y_POU).long().to(device)
labels_Y_NR = torch.tensor(Y_NR).long().to(device)
labels_Y_TB = torch.tensor(Y_TB).long().to(device)
labels_Y_WHH = torch.tensor(Y_WHH).long().to(device)

sequences = sequences.unsqueeze(1)
print("sequences: ", sequences.shape)
# Create a DataLoader for batch processing
dataset = TensorDataset(sequences, labels, labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH) #
train_size = int(0.8 * len(Y))
test_size = len(Y) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

class MultiLossModel(nn.Module):
    """
    Multi-Loss Baseline Model that uses multiple loss functions but lacks LTN's interpretability.
    This model demonstrates that while multi-loss approaches can achieve good performance,
    they cannot provide per-rule satisfaction monitoring like LTN.
    """
    def __init__(self, input_size):
        super(MultiLossModel, self).__init__()
        # CNN backbone
        self.conv1 = nn.Conv1d(in_channels=input_size[0], out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(256 * (input_size[1] // 4), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Main TF/NTF classifier
        self.main_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Motif-specific auxiliary classifiers (for multi-task learning)
        self.motif_classifiers = nn.ModuleDict({
            'zf_classifier': nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'lz_classifier': nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'bhlh_classifier': nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'fh_classifier': nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'whh_classifier': nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        })
        
        self.to(device)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        
        # Shared features
        features = self.feature_extractor(x)
        
        # Main prediction
        main_pred = self.main_classifier(features)
        
        # Motif predictions
        motif_preds = {}
        for motif_name, classifier in self.motif_classifiers.items():
            motif_preds[motif_name] = classifier(features)
        
        return main_pred, motif_preds


model = MultiLossModel(input_size=(1, len(X[0])))

def compute_multi_loss(model, data, labels, motif_labels_list, alpha=0.7, beta=0.3):
    """
    Compute multi-loss that combines main task loss with auxiliary motif losses.
    Unlike LTN, this approach lacks interpretable per-rule satisfaction monitoring.
    
    Args:
        alpha: Weight for main TF/NTF classification loss
        beta: Weight for auxiliary motif classification losses
    """
    main_pred, motif_preds = model(data)
    
    # Main classification loss
    main_loss = F.binary_cross_entropy(main_pred.squeeze(), labels.float())
    
    # Auxiliary motif losses (multi-task learning)
    motif_Y_ZF, motif_Y_LZ, motif_Y_BHH, motif_Y_FH, motif_Y_WHH = motif_labels_list
    motif_labels_dict = {
        'zf_classifier': motif_Y_ZF.float(),
        'lz_classifier': motif_Y_LZ.float(),
        'bhlh_classifier': motif_Y_BHH.float(),
        'fh_classifier': motif_Y_FH.float(),
        'whh_classifier': motif_Y_WHH.float()
    }
    
    auxiliary_loss = torch.tensor(0.0, device=data.device)
    for motif_name, motif_pred in motif_preds.items():
        motif_target = motif_labels_dict[motif_name]
        # Ensure both prediction and target have the same shape
        motif_pred_squeezed = motif_pred.squeeze()
        motif_target_squeezed = motif_target.squeeze()
        aux_loss = F.binary_cross_entropy(motif_pred_squeezed, motif_target_squeezed)
        auxiliary_loss += aux_loss
    
    auxiliary_loss /= len(motif_preds)  # Average across motifs
    
    # Combined loss
    total_loss = alpha * main_loss + beta * auxiliary_loss
    
    return total_loss, main_loss, auxiliary_loss

def compute_pseudo_satisfaction(loader):
    """
    Compute a pseudo-satisfaction metric for comparison with LTN.
    NOTE: This is NOT real logical satisfaction - it's just performance metrics.
    This demonstrates that multi-loss approaches lack true interpretability.
    """
    model.eval()
    total_consistency = 0.0
    count = 0
    
    with torch.no_grad():
        for data, labels, labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH in loader:
            main_pred, motif_preds = model(data)
            
            # Compute "consistency" between main and motif predictions
            # This is NOT logical satisfaction, just empirical consistency
            main_binary = (main_pred.squeeze() > 0.5).float()
            
            # For TF samples, check if motif predictions are consistent
            tf_mask = (labels == 1)
            if tf_mask.any():
                tf_main = main_binary[tf_mask]
                motif_consistency = 0.0
                
                for motif_name, motif_pred in motif_preds.items():
                    motif_binary = (motif_pred.squeeze()[tf_mask] > 0.5).float()
                    # If main predicts TF, motif predictions should also be reasonable
                    consistency = (tf_main * motif_binary).mean()
                    motif_consistency += consistency.item()
                
                motif_consistency /= len(motif_preds)
                total_consistency += motif_consistency
                count += 1
    
    return total_consistency / count if count > 0 else 0.0
'''
def compute_sat_level(loader):
    # Initialize dictionaries to store cumulative satisfaction levels as LTNObjects
    sat_levels_sum = {
        "labels": torch.tensor(0.0).to(device),
        "labels_Y_ZF": torch.tensor(0.0).to(device),
        "labels_Y_LZ": torch.tensor(0.0).to(device),
        "labels_Y_BHH": torch.tensor(0.0).to(device),
        "labels_Y_FH": torch.tensor(0.0).to(device),
        "labels_Y_WHH": torch.tensor(0.0).to(device)
    }
    count = len(loader)

    for data, labels, labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH in loader:
        # Calculate satisfaction level for TF (positive) and NTF (negative) classes
        if (labels == 1).any():
            x_TF = ltn.Variable("x_TF", data[labels == 1])
            sat_levels_sum["labels"] += Forall(x_TF, P(x_TF)).value
        else:
            sat_levels_sum["labels"] += torch.tensor(1.0).to(device)

        if (labels == 0).any():
            x_NTF = ltn.Variable("x_NTF", data[labels == 0])
            sat_levels_sum["labels"] += Forall(x_NTF, Not(P(x_NTF))).value
        else:
            sat_levels_sum["labels"] += torch.tensor(1.0).to(device)

        # Calculate satisfaction level for each motif-related label
        for label, ltn_var, key in zip([labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH],
                                       ["x_ZF", "x_LZ", "x_BHH", "x_FH", "x_WHH"],
                                       ["labels_Y_ZF", "labels_Y_LZ", "labels_Y_BHH", "labels_Y_FH", "labels_Y_WHH"]):
            if (label == 1).any():
                x = ltn.Variable(ltn_var, data[label == 1].unsqueeze(1))
                sat_levels_sum[key] += Forall(x, P(x)).value
            else:
                sat_levels_sum[key] += torch.tensor(1.0).to(device)

    # Normalize the satisfaction levels by the number of batches
    mean_sat_levels = {key: (value / count).cpu().item() for key, value in sat_levels_sum.items()}

    return mean_sat_levels
'''
def compute_accuracy(loader):
    """Compute comprehensive accuracy metrics for the multi-loss model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, *_ in loader:
            main_pred, _ = model(data)  # Only use main prediction for accuracy
            predictions = (main_pred.squeeze() > 0.5).cpu().numpy().astype(float)
            labels_np = labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate all metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    }
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
def compute_accuracy(loader):
    all_predictions = []
    all_labels = []

    for data, labels, *_ in loader:
        predictions = P.model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    }
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

train_losses = []
train_main_losses = []
train_aux_losses = []
train_pseudo_sats = []
test_pseudo_sats = []
train_accs = []
test_accs = []

num_epochs = 100  # Reduced epochs for multi-loss baseline

print("🔬 Starting Multi-Loss Baseline Training...")
print("NOTE: This model uses multiple loss functions but LACKS LTN's interpretable per-rule satisfaction monitoring")
print(f"Model: Multi-Loss CNN with {num_epochs} epochs")
print(f"Architecture: Main classifier + 5 auxiliary motif classifiers")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    
    for batch in train_loader:
        sequences_batch, labels_batch, label_Y_ZF_batch, labels_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH = batch
        
        motif_labels_list = [label_Y_ZF_batch, labels_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH]
        
        optimizer.zero_grad()
        
        # Compute multi-loss (main + auxiliary motif losses)
        loss, main_loss, aux_loss = compute_multi_loss(
            model, sequences_batch, labels_batch, motif_labels_list, 
            alpha=0.7, beta=0.3  # 70% main task, 30% auxiliary tasks
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    avg_main_loss = total_main_loss / len(train_loader)
    avg_aux_loss = total_aux_loss / len(train_loader)
    
    # Update learning rate
    scheduler.step(avg_loss)
    
    with torch.no_grad():
        # Compute pseudo-satisfaction (NOT real logical satisfaction like LTN)
        train_pseudo_sat = compute_pseudo_satisfaction(train_loader)
        test_pseudo_sat = compute_pseudo_satisfaction(test_loader)

        train_metrics = compute_accuracy(train_loader)
        test_metrics = compute_accuracy(test_loader)

        train_losses.append(avg_loss)
        train_main_losses.append(avg_main_loss)
        train_aux_losses.append(avg_aux_loss)
        train_pseudo_sats.append(train_pseudo_sat)
        test_pseudo_sats.append(test_pseudo_sat)
        train_accs.append(train_metrics)
        test_accs.append(test_metrics)
        torch.cuda.empty_cache()

    if epoch % 5 == 0:
        print(f" epoch {epoch:3d} | loss {avg_loss:.4f} | main_loss {avg_main_loss:.4f} | aux_loss {avg_aux_loss:.4f} | "
              f"Train PseudoSat {train_pseudo_sat:.3f} | Test PseudoSat {test_pseudo_sat:.3f} | "
              f"Train Acc {train_metrics['accuracy']:.3f} | Test Acc {test_metrics['accuracy']:.3f} | "
              f"Test Prec {test_metrics['precision']:.3f} | Test Rec {test_metrics['recall']:.3f} | "
              f"Test F1 {test_metrics['f1_score']:.3f} | Test BAcc {test_metrics['balanced_accuracy']:.3f}")

print(f"\n🚨 IMPORTANT NOTE:")
print(f"Multi-Loss Model Final Performance:")
print(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  - Test F1: {test_metrics['f1_score']:.4f}")
print(f"  - Pseudo-Satisfaction: {test_pseudo_sat:.4f} (NOT real logical satisfaction)")
print(f"\n❌ LIMITATION: This multi-loss approach CANNOT provide:")
print(f"  - Per-rule logical satisfaction monitoring")
print(f"  - Interpretable constraint violation analysis")  
print(f"  - Principled logical reasoning like LTN")
print(f"  - True logical consistency guarantees")
print(f"\n✅ LTN ADVANTAGE: LTN provides real logical satisfaction scores")
print(f"   that can be interpreted as constraint satisfaction levels!")

# Save the trained model
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/Multi-Loss_model.pth')

# Save the metrics after training
with open('training_metrics_multi_loss.pkl', 'wb') as f:
    pickle.dump({
        'train_losses': train_losses,
        'train_main_losses': train_main_losses,
        'train_aux_losses': train_aux_losses,
        'train_pseudo_sats': train_pseudo_sats,
        'test_pseudo_sats': test_pseudo_sats,
        'train_accs': train_accs,
        'test_accs': test_accs
    }, f)

"""
# Plotting the metrics
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot satisfaction levels
plt.subplot(1, 3, 2)
plt.plot(train_sat['labels'], label='Train Satisfaction')
plt.plot(test_sat['labels'], label='Test Satisfaction')
plt.title('Satisfaction Levels')
plt.xlabel('Epoch')
plt.ylabel('Satisfaction')
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.legend()

# Plot accuracies
plt.subplot(1, 3, 3)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.legend()

plt.tight_layout()
plt.show()
"""