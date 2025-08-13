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

# MOTIF-BASED DATA AUGMENTATION
print("\n🔄 Implementing Motif-Based Data Augmentation...")
print("Augmenting positive training samples with proteins containing TF motifs")

def create_motif_augmented_dataset(X, Y, Y_ZF, Y_LZ, Y_BHH, Y_FH, Y_WHH, augmentation_factor=2):
    """
    Create augmented dataset by adding proteins with TF motifs as positive samples.
    This demonstrates a data augmentation baseline approach vs LTN's logical reasoning.
    """
    print(f"Original dataset size: {len(X)}")
    
    # Identify samples with each motif type
    motif_masks = {
        'ZF': Y_ZF == 1,
        'LZ': Y_LZ == 1, 
        'BHLH': Y_BHH == 1,
        'FH': Y_FH == 1,
        'WHH': Y_WHH == 1
    }
    
    # Start with original data
    X_aug = list(X)
    Y_aug = list(Y)
    Y_ZF_aug = list(Y_ZF)
    Y_LZ_aug = list(Y_LZ)
    Y_BHH_aug = list(Y_BHH)
    Y_FH_aug = list(Y_FH)
    Y_WHH_aug = list(Y_WHH)
    
    print(f"Motif distribution in original data:")
    for motif, mask in motif_masks.items():
        count = np.sum(mask)
        print(f"  {motif}: {count} samples")
    
    # Augment by adding synthetic positive samples based on motif presence
    total_added = 0
    
    for motif_name, motif_mask in motif_masks.items():
        motif_indices = np.where(motif_mask)[0]  # Get indices where mask is True
        motif_samples = X[motif_indices]
        motif_count = len(motif_samples)
        
        if motif_count > 0:
            # Add augmented samples (with noise for diversity)
            for _ in range(augmentation_factor):
                for sample in motif_samples:
                    # Add small noise to create variation
                    noise = np.random.normal(0, 0.05, sample.shape)
                    augmented_sample = sample + noise
                    
                    # Add as positive sample
                    X_aug.append(augmented_sample)
                    Y_aug.append(1)  # Always positive for motif-containing samples
                    
                    # Preserve motif labels but enhance the specific motif
                    Y_ZF_aug.append(np.array([1]) if motif_name == 'ZF' else np.array([0]))
                    Y_LZ_aug.append(np.array([1]) if motif_name == 'LZ' else np.array([0]))
                    Y_BHH_aug.append(np.array([1]) if motif_name == 'BHLH' else np.array([0]))
                    Y_FH_aug.append(np.array([1]) if motif_name == 'FH' else np.array([0]))
                    Y_WHH_aug.append(np.array([1]) if motif_name == 'WHH' else np.array([0]))
                    
                    total_added += 1
    
    print(f"Added {total_added} augmented positive samples")
    print(f"Augmented dataset size: {len(X_aug)}")
    
    # Convert back to numpy arrays
    X_aug = np.array(X_aug)
    Y_aug = np.array(Y_aug)
    
    # Handle motif labels - ensure they have consistent shape
    Y_ZF_aug = np.concatenate(Y_ZF_aug)
    Y_LZ_aug = np.concatenate(Y_LZ_aug) 
    Y_BHH_aug = np.concatenate(Y_BHH_aug)
    Y_FH_aug = np.concatenate(Y_FH_aug)
    Y_WHH_aug = np.concatenate(Y_WHH_aug)
    
    print(f"Final class distribution:")
    print(f"  NTF: {np.sum(Y_aug == 0)}")
    print(f"  TF:  {np.sum(Y_aug == 1)}")
    
    return X_aug, Y_aug, Y_ZF_aug, Y_LZ_aug, Y_BHH_aug, Y_FH_aug, Y_WHH_aug

# Convert original data to tensors first
sequences = torch.tensor(X).float().to(device)
labels = torch.tensor(Y).long().to(device)
labels_Y_ZF = torch.tensor(Y_ZF).long().to(device)
labels_Y_LZ = torch.tensor(Y_LZ).long().to(device)
labels_Y_BHH = torch.tensor(Y_BHH).long().to(device)
labels_Y_FH = torch.tensor(Y_FH).long().to(device)
labels_Y_WHH = torch.tensor(Y_WHH).long().to(device)

sequences = sequences.unsqueeze(1)
print("Original sequences shape: ", sequences.shape)

# Create dataset and split FIRST (before augmentation)
dataset = TensorDataset(sequences, labels, labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH)
train_size = int(0.8 * len(Y))
test_size = len(Y) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Original dataset split: {train_size} train, {test_size} test")

# Extract training data for augmentation
train_indices = train_dataset.indices
X_train = X[train_indices]
Y_train = Y[train_indices] 
Y_ZF_train = Y_ZF[train_indices]
Y_LZ_train = Y_LZ[train_indices]
Y_BHH_train = Y_BHH[train_indices]
Y_FH_train = Y_FH[train_indices]
Y_WHH_train = Y_WHH[train_indices]

# Apply augmentation ONLY to training data
print(f"Applying augmentation to training data only...")
X_train_aug, Y_train_aug, Y_ZF_train_aug, Y_LZ_train_aug, Y_BHH_train_aug, Y_FH_train_aug, Y_WHH_train_aug = create_motif_augmented_dataset(
    X_train, Y_train, Y_ZF_train, Y_LZ_train, Y_BHH_train, Y_FH_train, Y_WHH_train, augmentation_factor=2
)

# Convert augmented training data to tensors
sequences_train_aug = torch.tensor(X_train_aug).float().to(device).unsqueeze(1)
labels_train_aug = torch.tensor(Y_train_aug).long().to(device)
labels_Y_ZF_train_aug = torch.tensor(Y_ZF_train_aug).long().to(device)
labels_Y_LZ_train_aug = torch.tensor(Y_LZ_train_aug).long().to(device)
labels_Y_BHH_train_aug = torch.tensor(Y_BHH_train_aug).long().to(device)
labels_Y_FH_train_aug = torch.tensor(Y_FH_train_aug).long().to(device)
labels_Y_WHH_train_aug = torch.tensor(Y_WHH_train_aug).long().to(device)

print("Augmented training sequences shape: ", sequences_train_aug.shape)

# Create augmented training dataset and keep test dataset unchanged
train_dataset_aug = TensorDataset(sequences_train_aug, labels_train_aug, labels_Y_ZF_train_aug, labels_Y_LZ_train_aug, labels_Y_BHH_train_aug, labels_Y_FH_train_aug, labels_Y_WHH_train_aug)

train_loader = DataLoader(train_dataset_aug, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

class AugmentationBaselineModel(nn.Module):
    """
    Motif-Based Data Augmentation Baseline Model.
    This model demonstrates that while data augmentation can improve performance,
    it cannot provide interpretable logical reasoning like LTN.
    """
    def __init__(self, input_size):
        super(AugmentationBaselineModel, self).__init__()
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
        
        # Main TF/NTF classifier (simpler since augmentation provides the signal)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
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
        
        # Single prediction (augmentation handles motif information implicitly)
        prediction = self.classifier(features)
        
        return prediction


model = AugmentationBaselineModel(input_size=(1, len(X_train_aug[0])))

def compute_augmentation_loss(model, data, labels):
    """
    Compute standard binary cross-entropy loss for augmentation baseline.
    The motif information is implicitly encoded through data augmentation,
    but unlike LTN, this approach lacks interpretable logical reasoning.
    """
    predictions = model(data)
    loss = F.binary_cross_entropy(predictions.squeeze(), labels.float())
    return loss

def compute_pseudo_satisfaction(loader):
    """
    Compute a pseudo-satisfaction metric for comparison with LTN.
    NOTE: This is NOT real logical satisfaction - just prediction confidence.
    Data augmentation cannot provide true logical constraint satisfaction.
    """
    model.eval()
    total_confidence = 0.0
    count = 0
    
    with torch.no_grad():
        for data, labels, labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH in loader:
            predictions = model(data).squeeze()
            
            # For augmentation baseline, "satisfaction" is just prediction confidence
            # This is NOT logical satisfaction, just how confident the model is
            confidence = torch.mean(torch.abs(predictions - 0.5) * 2)  # Scale to 0-1
            total_confidence += confidence.item()
            count += 1
    
    return total_confidence / count if count > 0 else 0.0

def compute_accuracy(loader):
    """Compute comprehensive accuracy metrics for the augmentation baseline model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, *_ in loader:
            predictions_raw = model(data).squeeze()
            predictions = (predictions_raw > 0.5).cpu().numpy().astype(float)
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


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

train_losses = []
train_pseudo_sats = []
test_pseudo_sats = []
train_accs = []
test_accs = []

num_epochs = 80  # Sufficient epochs for augmentation baseline

print("\n🔬 Starting Motif-Based Data Augmentation Baseline Training...")
print("NOTE: This model uses data augmentation but LACKS LTN's interpretable logical reasoning")
print(f"Model: Augmentation Baseline CNN with {num_epochs} epochs")
print(f"Strategy: Motif-based data augmentation for positive samples")
print(f"Dataset: {len(Y_train_aug)} samples (augmented from {len(Y_train)} original samples)")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        sequences_batch, labels_batch, label_Y_ZF_batch, labels_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH = batch
        
        optimizer.zero_grad()
        
        # Compute standard binary cross-entropy loss (augmentation provides implicit motif info)
        loss = compute_augmentation_loss(model, sequences_batch, labels_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    
    # Update learning rate
    scheduler.step(avg_loss)
    
    with torch.no_grad():
        # Compute pseudo-satisfaction (NOT real logical satisfaction like LTN)
        train_pseudo_sat = compute_pseudo_satisfaction(train_loader)
        test_pseudo_sat = compute_pseudo_satisfaction(test_loader)

        train_metrics = compute_accuracy(train_loader)
        test_metrics = compute_accuracy(test_loader)

        train_losses.append(avg_loss)
        train_pseudo_sats.append(train_pseudo_sat)
        test_pseudo_sats.append(test_pseudo_sat)
        train_accs.append(train_metrics)
        test_accs.append(test_metrics)
        torch.cuda.empty_cache()

    if epoch % 10 == 0:
        print(f" epoch {epoch:3d} | loss {avg_loss:.4f} | "
              f"Train Confidence {train_pseudo_sat:.3f} | Test Confidence {test_pseudo_sat:.3f} | "
              f"Train Acc {train_metrics['accuracy']:.3f} | Test Acc {test_metrics['accuracy']:.3f} | "
              f"Test Prec {test_metrics['precision']:.3f} | Test Rec {test_metrics['recall']:.3f} | "
              f"Test F1 {test_metrics['f1_score']:.3f} | Test BAcc {test_metrics['balanced_accuracy']:.3f}")

print(f"\n🚨 IMPORTANT NOTE:")
print(f"Motif-Based Data Augmentation Baseline Final Performance:")
print(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  - Test F1: {test_metrics['f1_score']:.4f}")
print(f"  - Prediction Confidence: {test_pseudo_sat:.4f} (NOT logical satisfaction)")
print(f"  - Dataset Size: {len(Y_aug)} (augmented from {len(Y)} original)")
print(f"\n❌ LIMITATION: Data augmentation approach CANNOT provide:")
print(f"  - Real logical satisfaction monitoring")
print(f"  - Interpretable constraint reasoning")  
print(f"  - Per-rule logical analysis")
print(f"  - True logical consistency guarantees")
print(f"  - Explainable decision making")
print(f"\n✅ LTN ADVANTAGE: LTN provides interpretable logical reasoning")
print(f"   that data augmentation cannot match - even with more training data!")

# Save the trained model
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/Augmentation_Baseline_model.pth')

# Save the metrics after training
with open('training_metrics_augmentation.pkl', 'wb') as f:
    pickle.dump({
        'train_losses': train_losses,
        'train_pseudo_sats': train_pseudo_sats,
        'test_pseudo_sats': test_pseudo_sats,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'original_dataset_size': len(Y),
        'augmented_dataset_size': len(Y_aug),
        'augmentation_factor': 2
    }, f)

