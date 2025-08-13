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
import ltn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from Bio import SeqIO
import pathlib
import pickle
from fast_dataset_loader import load_dataset_fast
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

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

class TFModel_CNN(nn.Module):
    def __init__(self, input_size):
        super(TFModel_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size[0], out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * (input_size[1] // 4), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class TFModel_BiLSTM(nn.Module):
    def __init__(self, input_size):
        super(TFModel_BiLSTM, self).__init__()
        C = input_size[0]               # expect x: (B, C=1, T)

        # 1) Per-timestep feature expansion (learned embedding for each position)
        self.proj = nn.Conv1d(in_channels=C, out_channels=64, kernel_size=1, bias=False)

        # 2) BiLSTM over expanded features
        self.hidden_size = 256
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,       # (B, T, F)
            bidirectional=True,
            dropout=0.2 if self.num_layers > 1 else 0.0
        )

        # 3) Head (kept the same output pathway you used)
        # We’ll use mean+max pooling over time, so in_features = 2*hidden * 2
        self.fc1 = nn.Linear(self.hidden_size * 2 * 2, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

        # optional: keep your original device move if you rely on a global `device`
        try:
            self.to(device)
        except NameError:
            pass

    def forward(self, x):
        # Ensure model is on same device as input; refresh RNN flat weights when moved
        if next(self.parameters()).device != x.device:
            self.to(x.device)
            self.lstm.flatten_parameters()

        # x: (B, C=1, T)
        x = self.proj(x)               # (B, 64, T)
        x = x.transpose(1, 2)          # -> (B, T, 64) for LSTM

        out, _ = self.lstm(x)          # out: (B, T, 2*H)

        # 4) Sequence pooling (richer than only last states)
        mean_pool = out.mean(dim=1)                    # (B, 2*H)
        max_pool = out.max(dim=1).values               # (B, 2*H)
        feat = torch.cat([mean_pool, max_pool], dim=1) # (B, 4*H)

        z = F.relu(self.fc1(feat))
        z = self.dropout(z)
        z = torch.sigmoid(self.fc2(z))  # keep sigmoid to match your pipeline
        return z

class TFModel_CNN_BiLSTM(nn.Module):
    def __init__(self, input_size):
        super(TFModel_CNN_BiLSTM, self).__init__()
        C = input_size[0]               # expect x: (B, C=1, T)

        # --- CNN feature extractor ---
        # Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool
        self.conv1 = nn.Conv1d(in_channels=C, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- BiLSTM over CNN features ---
        # LSTM input_size must match conv out_channels (128)
        self.hidden_size = 256
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,           # (B, T, F)
            bidirectional=True,
            dropout=0.2 if self.num_layers > 1 else 0.0
        )

        # --- Head (kept identical shape: 512 -> 1 with Dropout) ---
        # We'll do mean+max pooling over time from LSTM output (2*hidden each) => concat => 4*hidden
        self.fc1 = nn.Linear(self.hidden_size * 2 * 2, 512)  # (mean + max) * (bi-dir)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

        # If you rely on a global `device`, keep parity with your original class
        try:
            self.to(device)
        except NameError:
            pass

    def forward(self, x):
        # Ensure model is on the same device as input; refresh RNN flat weights if moved
        if next(self.parameters()).device != x.device:
            self.to(x.device)
            self.lstm.flatten_parameters()

        # x: (B, C, T)
        x = F.relu(self.conv1(x))       # (B, 64, T)
        x = self.pool1(x)               # (B, 64, T/2)
        x = F.relu(self.conv2(x))       # (B, 128, T/2)
        x = self.pool2(x)               # (B, 128, T/4)

        # LSTM expects (B, T, F)
        x = x.transpose(1, 2)           # (B, T/4, 128)
        out, _ = self.lstm(x)           # (B, T/4, 2*H)

        # Mean + Max pooling over time
        mean_pool = out.mean(dim=1)                 # (B, 2*H)
        max_pool = out.max(dim=1).values            # (B, 2*H)
        feat = torch.cat([mean_pool, max_pool], 1)  # (B, 4*H)

        z = F.relu(self.fc1(feat))
        z = self.dropout(z)
        z = torch.sigmoid(self.fc2(z))  # keep sigmoid to match your pipeline
        return z


P = ltn.Predicate(TFModel_CNN(input_size=(1, len(X[0]))))

Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

def compute_sat_level(loader):
    mean_sat = 0  #
    for data, labels, labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH in loader:
        sat_levels = []

        for label, ltn_var in zip([labels_Y_ZF, labels_Y_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH],
                                  ["x_ZF", "x_LZ", "x_BHH", "x_FH", "x_WHH"]):
            if (label == 1).any():
                x = ltn.Variable(ltn_var, data[label == 1].unsqueeze(1))
                sat_levels.append(Forall(x, P(x)))
            else:
                sat_levels.append(torch.tensor(1.0).to(device))

        if (labels == 1).any():
            x_TF = ltn.Variable("x_TF", data[labels == 1])
            sat_levels.append(Forall(x_TF, P(x_TF)))
        else:
            sat_levels.append(torch.tensor(1.0).to(device))

        if (labels == 0).any():
            x_NTF = ltn.Variable("x_NTF", data[labels == 0])
            sat_levels.append(Forall(x_NTF, Not(P(x_NTF))))
        else:
            sat_levels.append(torch.tensor(1.0).to(device))

        mean_sat += SatAgg(*sat_levels)
    mean_sat /= len(loader)
    return mean_sat.cpu().item()
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
    from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
    
    all_predictions = []
    all_labels = []
    
    for data, labels, *_ in loader:
        predictions = P.model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        labels = labels.cpu().numpy()
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
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

optimizer = torch.optim.Adam(P.parameters(), lr=0.0001)
train_losses = []
train_sats = []
test_sats = []
train_accs = []
test_accs = []
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader: #, labels_LZ, labels_HH, labels_Y_BHH, labels_Y_HB, labels_Y_FH, labels_Y_POU, labels_Y_NR, labels_Y_TB, labels_Y_WHH
        sequences_batch, labels_batch, label_Y_ZF_batch, labels_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH = batch
        optimizer.zero_grad()
        sat_levels = []

        for label, ltn_var in zip([label_Y_ZF_batch, labels_LZ, labels_Y_BHH, labels_Y_FH, labels_Y_WHH],
                                  ["x_ZF", "x_LZ", "x_BHH", "x_HB", "x_FH", "x_POU", "x_WHH"]):
            if (label == 1).any():
                x = ltn.Variable(ltn_var, sequences_batch[label == 1].unsqueeze(1)) #
                sat_levels.append(Forall(x, P(x)))
            else:
                sat_levels.append(torch.tensor(1.0).to(device))

        if (labels_batch == 1).any():
            x_TF = ltn.Variable("x_TF", sequences_batch[labels_batch == 1])
            sat_levels.append(Forall(x_TF, P(x_TF)))
        else:
            sat_levels.append(torch.tensor(1.0).to(device))

        if (labels_batch == 0).any():
            x_NTF = ltn.Variable("x_NTF", sequences_batch[labels_batch == 0])
            sat_levels.append(Forall(x_NTF, Not(P(x_NTF))))
        else:
            sat_levels.append(torch.tensor(1.0).to(device))

        sat_level = SatAgg(*sat_levels)
        loss = 1. - sat_level
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        torch.cuda.empty_cache()

    train_loss = total_loss / len(train_loader)
    with torch.no_grad():
        train_sat = compute_sat_level(train_loader)
        test_sat = compute_sat_level(test_loader)

        train_metrics = compute_accuracy(train_loader)
        test_metrics = compute_accuracy(test_loader)

        train_losses.append(train_loss)
        train_sats.append(train_sat)
        test_sats.append(test_sat)
        train_accs.append(train_metrics)
        test_accs.append(test_metrics)
        torch.cuda.empty_cache()

    if epoch % 2 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f | Test Prec %.3f | Test Rec %.3f | Test F1 %.3f | Test BAcc %.3f"
              % (epoch, train_loss, train_sat, test_sat, train_metrics['accuracy'], test_metrics['accuracy'], 
                 test_metrics['precision'], test_metrics['recall'], test_metrics['f1_score'], test_metrics['balanced_accuracy']))

        """
        # Extract metrics
        train_acc = train_metrics['accuracy']
        test_acc = test_metrics['accuracy']
        test_pre = test_metrics['precision']
        test_rec = test_metrics['recall']
        test_f1 = test_metrics['f1_score']
        test_bacc = test_metrics['balanced_accuracy']

        train_losses.append(train_loss)
        train_sats.append(train_sat)
        test_sats.append(test_sat)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        torch.cuda.empty_cache()

        print(" epoch %d | loss %.4f | Train Sat: labels: %.3f, ZF: %.3f, LZ: %.3f, BHH: %.3f, FH: %.3f, WHH: %.3f | "
              "Test Sat: labels: %.3f, ZF: %.3f, LZ: %.3f, BHH: %.3f, FH: %.3f, WHH: %.3f"
              % (epoch, train_loss,
                 train_sat['labels'], train_sat['labels_Y_ZF'], train_sat['labels_Y_LZ'], train_sat['labels_Y_BHH'],
                 train_sat['labels_Y_FH'], train_sat['labels_Y_WHH'],
                 test_sat['labels'], test_sat['labels_Y_ZF'], test_sat['labels_Y_LZ'], test_sat['labels_Y_BHH'],
                 test_sat['labels_Y_FH'], test_sat['labels_Y_WHH']))
        print(
            " epoch %d | loss %.4f | Train Acc %.3f | Test Acc %.3f | Test Pre %.3f | Test Rec %.3f | Test F1 %.3f | Test BAcc %.3f"
            % (epoch, train_loss, train_acc, test_acc, test_pre, test_rec, test_f1, test_bacc))
        del train_sat, test_sat, train_acc, test_acc, test_pre, test_rec, test_f1, test_bacc
        """

# Save the trained model
torch.save(P.model.state_dict(), 'model/TF-LTN_model_v5.pth')

# Save the metrics after training
with open('training_metrics.pkl', 'wb') as f:
    pickle.dump({
        'train_losses': train_losses,
        'train_sats': train_sats,
        'test_sats': test_sats,
        'train_accs': train_accs,
        'test_accs': test_accs
    }, f)

# Statistical Analysis: Paired t-tests and Wilcoxon signed-rank tests
def perform_statistical_tests(metrics_list, metric_name):
    """
    Perform paired t-tests and Wilcoxon signed-rank tests to evaluate statistical significance
    of performance improvements over training epochs.
    """
    print(f"\n📊 STATISTICAL ANALYSIS FOR {metric_name.upper()}")
    print("=" * 60)
    
    # Extract metric values from dictionaries if needed
    if isinstance(metrics_list[0], dict):
        values = [m[metric_name] for m in metrics_list]
    else:
        values = metrics_list
    
    values = np.array(values)
    n_epochs = len(values)
    
    # Split into early (first 25%) and late (last 25%) epochs for comparison
    early_period = values[:n_epochs//4]
    late_period = values[-n_epochs//4:]
    
    print(f"Early training period ({len(early_period)} epochs): Mean = {np.mean(early_period):.4f} ± {np.std(early_period):.4f}")
    print(f"Late training period ({len(late_period)} epochs): Mean = {np.mean(late_period):.4f} ± {np.std(late_period):.4f}")
    print(f"Improvement: {np.mean(late_period) - np.mean(early_period):.4f}")
    
    # Paired t-test
    try:
        t_stat, t_pvalue = ttest_rel(late_period, early_period)
        print(f"\n🔬 PAIRED T-TEST:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {t_pvalue:.6f}")
        if t_pvalue < 0.001:
            print(f"  Result: HIGHLY SIGNIFICANT improvement (p < 0.001) ***")
        elif t_pvalue < 0.01:
            print(f"  Result: VERY SIGNIFICANT improvement (p < 0.01) **")
        elif t_pvalue < 0.05:
            print(f"  Result: SIGNIFICANT improvement (p < 0.05) *")
        else:
            print(f"  Result: No significant improvement (p ≥ 0.05)")
    except Exception as e:
        print(f"  Error in paired t-test: {e}")
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = wilcoxon(late_period, early_period, alternative='greater')
        print(f"\n🔬 WILCOXON SIGNED-RANK TEST:")
        print(f"  W-statistic: {w_stat}")
        print(f"  p-value: {w_pvalue:.6f}")
        if w_pvalue < 0.001:
            print(f"  Result: HIGHLY SIGNIFICANT improvement (p < 0.001) ***")
        elif w_pvalue < 0.01:
            print(f"  Result: VERY SIGNIFICANT improvement (p < 0.01) **")
        elif w_pvalue < 0.05:
            print(f"  Result: SIGNIFICANT improvement (p < 0.05) *")
        else:
            print(f"  Result: No significant improvement (p ≥ 0.05)")
    except Exception as e:
        print(f"  Error in Wilcoxon test: {e}")
    
    # Additional analysis: trend over time
    epochs = np.arange(len(values))
    correlation_coeff, correlation_p = stats.pearsonr(epochs, values)
    print(f"\n📈 TREND ANALYSIS:")
    print(f"  Pearson correlation with epoch: r = {correlation_coeff:.4f}, p = {correlation_p:.6f}")
    if correlation_p < 0.05:
        if correlation_coeff > 0:
            print(f"  Result: SIGNIFICANT positive trend (improving over time) *")
        else:
            print(f"  Result: SIGNIFICANT negative trend (declining over time) *")
    else:
        print(f"  Result: No significant trend over time")
    
    return {
        'early_mean': np.mean(early_period),
        'late_mean': np.mean(late_period),
        'improvement': np.mean(late_period) - np.mean(early_period),
        't_stat': t_stat if 't_stat' in locals() else None,
        't_pvalue': t_pvalue if 't_pvalue' in locals() else None,
        'w_stat': w_stat if 'w_stat' in locals() else None,
        'w_pvalue': w_pvalue if 'w_pvalue' in locals() else None,
        'correlation_coeff': correlation_coeff,
        'correlation_p': correlation_p
    }

def compare_metrics_between_periods(metrics_list, metric_name, period1_name, period2_name, period1_indices, period2_indices):
    """
    Compare specific periods using statistical tests
    """
    print(f"\n🔍 COMPARING {period1_name} vs {period2_name} for {metric_name.upper()}")
    print("=" * 60)
    
    # Extract metric values
    if isinstance(metrics_list[0], dict):
        values = [m[metric_name] for m in metrics_list]
    else:
        values = metrics_list
    
    values = np.array(values)
    period1_values = values[period1_indices]
    period2_values = values[period2_indices]
    
    print(f"{period1_name}: Mean = {np.mean(period1_values):.4f} ± {np.std(period1_values):.4f}")
    print(f"{period2_name}: Mean = {np.mean(period2_values):.4f} ± {np.std(period2_values):.4f}")
    
    # Paired t-test
    t_stat, t_pvalue = ttest_rel(period2_values, period1_values)
    print(f"\nPaired t-test: t = {t_stat:.4f}, p = {t_pvalue:.6f}")
    
    # Wilcoxon signed-rank test
    w_stat, w_pvalue = wilcoxon(period2_values, period1_values, alternative='greater')
    print(f"Wilcoxon test: W = {w_stat}, p = {w_pvalue:.6f}")
    
    return {
        'period1_mean': np.mean(period1_values),
        'period2_mean': np.mean(period2_values),
        't_pvalue': t_pvalue,
        'w_pvalue': w_pvalue
    }

# Perform statistical analysis on all metrics
print("\n" + "="*80)
print("🔬 STATISTICAL SIGNIFICANCE TESTING")
print("="*80)

# Test satisfaction levels
train_sat_stats = perform_statistical_tests(train_sats, "train_satisfaction")
test_sat_stats = perform_statistical_tests(test_sats, "test_satisfaction")

# Test accuracy metrics
train_acc_stats = perform_statistical_tests(train_accs, "accuracy")
test_acc_stats = perform_statistical_tests(test_accs, "accuracy")

# Test precision, recall, F1, balanced accuracy
test_prec_stats = perform_statistical_tests(test_accs, "precision")
test_recall_stats = perform_statistical_tests(test_accs, "recall")
test_f1_stats = perform_statistical_tests(test_accs, "f1_score")
test_bacc_stats = perform_statistical_tests(test_accs, "balanced_accuracy")

# Compare specific training phases
n_epochs = len(train_losses)
initial_phase = slice(0, n_epochs//8)  # First 12.5%
middle_phase = slice(n_epochs//2 - n_epochs//8, n_epochs//2 + n_epochs//8)  # Middle 25%
final_phase = slice(-n_epochs//8, None)  # Last 12.5%

print("\n" + "="*80)
print("📊 PHASE-WISE COMPARISONS")
print("="*80)

# Compare initial vs final phases
compare_metrics_between_periods(test_accs, "accuracy", "Initial Phase", "Final Phase", 
                               range(n_epochs//8), range(n_epochs - n_epochs//8, n_epochs))

compare_metrics_between_periods(test_sats, "test_satisfaction", "Initial Phase", "Final Phase",
                               range(n_epochs//8), range(n_epochs - n_epochs//8, n_epochs))

# Summary report
print("\n" + "="*80)
print("📋 STATISTICAL SUMMARY REPORT")
print("="*80)

summary_data = {
    'Metric': ['Train Satisfaction', 'Test Satisfaction', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test Balanced Acc'],
    'Early_Mean': [train_sat_stats['early_mean'], test_sat_stats['early_mean'], 
                   train_acc_stats['early_mean'], test_prec_stats['early_mean'],
                   test_recall_stats['early_mean'], test_f1_stats['early_mean'], test_bacc_stats['early_mean']],
    'Late_Mean': [train_sat_stats['late_mean'], test_sat_stats['late_mean'],
                  train_acc_stats['late_mean'], test_prec_stats['late_mean'],
                  test_recall_stats['late_mean'], test_f1_stats['late_mean'], test_bacc_stats['late_mean']],
    'Improvement': [train_sat_stats['improvement'], test_sat_stats['improvement'],
                    train_acc_stats['improvement'], test_prec_stats['improvement'],
                    test_recall_stats['improvement'], test_f1_stats['improvement'], test_bacc_stats['improvement']],
    'T_Test_P': [train_sat_stats['t_pvalue'], test_sat_stats['t_pvalue'],
                 train_acc_stats['t_pvalue'], test_prec_stats['t_pvalue'],
                 test_recall_stats['t_pvalue'], test_f1_stats['t_pvalue'], test_bacc_stats['t_pvalue']],
    'Wilcoxon_P': [train_sat_stats['w_pvalue'], test_sat_stats['w_pvalue'],
                   train_acc_stats['w_pvalue'], test_prec_stats['w_pvalue'],
                   test_recall_stats['w_pvalue'], test_f1_stats['w_pvalue'], test_bacc_stats['w_pvalue']]
}

# Create DataFrame for better visualization
import pandas as pd
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False, float_format='%.6f'))

# Save statistical results
with open('statistical_analysis_results.pkl', 'wb') as f:
    pickle.dump({
        'train_sat_stats': train_sat_stats,
        'test_sat_stats': test_sat_stats,
        'train_acc_stats': train_acc_stats,
        'test_acc_stats': test_acc_stats,
        'test_prec_stats': test_prec_stats,
        'test_recall_stats': test_recall_stats,
        'test_f1_stats': test_f1_stats,
        'test_bacc_stats': test_bacc_stats,
        'summary_df': summary_df
    }, f)

print(f"\n💾 Statistical analysis results saved to 'statistical_analysis_results.pkl'")
print("="*80)

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