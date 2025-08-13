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
import torch.optim as optim
from tqdm import tqdm
from deeptfactor.tf_model import DeepTFactor


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_protein_sequence(fasta_file, protein_id):
    for record in SeqIO.parse(fasta_file, "fasta"):
        #print("record.id:", record.id)
        #print("protein_id:", protein_id)
        if record.id == protein_id:
            return str(record.seq)
    return None

def extract_protein_id(file_path):
    # Extract the filename from the path
    filename = os.path.basename(file_path)
    # Use regex to extract the string between "sp|" and the next "|"
    #match = re.search(r'sp\|([^|]+)\|', filename)
    #if match:
    #    return match.group(1)
    return filename[:-3]

def has_zinc_finger(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    c2h2_pattern = r'C.{2}C.{2,4}C.{12}H.{3,5}H' #r'C.{2,4}C.{3}[FYW].{5}[LIVMFY].{12}H.{3,5}H'
    c4_pattern = r'C.{2}C.{13}C.{2}C'
    c6_pattern = r'C.{2}C.{6}C.{5,6}C.{2}C.{6}C' #r'C.{2}C.{6}C.{6}C.{2}C.{6}C'
    treble_clef_pattern = r'C.{2}C.{9,14}C.{2}C'
    ribbon_pattern = r'C.{2}C.{7,12}C.{2}C'
    gag_knuckle_pattern = r'C.{2}C.{4}H.{4}C'

    limbo_pattern = r'C.{2}C.{16,23}H.{2}H'
    pdz_pattern = r'.{2}C.{2}C.{12}H.{3}H'
    ranbp2_pattern = r'C.{2}C.{9}C.{2}C'
    taz2_pattern = r'C.{2}C.{9}C.{2}C'

    c2h2_match = re.search(c2h2_pattern, sequence)
    c4_match = re.search(c4_pattern, sequence)
    c6_match = re.search(c6_pattern, sequence)
    treble_clef_match = re.search(treble_clef_pattern, sequence)
    ribbon_match = re.search(ribbon_pattern, sequence)
    gag_knuckle_match = re.search(gag_knuckle_pattern, sequence)
    limbo_match = re.search(limbo_pattern, sequence)
    pdz_clef_match = re.search(pdz_pattern, sequence)
    ranbp2_match = re.search(ranbp2_pattern, sequence)
    taz2_match = re.search(taz2_pattern, sequence)

    return any([c2h2_match, c4_match, c6_match, treble_clef_match, ribbon_match, gag_knuckle_match, limbo_match, pdz_clef_match, \
                ranbp2_match, taz2_match])

def has_leucine_zipper(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    leucine_zipper_pattern = r'(L[A-Z]{6}){4,}L'
    return re.search(leucine_zipper_pattern, sequence) is not None

def has_helix_turn_helix(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    helix_turn_helix_pattern = r'[A-Z]{3,4}H[A-Z]{4,5}H[A-Z]{3,4}'#r'([DEKR].{20,30}){2}'
    return re.search(helix_turn_helix_pattern, sequence) is not None

def has_basic_helix_loop_helix(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    #bHLH_pattern =  r'R[A-Z]{5}R[A-Z]{2}R[A-Z]{5}R[A-Z]{2}R[A-Z]{2,5}R' #r'([A-Z]{1,2}E.{4}R.{3,6}R)'
    bHLH_pattern = r'.{2}H.{5}H.{5}H'
    return re.search(bHLH_pattern, sequence) is not None

def has_homeobox(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    #homeobox_pattern = r'RTA[YF]SAR[FY][LIVM]E[A-Z]{10}R[KQ][A-Z]{10}[WF]N[A-Z]{10}RR[A-Z]{3}W[A-Z]{1,3}K'
    #homeobox_pattern = r'([A-Z]{1,4}N[A-Z]{2,4}R[A-Z]{0,3}H)'
    homeobox_pattern = r'.{2}N.{2}S.{2}K.{2}Q.{2}R'
    return re.search(homeobox_pattern, sequence) is not None

def has_forkhead(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    #forkhead_pattern = r'RY[A-Z]{3}Y[A-Z]{4}R[A-Z]{2}K[A-Z]{1,2}W[A-Z]{2,3}[RKH][A-Z]{2,4}R[A-Z]{3}Y[A-Z]{4,6}K' #r'([A-Z]{1,2}R[A-Z]{3}W[A-Z]{4,6}[RK][A-Z]{2,4}R[A-Z]{2,3}W)'
    forkhead_pattern = r'.{2}H.{3}H.{3}H'
    return re.search(forkhead_pattern, sequence) is not None

def has_pou(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    pou_pattern = r'([A-Z]{2}W[A-Z]{2,4}Y[A-Z]{3}R[A-Z]{2,4}[RK]W[A-Z]{2,4}K)'
    return re.search(pou_pattern, sequence) is not None

def has_nuclear_receptor(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    #nuclear_receptor_pattern =  r'C[A-Z]{4}C[A-Z]{4}C[A-Z]{17}C' #r'([A-Z]{4}C[A-Z]{3,6}C[A-Z]{3,5}C[A-Z]{2,5}C)'
    nuclear_receptor_pattern = r'.{2}C.{2}C.{13}C.{2}C'
    return re.search(nuclear_receptor_pattern, sequence) is not None

def has_t_box(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    #t_box_pattern = r'([A-Z]{1,3}[LIVMFY][A-Z]{3,6}[FYW][A-Z]{3,6}L[A-Z]{2,4}[FYW][A-Z]{1,4})'
    t_box_pattern = r'.{2}G.{2}G.{2}G'
    return re.search(t_box_pattern, sequence) is not None

def has_winged_helix_turn_helix(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    winged_helix_turn_helix_pattern = r'([A-Z]{1,3}G[A-Z]{2,6}Y[A-Z]{1,5}G[A-Z]{1,3}K[A-Z]{2,5}R[A-Z]{2,4}[DEKR])'
    return re.search(winged_helix_turn_helix_pattern, sequence) is not None

fasta_tf = 'datasets/tf.fasta'
fasta_ntf = 'datasets/ntf.fasta'

path_tf = 'datasets/ESM_avg_L33_tf/*.*'
files_tf = glob.glob(path_tf)
path_ntf = 'datasets/ESM_avg_L33_ntf/*.*'
files_ntf = glob.glob(path_ntf)
X = []
Y = []
TF_num = 0
NTF_num = 0
Y_ZF = []
Y_LZ = []
Y_HH = []
Y_BHH = []
Y_HB = []
Y_FH = []
Y_POU = []
Y_NR = []
Y_TB = []
Y_WHH = []

for name in files_tf:
    #if TF_num < 5000:
    with open(name) as f:
        seq = get_protein_sequence(fasta_tf, extract_protein_id(name))
        if has_zinc_finger(seq) or has_leucine_zipper(seq) or has_basic_helix_loop_helix(
                seq) or has_forkhead(seq) or has_winged_helix_turn_helix(seq): # has_pou(seq) or has_homeobox(seq) or
            embedding_tf = torch.load(name)
            embedding_tf = embedding_tf['mean_representations'][33].numpy()
            X.append(embedding_tf)
            Y.append(1)
            #print("name: ", extract_protein_id(name))

            Y_ZF.append([1 if has_zinc_finger(seq) else 0])
            Y_LZ.append([1 if has_leucine_zipper(seq) else 0])
            #Y_HH.append([1 if has_helix_turn_helix(seq) else 0])
            Y_BHH.append([1 if has_basic_helix_loop_helix(seq) else 0])
            #Y_HB.append([1 if has_homeobox(seq) else 0])
            Y_FH.append([1 if has_forkhead(seq) else 0])
            #Y_POU.append([1 if has_pou(seq) else 0])
            #Y_NR.append([1 if has_nuclear_receptor(seq) else 0])
            #Y_TB.append([1 if has_t_box(seq) else 0])
            Y_WHH.append([1 if has_winged_helix_turn_helix(seq) else 0])
        else:
            continue
        #TF_num += 1
    #else:
    #    break

for name in files_ntf:
    if NTF_num<5134:
        with open(name) as f:
            seq = get_protein_sequence(fasta_ntf, extract_protein_id(name))
            if has_zinc_finger(seq) or has_leucine_zipper(seq) or has_basic_helix_loop_helix(
                    seq) or has_homeobox(seq) or has_forkhead(seq) or has_pou(seq) or has_winged_helix_turn_helix(seq):
                continue
            else:
                embedding_ntf = torch.load(name)
                embedding_ntf = embedding_ntf['mean_representations'][33].numpy()
                X.append(embedding_ntf)
                Y.append(0)
                Y_ZF.append([0])
                Y_LZ.append([0])
                Y_HH.append([0])
                Y_BHH.append([0])
                Y_HB.append([0])
                Y_FH.append([0])
                Y_POU.append([0])
                Y_NR.append([0])
                Y_TB.append([0])
                Y_WHH.append([0])
        NTF_num+=1
    else:
        break

df = pd.read_csv('datasets/trainingmodellarge_0620.csv')
df.rename(columns={"Uniprot/Description": "description", "Fasta Seq": "seq", "Identification (output)": "output"}, inplace=True)
path_0620 = 'datasets/ESM_avg_L33/*.*'
files_0620 = glob.glob(path_0620)
X_0620 = []
Y_0620 = []
"""
for name in files_0620:
    with open(name) as f:
        seq = df.loc[df['description'].str.contains(name[8:-3])]['seq'].values[0]
        if 'NTF' in name:
            if has_zinc_finger(seq) or has_leucine_zipper(seq) or has_basic_helix_loop_helix(
                    seq) or has_homeobox(seq) or has_forkhead(seq) or has_pou(seq) or has_winged_helix_turn_helix(seq):
                continue
            else:
                embedding_0620 = torch.load(name)
                embedding_0620 = embedding_0620['mean_representations'][33].numpy()
                X.append(embedding_0620)
                Y.append(0)
                Y_ZF.append([0])
                Y_LZ.append([0])
                Y_HH.append([0])
                Y_BHH.append([0])
                Y_HB.append([0])
                Y_FH.append([0])
                Y_POU.append([0])
                Y_NR.append([0])
                Y_TB.append([0])
                Y_WHH.append([0])
                NTF_num += 1
        elif has_zinc_finger(seq) or has_leucine_zipper(seq) or has_basic_helix_loop_helix(
                seq) or has_homeobox(seq) or has_forkhead(seq) or has_pou(seq) or has_winged_helix_turn_helix(seq):
            embedding_0620 = torch.load(name)
            embedding_0620 = embedding_0620['mean_representations'][33].numpy()
            X.append(embedding_0620)
            Y.append(1)
            Y_ZF.append([1 if has_zinc_finger(seq) else 0])
            Y_LZ.append([1 if has_leucine_zipper(seq) else 0])
            Y_HH.append([1 if has_helix_turn_helix(seq) else 0])
            Y_BHH.append([1 if has_basic_helix_loop_helix(seq) else 0])
            Y_HB.append([1 if has_homeobox(seq) else 0])
            Y_FH.append([1 if has_forkhead(seq) else 0])
            Y_POU.append([1 if has_pou(seq) else 0])
            Y_NR.append([1 if has_nuclear_receptor(seq) else 0])
            Y_TB.append([1 if has_t_box(seq) else 0])
            Y_WHH.append([1 if has_winged_helix_turn_helix(seq) else 0])
        else:
            continue
"""
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
dataset = TensorDataset(sequences, labels) #
train_size = int(0.8 * len(Y))
test_size = len(Y) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)


class TFModel(nn.Module):
    def __init__(self, input_size):
        super(TFModel, self).__init__()
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

# Define the training function
# Modify train function to return accuracy
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1).float()).sum().item()

    accuracy = correct / total
    return running_loss / len(train_loader), accuracy


# Define the evaluation function
'''
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()

    accuracy = correct / total
    return running_loss / len(test_loader), accuracy
'''

from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total

    # Convert lists to numpy arrays for metric calculation
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()

    # Calculate additional metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

    return {
        'loss': running_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    }


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
input_size = (1, len(X[0]))  # Example input size, adjust accordingly
model = TFModel(input_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Training loop
num_epochs = 100
best_accuracy = 0.0

# Training loop with added tracking of accuracy and loss
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    #val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)

    metrics = evaluate(model, test_loader, criterion, device)
    test_loss = metrics['loss']
    test_acc = metrics['accuracy']
    test_pre = metrics['precision']
    test_rec = metrics['recall']
    test_f1 = metrics['f1_score']
    test_bacc = metrics['balanced_accuracy']

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(test_loss)
    val_accuracies.append(test_acc)

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f} ")
    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Precision: {test_pre:.4f}, Recall: {test_rec:.4f}, F1 Score: {test_f1:.4f}, Balanced Accuracy: {test_bacc:.4f}")

import matplotlib.pyplot as plt

# Plotting the loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()




