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
from Bio import SeqIO
import pathlib
import torch.optim as optim
from tqdm import tqdm
from deeptfactor.tf_model import DeepTFactor

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to retrieve protein sequences based on a given protein ID
def get_protein_sequence(fasta_file, protein_id):
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id == protein_id:
            return str(record.seq)
    return None


# Function to extract protein ID from the filename
def extract_protein_id(file_path):
    filename = os.path.basename(file_path)
    return filename[:-3]  # Assuming the file extension is .pt (3 characters)


# Various pattern matching functions for different TF-related domains
def has_zinc_finger(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')

    c2h2_pattern = r'C.{2}C.{2,4}C.{12}H.{3,5}H'
    c4_pattern = r'C.{2}C.{13}C.{2}C'
    c6_pattern = r'C.{2}C.{6}C.{5,6}C.{2}C.{6}C'
    treble_clef_pattern = r'C.{2}C.{9,14}C.{2}C'
    ribbon_pattern = r'C.{2}C.{7,12}C.{2}C'
    gag_knuckle_pattern = r'C.{2}C.{4}H.{4}C'
    limbo_pattern = r'C.{2}C.{16,23}H.{2}H'
    pdz_pattern = r'.{2}C.{2}C.{12}H.{3}H'
    ranbp2_pattern = r'C.{2}C.{9}C.{2}C'
    taz2_pattern = r'C.{2}C.{9}C.{2}C'

    patterns = [c2h2_pattern, c4_pattern, c6_pattern, treble_clef_pattern, ribbon_pattern, gag_knuckle_pattern,
                limbo_pattern, pdz_pattern, ranbp2_pattern, taz2_pattern]

    return any(re.search(pattern, sequence) for pattern in patterns)


# Other motif detection functions
def has_leucine_zipper(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    leucine_zipper_pattern = r'(L[A-Z]{6}){4,}L'
    return re.search(leucine_zipper_pattern, sequence) is not None


def has_helix_turn_helix(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    helix_turn_helix_pattern = r'[A-Z]{3,4}H[A-Z]{4,5}H[A-Z]{3,4}'
    return re.search(helix_turn_helix_pattern, sequence) is not None


def has_basic_helix_loop_helix(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    bHLH_pattern = r'.{2}H.{5}H.{5}H'
    return re.search(bHLH_pattern, sequence) is not None


def has_homeobox(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    homeobox_pattern = r'.{2}N.{2}S.{2}K.{2}Q.{2}R'
    return re.search(homeobox_pattern, sequence) is not None


def has_forkhead(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
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
    nuclear_receptor_pattern = r'.{2}C.{2}C.{13}C.{2}C'
    return re.search(nuclear_receptor_pattern, sequence) is not None


def has_t_box(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    t_box_pattern = r'.{2}G.{2}G.{2}G'
    return re.search(t_box_pattern, sequence) is not None


def has_winged_helix_turn_helix(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    winged_helix_turn_helix_pattern = r'([A-Z]{1,3}G[A-Z]{2,6}Y[A-Z]{1,5}G[A-Z]{1,3}K[A-Z]{2,5}R[A-Z]{2,4}[DEKR])'
    return re.search(winged_helix_turn_helix_pattern, sequence) is not None


# File paths
fasta_tf = 'datasets/tf.fasta'
fasta_ntf = 'datasets/ntf.fasta'
path_tf = 'datasets/ESM_avg_L33_tf/*.*'
path_ntf = 'datasets/ESM_avg_L33_ntf/*.*'

files_tf = glob.glob(path_tf)
files_ntf = glob.glob(path_ntf)

X = []
Y = []
Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR, Y_TB, Y_WHH = ([] for _ in range(10))

# Load TF files and process
for name in files_tf:
    with open(name) as f:
        seq = get_protein_sequence(fasta_tf, extract_protein_id(name))
        if has_zinc_finger(seq) or has_leucine_zipper(seq) or has_basic_helix_loop_helix(seq) or has_forkhead(
                seq) or has_winged_helix_turn_helix(seq):
            embedding_tf = torch.load(name)
            embedding_tf = embedding_tf['mean_representations'][33].numpy()
            X.append(embedding_tf)
            Y.append(1)

            Y_ZF.append([1 if has_zinc_finger(seq) else 0])
            Y_LZ.append([1 if has_leucine_zipper(seq) else 0])
            Y_BHH.append([1 if has_basic_helix_loop_helix(seq) else 0])
            Y_FH.append([1 if has_forkhead(seq) else 0])
            Y_WHH.append([1 if has_winged_helix_turn_helix(seq) else 0])

# Load non-TF files and process
for name in files_ntf:
    with open(name) as f:
        seq = get_protein_sequence(fasta_ntf, extract_protein_id(name))
        if has_zinc_finger(seq) or has_leucine_zipper(seq) or has_basic_helix_loop_helix(seq) or has_homeobox(
                seq) or has_forkhead(seq) or has_pou(seq) or has_winged_helix_turn_helix(seq):
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

# Convert lists to numpy arrays for further processing
X = np.array(X)
Y = np.array(Y)
Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR, Y_TB, Y_WHH = map(np.array,
                                                                    [Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR,
                                                                     Y_TB, Y_WHH])

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

# Prepare dataset for model
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
dataset = TensorDataset(sequences, labels)
train_size = int(0.8 * len(Y))
test_size = len(Y) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)


# Define the model
def create_model(input_shape, num_classes):
    model = DeepTFactor(out_features=[1])
    return model


model = create_model((2000, 22, 1), 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.to(device)
model.train()

for epoch in range(10):  # Number of epochs can be adjusted
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/10] Loss: {running_loss / len(train_loader):.4f}")

# Evaluation on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1).float()).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
