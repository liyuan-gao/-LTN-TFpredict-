import torch
import ltn
import numpy as np
import torch.nn as nn
import pandas as pd
import glob
import os
import torch.nn.functional as F

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model(model_path, input_size):
    model = TFModel(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_protein_sequence(model, sequence):
    sequence = np.array(sequence).astype(np.float32)
    sequence = torch.tensor(sequence).unsqueeze(0).unsqueeze(0).to(device)
    prediction = model(sequence).item()
    return prediction

X_test = []
path_test = 'datasets/test_avg/sp|A0A1D5NS60|ZN16L_DANRE.pt'
embedding_test = torch.load(path_test)
embedding_test = embedding_test['mean_representations'][33].numpy()
X_test.append(embedding_test)


new_protein_sequence = np.array(X_test[0])  # Load or input the new protein sequence as an array
emb_dim = len(new_protein_sequence)
sequence_length = emb_dim

# Load the trained model
P = ltn.Predicate(TFModel(input_size=(1, sequence_length)))
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

model_path = 'saved_model/TF-LTN_model_v5.pth'
input_size = (1, sequence_length)  # Replace with actual input size
model = load_model(model_path, input_size)

# Test the model with a new protein sequence
prediction = predict_protein_sequence(model, new_protein_sequence)

# Interpretation
if prediction > 0.5:
    print("The protein is predicted to be a transcription factor.")
else:
    print("The protein is not predicted to be a transcription factor.")

def evaluate_satisfaction(model, sequence, rule):
    sequence = np.array(sequence).astype(np.float32)
    sequence = torch.tensor(sequence).unsqueeze(0).unsqueeze(0).to(device)
    x = ltn.Variable("x", sequence)
    satisfaction = rule(x).value.item()
    return satisfaction

# Define the satisfaction levels for individual rules
rules = {
    'zinc_finger': lambda x: Forall(x, P(x)),
    'leucine_zipper': lambda x: Forall(x, P(x)),
    'helix_turn_helix': lambda x: Forall(x, P(x)),
    'basic_helix_loop_helix': lambda x: Forall(x, P(x)),
    'homeobox': lambda x: Forall(x, P(x)),
    'forkhead': lambda x: Forall(x, P(x)),
    'pou': lambda x: Forall(x, P(x)),
    'nuclear_receptor': lambda x: Forall(x, P(x)),
    't_box': lambda x: Forall(x, P(x)),
    'winged_helix_turn_helix': lambda x: Forall(x, P(x)),
    'x_TF': lambda x: Forall(x, P(x)),
    'x_NTF': lambda x: Forall(x, P(x))
}

# Evaluate the satisfaction levels for each rule
satisfaction_levels = {rule_name: evaluate_satisfaction(P.model, new_protein_sequence, rule) for rule_name, rule in rules.items()}

# Print the satisfaction levels
for rule_name, satisfaction in satisfaction_levels.items():
    print(f"Satisfaction level for {rule_name}: {satisfaction}")
