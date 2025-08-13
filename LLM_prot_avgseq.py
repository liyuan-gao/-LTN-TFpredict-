import pathlib
import torch
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

# Use full precision if running on CPU
if device.type == 'cpu':
    model.float()
else:
    model.half()

def preprocess_sequence(sequence):
    sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = " ".join(list(sequence))
    return sequence

def extract_embeddings(fasta_file, output_dir, seq_length=1000):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(fasta_file, 'r') as f:
        sequences = []
        labels = []
        for line in f:
            if line.startswith('>'):
                labels.append(line.strip())
            else:
                sequences.append(line.strip())

    # Preprocess sequences
    sequences = [preprocess_sequence(seq) for seq in sequences]
    sequences = ["<AA2fold> " + seq if seq.isupper() else "<fold2AA> " + seq for seq in sequences]

    # Tokenize sequences
    tokenized_inputs = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest", return_tensors='pt').to(device)

    # Generate embeddings
    with torch.no_grad():
        embedding_repr = model(
            tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask
        )

    # Extract and save embeddings
    for i, label in enumerate(labels):
        entry_id = label.split()[0]
        filename = output_dir / f"{entry_id}.pt"
        emb = embedding_repr.last_hidden_state[i, 1:len(sequences[i].split()) + 1]  # Adjust for tokenization
        emb_per_protein = emb.mean(dim=0)  # Average across sequence to get per-protein embedding

        result = {"entry_id": entry_id, "representation": emb_per_protein.clone().cpu()}
        torch.save(result, filename)

# Example usage
fasta_file = pathlib.Path('datasets/trainingmodellarge_0620.fasta')
output_dir = pathlib.Path('datasets/prostT5_avg')
extract_embeddings(fasta_file, output_dir)
