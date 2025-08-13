# Replace the slow dataset loading section in train_LTN.py with this optimized version

import torch
import pandas as pd
import glob
import numpy as np
import re
import os
from Bio import SeqIO

def extract_protein_id(file_path):
    """Extract protein ID from file path"""
    filename = os.path.basename(file_path)
    return filename[:-3]

def load_dataset_fast():
    """Fast dataset loading with optimizations"""
    fasta_tf = 'datasets/tf.fasta'
    fasta_ntf = 'datasets/ntf.fasta'
    
    path_tf = 'datasets/ESM_avg_L33_tf/*.*'
    files_tf = glob.glob(path_tf)
    path_ntf = 'datasets/ESM_avg_L33_ntf/*.*'
    files_ntf = glob.glob(path_ntf)
    
    print(f"Processing {len(files_tf)} TF files and {len(files_ntf)} NTF files...")
    
    # Pre-load FASTA sequences into memory for faster access
    tf_sequences = {}
    ntf_sequences = {}
    
    print("Loading TF sequences...")
    for record in SeqIO.parse(fasta_tf, "fasta"):
        tf_sequences[record.id] = str(record.seq)
    
    print("Loading NTF sequences...")
    for record in SeqIO.parse(fasta_ntf, "fasta"):
        ntf_sequences[record.id] = str(record.seq)
    
    # Pre-compile regex patterns
    zinc_finger_patterns = [
        re.compile(r'C.{2}C.{2,4}C.{12}H.{3,5}H'),
        re.compile(r'C.{2}C.{13}C.{2}C'),
        re.compile(r'C.{2}C.{6}C.{5,6}C.{2}C.{6}C'),
        re.compile(r'C.{2}C.{9,14}C.{2}C'),
        re.compile(r'C.{2}C.{7,12}C.{2}C'),
        re.compile(r'C.{2}C.{4}H.{4}C'),
        re.compile(r'C.{2}C.{16,23}H.{2}H'),
        re.compile(r'.{2}C.{2}C.{12}H.{3}H'),
        re.compile(r'C.{2}C.{9}C.{2}C'),
        re.compile(r'C.{2}C.{9}C.{2}C')
    ]
    
    leucine_zipper_pattern = re.compile(r'(L[A-Z]{6}){4,}L')
    helix_turn_helix_pattern = re.compile(r'[A-Z]{3,4}H[A-Z]{4,5}H[A-Z]{3,4}')
    bhlh_pattern = re.compile(r'.{2}H.{5}H.{5}H')
    homeobox_pattern = re.compile(r'.{2}N.{2}S.{2}K.{2}Q.{2}R')
    forkhead_pattern = re.compile(r'.{2}H.{3}H.{3}H')
    pou_pattern = re.compile(r'([A-Z]{2}W[A-Z]{2,4}Y[A-Z]{3}R[A-Z]{2,4}[RK]W[A-Z]{2,4}K)')
    winged_helix_pattern = re.compile(r'([A-Z]{1,3}G[A-Z]{2,6}Y[A-Z]{1,5}G[A-Z]{1,3}K[A-Z]{2,5}R[A-Z]{2,4}[DEKR])')
    
    def has_zinc_finger_fast(sequence):
        return any(pattern.search(sequence) for pattern in zinc_finger_patterns)
    
    def has_leucine_zipper_fast(sequence):
        return leucine_zipper_pattern.search(sequence) is not None
    
    def has_helix_turn_helix_fast(sequence):
        return helix_turn_helix_pattern.search(sequence) is not None
    
    def has_basic_helix_loop_helix_fast(sequence):
        return bhlh_pattern.search(sequence) is not None
    
    def has_homeobox_fast(sequence):
        return homeobox_pattern.search(sequence) is not None
    
    def has_forkhead_fast(sequence):
        return forkhead_pattern.search(sequence) is not None
    
    def has_pou_fast(sequence):
        return pou_pattern.search(sequence) is not None
    
    def has_winged_helix_turn_helix_fast(sequence):
        return winged_helix_pattern.search(sequence) is not None
    
    X = []
    Y = []
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
    
    # Process TF files
    print("Processing TF files...")
    for i, name in enumerate(files_tf):
        if i % 100 == 0:
            print(f"TF progress: {i}/{len(files_tf)}")
            
        protein_id = extract_protein_id(name)
        seq = tf_sequences.get(protein_id)
        
        if seq is None:
            continue
            
        # Fast motif checking
        has_zf = has_zinc_finger_fast(seq)
        has_lz = has_leucine_zipper_fast(seq)
        has_hh = has_helix_turn_helix_fast(seq)
        has_bhh = has_basic_helix_loop_helix_fast(seq)
        has_hb = has_homeobox_fast(seq)
        has_fh = has_forkhead_fast(seq)
        has_pou = has_pou_fast(seq)
        has_whh = has_winged_helix_turn_helix_fast(seq)
        
        if has_zf or has_lz or has_bhh or has_hb or has_fh or has_pou or has_whh:
            try:
                embedding_tf = torch.load(name)
                embedding_tf = embedding_tf['mean_representations'][33].numpy()
                X.append(embedding_tf)
                Y.append(1)
                
                Y_ZF.append([1 if has_zf else 0])
                Y_LZ.append([1 if has_lz else 0])
                Y_HH.append([1 if has_hh else 0])
                Y_BHH.append([1 if has_bhh else 0])
                Y_HB.append([1 if has_hb else 0])
                Y_FH.append([1 if has_fh else 0])
                Y_POU.append([1 if has_pou else 0])
                Y_NR.append([0])  # Placeholder
                Y_TB.append([0])  # Placeholder
                Y_WHH.append([1 if has_whh else 0])
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
    
    # Process NTF files
    print("Processing NTF files...")
    NTF_num = 0
    for i, name in enumerate(files_ntf):
        if NTF_num >= 5134:
            break
            
        if i % 100 == 0:
            print(f"NTF progress: {i}/{len(files_ntf)}")
            
        protein_id = extract_protein_id(name)
        seq = ntf_sequences.get(protein_id)
        
        if seq is None:
            continue
            
        # Fast motif checking
        has_zf = has_zinc_finger_fast(seq)
        has_lz = has_leucine_zipper_fast(seq)
        has_hh = has_helix_turn_helix_fast(seq)
        has_bhh = has_basic_helix_loop_helix_fast(seq)
        has_hb = has_homeobox_fast(seq)
        has_fh = has_forkhead_fast(seq)
        has_pou = has_pou_fast(seq)
        has_whh = has_winged_helix_turn_helix_fast(seq)
        
        if not (has_zf or has_lz or has_bhh or has_hb or has_fh or has_pou or has_whh):
            try:
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
                NTF_num += 1
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
    
    # Process 0620 dataset
    print("Processing 0620 dataset...")
    df = pd.read_csv('datasets/trainingmodellarge_0620.csv')
    df.rename(columns={"Uniprot/Description": "description", "Fasta Seq": "seq", "Identification (output)": "output"}, inplace=True)
    
    path_0620 = 'datasets/ESM_avg_L33/*.*'
    files_0620 = glob.glob(path_0620)
    
    for i, name in enumerate(files_0620):
        if i % 100 == 0:
            print(f"0620 progress: {i}/{len(files_0620)}")
            
        if 'NTF' in name:
            continue
            
        protein_id = extract_protein_id(name)
        seq_row = df[df['description'].str.contains(protein_id, na=False)]
        
        if seq_row.empty:
            continue
            
        seq = seq_row['seq'].iloc[0]
        
        # Fast motif checking
        has_zf = has_zinc_finger_fast(seq)
        has_lz = has_leucine_zipper_fast(seq)
        has_hh = has_helix_turn_helix_fast(seq)
        has_bhh = has_basic_helix_loop_helix_fast(seq)
        has_hb = has_homeobox_fast(seq)
        has_fh = has_forkhead_fast(seq)
        has_pou = has_pou_fast(seq)
        has_whh = has_winged_helix_turn_helix_fast(seq)
        
        if has_zf or has_lz or has_bhh or has_hb or has_fh or has_pou or has_whh:
            try:
                embedding_0620 = torch.load(name)
                embedding_0620 = embedding_0620['mean_representations'][33].numpy()
                X.append(embedding_0620)
                Y.append(1)
                
                Y_ZF.append([1 if has_zf else 0])
                Y_LZ.append([1 if has_lz else 0])
                Y_HH.append([1 if has_hh else 0])
                Y_BHH.append([1 if has_bhh else 0])
                Y_HB.append([1 if has_hb else 0])
                Y_FH.append([1 if has_fh else 0])
                Y_POU.append([1 if has_pou else 0])
                Y_NR.append([0])  # Placeholder
                Y_TB.append([0])  # Placeholder
                Y_WHH.append([1 if has_whh else 0])
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
    
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
    
    print(f"Dataset loaded: {len(X)} samples")
    print(f"TF samples: {np.sum(Y == 1)}")
    print(f"NTF samples: {np.sum(Y == 0)}")
    
    return X, Y, Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR, Y_TB, Y_WHH

# Usage: Replace the slow dataset loading section in train_LTN.py with:
# X, Y, Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR, Y_TB, Y_WHH = load_dataset_fast() 