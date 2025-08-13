import pathlib
import pandas as pd
import glob
import numpy as np
import os
import re
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def has_bzip(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')
    winged_helix_turn_helix_pattern = r'.{2}L.{6}L.{6}L'
    return re.search(winged_helix_turn_helix_pattern, sequence) is not None

# Function to parse the FASTA file and return a list of sequences
def parse_fasta(fasta_path):
    with open(fasta_path, 'r') as file:
        sequences = []
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line
        # Append the last sequence if file does not end with a new header
        if sequence:
            sequences.append(sequence)
    return sequences


fasta_file = pathlib.Path('datasets/tf.fasta')
# Call the function and get the list of sequences
seqs = parse_fasta(fasta_file)

zf_num = 0
lz_num = 0
hth_num = 0
hlh_num = 0
hb_num = 0
fh_num = 0
pou_num = 0
nr_num = 0
tb_num = 0
whh_num = 0
bzip_num = 0
for seq in seqs:
    if has_zinc_finger(str(seq)) is True:
        print("zinc finger:", seq)
        zf_num+=1

    if has_leucine_zipper(str(seq)) is True:
        #print("leucine_zipper")
        lz_num += 1

    if has_helix_turn_helix(str(seq)) is True:
        #print("helix_turn_helix")
        hth_num += 1

    if has_basic_helix_loop_helix(str(seq)) is True:
        #print("basic_helix_loop_helix")
        hlh_num += 1

    if has_homeobox(str(seq)) is True:
        #print("homeobox")
        hb_num +=1

    if has_forkhead(str(seq)) is True:
        #print("forkhead")
        fh_num +=1

    if has_pou(str(seq)) is True:
        #print("pou")
        pou_num +=1

    if has_nuclear_receptor(str(seq)) is True:
        #print("nuclear_receptor")
        pou_num +=1

    if has_t_box(str(seq)) is True:
        #print("t_box")
        tb_num +=1

    if has_winged_helix_turn_helix(str(seq)) is True:
        #print("winged_helix_turn_helix")
        whh_num +=1

    if has_bzip(str(seq)) is True:
        bzip_num +=1

print("zf_num: ", zf_num)
print("lz_num: ", lz_num)
print("hth_num: ", hth_num)
print("hlh_num: ", hlh_num)
print("hb_num: ", hb_num)
print("fh_num: ", fh_num)
print("pou_num: ", pou_num)
print("nr_num: ", nr_num)
print("tb_num: ", tb_num)
print("whh_num: ", whh_num)
print("bzip_num: ", bzip_num)