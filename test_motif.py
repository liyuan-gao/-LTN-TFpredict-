import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
import numpy as np
import torch
import glob
import tensorflow as tf
from keras import backend as K
import re


def has_zinc_finger(sequence):
    # Assuming 'sequence' is a bytes-like object, we first decode it to a string
    if isinstance(sequence, bytes):
        sequence = sequence.decode('utf-8')  # You can specify the encoding if it's not utf-8

    # Zinc fingers can be Cys2His2 (C2H2) type, among others.
    # This is a simplified pattern that might represent a zinc finger motif.
    zinc_finger_pattern = 'C.{2,4}C.{12}H.{3,5}H'
    return re.search(zinc_finger_pattern, sequence) is not None

def has_leucine_zipper(sequence):
    # Leucine zippers have leucines every seventh position in a dimerization domain.
    # This is a highly simplified pattern and does not capture the true complexity.
    leucine_zipper_pattern = '(L.{6}){4,6}L'
    return re.search(leucine_zipper_pattern, sequence) is not None

def has_helix_turn_helix(sequence):
    # Helix-turn-helix motif involves two alpha helices separated by a turn.
    # Again, this is a highly simplified representation.
    helix_turn_helix_pattern = '([DEKR].{20,30}){2}'
    return re.search(helix_turn_helix_pattern, sequence) is not None


# load test data
path_0620 = '/home/liyuan/Project/DNA sequence/CCM/Fasta/trainingmodellarge_0620_L6_full_3000/*.*'  #max_length=1022, 320D, trainingmodellarge_0620_full_L6
files_0620 = glob.glob(path_0620)
# get file names
file_names = [os.path.basename(file) for file in files_0620]
sequence = []

# Read the CSV file
df = pd.read_csv('/home/liyuan/Project/DNA sequence/CCM/train_data/trainingmodellarge_0620.csv')
df.rename(columns={"Uniprot/Description": "description", "Fasta Seq": "seq", "Identification (output)": "output"}, inplace=True)
root_path = "/home/liyuan/Project/DNA sequence/CCM/Fasta/trainingmodellarge_0620_L6_full_3000/"
for name in file_names:
    # find rows that contain the string
    pos = df.loc[df['description'].str.contains(name[8:-3])]
    seq = pos['seq'].values
    #print(seq)
    print(has_zinc_finger(str(seq)))

    sequence.append(seq)


print("sequence:", len(sequence))

#sequence = sequence[0]
#print(sequence)

