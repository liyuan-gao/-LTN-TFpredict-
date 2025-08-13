import numpy as np
from fast_dataset_loader import load_dataset_fast

FIVE_KEYS = ('ZF','LZ','BHH','FH','WHH')

def load_filtered_five_motifs():
    X, Y, Y_ZF, Y_LZ, Y_HH, Y_BHH, Y_HB, Y_FH, Y_POU, Y_NR, Y_TB, Y_WHH = load_dataset_fast()
    X = np.array(X)
    Y = np.array(Y)
    Y_ZF = np.array(Y_ZF).flatten()
    Y_LZ = np.array(Y_LZ).flatten()
    Y_BHH = np.array(Y_BHH).flatten()
    Y_FH = np.array(Y_FH).flatten()
    Y_WHH = np.array(Y_WHH).flatten()
    five_any = (Y_ZF + Y_LZ + Y_BHH + Y_FH + Y_WHH) > 0
    keep = (Y == 0) | ((Y == 1) & five_any)
    X = X[keep]
    Y = Y[keep]
    motifs = {
        'ZF': Y_ZF[keep],
        'LZ': Y_LZ[keep],
        'BHH': Y_BHH[keep],
        'FH': Y_FH[keep],
        'WHH': Y_WHH[keep],
    }
    return X, Y, motifs 