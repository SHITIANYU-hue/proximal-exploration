import math
import numpy as np
from typing import Dict

GLOBAL_AA = list("ACDEFGHIKLMNPQRSTVWY")
ONEHOT = {'J': np.zeros(20)}
for aa, vec in zip(GLOBAL_AA, np.eye(20)):
    ONEHOT[aa] = vec

def process_cdr3(cdr3_seq):
    padding_len = 40 - len(cdr3_seq)
    if padding_len > 0:
        r = math.floor(padding_len / 2)
        l = math.ceil(padding_len / 2)
        aa = np.array([ONEHOT[c] for c in ('J' * l) + cdr3_seq + ('J' * r)]).T
        ranges = (40 - l - 20, r)
    else:
        truncate_len = - padding_len
        r = math.floor(truncate_len / 2)
        l = math.ceil(truncate_len / 2)
        aa = np.array([ONEHOT[c] for c in cdr3_seq[l:-r]]).T
        ranges = (0, 0)
    return aa, ranges
    

