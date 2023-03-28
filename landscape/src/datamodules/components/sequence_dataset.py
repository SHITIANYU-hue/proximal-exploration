import torch
import pandas as pd
import numpy as np

from typing import List
from torch.utils.data import Dataset
from src.utils.sequence_feature import process_cdr3

class CDRH3DataSet(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        feature_col: str,
        index_col: str, # index witch specific one sample
        label_col: str = "label",

    ):
        self.samples = samples
        self.feature_col = feature_col
        self.label_col = label_col
        self.index_col = index_col

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        item = self.samples.iloc[idx]
        aa_featue, ranges = process_cdr3(item[self.feature_col])
        aa_featue = torch.tensor(aa_featue, dtype=torch.float)
        label = torch.tensor(item[self.label_col], dtype=torch.float)
        return {'x': aa_featue, 'y': label, 'ranges': ranges, 'index': item[self.index_col]}

class CDRH3DataPred(Dataset):
    def __init__(
        self,
        samples: List,
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        aa_featue, ranges = process_cdr3(item)
        aa_featue = torch.tensor(aa_featue, dtype=torch.float)
        return {'x': aa_featue, 'index': item, "y":0}
    
    def update_data(self, samples):
        self.samples = samples