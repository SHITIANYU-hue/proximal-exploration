import os
import torch
import pandas as pd
import numpy as np

from typing import Optional, Dict, Any
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datamodules.components.sequence_dataset import CDRH3DataSet
from src.utils.worker_utils import seed_worker

class CDRH3DataModule(LightningDataModule):
    def __init__(
        self,
        train_data_loc: str,
        test_data_loc: str,
        pred_data_loc: str,
        num_workers: int,
        index_col: str, # index witch specific one sample
        feature_col: str = 'cdr3',
        label_col: str = 'label',
        batch_size: int = 32,
        train_valid_rate: float = 0.1,
        pin_memory: bool = True
    ):
        super().__init__()
        self.train_data_loc = train_data_loc
        self.test_data_loc = test_data_loc
        self.pred_data_loc = pred_data_loc
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_valid_rate = train_valid_rate
        self.pin_memory = pin_memory
        self.index_col = index_col
        self.label_col = label_col
        self.feature_col = feature_col
        self.data_train = None
        self.data_test = None
        self.data_pred = None
        self.data_val = None


    def prepare_data(self,):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            data_raw = pd.read_csv(self.train_data_loc)
            data_size = data_raw.shape[0]
            if self.train_valid_rate < 1:
                train_sample_rate = 1 - self.train_valid_rate
                valid_sample_rate = self.train_valid_rate
                trn_size, vld_size = int(data_size * train_sample_rate), int(
                            data_size * valid_sample_rate
                        )
                random_index = torch.randperm(data_size)
                trn_idx, vld_idx = random_index[:trn_size], random_index[trn_size:]
                train_samples = data_raw.iloc[trn_idx, :].reset_index()
                valid_samples = data_raw.iloc[vld_idx, :].reset_index()
            elif self.train_valid_rate == 1:
                train_samples = data_raw
                valid_samples = pd.read_csv(self.train_data_loc)
                print("全量验证集")
            else:
                raise ValueError("train_valid_rate value error.")

        if stage == 'fit' and not self.data_train:
            self.data_train = CDRH3DataSet(train_samples, feature_col=self.feature_col, index_col=self.index_col, label_col=self.label_col)

        if stage == 'fit' and not self.data_val:
            self.data_val = CDRH3DataSet(valid_samples, feature_col=self.feature_col, index_col=self.index_col, label_col=self.label_col)

        if stage == "test" and not self.data_test:
            test_raw = pd.read_csv(self.test_data_loc)
            self.data_test = CDRH3DataSet(test_raw, feature_col=self.feature_col, index_col=self.index_col, label_col=self.label_col)

        if stage == "predict" and not self.data_pred:
            pred_raw = pd.read_csv(self.pred_data_loc)
            if self.label_col not in pred_raw.columns:
                pred_raw[self.label_col] = 1
            self.data_pred = CDRH3DataSet(pred_raw, feature_col=self.feature_col, index_col=self.index_col, label_col=self.label_col)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
