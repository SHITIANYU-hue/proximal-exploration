import sys
import os
import yaml
import argparse
import pytorch_lightning as pl
import pandas as pd
from . import register_landscape
from typing import Dict, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import tape

env_path = '%s/' % os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_path)

from inference_non_specificity import single_inference
from src.datamodules.components.sequence_dataset import CDRH3DataPred
from src.models import get_module, module_collection
import logging
logging.getLogger("package").propagate = False



with open(os.path.join(env_path, "configs/affinity_prediction/inference.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    


@register_landscape("antibody")
class antiberty_lanescape:
    def __init__(self, args):
        with open(os.path.join(env_path, "configs/affinity_prediction/inference.yaml")) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        self.dataloader = DataLoader(
                dataset=CDRH3DataPred(samples=[]),
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
                pin_memory=True,
                drop_last=False,
                shuffle=False,
            )
        self.inferencer = Trainer(
            accelerator=cfg["accelerator"], 
            strategy=cfg["strategy"], 
            devices=cfg["devices"], 
            logger=False
            )
        with open(os.path.join(env_path, "configs/starting_sequence.json"),'r') as f:
            self.starting_sequence = json.load(f)
        self.device = args.device
        self.model = get_module(cfg["module_name"])()
    def get_fitness(self, cdr3_sequences):
        fitness_scores=[]
        self.model.eval()
        for seq in cdr3_sequences:
            print('seq',seq)
            self.dataloader.dataset.update_data([seq])
            prediction = self.inferencer.predict(self.model, self.dataloader, ckpt_path=cfg["ckpt_for_predict"])
            prediction = -1*prediction[0]['preds'].item()
            fitness_scores.append(prediction)
        return fitness_scores
