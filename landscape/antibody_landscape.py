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
                pin_memory=False,  ## disable cuda usage
                drop_last=False,
                shuffle=False,
            )
        self.inferencer = Trainer(
            accelerator=cfg["accelerator"], 
            strategy=cfg["strategy"], 
            devices=cfg["devices"], 
            logger=False
            )
        if cfg['checkpoint'] == 'bsa':
            self.starting_sequence = 'ELQGWLRYWQHGQLDY'
        if cfg['checkpoint'] == 'bv':
            self.starting_sequence = 'VVDRRSSSYFDY'
        if cfg['checkpoint'] == 'tgfb':
            self.starting_sequence = 'WGGYSRVFYFEAPFDY'  
        if cfg['checkpoint'] == 'phage_display':
            self.starting_sequence= 'GDGPWRVWRSAFDY'
            
        self.device = args.device
        model_config = cfg["checkpoint_configs"][cfg["checkpoint"]]
        model = get_module(model_config["module_name"])(net_configs=model_config["net_configs"])
        self.model = model
    def get_fitness(self, cdr3_sequences):
        fitness_scores=[]
        self.model.eval()
        for seq in cdr3_sequences:
            self.dataloader.dataset.update_data([seq])
            prediction = self.inferencer.predict(self.model, self.dataloader)
            prediction = 1*prediction[0]['preds'].item()
            fitness_scores.append(prediction)
        return fitness_scores
