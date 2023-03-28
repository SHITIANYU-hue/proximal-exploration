import sys
import os
import torch
import argparse
import pytorch_lightning as pl
import pandas as pd

from typing import Dict
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

env_path = '%s/' % os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_path)

import src.models
from src.callbacks import CSVPredictionWriter
from src.datamodules import CDRH3DataModule
from src.models import get_module, module_collection
from src.utils.load_configs import load_configs_from_yaml
from src.utils.logger import Logger
logger = Logger.logger

PROTEINS = ["bsa", "bv", "tgfb"]

def single_inference(model: LightningModule, ckpt_path: str, datamodule: LightningDataModule, **kwargs):
    inferencer = Trainer(**kwargs)
    prediction = inferencer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    return prediction

def post_process(predictions: Dict):
    result = pd.DataFrame()
    for pt in PROTEINS:
        preds = predictions[pt][0]
        for k, v in preds.items():
            if torch.is_tensor(v):
                preds[k] = v.tolist()
        df = pd.DataFrame(preds).rename(columns={"preds": "%s_preds" % pt, "std": "%s_std" % pt})
        if len(result) == 0:
            result = df
        else:
            result = pd.merge(result, df, on="index", how="left")

    def get_emsemble(row):
        if (row["bsa_preds"] > 0.5) or (row["bv_preds"] > 0.5) or (row["tgfb_preds"] > 0.5):
            return 0    # bind to at least one protein, non specific
        else:
            return 1

    result["specific"] = result.apply(get_emsemble, axis=1)
    return result

def inference(cfg: Dict):
    logger.info("Init datamodule")
    datamodule_cfg = cfg["datamodule"]
    datamodule = CDRH3DataModule(**datamodule_cfg)

    logger.info("Init model")
    model_cfg = cfg["model"]
    model = get_module(model_cfg["module_name"])()

    logger.info("Start inference")
    predictions = {}
    for pt in PROTEINS:
        predictions[pt] = single_inference(model=model, datamodule=datamodule, ckpt_path=model_cfg[pt])

    return predictions

def init_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=os.path.join(env_path, 'configs/non_specific/inference/model.yaml'))
    parser.add_argument('--datamodule', type=str, default=os.path.join(env_path, 'configs/non_specific/inference/datamodule.yaml'))
    parser.add_argument('--inferencer', type=str, default=os.path.join(env_path, 'configs/non_specific/inference/inferencer.yaml'))

    args = parser.parse_args()
    cfg = load_configs_from_yaml(args, main_cfg_name="inferencer")
    if not cfg["output_dir"]:
        output_dir = os.path.join(env_path, "output")
        output_dir = os.path.join(output_dir, cfg["name"])
        logger.info("Output dir: %s" % output_dir)
        os.makedirs(output_dir, exist_ok=True)
        cfg["output_dir"] = output_dir
    return cfg

if __name__ == "__main__":
    cfg = init_configs()
    predictions = inference(cfg)
    result = post_process(predictions)
    preds_loc = os.path.join(cfg["output_dir"], "specificity.csv")
    logger.info("Inference complete, saving results to: %s" % preds_loc)
    result.to_csv(preds_loc, index=False)
    logger.info("Done.")