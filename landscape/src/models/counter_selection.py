import torch
import logging

from typing import Any, List, Dict
from torch import optim
from torch.nn.functional import softmax
from torch.nn import ModuleList
from pytorch_lightning import LightningModule
from torchmetrics import (
    MaxMetric,
    MeanAbsoluteError,
    MeanMetric,
    MeanSquaredError,
    MetricCollection,
    SpearmanCorrCoef,
)
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryRecall,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryF1Score,
)
from functools import partial
from src.models.components.nets import get_net

class EnsembleInference(LightningModule):
    """
        Inference by models
    """
    def __init__(
        self,
        net_configs: Dict, # {net_name: state_dict_pth}
        net_names: List[str] = ["Seq32x1_16", "Seq32x2_16", "Seq64x1_16", "Seq_emb_32x1_16", "Seq32x1_16_filt3", "Seq_32_32"],  # names of neural networks
        activation_function = lambda x: x,
        output_dim: int = 1,
    ):
        super().__init__()
        nets = []
        for net_name in net_names:
            pth = net_configs[net_name]
            net = get_net(net_name)(output_dim=output_dim)
            net.load_state_dict(torch.load(pth))
            nets.append(net)
        self.nets = ModuleList(nets)
        self.activation_function = activation_function

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch["x"], batch["y"]
        preds = torch.concat([self.activation_function(net(x)) for net in self.nets], dim=-1)
        emsemble = torch.mean(preds, dim=-1, dtype=torch.float)
        std = torch.std(torch.sigmoid(preds), dim=-1)

        out = {
                "index": batch["index"],
                "preds": emsemble,
                "std": std,
            }
        return out

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = optim.Adam(params=self.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=0.000001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                },
            }


class EnsembleInferenceBCE(EnsembleInference):

    def __init__(
        self,
        net_configs: Dict, # {net_name: state_dict_pth}
        net_names: List[str] = ["Seq32x1_16", "Seq32x2_16", "Seq64x1_16", "Seq_emb_32x1_16", "Seq32x1_16_filt3", "Seq_32_32"],  # names of neural networks
        activation_function = torch.sigmoid,
        output_dim: int = 1,
    ):
        super().__init__(
            net_configs=net_configs,
            net_names=net_names, 
            activation_function=activation_function, 
            output_dim=output_dim
        )
