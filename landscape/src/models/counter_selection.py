import torch
import logging

from typing import Any, List
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
from src.models.components.nets import get_net

class BasicModel(LightningModule):
    "Only one model"
    def __init__(
        self,
        net_name: str,
        lossfunc: torch.nn.modules.loss = torch.nn.MSELoss(),
        output_dim: int = 1,
    ):
        super().__init__()
        self.net = get_net(net_name)(output_dim=output_dim)
        self.lossfunc = lossfunc

    def step(self, batch: Any):
        x, y = batch["x"], batch["y"]
        preds = net(x)
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = self.lossfunc(preds, y) + (0.0000001 * l1_norm)
        return y, x, loss, preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())

class EnsembleModule(LightningModule):
    """Base CounterSelection Ensemble model
    """
    def __init__(
        self,
        net_names: List[str] = ["Seq32x1_16", "Seq32x2_16", "Seq64x1_16", "Seq_emb_32x1_16", "Seq32x1_16_filt3", "Seq_32_32"],  # names of neural networks
        lossfunc: torch.nn.modules.loss = torch.nn.MSELoss(),
        output_dim: int = 1,
    ):
        super().__init__()
        self.nets = ModuleList([get_net(net_name)(output_dim=output_dim) for net_name in net_names])
        self.lossfunc = lossfunc

    def step(self, batch: Any):
        x, y = batch["x"], batch["y"]
        preds = torch.concat([net(x) for net in self.nets], dim=-1)
        emsemble = torch.mean(preds, dim=-1, dtype=torch.float)
        std = torch.std(preds, dim=-1)
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = self.lossfunc(emsemble, y) + (0.0000001 * l1_norm)
        return y, x, loss, emsemble, std

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
    
class BinaryClassificationModel(BasicModel):
    """ Single model for Binary Classification
    """
    def __init__(
        self,
        net_name: str,  # name of neural network
        lossfunc: torch.nn.modules.loss = torch.nn.MSELoss(),
    ):
        super().__init__(net_name=net_name, lossfunc=lossfunc)

        # metric objects for calculating and averaging accuracy across batches, also in DDP
        self.train_metrics: MetricCollection = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryRecall(),
                BinaryAUROC(),
                BinaryPrecision(),
            ],
            prefix="train/",
        )

        self.validation_metrics: MetricCollection = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryRecall(),
                BinaryAUROC(),
                BinaryPrecision(),
            ],
            prefix="val/",
        )

        self.test_metrics: MetricCollection = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryRecall(),
                BinaryAUROC(),
                BinaryPrecision(),
            ],
            prefix="test/",
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_auc_best = MaxMetric()

    def training_step(self, batch: Any, batch_idx: int):
        targets, _, loss, logits = self.step(batch)
        preds = torch.sigmoid(logits)
        outputs = self.train_metrics(preds, targets)
        self.train_loss(loss)
        outputs['train/loss'] = self.train_loss
        self.log_dict(outputs, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        targets, _, loss, logits = self.step(batch)
        preds = torch.sigmoid(logits)
        self.validation_metrics.update(preds, targets)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        targets, _, loss, logits = self.step(batch)
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, targets)
        self.test_loss(loss)
        self.log("val/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_auc_best.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        output = self.validation_metrics.compute()
        self.val_auc_best(output["val/BinaryAUROC"])  # update best so far val acc
        output["val/BinaryAUROC_best"] = self.val_auc_best.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_epoch_end(self, outputs: List[Any]):
        metrics_epoch = self.test_metrics.compute()
        self.log_dict(metrics_epoch, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch["x"], batch["y"]
        logits = self.net(x)
        y_hat = torch.sigmoid(logits)

        out = {
                "index": batch["index"],
                "preds": y_hat,
            }
        return out

class EnsembleBinaryClassificationModel(EnsembleModule):
    """ CounterSelection models for Binary Classification
    """
    def __init__(
        self,
        net_names: List[str] = ["Seq32x1_16", "Seq32x2_16", "Seq64x1_16", "Seq_emb_32x1_16", "Seq32x1_16_filt3", "Seq_32_32"],  # names of neural networks
        lossfunc: torch.nn.modules.loss = torch.nn.modules.loss.BCEWithLogitsLoss(),
        output_dim: int = 1,
    ):
        super().__init__(net_names=net_names, lossfunc=lossfunc, output_dim=output_dim)

        # metric objects for calculating and averaging accuracy across batches, also in DDP
        self.train_metrics: MetricCollection = MetricCollection(
            [
                BinaryF1Score(),
                BinaryAUROC(),
                BinaryAccuracy(),
                BinaryRecall(),
                BinaryPrecision(),
            ],
            prefix="train/",
        )

        self.validation_metrics: MetricCollection = MetricCollection(
            [
                BinaryF1Score(),
                BinaryAUROC(),
                BinaryAccuracy(),
                BinaryRecall(),
                BinaryPrecision(),
            ],
            prefix="val/",
        )

        self.test_metrics: MetricCollection = MetricCollection(
            [
                BinaryF1Score(),
                BinaryAUROC(),
                BinaryAccuracy(),
                BinaryRecall(),
                BinaryPrecision(),
            ],
            prefix="test/",
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_auc_best = MaxMetric()

    def training_step(self, batch: Any, batch_idx: int):
        targets, _, loss, logits, std = self.step(batch)
        preds = torch.sigmoid(logits)
        outputs = self.train_metrics(preds, targets)
        self.train_loss(loss)
        outputs['train/loss'] = self.train_loss
        self.log_dict(outputs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits, "std": std}

    def validation_step(self, batch: Any, batch_idx: int):
        targets, _, loss, logits, std = self.step(batch)
        preds = torch.sigmoid(logits)
        self.validation_metrics.update(preds, targets)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits, "std": std}

    def test_step(self, batch: Any, batch_idx: int):
        targets, _, loss, logits, std = self.step(batch)
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, targets)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits, "std": std}

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_auc_best.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        output = self.validation_metrics.compute()
        self.val_auc_best(output["val/BinaryAUROC"])  # update best so far val acc
        output["val/BinaryAUROC_best"] = self.val_auc_best.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/best_auc", self.val_auc_best.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def test_epoch_end(self, outputs: List[Any]):
        metrics_epoch = self.test_metrics.compute()
        self.log_dict(metrics_epoch, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch["x"], batch["y"]
        preds = torch.concat([net(x) for net in self.nets], dim=-1)
        emsemble = torch.mean(preds, dim=-1, dtype=torch.float)
        std = torch.std(torch.sigmoid(preds), dim=-1)
        y_hat = torch.sigmoid(emsemble)

        out = {
                "index": batch["index"],
                "preds": y_hat,
                "std": std,
            }
        return out

class EnsembleRegressionModel(EnsembleModule):
    """ CounterSelection models for Regression
    """
    def __init__(
        self,
        net_names: List[str] = ["Seq32x1_16", "Seq32x2_16", "Seq64x1_16", "Seq_emb_32x1_16", "Seq32x1_16_filt3", "Seq_32_32"],  # names of neural networks
        lossfunc: torch.nn.modules.loss = torch.nn.modules.loss.MSELoss(),
        output_dim: int = 1,
    ):
        super().__init__(net_names=net_names, lossfunc=lossfunc, output_dim=output_dim)

        # metric objects for calculating and averaging accuracy across batches, also in DDP
        self.train_metrics: MetricCollection = MetricCollection(
            [
                MeanSquaredError(),
                SpearmanCorrCoef(),
                MeanAbsoluteError(),
            ],
            prefix="train/",
        )

        self.validation_metrics: MetricCollection = MetricCollection(
            [
                MeanSquaredError(),
                SpearmanCorrCoef(),
                MeanAbsoluteError(),
            ],
            prefix="val/",
        )

        self.test_metrics: MetricCollection = MetricCollection(
            [
                MeanSquaredError(),
                SpearmanCorrCoef(),
                MeanAbsoluteError(),
            ],
            prefix="test/",
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_spearman_best = MaxMetric()

    def training_step(self, batch: Any, batch_idx: int):
        targets, _, loss, preds, std = self.step(batch)
        outputs = self.train_metrics(preds, targets)
        self.train_loss(loss)
        outputs['train/loss'] = self.train_loss
        self.log_dict(outputs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets, "std": std}

    def validation_step(self, batch: Any, batch_idx: int):
        targets, _, loss, preds, std = self.step(batch)
        self.validation_metrics.update(preds, targets)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets, "std": std}

    def test_step(self, batch: Any, batch_idx: int):
        targets, _, loss, preds, std = self.step(batch)
        self.test_metrics.update(preds, targets)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets, "std": std}

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_spearman_best.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        output = self.validation_metrics.compute()
        self.val_spearman_best(output["val/SpearmanCorrCoef"])  # update best so far val acc
        output["val/SpearmanCorrCoef_best"] = self.val_spearman_best.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/spearman_best", self.val_spearman_best.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def test_epoch_end(self, outputs: List[Any]):
        metrics_epoch = self.test_metrics.compute()
        self.log_dict(metrics_epoch, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch["x"], batch["y"]
        preds = torch.concat([net(x) for net in self.nets], dim=-1)
        emsemble = torch.mean(preds, dim=-1, dtype=torch.float)
        std = torch.std(torch.sigmoid(preds), dim=-1)

        out = {
                "index": batch["index"],
                "preds": emsemble,
                "std": std,
            }
        return out