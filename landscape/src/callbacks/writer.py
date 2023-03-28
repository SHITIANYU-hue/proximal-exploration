import os
import pickle
from collections import defaultdict
from typing import Any, Optional, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

class CSVPredictionWriter(BasePredictionWriter):
    """Prediction Writer for model that output multiple columns of results.

    It is a child class of `pytorch_lightning.callbacks.BasePredictionWriter`, which is a base
    class to implement how the predictions should be stored.

    Docs:
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.BasePredictionWriter.html
    """

    def __init__(
        self, output_dir: str, enable: bool = False, write_interval: str = "epoch"
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.enable = enable

    def on_predict_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isdir(self.output_dir):
            raise ValueError(f"target output dir {self.output_dir} is a file!")

        if not predictions or len(predictions) == 0:
            # log.warning("no prediction output")
            print("no prediction output.")

        elif self.enable:
            df_dict = defaultdict(list)
            for dataset_index, dataset_preds in enumerate(predictions):
                for batch_id, batch_preds in enumerate(dataset_preds):
                    for key in batch_preds.keys():
                        df_dict[key].extend(batch_preds[key])

            for key in df_dict:
                if type(df_dict[key][0]) == torch.Tensor:
                    df_dict[key] = torch.tensor(df_dict[key]).numpy()

            preds_df = pd.DataFrame(df_dict)
            output_path = os.path.join(self.output_dir, "pred.csv")
            preds_df.to_csv(output_path, index=False)
