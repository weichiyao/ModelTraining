"""Run a small CPU example after installing ModelTraining."""

from argparse import Namespace

import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_training.lightning import TrainPLModule


class RegressionModule(TrainPLModule):
    @property
    def metric_count_keys(self) -> dict[str, str]:
        # Map each summed metric to the key holding its sample count.
        return {"squared_error": "n_samples"}

    def compute_loss(self, batch):
        inputs, targets = batch
        squared_error = (self.model(inputs) - targets).square()
        return squared_error.mean(), {
            "squared_error": squared_error.sum(),
            "n_samples": squared_error.new_tensor(targets.numel()),
        }


def main():
    cfg = Namespace(
        lr=1e-3,
        optimizer="adamw",
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        weight_decay=0.01,
        lr_scheduler="none",
        warmup_proportion=0.1,
        power=1.0,
        num_cycles=0.5,
        lr_end=0.0,
        lr_interval="step",
        standparam=True,
        log_every_n_steps=1,
    )
    module = RegressionModule(nn.Linear(4, 1), cfg)
    inputs = torch.randn(16, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)


if __name__ == "__main__":
    main()
