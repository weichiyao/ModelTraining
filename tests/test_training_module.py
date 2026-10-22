from argparse import Namespace

import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_training.lightning import TrainPLModule


def test_generic_training_with_subclass_hooks_and_no_loss_specific_config(tmp_path):
    class RegressionModule(TrainPLModule):
        def __init__(self, cfg):
            super().__init__(nn.Linear(2, 1), cfg)
            self.epochs_started = []
            self.batch_steps = []

        def on_train_epoch_start(self):
            super().on_train_epoch_start()
            self.epochs_started.append(self.current_epoch)

        def on_train_batch_start(self, batch, batch_idx):
            result = super().on_train_batch_start(batch, batch_idx)
            self.batch_steps.append(self.global_step)
            return result

        def compute_loss(self, batch):
            inputs, targets = batch
            return (self.model(inputs) - targets).square().mean(), {}

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
    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    loader = DataLoader(
        TensorDataset(inputs, inputs.sum(dim=1, keepdim=True)),
        batch_size=1,
    )
    module = RegressionModule(cfg)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert module.epochs_started == [0]
    assert module.batch_steps == [0, 0, 1, 1]
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["val_loss"])

    results = trainer.validate(module, dataloaders=loader, verbose=False)
    assert torch.isfinite(torch.tensor(results[0]["val_loss"]))
