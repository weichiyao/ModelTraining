"""Reusable helpers for loading trusted PyTorch Lightning checkpoints."""

import inspect

import fsspec
import lightning.pytorch as pl
import torch


def load_training_checkpoint(path, map_location="cpu", *, weights_only=False):
    """Load a checkpoint with standard PyTorch serialization."""
    with fsspec.open(str(path), "rb") as stream:
        return torch.load(stream, map_location=map_location, weights_only=weights_only)


def load_training_weights(module:pl.LightningModule, path):
    """Restore a constructed Lightning module with its existing model and config."""
    checkpoint = load_training_checkpoint(path)
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    return module


def fit_trusted_checkpoint(trainer:pl.Trainer, lightning_module:pl.LightningModule, dm:pl.LightningDataModule, ckpt_path=None):
    """Fit, optionally restoring a full checkpoint from a trusted training run."""
    if ckpt_path is None:
        trainer.fit(lightning_module, datamodule=dm)
        return

    fit_signature = inspect.signature(trainer.fit)
    if "weights_only" in fit_signature.parameters:
        trainer.fit(lightning_module, datamodule=dm, ckpt_path=ckpt_path, weights_only=False)
    else:
        trainer.fit(lightning_module, datamodule=dm, ckpt_path=ckpt_path)
