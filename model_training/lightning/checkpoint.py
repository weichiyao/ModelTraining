from __future__ import annotations

import inspect

import lightning.pytorch as pl


def fit_trusted_checkpoint(
    trainer,
    lightning_module: pl.LightningModule,
    dm: pl.LightningDataModule,
    ckpt_path: str | None = None,
):
    """Resume full Lightning checkpoints created by this training code.

    PyTorch 2.6+ defaults ``torch.load`` toward weights-only loading. Full
    Lightning checkpoints include loop, optimizer, scheduler, and config
    objects, so trusted resume checkpoints need full pickle loading.
    """
    if ckpt_path is None:
        trainer.fit(lightning_module, datamodule=dm)
        return

    fit_signature = inspect.signature(trainer.fit)
    if "weights_only" in fit_signature.parameters:
        trainer.fit(
            lightning_module,
            datamodule=dm,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
    else:
        trainer.fit(lightning_module, datamodule=dm, ckpt_path=ckpt_path)
