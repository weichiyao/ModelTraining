import warnings
from argparse import Namespace

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger

from .callbacks import EMA, EMAModelCheckpoint
from .callbacks.validation import OptimizerStepValidation


class _EpochLastCheckpoint(ModelCheckpoint):
    """Keep the final full state even when training ends between epoch saves."""

    @property
    def state_key(self):
        # Preserve the callback identity used by existing resume checkpoints.
        return super().state_key.replace(
            type(self).__qualname__, ModelCheckpoint.__qualname__, 1
        )

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if not self._should_skip_saving_checkpoint(trainer):
            self._save_topk_checkpoint(trainer, self._monitor_candidates(trainer))


def get_trainer_inputs(
    log_dir: str,
    args: Namespace,
    logger: Logger | list[Logger] | bool | None,
    monitor: str | list[str] | None = None,
    mode: str | list[str] | None = None,
    filename: str | list[str] | None = None,
):
    """Build Trainer options and callbacks for epoch- or step-limited training."""
    max_steps = getattr(args, "max_steps", -1)
    if max_steps > 0 and args.max_epochs > 0:
        warnings.warn(
            "Both max_steps and max_epochs are positive; max_epochs takes "
            "precedence. Overriding max_steps=-1 for the Trainer.",
            UserWarning,
            stacklevel=2,
        )
        max_steps = -1

    if monitor is None:
        monitor = ["val_loss"]
    if mode is None:
        mode = ["min"]
    if filename is None:
        filename = ["best-epoch{epoch}-step{step}-val_loss{val_loss:.4f}"]
    if isinstance(monitor, str):
        monitor = [monitor]
    if isinstance(mode, str):
        mode = [mode]
    if isinstance(filename, str):
        filename = [filename]

    step_limited_training = max_steps > 0 and args.max_epochs < 0
    epoch_interval = max(1, args.log_interval)
    step_val_interval = getattr(args, "step_val_interval", -1)
    step_validation = step_limited_training and step_val_interval > 0
    last_cadence = (
        {"every_n_train_steps": max(500, int(args.log_every_n_steps))}
        if step_limited_training
        else {"every_n_epochs": epoch_interval}
    )

    callbacks: list[Callback] = []
    monitor_checkpoint_cls = ModelCheckpoint
    if getattr(args, "ema", False):
        callbacks.append(
            EMA(
                decay=0.9999,
                validate_original_weights=False,
                every_n_steps=1,
                cpu_offload=False,
            ),
        )
        monitor_checkpoint_cls = EMAModelCheckpoint

    if step_validation:
        callbacks.append(OptimizerStepValidation(step_val_interval))

    last_checkpoint_cls = (
        ModelCheckpoint if step_limited_training else _EpochLastCheckpoint
    )
    callbacks.append(
        last_checkpoint_cls(
            filename="last",
            save_last=False,
            save_top_k=1,
            dirpath=log_dir,
            monitor=None,
            save_weights_only=False,
            save_on_exception=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
            save_on_train_epoch_end=not step_limited_training,
            **last_cadence,
        )
    )
    for j in range(len(monitor)):
        callbacks.append(
            monitor_checkpoint_cls(
                monitor=monitor[j],
                dirpath=log_dir,
                save_last=False,
                save_top_k=1,
                save_weights_only=True,
                mode=mode[j],
                filename=filename[j],
                auto_insert_metric_name=False,
                every_n_epochs=1 if step_validation else epoch_interval,
                # Save the weights that produced the fresh validation metric.
                save_on_train_epoch_end=False,
            )
        )

    kwargs_trainer = {
        "num_sanity_val_steps": 0,
        "max_epochs": args.max_epochs,
        "min_epochs": max(0, args.min_epochs),
        "max_steps": max_steps,
        "enable_model_summary": True,
        "enable_progress_bar": args.progress_bar,
        "check_val_every_n_epoch": None if step_validation else epoch_interval,
        "val_check_interval": 1 if step_validation else None,
        "default_root_dir": log_dir,
        "enable_checkpointing": True,
        "logger": logger,
        "deterministic": True,
        "accelerator": args.device,
        "strategy": "ddp_find_unused_parameters_true" if args.use_ddp else "auto",
        "devices": args.n_devices,
        "num_nodes": args.num_nodes,
        "gradient_clip_val": args.gradient_clip_val,
        "gradient_clip_algorithm": args.gradient_clip_algorithm,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "use_distributed_sampler": False,
        "limit_train_batches": getattr(args, "limit_train_batches", 1.0),
        "limit_val_batches": getattr(args, "limit_val_batches", 1.0),
        "reload_dataloaders_every_n_epochs": 1 if args.use_ddp else 0,
        "log_every_n_steps": getattr(args, "log_every_n_steps", 100),
        "precision": args.precision,
    }

    return kwargs_trainer, callbacks
