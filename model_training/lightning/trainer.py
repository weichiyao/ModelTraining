from argparse import Namespace

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from .callbacks import EMA, EMAModelCheckpoint


def get_trainer_inputs(
    log_dir: str,
    args: Namespace,
    logger: loggers,
    monitor: list | None = None,
    mode: list | None = None,
    filename: list | None = None,
):
    if monitor is None:
        monitor = ["val_loss"]
    if mode is None:
        mode = ["min"]
    if filename is None:
        filename = ["best-epoch{epoch}-step{step}-val_loss{val_loss}"]
    if isinstance(monitor, str):
        monitor = [monitor]
    if isinstance(mode, str):
        mode = [mode]
    if isinstance(filename, str):
        filename = [filename]

    step_limited_training = (
        hasattr(args, "max_steps") and args.max_steps > 0 and args.max_epochs < 0
    )

    save_on_train_epoch_end = True
    every_n_epochs = max(1, args.log_interval)  # checkpoint/validate every N epochs
    check_val_every_n_epoch = max(1, args.log_interval)  # validate every N epochs
    every_n_train_steps = None  # don't checkpoint by steps
    val_check_interval = None  # not used in epoch mode

    if step_limited_training:
        last_every_n_epochs = None
        last_save_on_train_epoch_end = False
        last_every_n_train_steps = max(500, int(args.log_every_n_steps))

    checkpoint_every_n_train_steps = max(
        0, int(getattr(args, "checkpoint_every_n_train_steps", 0))
    )
    if checkpoint_every_n_train_steps:
        last_every_n_epochs = None
        last_every_n_train_steps = checkpoint_every_n_train_steps
        last_save_on_train_epoch_end = False
    else:
        last_every_n_epochs = every_n_epochs
        last_every_n_train_steps = every_n_train_steps
        last_save_on_train_epoch_end = save_on_train_epoch_end

    callbacks: list[Callback] = []

    MonitorCheckpointCls = ModelCheckpoint
    if hasattr(args, "ema") and args.ema:
        callbacks = [
            EMA(
                decay=0.9999,
                validate_original_weights=False,
                every_n_steps=1,
                cpu_offload=False,
            ),
        ]
        MonitorCheckpointCls = EMAModelCheckpoint

    callbacks.append(
        ModelCheckpoint(
            filename="last",
            save_last=False,
            save_top_k=1,
            dirpath=log_dir,
            monitor=None,
            save_weights_only=False,
            save_on_exception=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
            every_n_epochs=last_every_n_epochs,
            every_n_train_steps=last_every_n_train_steps,
            save_on_train_epoch_end=last_save_on_train_epoch_end,
        )
    )
    n_monitor = len(monitor)
    for j in range(n_monitor):
        callbacks.append(
            # Save the weights used to compute the monitored metric.
            MonitorCheckpointCls(
                monitor=monitor[j],
                dirpath=log_dir,
                save_last=False,
                save_top_k=1,
                save_weights_only=True,
                mode=mode[j],
                filename=filename[j],
                auto_insert_metric_name=False,
                every_n_epochs=every_n_epochs,
                every_n_train_steps=every_n_train_steps,
                save_on_train_epoch_end=save_on_train_epoch_end,
            )
        )

    kwargs_trainer = {
        "num_sanity_val_steps": 0,  # Disable the sanity check
        "max_epochs": args.max_epochs,
        "min_epochs": max(0, args.min_epochs),
        "max_steps": getattr(args, "max_steps", -1),
        "enable_model_summary": True,
        "enable_progress_bar": args.progress_bar,
        "check_val_every_n_epoch": check_val_every_n_epoch,  # epoch-based validation cadence; validate every n epoch
        "val_check_interval": val_check_interval,  # within-epoch validation cadence
        "default_root_dir": log_dir,
        "enable_checkpointing": True,
        "logger": logger,
        "deterministic": True,
        "accelerator": args.device,
        "strategy": "ddp_find_unused_parameters_true" if args.use_ddp else "auto",
        "devices": args.n_devices,  # Number of GPUs per node to use
        "num_nodes": args.num_nodes,  # Number of nodes for distributed training
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
