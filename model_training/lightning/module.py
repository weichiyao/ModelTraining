from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim import AdamW, Optimizer, lr_scheduler
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


def _as_log_scalar(value):
    """Detach metric tensors and reduce them to floating-point scalars."""
    if torch.is_tensor(value):
        return value.detach().float().mean()
    return value


class Queue:
    """Bounded gradient-norm history persisted in training checkpoints."""

    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items) if self.items else 0.0

    def std(self):
        return np.std(self.items) if self.items else 0.0

    def state_dict(self):
        """Return the state of the queue for saving in a checkpoint."""
        return {"items": self.items}

    def load_state_dict(self, state):
        """Load the state of the queue from a checkpoint."""
        self.items = state["items"]


class TrainPLModule(LightningModule):
    def __init__(self, model: nn.Module, cfg):
        """Wrap a model and optimizer/scheduler config for Lightning training.

        Subclasses implement compute_loss and map summed metrics to their counts
        through metric_count_keys.
        """
        super().__init__()
        self.model              = model
        self.cfg                = cfg
        # Optimizer and LR scheduler
        self.lr                 = cfg.lr
        self.optimizer          = cfg.optimizer
        self.adam_betas         = cfg.adam_betas
        self.adam_eps           = cfg.adam_eps
        self.weight_decay       = cfg.weight_decay
        self.lr_scheduler       = cfg.lr_scheduler
        self.warmup_proportion  = cfg.warmup_proportion
        self.power              = cfg.power
        self.num_cycles         = cfg.num_cycles
        self.lr_end             = cfg.lr_end
        self.lr_interval        = cfg.lr_interval
        self.standparam         = cfg.standparam
        self.log_every_n_steps  = cfg.log_every_n_steps
        self.num_decay_steps    = getattr(cfg, 'num_decay_steps',  -1)
        self.gradient_clip_adaptive = getattr(cfg, 'gradient_clip_adaptive', False)
        self.track_grad_norms   = getattr(cfg, 'track_grad_norms', False)

        if self.gradient_clip_adaptive == "queue":
            self.gradnorm_queue = Queue()  # Shared gradient norm queue
            self.gradnorm_queue.add(3000)  # Add large value that will be flushed.

        # Default value assuming single-GPU or CPU training
        self.sync_dist = False

    @property
    def metric_count_keys(self):
        raise ValueError("Subclasses must define the `metric_count_keys` property")

    def on_train_start(self) -> None:
        self.sync_dist = self.trainer.world_size > 1

    def on_validation_start(self) -> None:
        self.sync_dist = self.trainer.world_size > 1

    def compute_loss(self, batch):
        raise NotImplementedError("Subclasses must implement the `compute_loss` method")

    def _log_scalar_dict(
        self,
        values: dict,
        *,
        step: str,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool,
        reduce_fx: str = "mean",
        batch_size: int | None = None,
    ) -> None:
        if not values:
            return

        self.log_dict(
            {f"{step}_{key}": _as_log_scalar(value) for key, value in values.items()},
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            sync_dist=sync_dist,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )

    def general_step(
        self,
        batch,
        step="train",
        on_step: bool = True,
        on_epoch: bool = True,
        sync_dist: bool = False,
    ):
        loss, loss_dict = self.compute_loss(batch)
        loss = loss.mean()

        if step != "train" or (self.global_step + 1) % self.log_every_n_steps == 0:
            log_kwargs = {
                "step": step,
                "on_step": on_step,
                "on_epoch": on_epoch,
                "sync_dist": sync_dist,
            }
            self._log_scalar_dict({"loss": loss}, **log_kwargs)
            # metric_count_keys pairs batch totals with their counts.
            self._log_scalar_dict(loss_dict, reduce_fx="sum", **log_kwargs)
        return loss

    def general_on_epoch_end(
        self, step_type: str = "train", on_step: bool = True, on_epoch: bool = True
    ):
        metrics = self.trainer.logged_metrics
        suffix = "_epoch" if on_step and on_epoch else ""
        for metric_name, total_n in self.metric_count_keys.items():
            total_value = metrics.get(f"{step_type}_{metric_name}{suffix}", 0.0)
            total_count = metrics.get(f"{step_type}_{total_n}{suffix}", 0.0)
            avg_loss = total_value / (
                total_count if total_count > 0 else -1.0
            )  # Avoid division by zeros
            self.log(
                f"{step_type}_Avg_{metric_name}",
                avg_loss,
                prog_bar=True,
                sync_dist=self.sync_dist,
            )

    def training_step(
        self, batch, batch_idx, on_step: bool = True, on_epoch: bool = True
    ):
        return self.general_step(
            batch,
            step="train",
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=self.sync_dist,
        )

    def on_train_epoch_end(self):
        self.general_on_epoch_end("train", on_step=True, on_epoch=True)

    def _has_multiple_val_loaders(self) -> bool:
        trainer = self._trainer
        if trainer is None:
            return False
        num_val_dataloaders = getattr(trainer, "num_val_dataloaders", 1) or 1
        for attribute in ("val_dataloaders", "num_val_batches"):
            loaders = getattr(trainer, attribute, None)
            if isinstance(loaders, (list, tuple)):
                num_val_dataloaders = max(int(num_val_dataloaders), len(loaders))
        return int(num_val_dataloaders) > 1

    def _validation_loader_name(self, dataloader_idx: int) -> str:
        trainer = self._trainer
        datamodule = getattr(trainer, "datamodule", None)
        names = getattr(datamodule, "val_loader_names", None)
        if names and dataloader_idx < len(names):
            return names[dataloader_idx]
        return f"loader{dataloader_idx}"

    def _validation_step_name(self, dataloader_idx: int) -> str:
        trainer = self._trainer
        datamodule = getattr(trainer, "datamodule", None)
        names = getattr(datamodule, "val_loader_names", None)
        dataloader_idx = int(dataloader_idx)
        if names or self._has_multiple_val_loaders():
            return f"val_{self._validation_loader_name(dataloader_idx)}"
        return "val"

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        return self.general_step(
            batch,
            step=self._validation_step_name(dataloader_idx),
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=self.sync_dist,
        )

    def on_validation_epoch_end(self):
        for dataloader_idx, num_batches in enumerate(self.trainer.num_val_batches):
            if num_batches:
                self.general_on_epoch_end(
                    self._validation_step_name(dataloader_idx),
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        """Build AdamW and the scheduler using Lightning's training horizon."""
        if self.optimizer == "adamw":
            if self.standparam:
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.lr,
                    betas=self.adam_betas,
                    weight_decay=self.weight_decay,
                    eps=self.adam_eps,
                )
            else:
                assert hasattr(self.model, "configure_AdamW"), (
                    "Model must implement configure_AdamW when standparam is False"
                )
                optimizer = self.model.configure_AdamW(
                    learning_rate=self.lr,
                    betas=self.adam_betas,
                    weight_decay=self.weight_decay,
                    eps=self.adam_eps,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lr * param_group.get("lr_scale", 1.0)

        else:
            raise ValueError(
                f"Optimizer only implemented for 'adamw', received {self.optimizer}."
            )

        # Step-based schedules count optimizer updates, including accumulation.
        if self.lr_scheduler == "none":
            return {"optimizer": optimizer}

        self.lr_interval = (
            "epoch" if self.lr_scheduler in {"multistep", "step"} else "step"
        )
        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_warmup_steps = int(
            self.warmup_proportion * num_training_steps
        )  # e.g., 10% of steps for warm-up

        if self.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.num_cycles,
            )
        elif self.lr_scheduler == "cosine_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps, eta_min=self.lr_end
            )
        elif self.lr_scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps
            )
        elif self.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.lr_scheduler == "polynomial_decay":
            if self.num_decay_steps < 0:
                self.num_decay_steps = num_training_steps
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.num_decay_steps,
                lr_end=self.lr_end,
                power=self.power,
            )
        elif self.lr_scheduler == "cosine_with_lr_end":
            if self.num_decay_steps < 0:
                self.num_decay_steps = num_training_steps
            if not 0 <= num_warmup_steps < self.num_decay_steps:
                raise ValueError(
                    "cosine_with_lr_end requires 0 <= warmup steps < decay steps"
                )
            base_lrs = [group["lr"] for group in optimizer.param_groups]
            if any(lr <= 0 for lr in base_lrs) or self.lr_end < 0:
                raise ValueError(
                    "cosine_with_lr_end requires positive base learning rates and lr_end >= 0"
                )
            scheds, milestones = [], []
            if num_warmup_steps > 0:
                # Retain the original first-update LR of base_lr / warmup_steps.
                def warmup_lambda(step):
                    return min((step + 1) / float(num_warmup_steps), 1.0)

                scheds.append(lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda))
                milestones.append(num_warmup_steps)
            # Decay to lr_end, then hold the floor.
            scheds.append(
                lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.num_decay_steps - num_warmup_steps,
                    eta_min=self.lr_end,
                )
            )
            milestones.append(self.num_decay_steps)

            # LambdaLR uses initial_lr, not the LR already changed by warmup.
            lr_lambdas = [(lambda _, blr=blr: self.lr_end / blr) for blr in base_lrs]
            scheds.append(lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas))

            scheduler = lr_scheduler.SequentialLR(
                optimizer, schedulers=scheds, milestones=milestones
            )
        elif self.lr_scheduler == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 500, 5000], gamma=0.2
            )
        elif self.lr_scheduler == "step":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        else:
            raise ValueError(
                f"Unknown scheduler functions: {self.lr_scheduler}. Only 'cosine', 'constant', 'linear', 'polynomial_decay'."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_interval,
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        """Save adaptive gradient-clipping history."""
        if self.gradient_clip_adaptive == "queue":
            checkpoint["gradnorm_queue"] = self.gradnorm_queue.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Restore adaptive gradient-clipping history."""
        if self.gradient_clip_adaptive == "queue":
            if "gradnorm_queue" in checkpoint:
                self.gradnorm_queue.load_state_dict(checkpoint["gradnorm_queue"])
            else:
                print("Warning: No gradnorm_queue found in checkpoint.")
                self.gradnorm_queue = Queue()  # Shared gradient norm queue
                self.gradnorm_queue.add(3000)  # Add large value that will be flushed.

    def on_before_optimizer_step(self, optimizer):
        """Apply adaptive clipping before Lightning's configured clipping."""
        if self.gradient_clip_adaptive == "queue":
            max_grad_norm = (
                1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
            )

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm, norm_type=2.0
            )

            # Retain the clipping history used by existing checkpoints.
            if float(grad_norm) > max_grad_norm:
                self.gradnorm_queue.add(float(max_grad_norm))
            else:
                self.gradnorm_queue.add(float(grad_norm))

            if self.track_grad_norms:
                self.log("unclipped grad norm", grad_norm)

                if float(grad_norm) > max_grad_norm:
                    print(
                        f"Clipped gradient with value {grad_norm:.1f} "
                        f"while allowed {max_grad_norm:.1f}"
                    )

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        if not gradient_clip_val or gradient_clip_val <= 0:
            return
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return
        if gradient_clip_algorithm == "norm":
            total_norm = torch.nn.utils.clip_grad_norm_(
                params, max_norm=float(gradient_clip_val)
            )
            if (
                getattr(self, "track_grad_norms", False)
                and (self.global_step + 1) % self.log_every_n_steps == 0
            ):
                self.log(
                    "grad_norm/unclipped",
                    total_norm,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=self.sync_dist,
                )
        else:
            torch.nn.utils.clip_grad_value_(params, clip_value=float(gradient_clip_val))
