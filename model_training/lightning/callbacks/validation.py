"""Validation cadence measured in completed Lightning optimizer steps."""

import sys

from lightning.pytorch import Callback


class OptimizerStepValidation(Callback):
    """Gate Lightning's batch validation check using global_step.

    Multiplying a batch interval by gradient accumulation drifts when an epoch
    ends with a partial accumulation window. Read the completed step instead.
    Lightning still owns the validation loop, including distributed execution.
    """

    def __init__(self, every_n_steps):
        if every_n_steps < 1:
            raise ValueError("Validation interval must be positive")
        self.every_n_steps = every_n_steps
        self.last_validated_step = 0

    def state_dict(self):
        return {"last_validated_step": self.last_validated_step}

    def load_state_dict(self, state_dict):
        self.last_validated_step = state_dict["last_validated_step"]

    def _set_interval(self, trainer):
        step = trainer.global_step
        due = step > self.last_validated_step and step % self.every_n_steps == 0
        # A finite sentinel avoids Lightning's special end-of-epoch behavior
        # for val_check_batch == inf. These hooks precede its validation check.
        trainer.val_check_batch = 1 if due else sys.maxsize

    def on_train_start(self, trainer, pl_module):
        self._set_interval(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._set_interval(trainer)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.last_validated_step = trainer.global_step
            self._set_interval(trainer)
