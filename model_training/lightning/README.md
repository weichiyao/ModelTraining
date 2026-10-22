# Lightning training helpers

`TrainPLModule` provides shared loss logging, optimization, and clipping.
`get_trainer_inputs` configures the Trainer and checkpoint callbacks. The
application supplies the model, datamodule, arguments, logger, and output directory.

## Trainer setup

```python
from lightning.pytorch import Trainer
from model_training.lightning import get_trainer_inputs

trainer_kwargs, callbacks = get_trainer_inputs(log_dir, args, logger)
trainer = Trainer(**trainer_kwargs, callbacks=callbacks)
trainer.fit(module, datamodule=dm)
```

The factory returns callbacks separately; pass both outputs to `Trainer`.
Checkpoint files are written under `log_dir`.

## Epoch-based and step-based configuration

Use `-1` to disable the stopping limit you are not using:

| Mode | `max_epochs` | `max_steps` | `step_val_interval` |
| --- | --- | --- | --- |
| Epoch-based | Positive | `-1` | Ignored |
| Step-based with step validation | `-1` | Positive | Positive |
| Step-based with epoch validation | `-1` | Positive | Missing, zero, or negative |

If both stopping limits are positive, the factory emits a warning and passes
`max_steps=-1` to the Trainer. The epoch limit takes precedence. This override
applies to the returned Trainer options; the input `args` is not modified.

For the cadence table below:

- `N = max(1, args.log_interval)` is an interval in epochs.
- `V = args.step_val_interval` is an interval in completed Lightning optimizer steps.
- `S = max(500, int(args.log_every_n_steps))` is the step-based recovery interval.

| Mode | Validation | Best checkpoint | Full `last.ckpt` | At successful completion |
| --- | --- | --- | --- | --- |
| Epoch-based | At the end of every `N` epochs | Considered after each scheduled validation | At the end of every `N` epochs | Save the final full state if it has not already been saved |
| Step-based, `V > 0` | Every `V` optimizer steps, across epoch boundaries | Considered after each validation | Every `S` optimizer steps | No additional validation or checkpoint save is guaranteed |
| Step-based, no positive `V` | At the end of every `N` epochs | Considered after each scheduled validation | Every `S` optimizer steps | No additional validation or checkpoint save is guaranteed |

"Considered" means the monitored metric is compared with its previous best.
The factory keeps one best checkpoint per monitored metric and writes a new
file when the metric improves. The default is `monitor="val_loss"`, `mode="min"`.

Validation requires an enabled, nonempty validation loader. Sanity validation
is disabled. A positive `step_val_interval` only takes effect in step-based mode;
in that case, `log_interval` does not control validation or best-checkpoint timing.
Another callback requesting early stopping may cause an extra validation pass.

### Examples

| Settings | Validation / best-checkpoint checks | `last.ckpt` saves |
| --- | --- | --- |
| `max_epochs=7`, `max_steps=-1`, `log_interval=2` | After epochs 2, 4, and 6 | After epochs 2, 4, 6, and final epoch 7 |
| `max_epochs=-1`, `max_steps=1250`, `step_val_interval=200`, `log_every_n_steps=100` | At steps 200, 400, 600, 800, 1000, and 1200 | At steps 500 and 1000; the saved state can lag the final step 1250 |
| `max_epochs=-1`, `max_steps=1250`, `step_val_interval=-1`, `log_interval=2`, `log_every_n_steps=100` | After every two epochs completed before training stops | At steps 500 and 1000, independently of validation |

These examples assume normal completion without an early-stopping request.
Epoch numbers above count completed epochs starting at 1; `{epoch}` in a
checkpoint filename uses Lightning's epoch index.

Saving the final epoch state does **not** trigger another validation pass.
For the first example, `last.ckpt` contains epoch 7's state, while the best
checkpoint is selected from validation results through epoch 6.

### Step counting and logging

Step intervals use completed Lightning optimizer steps, including gradient
accumulation. They do not count individual microbatches. Step validation handles
partial accumulation windows at epoch boundaries and saves its cadence state
in full checkpoints.

`log_every_n_steps` controls training metric logging and also determines `S`
through the 500-step minimum. Training epoch averages use the batches selected
for logging. Validation metrics include every validation batch.

The stopping mode does not select the LR scheduler's interval. `step` and
`multistep` schedulers advance by epoch; all other enabled schedulers advance
after optimizer steps.

## Checkpoint contents

| Checkpoint | Saved weights | Other state | Purpose |
| --- | --- | --- | --- |
| `last.ckpt` | Live training weights | Optimizer, scheduler, loop and callback state, precision state where applicable, and adaptive clipping history | Full training resume |
| Best checkpoint | Live weights, or EMA weights when EMA is enabled | No optimizer, scheduler, or callback state | Inference, evaluation, or starting a new training run from weights |

The recovery callback overwrites the fixed `last.ckpt` path. It explicitly uses
`filename="last"`; `save_last=False` disables an additional copy, not this file.
It also attempts a full save on exceptions, outside the regular cadence.

Best checkpoints are saved after validation so their weights match the metric
used to rank them. The default filename is
`best-epoch{epoch}-step{step}-val_loss{val_loss:.4f}.ckpt`.

The first checkpoint callback is the recovery callback. To obtain the actual
best path, select the callback for the monitored metric rather than assuming
`trainer.checkpoint_callback` refers to the best-model callback:

```python
best_callback = next(
    cb for cb in trainer.checkpoint_callbacks if cb.monitor == "val_loss"
)
best_path = best_callback.best_model_path
```

## Resuming training or loading weights

The factory does not automatically load a checkpoint. For full resume, construct
the module and Trainer, then pass a full checkpoint explicitly:

```python
from pathlib import Path
from model_training.lightning import fit_trusted_checkpoint, load_training_weights

fit_trusted_checkpoint(trainer, module, dm, ckpt_path=Path(log_dir) / "last.ckpt")
```

This restores model parameters, optimizer moments, scheduler position, training
progress, callback state, and adaptive clipping history. Keep the scheduler
configuration and planned training horizon consistent: lambda schedules rebuild
their functions from the current configuration. Standard dataloaders do not
guarantee restoration of shuffled sample order when resuming mid-epoch.

To load only weights into a constructed module:

```python
load_training_weights(module, best_path)
```

A best checkpoint cannot provide full training resume. For W&B continuity,
also pass the existing `wandb_run_id` and checkpoint path as
`resume_from_checkpoint` to `wandb_logger`. Resuming the logger does not load
model state. The default W&B helper uses `log_model=False`, so checkpoint files
are not uploaded as W&B model artifacts.

## Module metrics and multiple validation loaders

Subclasses implement `compute_loss(batch)`, returning `(loss, loss_dict)`, and
the `metric_count_keys` property, mapping summed metric names to count names.
For example, `{"squared_error": "n_samples"}` produces an average from total
squared error divided by total sample count. Values in `loss_dict` should be
scalar batch totals; tensor values are detached and reduced to scalar means
before logging. See the [basic training example](../../examples/basic_training.py).

| Validation loader configuration | Loss metric names |
| --- | --- |
| One unnamed loader | `val_loss` |
| Multiple unnamed loaders | `val_loader0_loss`, `val_loader1_loss`, ... |
| `dm.val_loader_names = ["iid", "ood"]` | `val_iid_loss`, `val_ood_loss` |

Names follow dataloader order. Missing entries fall back to `loader{index}`.
Each loader has separate totals and averages. Named loaders use their names
even when there is only one loader.

Choose an existing logged metric explicitly when using multiple or named loaders.
For multiple monitored metrics, supply equally sized `monitor`, `mode`, and
`filename` lists:

```python
trainer_kwargs, callbacks = get_trainer_inputs(
    log_dir, args, logger,
    monitor=["val_iid_loss", "val_ood_loss"],
    mode=["min", "min"],
    filename=[
        "best-iid-step{step}-loss{val_iid_loss:.4f}",
        "best-ood-step{step}-loss{val_ood_loss:.4f}",
    ],
)
```

## EMA

With `args.ema=True`, the factory attaches EMA with decay `0.9999`, updates
every optimizer step, and validates using averaged weights. Best checkpoints
contain those averaged weights. Full checkpoints keep live weights together
with optimizer state and EMA averages for consistent resume.

The shared EMA implementation also supports CPU offload and fused AdamW with
AMP. Overflow-skipped optimizer updates do not advance the average. If
evaluation raises while EMA weights are active, live parameters are restored
before exception checkpoints are saved.
