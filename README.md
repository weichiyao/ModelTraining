# ModelTraining

Reusable training utilities for machine learning projects. Currently under
active development. Requires Python 3.12+.

## Installation

From the repository root, in your target environment:

```bash
pip install -e .
```

Includes PyTorch, Lightning, Transformers, NumPy, and W&B.

## Usage

```python
from model_training.schedules import get_kl_weight
from model_training.lightning import TrainPLModule, EMA, get_trainer_inputs
```

## Structure

```text
model_training/
├── schedules.py       # Shared training schedules
└── lightning/         # Lightning modules, trainer setup, and logging
    └── callbacks/     # EMA and checkpoint callbacks
examples/              # Basic training example
tests/
```

## Development

Version 0.0.2 extracts EF-TALFM's training utilities and fixes confirmed defects
in cosine floors, value clipping, optimizer-step validation, and shared EMA.
Optimizer parameter groups and the SequentialLR checkpoint structure are
preserved.

The trainer factory takes the output directory explicitly; consumers retain
control of their stage-specific paths:

```python
trainer_kwargs, callbacks = get_trainer_inputs(log_dir, args, logger)
```

Subclasses implement `compute_loss` and the `metric_count_keys` property,
which maps summed metric names to their count names. This property was renamed
for clarity; its calculations are unchanged.

`cosine_with_lr_end` retains the original first-update warmup rate of
`base_lr / warmup_steps`, decays to the absolute `lr_end` in each parameter
group, and holds that floor. The hold multiplier now uses the original base
LR, avoiding a floor incorrectly multiplied by the number of warmup steps.
This is a correction to the original EF-TALFM implementation, rather than an
unchanged copy of that implementation. Warmup and cosine decay are unchanged;
optimizer updates differ only if training continues into the final hold phase
after the decay endpoint. With no warmup or just one warmup step, the original
hold floor was already correct.
With the default `num_decay_steps=-1`, decay ends at the training endpoint,
so no optimizer update within that run uses the old incorrect hold floor.
Decay must end after warmup. This schedule advances after optimizer steps,
including gradient accumulation.

With step-limited training (`max_steps > 0`, `max_epochs < 0`), a positive
`step_val_interval` validates every N completed Lightning optimizer steps.
This works across epoch boundaries with partial accumulation windows. Its
callback state is checkpointed to avoid duplicate validation after resume.
Lightning may also validate when another callback requests early stopping.
Monitored checkpoints save after validation, using the fresh metrics. With
no positive step interval, the original epoch cadence remains in effect.
Full resume checkpoints retain the existing periodic cadence (every
`max(500, log_every_n_steps)` optimizer steps for step-limited runs); this
does not guarantee a final checkpoint for runs that stop between saves.

The optional shared EMA implementation supports CPU, CUDA, CPU offload, and
fused AdamW with AMP. Overflow-skipped updates do not advance the average.
Repeated fit setup retains wrapped optimizers, and nested saves during
validation preserve averaged weights. EMA-aware weights-only checkpoints
contain the averages; full checkpoints retain live weights and optimizer
state, including averages, for consistent resume.

The KL schedule remains this package's corrected, bounded implementation.
That was an explicitly selected change in the EF-TALFM integration.

Use `fit_trusted_checkpoint` for full resume or
`load_training_weights(module, path)` for a constructed module. These helpers
use standard PyTorch and Lightning checkpoint loading. Compatibility with
application-specific checkpoint formats belongs in the consuming application.
No EF-TALFM class-path remapping is included.

Full resume restores parameters, Adam moments, scheduler position,
`global_step`, and adaptive clipping history. Keep the same scheduler
configuration and training horizon: lambda schedules reconstruct their
functions from the current config. Original SequentialLR checkpoints remain
loadable. Their saved LR is restored faithfully. For a checkpoint already in
hold, the corrected cosine floor takes effect on the next scheduler update;
otherwise it applies when hold begins. Compatibility with the checkpoint
format does not imply reproducing the old incorrect hold LR indefinitely.
Weights-only checkpoints
cannot restore optimizer or loop state. Standard dataloaders do not guarantee
recovery of shuffled sample order when resuming mid-epoch.

Parity tests store reference traces generated from EF-TALFM's pre-migration
code for unaffected behavior. Independent regression tests cover the corrected
floor, clipping, optimizer-step validation, AMP, and checkpoint contents.
Resume tests exercise warmup, cosine decay, and hold phases, comparing Adam
moments, scheduler state, parameters, step counts, and clipping history.
CUDA cases run when CUDA hardware is available.

```bash
pip install -e ".[dev]"
python -m pytest
```
