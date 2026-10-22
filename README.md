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

```bash
pip install -e ".[dev]"
python -m pytest
```
