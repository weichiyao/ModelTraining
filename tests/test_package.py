"""Check the installed API from a separate project's working directory."""

import subprocess
import sys
import textwrap


def test_schedules_import_without_training_frameworks(tmp_path):
    code = textwrap.dedent("""\
        import builtins
        import sys

        frameworks = {"lightning", "torch", "transformers", "wandb"}
        original_import = builtins.__import__
        def without_frameworks(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name.split(".", 1)[0] in frameworks:
                raise AssertionError(f"Shared schedules imported {name}")
            return original_import(name, globals, locals, fromlist, level)
        builtins.__import__ = without_frameworks

        from model_training import get_kl_weight
        from model_training.schedules import get_kl_weight as schedule

        assert get_kl_weight is schedule
        assert schedule(10, 100, method="linear") == 0.5
        assert not frameworks.intersection(sys.modules)
    """)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_import_from_another_project(tmp_path):
    # These common names must not intercept the library's internal imports.
    for name in ("modules", "callbacks", "model_loading"):
        (tmp_path / f"{name}.py").write_text("raise RuntimeError('import collision')\n")

    code = textwrap.dedent("""\
        from model_training import get_kl_weight
        from model_training.lightning import (
            EMA, EMAModelCheckpoint, EMAOptimizer, TrainPLModule,
            fit_trusted_checkpoint, get_trainer_inputs, wandb_logger,
        )
        from model_training.lightning.callbacks import EMA as CallbackEMA
        from model_training.lightning.utils import is_rank_zero_process

        assert CallbackEMA is EMA
        assert get_kl_weight(10, 100, method="linear") == 0.5
        assert callable(is_rank_zero_process)
        assert callable(wandb_logger)
    """)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
