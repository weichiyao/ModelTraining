"""Exercise logger setup and resume behavior without contacting W&B."""

import importlib
import pickle
from argparse import Namespace
from unittest.mock import Mock

import pytest

wandb_module = importlib.import_module("model_training.lightning.wandb")


@pytest.fixture
def logger_class(monkeypatch):
    monkeypatch.setattr(wandb_module.wandb, "Settings", Mock())
    logger_class = Mock()
    logger_class.return_value.experiment.id = "newrun123"
    monkeypatch.setattr(wandb_module, "WandbLogger", logger_class)
    monkeypatch.setattr(wandb_module, "is_rank_zero_process", lambda: True)
    return logger_class


def test_new_run_returns_logger_without_optional_args(tmp_path, logger_class):
    logger = wandb_module.wandb_logger("team", str(tmp_path), "run", "project")

    assert logger is logger_class.return_value
    assert "resume" not in logger_class.call_args.kwargs
    assert (tmp_path / "wandb_run_id.txt").read_text().strip() == "newrun123"


@pytest.mark.parametrize("explicit_id", [None, "explicit123"])
def test_resume_uses_explicit_or_saved_run_id(tmp_path, logger_class, explicit_id):
    (tmp_path / "wandb_run_id.txt").write_text("saved123\n")
    wandb_module.wandb_logger(
        "team",
        str(tmp_path),
        "run",
        "project",
        wandb_run_id=explicit_id,
        resume_from_checkpoint="last.ckpt",
    )

    kwargs = logger_class.call_args.kwargs
    assert kwargs["id"] == (explicit_id or "saved123")
    assert kwargs["resume"] == "must"


def test_optional_args_are_saved_when_requested(tmp_path, logger_class):
    args = Namespace(lr=0.01)
    wandb_module.wandb_logger(
        "team",
        str(tmp_path),
        "run",
        "project",
        args=args,
        args_folder=str(tmp_path),
    )

    assert args.wandb_run_id == "newrun123"
    with (tmp_path / "args.pickle").open("rb") as stream:
        assert pickle.load(stream).wandb_run_id == "newrun123"


def test_args_without_args_folder(tmp_path, logger_class):
    args = Namespace()
    wandb_module.wandb_logger(
        "team",
        str(tmp_path),
        "run",
        "project",
        args=args,
    )

    assert args.wandb_run_id == "newrun123"


def test_disabled_logging_does_not_resume_or_initialize_experiment(
    tmp_path, logger_class
):
    logger = wandb_module.wandb_logger(
        "team",
        str(tmp_path),
        "run",
        "project",
        wandb_run_id="old123",
        resume_from_checkpoint="last.ckpt",
        no_wandb=True,
    )

    assert logger is logger_class.return_value
    assert logger_class.call_args.kwargs["mode"] == "disabled"
    assert "resume" not in logger_class.call_args.kwargs
    assert not (tmp_path / "wandb_run_id.txt").exists()
