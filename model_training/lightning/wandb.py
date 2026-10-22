import os
import re
from argparse import Namespace

import wandb
from lightning.pytorch.loggers import WandbLogger

from .utils import is_rank_zero_process, persist_args_pickle

_WANDB_RUN_DIR_RE = re.compile(r"^(?:offline-)?run-[^-]+-([A-Za-z0-9]+)$")


def extract_wandb_run_id_from_name(name: str):
    match = _WANDB_RUN_DIR_RE.match(name)
    return match.group(1) if match else None


def find_existing_wandb_run_id(wandb_dir: str):
    if (wandb_dir is None) or (not os.path.isdir(wandb_dir)):
        return None

    run_id_path = os.path.join(wandb_dir, "wandb_run_id.txt")
    if os.path.isfile(run_id_path):
        try:
            with open(run_id_path, "r", encoding="utf-8") as f:
                run_id = f.read().strip()
            if run_id:
                return run_id
        except OSError:
            pass

    latest_run = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_run):
        latest_run_name = os.path.basename(os.path.realpath(latest_run))
        run_id = extract_wandb_run_id_from_name(latest_run_name)
        if run_id is not None:
            return run_id

    candidates = []
    try:
        for entry in os.scandir(wandb_dir):
            if not entry.is_dir():
                continue
            run_id = extract_wandb_run_id_from_name(entry.name)
            if run_id is not None:
                candidates.append((entry.stat().st_mtime, run_id))
    except OSError:
        return None

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def persist_wandb_run_id(wandb_dir: str, run_id: str):
    if (wandb_dir is None) or (run_id is None):
        return

    try:
        os.makedirs(wandb_dir, exist_ok=True)
        with open(
            os.path.join(wandb_dir, "wandb_run_id.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(run_id)
            f.write("\n")
    except OSError:
        pass


def coerce_wandb_run_id(run_id):
    if callable(run_id):
        try:
            run_id = run_id()
        except TypeError:
            return None
    if run_id is None:
        return None
    if isinstance(run_id, bytes):
        run_id = run_id.decode("utf-8", errors="ignore")
    if not isinstance(run_id, str):
        run_id = str(run_id)
    run_id = run_id.strip()
    return run_id or None


def wandb_logger(
    wandb_usr: str,
    wandb_dir: str,
    wandb_expname: str,
    wandb_project: str,
    wandb_run_id: str | int | None = None,
    resume_from_checkpoint: str | None = "none",
    no_wandb: bool = False,
    wandb_online: bool = True,
    args: Namespace | None = None,
    args_folder: str | None = None,
) -> WandbLogger:
    """Create a W&B logger and optionally resume a checkpoint's run."""
    if no_wandb:
        mode = "disabled"
    else:
        mode = "online" if wandb_online else "offline"
    os.makedirs(wandb_dir, exist_ok=True)

    should_resume_wandb = (not no_wandb) and (
        resume_from_checkpoint not in (None, "none")
    )
    resume_wandb_run_id = (
        coerce_wandb_run_id(wandb_run_id) if should_resume_wandb else None
    )
    if should_resume_wandb and resume_wandb_run_id is None:
        resume_wandb_run_id = find_existing_wandb_run_id(wandb_dir)
    if should_resume_wandb:
        if resume_wandb_run_id is not None:
            print(f"Resuming W&B run {resume_wandb_run_id} from '{wandb_dir}'.")
        else:
            print(
                f"Checkpoint resume was requested, but no existing W&B run ID was found in "
                f"'{wandb_dir}'. Starting a new W&B run."
            )
    kwargs = {
        "entity": wandb_usr,
        "name": wandb_expname,
        "project": wandb_project,
        "config": args,
        "settings": wandb.Settings(
            x_disable_stats=False,
            init_timeout=120,
            x_stats_sampling_interval=120,
            x_save_requirements=False,
        ),
        "reinit": "finish_previous",
        "mode": mode,
        "dir": wandb_dir,
        "save_dir": wandb_dir,
    }
    if resume_wandb_run_id is not None:
        kwargs["id"] = resume_wandb_run_id
        kwargs["resume"] = "must"
    logger = WandbLogger(log_model=False, **kwargs)
    if (not no_wandb) and is_rank_zero_process():
        experiment = logger.experiment
        run_id = coerce_wandb_run_id(getattr(experiment, "id", None))
        if run_id is not None:
            persist_wandb_run_id(wandb_dir, run_id)
            if args is not None:
                args.wandb_run_id = run_id
                if args_folder is not None:
                    persist_args_pickle(os.path.join(args_folder, "args.pickle"), args)
    return logger
