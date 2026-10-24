import argparse
import pickle


def merge_resume_args(
        saved_args: argparse.Namespace,
        current_args: argparse.Namespace,
    ) -> argparse.Namespace:
    """Merge CLI overrides while preserving checkpoint-shaped VAE options."""
    saved = vars(saved_args).copy()
    current = vars(current_args).copy() 
    saved.update(current)
    return argparse.Namespace(**saved)


def persist_args_pickle(args_path: str, args_namespace: argparse.Namespace):
    if args_path is None:
        return

    try:
        with open(args_path, "wb") as f:
            pickle.dump(args_namespace, f)
    except OSError:
        pass

