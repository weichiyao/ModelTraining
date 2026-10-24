import os 
from lightning.pytorch.utilities.rank_zero import rank_zero_only

def is_rank_zero_process() -> bool:
    for env_name in ("RANK", "SLURM_PROCID"):
        env_value = os.environ.get(env_name)
        if env_value not in (None, ""):
            try:
                return int(env_value) == 0
            except ValueError:
                pass

    rank = getattr(rank_zero_only, "rank", None)
    if rank is not None:
        try:
            return int(rank) == 0
        except (TypeError, ValueError):
            return False

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, ""):
        try:
            return int(local_rank) == 0
        except ValueError:
            pass

    return True
