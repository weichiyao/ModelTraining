"""Shared training utilities; Lightning helpers live in ``model_training.lightning``."""

from .schedules import get_kl_weight

__all__ = ["get_kl_weight"]
