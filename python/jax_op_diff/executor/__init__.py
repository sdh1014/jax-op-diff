"""Executor package: test execution engine + precision metrics."""

from .engine import execute_single_test, execute_jax_only, replay_single_case
from .metrics import compute_metrics, compute_all_close, make_result

# Re-export PrecisionResult for backwards compatibility
from ..core import PrecisionResult

__all__ = [
    "execute_single_test",
    "execute_jax_only",
    "replay_single_case",
    "compute_metrics",
    "compute_all_close",
    "make_result",
    "PrecisionResult",
]
