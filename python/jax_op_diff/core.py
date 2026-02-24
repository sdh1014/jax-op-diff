"""Cross-module shared data types. Depends only on stdlib + numpy."""

import dataclasses
from typing import NamedTuple, Optional, Set

import numpy as np


@dataclasses.dataclass(frozen=True)
class RunFilters:
    """Runtime filter criteria. Built by cli, consumed by pipeline / dump_store."""
    categories: Optional[Set[str]] = None
    op_names: Optional[Set[str]] = None


@dataclasses.dataclass
class PrecisionResult:
    """Precision result for a single (op, shape, dtype) test case."""
    op_name: str
    category: str
    dtype: str
    shape: str
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    max_ulp_diff: float
    mean_ulp_diff: float
    all_close: bool
    jax_has_nan: bool
    torch_has_nan: bool
    torch_missing: bool
    matrix_rel_fro_error: float = 0.0
    error_msg: str = ""
    notes: str = ""

    @classmethod
    def error(cls, op_name: str, category: str, dtype: str,
              shape: str, msg: str) -> "PrecisionResult":
        """Build an error/skip result with zeroed metric fields."""
        return cls(
            op_name=op_name, category=category, dtype=dtype, shape=shape,
            max_abs_error=0, mean_abs_error=0, max_rel_error=0,
            mean_rel_error=0, max_ulp_diff=0, mean_ulp_diff=0,
            all_close=False, jax_has_nan=False, torch_has_nan=False,
            torch_missing=False, error_msg=msg,
        )

    @classmethod
    def missing(cls, op_name: str, category: str, dtype: str,
                shape: str, notes: str = "") -> "PrecisionResult":
        """Build a torch_missing result."""
        return cls(
            op_name=op_name, category=category, dtype=dtype, shape=shape,
            max_abs_error=0, mean_abs_error=0, max_rel_error=0,
            mean_rel_error=0, max_ulp_diff=0, mean_ulp_diff=0,
            all_close=True, jax_has_nan=False, torch_has_nan=False,
            torch_missing=True, notes=notes,
        )


class CaseData(NamedTuple):
    """Single case read from HDF5 dump, passed to replay_single_case."""
    case_name: str
    op_name: str
    category: str
    dtype_key: str
    shape_str: str
    arity: str
    inputs: dict
    stored_output: np.ndarray
