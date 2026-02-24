"""Precision metrics computation. Pure numpy, no JAX/Torch dependency."""

import numpy as np

from ..core import PrecisionResult
from ..config import NP_DTYPE_MAP, ATOL_MAP


def compute_metrics(jax_result: np.ndarray, torch_result: np.ndarray,
                    dtype_key: str) -> dict:
    """Compute precision metrics between JAX and PyTorch results.

    Both inputs should be numpy arrays. Computation is done in float64.
    """
    j_arr = np.asarray(jax_result)
    t_arr = np.asarray(torch_result)
    j = j_arr.astype(np.float64).flatten()
    t = t_arr.astype(np.float64).flatten()

    if j.size == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
            "max_ulp_diff": 0.0,
            "mean_ulp_diff": 0.0,
            "jax_has_nan": False,
            "torch_has_nan": False,
        }

    abs_diff = np.abs(j - t)

    matrix_rel_fro_error = 0.0
    if j_arr.ndim >= 1:
        delta = (j_arr.astype(np.float64) - t_arr.astype(np.float64)).reshape(-1)
        numer = np.linalg.norm(delta)
        denom_fro = np.linalg.norm(j_arr.astype(np.float64).reshape(-1))
        denom_fro = max(float(denom_fro), float(ATOL_MAP.get(dtype_key, 1e-6)))
        matrix_rel_fro_error = float(numer / denom_fro)

    # Relative error: |j - t| / max(|j|, |t|, eps)
    # Use max of both values to avoid inf when one side ≈ 0
    denom = np.maximum(np.maximum(np.abs(j), np.abs(t)),
                       np.float64(ATOL_MAP.get(dtype_key, 1e-6)))
    rel_diff = abs_diff / denom

    # ULP difference in the original dtype
    # spacing must be computed in orig_dtype, NOT float64, otherwise
    # the float64 ULP (~2.2e-16) is used instead of e.g. bfloat16 ULP (~7.8e-3)
    orig_dtype = NP_DTYPE_MAP.get(dtype_key, np.float32)
    jax_in_orig = jax_result.astype(orig_dtype).flatten()
    torch_in_orig = torch_result.astype(orig_dtype).flatten()
    ref_vals = np.maximum(np.abs(jax_in_orig), np.abs(torch_in_orig))
    try:
        ulp_at_point = np.abs(np.spacing(ref_vals.astype(orig_dtype))).astype(np.float64)
    except (TypeError, ValueError):
        # Fallback for dtypes where np.spacing is unsupported (e.g. fp8):
        # approximate ULP via float32 spacing scaled by mantissa-bit ratio
        f32_vals = ref_vals.astype(np.float32)
        ulp_at_point = np.abs(np.spacing(f32_vals)).astype(np.float64)
        # Scale: float32 has 23 mantissa bits; estimate orig mantissa bits from ATOL
        # For fp8_e4m3fn (3 bits) scale = 2^(23-3) = 2^20; for fp8_e5m2 (2 bits) = 2^21
        if dtype_key == "float8_e4m3fn":
            ulp_at_point *= 2.0 ** 20
        elif dtype_key == "float8_e5m2":
            ulp_at_point *= 2.0 ** 21
    ulp_at_point = np.maximum(ulp_at_point, np.finfo(np.float64).tiny)
    ulp_diff = abs_diff / ulp_at_point

    return {
        "max_abs_error": float(np.nanmax(abs_diff)),
        "mean_abs_error": float(np.nanmean(abs_diff)),
        "max_rel_error": float(np.nanmax(rel_diff)),
        "mean_rel_error": float(np.nanmean(rel_diff)),
        "max_ulp_diff": float(np.nanmax(ulp_diff)),
        "mean_ulp_diff": float(np.nanmean(ulp_diff)),
        "matrix_rel_fro_error": matrix_rel_fro_error,
        "jax_has_nan": bool(np.any(np.isnan(j))),
        "torch_has_nan": bool(np.any(np.isnan(t))),
    }


def compute_all_close(jax_result: np.ndarray, torch_result: np.ndarray, dtype_key: str) -> bool:
    """Compute verdict with dtype-aware semantics.

    Discrete outputs (bool/int) require exact match; floating outputs use allclose.
    """
    j_arr = np.asarray(jax_result)
    t_arr = np.asarray(torch_result)
    if j_arr.shape != t_arr.shape:
        return False

    if (j_arr.dtype.kind in ("b", "i", "u")) or (t_arr.dtype.kind in ("b", "i", "u")):
        return bool(np.array_equal(j_arr, t_arr))

    atol = ATOL_MAP.get(dtype_key, 1e-6)
    return bool(np.allclose(
        j_arr.astype(np.float64),
        t_arr.astype(np.float64),
        atol=atol, rtol=1e-3,
    ))


def make_result(op_name: str, category: str, dtype: str, shape: str,
                jax_result: np.ndarray, torch_result: np.ndarray,
                dtype_key: str, notes: str = "") -> PrecisionResult:
    """Compute metrics and build a PrecisionResult."""
    metrics = compute_metrics(jax_result, torch_result, dtype_key)
    all_close = compute_all_close(jax_result, torch_result, dtype_key)
    return PrecisionResult(
        op_name=op_name,
        category=category,
        dtype=dtype,
        shape=shape,
        all_close=all_close,
        torch_missing=False,
        notes=notes,
        **metrics,
    )
