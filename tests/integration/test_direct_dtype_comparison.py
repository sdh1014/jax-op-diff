"""Strict direct-dtype comparison tests.

Goal: verify what happens if comparison is done strictly in the original
dtype, without promotion to float64.
"""

from dataclasses import dataclass

import numpy as np
import pytest

# ── project imports ──────────────────────────────────────────────────────────
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from jax_op_diff.config import ATOL_MAP, NP_DTYPE_MAP
from jax_op_diff.executor import compute_metrics, compute_all_close
from jax_op_diff.op_registry import generate_inputs, get_all_ops

# make sure ops are registered
import jax_op_diff.ops.all_ops  # noqa: F401


def compute_metrics_direct(jax_result: np.ndarray, torch_result: np.ndarray, dtype_key: str) -> dict:
    """Strict original-dtype arithmetic with no float32/float64 promotion."""
    dtype = NP_DTYPE_MAP[dtype_key]
    j = np.asarray(jax_result).astype(dtype).reshape(-1)
    t = np.asarray(torch_result).astype(dtype).reshape(-1)

    if j.size == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
            "jax_has_nan": False,
            "torch_has_nan": False,
        }

    abs_diff = np.abs(j - t)
    atol = np.array(ATOL_MAP[dtype_key], dtype=dtype)
    denom = np.maximum(np.maximum(np.abs(j), np.abs(t)), atol)
    rel_diff = abs_diff / denom

    return {
        "max_abs_error": float(np.nanmax(abs_diff)),
        "mean_abs_error": float(np.nanmean(abs_diff)),
        "max_rel_error": float(np.nanmax(rel_diff)),
        "mean_rel_error": float(np.nanmean(rel_diff)),
        "jax_has_nan": bool(np.any(np.isnan(j))),
        "torch_has_nan": bool(np.any(np.isnan(t))),
    }


def allclose_direct(jax_result: np.ndarray, torch_result: np.ndarray, dtype_key: str) -> bool:
    """Strict original-dtype allclose with tolerance computed in original dtype."""
    dtype = NP_DTYPE_MAP[dtype_key]
    j = np.asarray(jax_result).astype(dtype).reshape(-1)
    t = np.asarray(torch_result).astype(dtype).reshape(-1)
    atol = np.array(ATOL_MAP[dtype_key], dtype=dtype)
    rtol = np.array(1e-3, dtype=dtype)
    tol = atol + rtol * np.abs(t)
    return bool(np.all(np.abs(j - t) <= tol))


def allclose_f64(jax_result: np.ndarray, torch_result: np.ndarray, dtype_key: str) -> bool:
    j = np.asarray(jax_result).astype(np.float64).reshape(-1)
    t = np.asarray(torch_result).astype(np.float64).reshape(-1)
    return bool(np.allclose(j, t, atol=ATOL_MAP[dtype_key], rtol=1e-3))


@pytest.mark.parametrize(
    "dtype_key,j_val,t_val",
    [
        ("bfloat16", -0.01507568359375, -1.8984375),
        ("float8_e4m3fn", -7.5, -1.25),
        ("float8_e5m2", -0.00341796875, -0.015625),
    ],
)
def test_direct_metrics_smaller_with_same_dtype_inputs(dtype_key, j_val, t_val):
    """Both sides start in the same low-precision dtype; direct arithmetic can hide error."""
    dtype = NP_DTYPE_MAP[dtype_key]
    j = np.array([j_val], dtype=dtype)
    t = np.array([t_val], dtype=dtype)

    m_f64 = compute_metrics(j, t, dtype_key)
    m_direct = compute_metrics_direct(j, t, dtype_key)

    assert m_f64["max_abs_error"] > 0.0
    assert m_direct["max_abs_error"] < m_f64["max_abs_error"]


def test_fp8_allclose_verdict_flips_with_same_dtype_inputs():
    """With same-dtype fp8 inputs, float64 can fail while direct allclose passes."""
    dtype = NP_DTYPE_MAP["float8_e4m3fn"]
    j = np.array([0.125], dtype=dtype)
    t = np.array([-0.00390625], dtype=dtype)

    assert allclose_f64(j, t, "float8_e4m3fn") is False
    assert allclose_direct(j, t, "float8_e4m3fn") is True


def test_float32_direct_close_to_f64_metrics():
    """For float32, direct and float64 metrics are typically very close."""
    rng = np.random.RandomState(0)
    j = rng.uniform(-2.0, 2.0, size=4096).astype(np.float32)
    t = (j.astype(np.float64) + rng.uniform(-1e-6, 1e-6, size=4096)).astype(np.float32)

    m_f64 = compute_metrics(j, t, "float32")
    m_direct = compute_metrics_direct(j, t, "float32")

    diff = abs(m_f64["max_abs_error"] - m_direct["max_abs_error"])
    assert diff < 1e-7


def test_discrete_verdict_requires_exact_match():
    """Discrete outputs (e.g. argmax indices) must be exact match."""
    j = np.array([1000], dtype=np.int32)
    t = np.array([1001], dtype=np.int32)
    assert compute_all_close(j, t, "float32") is False


def test_float_verdict_keeps_allclose_behavior():
    """Floating outputs still use tolerance-based allclose."""
    j = np.array([1000.0], dtype=np.float32)
    t = np.array([1001.0], dtype=np.float32)
    assert compute_all_close(j, t, "float32") is True


_FAST_OP_NAMES = {"add", "mul", "broadcasted_iota"}
_TEST_DTYPES = ["float32", "bfloat16"]
_TEST_SHAPES = [(128,), (4, 8)]


def _get_test_ops():
    return [
        op
        for op in get_all_ops()
        if op.name in _FAST_OP_NAMES and op.torch_fn is not None
    ]


def _run_op_pair(op, shape, dtype_key, seed=42):
    import jax

    from jax_op_diff.executor import _execute_jax, _execute_torch

    inputs = generate_inputs(op, shape, dtype_key, seed)
    jax_device = jax.devices("cpu")[0]
    jax_out = _execute_jax(op, inputs, dtype_key, jax_device, is_complex=False)
    torch_out = _execute_torch(op, inputs, dtype_key, "cpu", is_complex=False)

    jax_result = np.array(jax_out)
    torch_out_cpu = torch_out.detach().cpu()
    try:
        torch_result = torch_out_cpu.numpy()
    except (TypeError, RuntimeError):
        torch_result = torch_out_cpu.float().numpy()

    return jax_result, torch_result


@dataclass
class ComparisonRow:
    op_name: str
    dtype: str
    f64_max_abs: float
    direct_max_abs: float


def test_get_test_ops_is_really_fast_subset():
    ops = _get_test_ops()
    assert len(ops) > 0
    assert all(op.name in _FAST_OP_NAMES for op in ops)


def test_broadcasted_iota_uses_explicit_builder_and_adapter():
    op = next(op for op in get_all_ops() if op.name == "broadcasted_iota")
    assert op.torch_fn is not None
    assert op.torch_fn_builder is not None
    assert op.torch_output_adapter is not None


def test_real_ops_direct_not_larger_than_f64():
    rows: list[ComparisonRow] = []
    for op in _get_test_ops():
        for dtype_key in _TEST_DTYPES:
            if op.supported_dtypes and dtype_key not in op.supported_dtypes:
                continue
            for shape in _TEST_SHAPES:
                jax_res, torch_res = _run_op_pair(op, shape, dtype_key)
                m_f64 = compute_metrics(jax_res, torch_res, dtype_key)
                m_direct = compute_metrics_direct(jax_res, torch_res, dtype_key)
                rows.append(
                    ComparisonRow(
                        op_name=op.name,
                        dtype=dtype_key,
                        f64_max_abs=m_f64["max_abs_error"],
                        direct_max_abs=m_direct["max_abs_error"],
                    )
                )

    assert len(rows) > 0
    for row in rows:
        assert row.direct_max_abs <= row.f64_max_abs + 1e-12
