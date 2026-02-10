"""Test execution engine and precision metrics."""

import dataclasses
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import torch

from .op_registry import OpSpec, OpArity, InputDomain, generate_inputs
from .config import TestConfig, DTYPE_MAP, NP_DTYPE_MAP, ATOL_MAP, is_fp8, safe_shape_str


# =============================================================================
# FP8 + CPU pre-check
# =============================================================================

# JAX lax ops that have known FP8 support (typically via promotion or dedicated kernels on GPU).
# On CPU backend, most FP8 arithmetic is unsupported.
_FP8_CPU_UNSUPPORTED_NOTE = (
    "SKIPPED: FP8 dtype ({dtype}) is not supported on CPU backend ({backend}). "
    "FP8 computation requires GPU/TPU hardware support."
)


def _is_cpu_backend(config: TestConfig) -> bool:
    """Check if the effective JAX backend is CPU."""
    if config.jax_backend == "cpu":
        return True
    # Also detect fallback: if requested backend is unavailable, JAX falls back to CPU
    try:
        jax.devices(config.jax_backend)
        return False
    except RuntimeError:
        return True


def _is_cpu_torch_device(config: TestConfig) -> bool:
    """Check if the effective PyTorch device is CPU."""
    if config.torch_device == "cpu":
        return True
    if config.torch_device == "cuda" and not torch.cuda.is_available():
        return True
    return False


def _should_skip_fp8_on_cpu(dtype_key: str, config: TestConfig) -> str | None:
    """Return a skip reason string if FP8 dtype should be skipped on CPU, else None."""
    if not is_fp8(dtype_key):
        return None

    backends = []
    if _is_cpu_backend(config):
        backends.append(f"JAX/{config.jax_backend}->cpu")
    if _is_cpu_torch_device(config):
        backends.append(f"PyTorch/{config.torch_device}->cpu")

    if backends:
        return _FP8_CPU_UNSUPPORTED_NOTE.format(
            dtype=dtype_key, backend=", ".join(backends)
        )
    return None


# =============================================================================
# Precision Result
# =============================================================================


@dataclasses.dataclass
class PrecisionResult:
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


# =============================================================================
# Metrics Computation
# =============================================================================


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
    if j_arr.ndim >= 2:
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


def make_result(op_name: str, category: str, dtype: str, shape: str,
                jax_result: np.ndarray, torch_result: np.ndarray,
                dtype_key: str, notes: str = "") -> PrecisionResult:
    """Compute metrics and build a PrecisionResult."""
    metrics = compute_metrics(jax_result, torch_result, dtype_key)
    atol = ATOL_MAP.get(dtype_key, 1e-6)
    all_close = bool(np.allclose(
        jax_result.astype(np.float64),
        torch_result.astype(np.float64),
        atol=atol, rtol=1e-3,
    ))
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


# =============================================================================
# Array Conversion Helpers
# =============================================================================


def _numpy_to_jax(arr: np.ndarray, dtype_key: str, device=None):
    """Convert numpy array to JAX array on target device."""
    jax_dtype = DTYPE_MAP[dtype_key]["jax"]
    x = jnp.array(arr.astype(np.float32), dtype=jax_dtype)
    if device is not None:
        x = jax.device_put(x, device)
    return x


def _numpy_to_jax_complex(arr: np.ndarray, device=None):
    """Convert complex numpy array to JAX complex array."""
    x = jnp.array(arr)
    if device is not None:
        x = jax.device_put(x, device)
    return x


def _numpy_to_torch(arr: np.ndarray, dtype_key: str, device: str = "cpu"):
    """Convert numpy array to PyTorch tensor on target device."""
    if is_fp8(dtype_key):
        t = torch.tensor(arr.astype(np.float32), dtype=torch.float32, device=device)
        return t
    torch_dtype = DTYPE_MAP[dtype_key]["torch"]
    t = torch.tensor(arr.astype(np.float32), dtype=torch_dtype, device=device)
    return t


def _numpy_to_torch_complex(arr: np.ndarray, device: str = "cpu"):
    """Convert complex numpy array to PyTorch complex tensor."""
    # Ensure complex64
    arr_c = arr.astype(np.complex64)
    return torch.tensor(arr_c, device=device)


# =============================================================================
# Test Execution
# =============================================================================


def execute_single_test(op: OpSpec, shape, dtype_key: str,
                        config: TestConfig, seed: int = 42,
                        ) -> tuple["PrecisionResult", "np.ndarray | None"]:
    """Execute a single precision test for one op, one shape, one dtype.

    Returns (PrecisionResult, jax_result_np_or_None).
    The second element is the raw JAX output as numpy, for dump reuse.
    """
    shape_str = safe_shape_str(shape)

    def _skip_result(**kw):
        """Helper to build (PrecisionResult, None) for skip/error returns."""
        defaults = dict(
            op_name=op.name, category=op.category, dtype=dtype_key,
            shape=shape_str, max_abs_error=0, mean_abs_error=0,
            max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
            mean_ulp_diff=0, all_close=True, jax_has_nan=False,
            torch_has_nan=False, torch_missing=False,
        )
        defaults.update(kw)
        return PrecisionResult(**defaults), None

    # Check dtype support
    if op.supported_dtypes and dtype_key not in op.supported_dtypes:
        return _skip_result(error_msg="SKIPPED: dtype not supported by this op")

    # Check FP8 on CPU: skip with explicit message
    fp8_skip = _should_skip_fp8_on_cpu(dtype_key, config)
    if fp8_skip is not None:
        return _skip_result(error_msg=fp8_skip)

    # Check torch availability
    if op.torch_fn is None:
        return _skip_result(
            torch_missing=True,
            notes=f"MISSING: no PyTorch equivalent. {op.notes}")

    try:
        inputs = generate_inputs(op, shape, dtype_key, seed)
        is_complex = op.input_domain == InputDomain.COMPLEX

        # --- JAX execution ---
        try:
            jax_device = jax.devices(config.jax_backend)[0]
        except RuntimeError:
            jax_device = jax.devices("cpu")[0]

        jax_out = _execute_jax(op, inputs, dtype_key, jax_device, is_complex)
        jax_result = np.array(jax_out)

        # --- PyTorch execution ---
        torch_device = config.torch_device
        if torch_device == "cuda" and not torch.cuda.is_available():
            torch_device = "cpu"

        torch_out = _execute_torch(op, inputs, dtype_key, torch_device, is_complex)

        # For FP8 torch: cast result back to FP8 dtype for fair comparison
        if is_fp8(dtype_key):
            torch_out = torch_out.to(DTYPE_MAP[dtype_key]["torch"])

        torch_out_cpu = torch_out.detach().cpu()
        # bfloat16 and fp8 tensors can't be directly converted to numpy
        try:
            torch_result = torch_out_cpu.numpy()
        except (TypeError, RuntimeError):
            torch_result = torch_out_cpu.float().numpy()

        # --- Compute metrics ---
        # Handle complex results: compare real and imaginary parts separately
        if np.iscomplexobj(jax_result) or np.iscomplexobj(torch_result):
            jax_r = np.real(jax_result).astype(np.float64)
            jax_i = np.imag(jax_result).astype(np.float64)
            torch_r = np.real(torch_result).astype(np.float64)
            torch_i = np.imag(torch_result).astype(np.float64)
            jax_combined = np.concatenate([jax_r.flatten(), jax_i.flatten()])
            torch_combined = np.concatenate([torch_r.flatten(), torch_i.flatten()])
            jax_for_metric = jax_combined.astype(np.float32)
            torch_for_metric = torch_combined.astype(np.float32)
            metric_dtype = "float32"
        else:
            jax_for_metric = jax_result
            torch_for_metric = torch_result
            metric_dtype = dtype_key

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_metrics(jax_for_metric, torch_for_metric, metric_dtype)
            atol = ATOL_MAP.get(dtype_key, 1e-6)
            all_close = bool(np.allclose(
                jax_for_metric.astype(np.float64),
                torch_for_metric.astype(np.float64),
                atol=atol, rtol=1e-3,
            ))

        result = PrecisionResult(
            op_name=op.name, category=op.category, dtype=dtype_key,
            shape=shape_str, all_close=all_close, torch_missing=False,
            notes=op.notes, **metrics)
        return result, jax_result

    except Exception as e:
        return _skip_result(
            all_close=False,
            error_msg=f"ERROR: {type(e).__name__}: {str(e)[:200]}")


def _execute_jax(op: OpSpec, inputs: dict, dtype_key: str, device, is_complex: bool = False):
    """Run the JAX function with given inputs."""
    if op.arity == OpArity.UNARY:
        if is_complex:
            x = _numpy_to_jax_complex(inputs["x"], device)
        else:
            x = _numpy_to_jax(inputs["x"], dtype_key, device)
        return op.jax_fn(x)
    elif op.arity == OpArity.BINARY:
        if is_complex:
            x = _numpy_to_jax_complex(inputs["x"], device)
            y = _numpy_to_jax_complex(inputs["y"], device)
        else:
            x = _numpy_to_jax(inputs["x"], dtype_key, device)
            y = _numpy_to_jax(inputs["y"], dtype_key, device)
        return op.jax_fn(x, y)
    elif op.arity == OpArity.TERNARY:
        lo = _numpy_to_jax(inputs["lo"], dtype_key, device)
        x = _numpy_to_jax(inputs["x"], dtype_key, device)
        hi = _numpy_to_jax(inputs["hi"], dtype_key, device)
        return op.jax_fn(lo, x, hi)
    elif op.arity == OpArity.REDUCTION:
        x = _numpy_to_jax(inputs["x"], dtype_key, device)
        ndim = x.ndim
        axis = ndim - 1 if ndim > 0 else 0
        return op.jax_fn(x, axis)
    elif op.arity == OpArity.MATMUL:
        x = _numpy_to_jax(inputs["x"], dtype_key, device)
        y = _numpy_to_jax(inputs["y"], dtype_key, device)
        return op.jax_fn(x, y)
    elif op.arity == OpArity.FFT:
        x = _numpy_to_jax_complex(inputs["x"], device)
        return op.jax_fn(x)
    elif op.arity == OpArity.CONV:
        inp = _numpy_to_jax(inputs["input"], dtype_key, device)
        ker = _numpy_to_jax(inputs["kernel"], dtype_key, device)
        strides = inputs["strides"]
        padding = inputs["padding"]
        return op.jax_fn(inp, ker, strides, padding)
    elif op.arity == OpArity.TYPE_CAST:
        x = _numpy_to_jax(inputs["x"], dtype_key, device)
        return op.jax_fn(x)
    else:
        raise ValueError(f"Unsupported arity: {op.arity}")


def _execute_torch(op: OpSpec, inputs: dict, dtype_key: str, device: str,
                    is_complex: bool = False):
    """Run the PyTorch function with given inputs."""
    if op.arity == OpArity.UNARY:
        if is_complex:
            x = _numpy_to_torch_complex(inputs["x"], device)
        else:
            x = _numpy_to_torch(inputs["x"], dtype_key, device)
        return op.torch_fn(x)
    elif op.arity == OpArity.BINARY:
        if is_complex:
            x = _numpy_to_torch_complex(inputs["x"], device)
            y = _numpy_to_torch_complex(inputs["y"], device)
        else:
            x = _numpy_to_torch(inputs["x"], dtype_key, device)
            y = _numpy_to_torch(inputs["y"], dtype_key, device)
        return op.torch_fn(x, y)
    elif op.arity == OpArity.TERNARY:
        lo = _numpy_to_torch(inputs["lo"], dtype_key, device)
        x = _numpy_to_torch(inputs["x"], dtype_key, device)
        hi = _numpy_to_torch(inputs["hi"], dtype_key, device)
        return op.torch_fn(lo, x, hi)
    elif op.arity == OpArity.REDUCTION:
        x = _numpy_to_torch(inputs["x"], dtype_key, device)
        ndim = x.ndim
        axis = ndim - 1 if ndim > 0 else 0
        return op.torch_fn(x, axis)
    elif op.arity == OpArity.MATMUL:
        x = _numpy_to_torch(inputs["x"], dtype_key, device)
        y = _numpy_to_torch(inputs["y"], dtype_key, device)
        return op.torch_fn(x, y)
    elif op.arity == OpArity.FFT:
        x = _numpy_to_torch_complex(inputs["x"], device)
        return op.torch_fn(x)
    elif op.arity == OpArity.CONV:
        inp = _numpy_to_torch(inputs["input"], dtype_key, device)
        ker = _numpy_to_torch(inputs["kernel"], dtype_key, device)
        return op.torch_fn(inp, ker)
    elif op.arity == OpArity.TYPE_CAST:
        x = _numpy_to_torch(inputs["x"], dtype_key, device)
        return op.torch_fn(x)
    else:
        raise ValueError(f"Unsupported arity: {op.arity}")
