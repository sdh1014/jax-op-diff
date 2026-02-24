"""Test execution engine: JAX/Torch execution and single-test orchestration."""

import ast
import warnings
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import torch

from ..core import CaseData, PrecisionResult
from ..op_registry import OpSpec, OpArity, InputDomain, generate_inputs
from ..config import TestConfig, DTYPE_MAP, NP_DTYPE_MAP, ATOL_MAP, is_fp8, safe_shape_str
from .metrics import compute_metrics, compute_all_close


# =============================================================================
# FP8 + CPU pre-check
# =============================================================================

_FP8_CPU_UNSUPPORTED_NOTE = (
    "ERROR: FP8 dtype ({dtype}) is not supported on CPU backend ({backend}). "
    "FP8 computation requires GPU/TPU hardware support."
)


def _is_cpu_backend(config: TestConfig) -> bool:
    """Check if the effective JAX backend is CPU."""
    return config.jax_backend == "cpu"


def _is_cpu_torch_device(config: TestConfig) -> bool:
    """Check if the effective PyTorch device is CPU."""
    return config.torch_device == "cpu"


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
    torch_dtype = DTYPE_MAP[dtype_key]["torch"]
    t = torch.tensor(arr.astype(np.float32), dtype=torch_dtype, device=device)
    return t


def _numpy_to_torch_complex(arr: np.ndarray, device: str = "cpu"):
    """Convert complex numpy array to PyTorch complex tensor."""
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

    # Check FP8 on CPU: mark as ERROR with explicit message
    fp8_skip = _should_skip_fp8_on_cpu(dtype_key, config)
    if fp8_skip is not None:
        return _skip_result(all_close=False, error_msg=fp8_skip)

    # Check torch availability
    if op.torch_fn is None:
        return _skip_result(
            torch_missing=True,
            notes=f"MISSING: no PyTorch equivalent. {op.notes}")

    try:
        inputs = generate_inputs(op, shape, dtype_key, seed)
        is_complex = op.input_domain == InputDomain.COMPLEX

        # --- JAX execution ---
        jax_device = jax.devices(config.jax_backend)[0]

        jax_out = _execute_jax(op, inputs, dtype_key, jax_device, is_complex)
        jax_result = np.array(jax_out)

        # --- PyTorch execution ---
        torch_device = config.torch_device
        if torch_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"PyTorch device '{torch_device}' is unavailable: CUDA is not available"
            )

        torch_out = _execute_torch(op, inputs, dtype_key, torch_device, is_complex)

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
            all_close = compute_all_close(jax_for_metric, torch_for_metric, dtype_key)

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


def _resolve_torch_callable(op: OpSpec):
    """Resolve callable from unified torch mapping."""
    if op.torch_fn is None:
        raise ValueError(f"No torch mapping for op: {op.name}")
    return op.torch_fn


def _prepare_torch_inputs(inputs: dict, dtype_key: str, device: str,
                          is_complex: bool = False) -> dict:
    """Convert numeric inputs to torch tensors; keep metadata fields unchanged."""
    prepared = {}
    for key, value in inputs.items():
        if isinstance(value, (str, bytes)):
            prepared[key] = value
            continue

        if isinstance(value, np.ndarray) or np.isscalar(value):
            arr = np.asarray(value)
            if is_complex or np.iscomplexobj(arr):
                prepared[key] = _numpy_to_torch_complex(arr, device)
            else:
                prepared[key] = _numpy_to_torch(arr, dtype_key, device)
        else:
            prepared[key] = value
    return prepared


def _execute_torch(op: OpSpec, inputs: dict, dtype_key: str, device: str,
                   is_complex: bool = False):
    """Run the PyTorch function with per-op call plans from registry."""
    torch_call = _resolve_torch_callable(op)
    call_inputs = _prepare_torch_inputs(inputs, dtype_key, device, is_complex)

    builder = op.torch_fn_builder
    if builder is None:
        raise ValueError(f"No torch builder for op: {op.name}")

    args, kwargs = builder(call_inputs)
    out = torch_call(*args, **kwargs)

    if op.torch_output_adapter is not None:
        out = op.torch_output_adapter(out, call_inputs)

    return out


# =============================================================================
# Complex metric handling helper
# =============================================================================


def _prepare_for_metrics(a: np.ndarray, b: np.ndarray, dtype_key: str):
    """Handle complex arrays for metric computation.

    Returns (a_metric, b_metric, metric_dtype).
    """
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        a_r = np.real(a).astype(np.float64).flatten()
        a_i = np.imag(a).astype(np.float64).flatten()
        b_r = np.real(b).astype(np.float64).flatten()
        b_i = np.imag(b).astype(np.float64).flatten()
        return (
            np.concatenate([a_r, a_i]).astype(np.float32),
            np.concatenate([b_r, b_i]).astype(np.float32),
            "float32",
        )
    return a, b, dtype_key


# =============================================================================
# JAX-only execution (for dump mode)
# =============================================================================


def execute_jax_only(op: OpSpec, shape, dtype_key: str,
                     config: TestConfig) -> Optional[np.ndarray]:
    """Execute only JAX, return numpy output. Returns None on failure."""
    try:
        inputs = generate_inputs(op, shape, dtype_key, config.seed)
        is_complex = op.input_domain == InputDomain.COMPLEX
        device = jax.devices(config.jax_backend)[0]
        jax_out = _execute_jax(op, inputs, dtype_key, device, is_complex)
        return np.array(jax_out)
    except Exception:
        return None


# =============================================================================
# Replay from dump
# =============================================================================


def replay_single_case(case: CaseData, op_map: dict,
                       config: TestConfig) -> PrecisionResult:
    """Replay a single dumped case on the target JAX backend.

    Compares stored output (baseline) against freshly computed output.
    """
    try:
        op_name = case.op_name
        category = case.category
        dtype_key = case.dtype_key
        shape_str = case.shape_str

        if op_name not in op_map:
            return PrecisionResult.error(
                op_name, category, dtype_key, shape_str,
                f"ERROR: op '{op_name}' not found in registry")

        op = op_map[op_name]

        # FP8 CPU check
        fp8_skip = _should_skip_fp8_on_cpu(dtype_key, config)
        if fp8_skip is not None:
            return PrecisionResult.error(
                op_name, category, dtype_key, shape_str, fp8_skip)

        stored_output = case.stored_output
        if stored_output.size == 0:
            return PrecisionResult.error(
                op_name, category, dtype_key, shape_str,
                "ERROR: no stored jax_output in dump file")

        inputs = dict(case.inputs)

        # Restore conv metadata from shape string
        if op.arity == OpArity.CONV:
            conv_shape = ast.literal_eval(shape_str)
            inputs["strides"] = tuple(conv_shape["strides"])
            inputs["padding"] = conv_shape["padding"]

        # Validate expected input keys
        expected_keys = {
            OpArity.UNARY: ("x",),
            OpArity.BINARY: ("x", "y"),
            OpArity.TERNARY: ("lo", "x", "hi"),
            OpArity.REDUCTION: ("x",),
            OpArity.MATMUL: ("x", "y"),
            OpArity.FFT: ("x",),
            OpArity.CONV: ("input", "kernel", "strides", "padding"),
            OpArity.TYPE_CAST: ("x",),
        }[op.arity]
        for key in expected_keys:
            if key not in inputs:
                return PrecisionResult.error(
                    op_name, category, dtype_key, shape_str,
                    "SKIPPED: dump missing stored inputs")

        # Execute on target backend
        is_complex = op.input_domain == InputDomain.COMPLEX
        device = jax.devices(config.jax_backend)[0]
        fresh_output = np.array(
            _execute_jax(op, inputs, dtype_key, device, is_complex)
        )

        # Compute metrics
        stored_metric, fresh_metric, metric_dtype = _prepare_for_metrics(
            stored_output, fresh_output, dtype_key)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_metrics(stored_metric, fresh_metric, metric_dtype)
            all_close = compute_all_close(stored_metric, fresh_metric, dtype_key)

        return PrecisionResult(
            op_name=op_name, category=category, dtype=dtype_key,
            shape=shape_str, all_close=all_close, torch_missing=False,
            notes=op.notes, **metrics)

    except Exception as e:
        return PrecisionResult.error(
            case.case_name, "", "", "",
            f"ERROR: {type(e).__name__}: {str(e)[:200]}")
