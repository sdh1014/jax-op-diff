"""Top-level test runner and data dump."""

import ast
import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Set

import ml_dtypes
import numpy as np

from .config import TestConfig, DEFAULT_CONFIG, ATOL_MAP, safe_shape_str
from .op_registry import get_all_ops, OpSpec, OpArity, InputDomain, generate_inputs
from .executor import (
    execute_single_test, PrecisionResult, _execute_jax,  # _execute_jax used by replay
    compute_metrics, _should_skip_fp8_on_cpu,
)


# =============================================================================
# Data Dump
# =============================================================================


class DumpManager:
    """Dumps JAX results to .npz files for cross-backend comparison."""

    def __init__(self, dump_dir: str):
        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)

    def _make_filename(self, op_name: str, dtype_key: str, shape_str: str) -> str:
        safe = shape_str.replace(" ", "").replace(",", "x").replace("(", "").replace(")", "")
        return f"{op_name}__{dtype_key}__{safe}"

    def dump_test_case(self, op: OpSpec, shape, dtype_key: str,
                       seed: int = 42,
                       jax_output: np.ndarray = None):
        """Save a single test case to .npz.

        Contents:
          - input_x, input_y, ... : input arrays
          - jax_output: JAX result (if provided)
          - metadata_json: JSON string with op metadata
        """
        shape_str = safe_shape_str(shape)
        base_name = self._make_filename(op.name, dtype_key, shape_str)
        filepath = os.path.join(self.dump_dir, base_name + ".npz")

        inputs = generate_inputs(op, shape, dtype_key, seed)

        save_dict = {}
        for key, arr in inputs.items():
            if isinstance(arr, np.ndarray):
                save_dict[f"input_{key}"] = arr
            elif np.isscalar(arr):
                save_dict[f"input_{key}"] = np.asarray(arr)
            # skip non-array items (strides, padding for conv)

        if jax_output is not None:
            save_dict["jax_output"] = jax_output

        metadata = {
            "op_name": op.name,
            "category": op.category,
            "dtype": dtype_key,
            "shape": str(shape),
            "notes": op.notes,
            "arity": op.arity.value,
        }
        save_dict["metadata_json"] = np.array(json.dumps(metadata))

        np.savez_compressed(filepath, **save_dict)
        return filepath


# =============================================================================
# Test Runner
# =============================================================================


def get_shapes_for_op(op: OpSpec, config: TestConfig) -> list:
    """Return applicable shapes for an operator."""
    if op.shape_type == "matmul":
        return list(config.matmul_shapes)
    elif op.shape_type == "batch_matmul":
        return list(config.batch_matmul_shapes)
    elif op.shape_type == "conv":
        return list(config.conv_shapes)
    elif op.shape_type == "fft":
        return list(config.vector_shapes)
    elif op.shape_type == "reduction":
        # Reductions need at least 1D
        return list(config.vector_shapes + config.matrix_shapes + config.higher_dim_shapes)
    elif op.shape_type == "linalg":
        return list(config.linalg_shapes)
    elif op.shape_type == "linalg_solve":
        return list(config.linalg_solve_shapes)
    else:  # elementwise
        return list(
            config.scalar_shapes + config.vector_shapes +
            config.matrix_shapes + config.higher_dim_shapes
        )


def get_dtypes_for_op(op: OpSpec, config: TestConfig) -> list:
    """Return applicable dtypes for an operator."""
    if op.supported_dtypes:
        return [d for d in op.supported_dtypes if d in config.dtypes]
    return list(config.dtypes)


def run_all_tests(
    config: TestConfig = None,
    categories: Optional[Set[str]] = None,
    op_names: Optional[Set[str]] = None,
    dump: bool = True,
) -> List[PrecisionResult]:
    """Run all registered operator tests across all shapes and dtypes."""
    if config is None:
        config = DEFAULT_CONFIG

    results: List[PrecisionResult] = []
    dumper = DumpManager(config.dump_dir) if dump else None

    ops = get_all_ops()

    # Filter by category
    if categories:
        ops = [op for op in ops if op.category in categories]

    # Filter by op name
    if op_names:
        ops = [op for op in ops if op.name in op_names]

    total = sum(
        len(get_shapes_for_op(op, config)) * len(get_dtypes_for_op(op, config))
        for op in ops
    )
    done = 0

    print(f"Running {total} test cases across {len(ops)} operators...")

    for op in ops:
        shapes = get_shapes_for_op(op, config)
        dtypes = get_dtypes_for_op(op, config)

        for dtype_key in dtypes:
            for shape in shapes:
                result, jax_output = execute_single_test(
                    op, shape, dtype_key, config, config.seed)
                results.append(result)

                # Dump JAX results — reuse jax_output from test (no recompute)
                if dumper and jax_output is not None and not result.error_msg:
                    try:
                        dumper.dump_test_case(
                            op, shape, dtype_key, config.seed, jax_output)
                    except Exception:
                        pass  # dump failure is non-critical

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({100*done//total}%)")

    return results


# =============================================================================
# Replay from Dumps
# =============================================================================


def _replay_one_dump(
    npz_path: Path,
    op_map: dict,
    device,
    config: TestConfig,
    categories: Optional[Set[str]] = None,
    op_names: Optional[Set[str]] = None,
) -> Optional[PrecisionResult]:
    """Replay a single .npz dump on the target JAX backend.

    Returns None if the dump is filtered out by categories/op_names.
    Returns a PrecisionResult comparing stored output (baseline) vs fresh output.
    """
    try:
        with np.load(npz_path) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            op_name = metadata["op_name"]
            dtype_key = metadata["dtype"]
            shape_str = metadata["shape"]
            category = metadata.get("category", "")

            # Apply filters early before any computation
            if categories and category not in categories:
                return None
            if op_names and op_name not in op_names:
                return None

            if op_name not in op_map:
                return PrecisionResult(
                    op_name=op_name, category=category, dtype=dtype_key,
                    shape=shape_str, max_abs_error=0, mean_abs_error=0,
                    max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
                    mean_ulp_diff=0, all_close=False, jax_has_nan=False,
                    torch_has_nan=False, torch_missing=False,
                    matrix_rel_fro_error=0.0,
                    error_msg=f"ERROR: op '{op_name}' not found in registry")

            op = op_map[op_name]

            # FP8 CPU check
            fp8_skip = _should_skip_fp8_on_cpu(dtype_key, config)
            if fp8_skip is not None:
                return PrecisionResult(
                    op_name=op_name, category=category, dtype=dtype_key,
                    shape=shape_str, max_abs_error=0, mean_abs_error=0,
                    max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
                    mean_ulp_diff=0, all_close=True, jax_has_nan=False,
                    torch_has_nan=False, torch_missing=False,
                    matrix_rel_fro_error=0.0,
                    error_msg=fp8_skip)

            if "jax_output" not in data.files:
                return PrecisionResult(
                    op_name=op_name, category=category, dtype=dtype_key,
                    shape=shape_str, max_abs_error=0, mean_abs_error=0,
                    max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
                    mean_ulp_diff=0, all_close=False, jax_has_nan=False,
                    torch_has_nan=False, torch_missing=False,
                    matrix_rel_fro_error=0.0,
                    error_msg="ERROR: no stored jax_output in dump file")

            stored_output = np.array(data["jax_output"])
            if stored_output.dtype.kind == "V":
                stored_output = stored_output.view(ml_dtypes.bfloat16)

            # Extract inputs from dump
            inputs = {
                key[len("input_"):]: np.array(data[key])
                for key in data.files
                if key.startswith("input_")
            }

            for key, value in list(inputs.items()):
                if isinstance(value, np.ndarray) and value.dtype.kind == "V":
                    inputs[key] = value.view(ml_dtypes.bfloat16)
            if op.arity == OpArity.CONV:
                conv_shape = ast.literal_eval(shape_str)
                inputs["strides"] = tuple(conv_shape["strides"])
                inputs["padding"] = conv_shape["padding"]

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
                    return PrecisionResult(
                        op_name=op_name, category=category, dtype=dtype_key,
                        shape=shape_str, max_abs_error=0, mean_abs_error=0,
                        max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
                        mean_ulp_diff=0, all_close=True, jax_has_nan=False,
                        torch_has_nan=False, torch_missing=False,
                        matrix_rel_fro_error=0.0,
                        error_msg="SKIPPED: dump missing stored inputs")

            # Execute on target backend
            is_complex = op.input_domain == InputDomain.COMPLEX
            fresh_output = np.array(
                _execute_jax(op, inputs, dtype_key, device, is_complex)
            )

            # Compute metrics (stored = baseline, fresh = replay target)
            if np.iscomplexobj(stored_output) or np.iscomplexobj(fresh_output):
                s_r = np.real(stored_output).astype(np.float64).flatten()
                s_i = np.imag(stored_output).astype(np.float64).flatten()
                f_r = np.real(fresh_output).astype(np.float64).flatten()
                f_i = np.imag(fresh_output).astype(np.float64).flatten()
                stored_metric = np.concatenate([s_r, s_i]).astype(np.float32)
                fresh_metric = np.concatenate([f_r, f_i]).astype(np.float32)
                metric_dtype = "float32"
            else:
                stored_metric = stored_output
                fresh_metric = fresh_output
                metric_dtype = dtype_key

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = compute_metrics(stored_metric, fresh_metric, metric_dtype)
                atol = ATOL_MAP.get(dtype_key, 1e-6)
                all_close = bool(np.allclose(
                    stored_metric.astype(np.float64),
                    fresh_metric.astype(np.float64),
                    atol=atol, rtol=1e-3,
                ))

            return PrecisionResult(
                op_name=op_name, category=category, dtype=dtype_key,
                shape=shape_str, all_close=all_close, torch_missing=False,
                notes=op.notes, **metrics)

    except Exception as e:
        return PrecisionResult(
            op_name=npz_path.stem, category="", dtype="",
            shape="", max_abs_error=0, mean_abs_error=0,
            max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
            mean_ulp_diff=0, all_close=False, jax_has_nan=False,
            torch_has_nan=False, torch_missing=False,
            matrix_rel_fro_error=0.0,
            error_msg=f"ERROR: {type(e).__name__}: {str(e)[:200]}")


def run_replay_tests(
    config: TestConfig,
    dump_dir: str,
    categories: Optional[Set[str]] = None,
    op_names: Optional[Set[str]] = None,
) -> List[PrecisionResult]:
    """Replay tests from existing .npz dump files on the target JAX backend.

    Compares stored JAX outputs (baseline) against freshly computed outputs
    on the specified jax_backend.
    """
    import jax

    op_map = {op.name: op for op in get_all_ops()}

    try:
        device = jax.devices(config.jax_backend)[0]
    except RuntimeError:
        device = jax.devices("cpu")[0]

    dump_path = Path(dump_dir)
    npz_files = sorted(dump_path.glob("*.npz"))

    if not npz_files:
        print(f"  No .npz files found in {dump_dir}")
        return []

    print(f"Found {len(npz_files)} dump files in {dump_dir}")
    print(f"Replaying on JAX backend: {config.jax_backend}")

    results: List[PrecisionResult] = []
    total = len(npz_files)
    for i, npz_path in enumerate(npz_files, 1):
        result = _replay_one_dump(
            npz_path, op_map, device, config, categories, op_names,
        )
        if result is not None:
            results.append(result)
        if i % 50 == 0 or i == total:
            print(f"  Progress: {i}/{total} ({100*i//total}%)")

    return results
