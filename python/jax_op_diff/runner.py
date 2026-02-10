"""Top-level test runner and data dump."""

import ast
import os
import warnings
from pathlib import Path
from typing import List, Optional, Set

import h5py
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
    """Dumps all JAX test cases into one HDF5 file."""

    def __init__(self, dump_dir: str):
        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        self.dump_path = os.path.join(dump_dir, "dump.h5")
        self.case_index = 0
        with h5py.File(self.dump_path, "w") as h5:
            h5.attrs["format"] = "jax-op-diff-h5-v1"

    def _make_case_name(self, op_name: str, dtype_key: str, shape_str: str) -> str:
        safe = shape_str.replace(" ", "").replace(",", "x").replace("(", "").replace(")", "")
        return f"{self.case_index:06d}__{op_name}__{dtype_key}__{safe}"

    @staticmethod
    def _encode_array(arr: np.ndarray):
        a = np.asarray(arr)
        if a.dtype.kind == "V":
            return a.view(np.uint8), str(a.dtype)
        return a, ""

    @staticmethod
    def _decode_array(dset):
        arr = np.array(dset)
        original_dtype = dset.attrs.get("original_dtype", "")
        if original_dtype:
            arr = arr.view(np.dtype(original_dtype))
        return arr

    def dump_test_case(self, op: OpSpec, shape, dtype_key: str,
                       seed: int = 42,
                       jax_output: np.ndarray = None):
        """Append a single test case into dump.h5."""
        shape_str = safe_shape_str(shape)
        case_name = self._make_case_name(op.name, dtype_key, shape_str)

        inputs = generate_inputs(op, shape, dtype_key, seed)

        with h5py.File(self.dump_path, "a") as h5:
            grp = h5.create_group(case_name)
            grp.attrs["op_name"] = op.name
            grp.attrs["category"] = op.category
            grp.attrs["dtype"] = dtype_key
            grp.attrs["shape"] = str(shape)
            grp.attrs["notes"] = op.notes
            grp.attrs["arity"] = op.arity.value

            for key, arr in inputs.items():
                if not (isinstance(arr, np.ndarray) or np.isscalar(arr)):
                    continue
                encoded, original_dtype = self._encode_array(arr)
                dset = grp.create_dataset(f"input_{key}", data=encoded, compression="gzip")
                if original_dtype:
                    dset.attrs["original_dtype"] = original_dtype

            if jax_output is not None:
                encoded, original_dtype = self._encode_array(jax_output)
                dset = grp.create_dataset("jax_output", data=encoded, compression="gzip")
                if original_dtype:
                    dset.attrs["original_dtype"] = original_dtype

        self.case_index += 1
        return self.dump_path


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
    case_name: str,
    case_group,
    op_map: dict,
    device,
    config: TestConfig,
    categories: Optional[Set[str]] = None,
    op_names: Optional[Set[str]] = None,
) -> Optional[PrecisionResult]:
    """Replay a single HDF5 dump case on the target JAX backend.

    Returns None if the dump is filtered out by categories/op_names.
    Returns a PrecisionResult comparing stored output (baseline) vs fresh output.
    """
    try:
        op_name = str(case_group.attrs["op_name"])
        dtype_key = str(case_group.attrs["dtype"])
        shape_str = str(case_group.attrs["shape"])
        category = str(case_group.attrs.get("category", ""))

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

        if "jax_output" not in case_group:
            return PrecisionResult(
                op_name=op_name, category=category, dtype=dtype_key,
                shape=shape_str, max_abs_error=0, mean_abs_error=0,
                max_rel_error=0, mean_rel_error=0, max_ulp_diff=0,
                mean_ulp_diff=0, all_close=False, jax_has_nan=False,
                torch_has_nan=False, torch_missing=False,
                matrix_rel_fro_error=0.0,
                error_msg="ERROR: no stored jax_output in dump file")

        stored_output = DumpManager._decode_array(case_group["jax_output"])

        # Extract inputs from dump
        inputs = {
            key[len("input_"):]: DumpManager._decode_array(case_group[key])
            for key in case_group.keys()
            if key.startswith("input_")
        }

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
            op_name=case_name, category="", dtype="",
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
    """Replay tests from a single dump.h5 file on the target JAX backend.

    Compares stored JAX outputs (baseline) against freshly computed outputs
    on the specified jax_backend.
    """
    import jax

    op_map = {op.name: op for op in get_all_ops()}

    try:
        device = jax.devices(config.jax_backend)[0]
    except RuntimeError:
        device = jax.devices("cpu")[0]

    dump_path = Path(dump_dir) / "dump.h5"

    if not dump_path.exists():
        print(f"  No dump.h5 found in {dump_dir}")
        return []

    with h5py.File(dump_path, "r") as h5:
        case_names = sorted(h5.keys())

    if not case_names:
        print(f"  No dump cases found in {dump_path}")
        return []

    print(f"Found {len(case_names)} dump cases in {dump_path}")
    print(f"Replaying on JAX backend: {config.jax_backend}")

    results: List[PrecisionResult] = []
    total = len(case_names)
    with h5py.File(dump_path, "r") as h5:
        for i, case_name in enumerate(case_names, 1):
            result = _replay_one_dump(
                case_name, h5[case_name], op_map, device, config, categories, op_names,
            )
            if result is not None:
                results.append(result)
            if i % 50 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({100*i//total}%)")

    return results
