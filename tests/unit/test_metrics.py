"""Unit tests for executor/metrics.py. Pure numpy, no JAX/Torch/GPU needed."""

import numpy as np
import pytest
import ml_dtypes

from jax_op_diff.executor.metrics import compute_metrics, compute_all_close


class TestComputeMetrics:
    def test_identical_arrays_zero_error(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = compute_metrics(a, a, "float32")
        assert m["max_abs_error"] == 0.0
        assert m["mean_abs_error"] == 0.0
        assert m["max_rel_error"] == 0.0
        assert m["mean_rel_error"] == 0.0
        assert m["max_ulp_diff"] == 0.0
        assert m["mean_ulp_diff"] == 0.0
        assert m["jax_has_nan"] is False
        assert m["torch_has_nan"] is False

    def test_empty_arrays(self):
        a = np.array([], dtype=np.float32)
        m = compute_metrics(a, a, "float32")
        assert m["max_abs_error"] == 0.0
        assert m["jax_has_nan"] is False

    def test_nan_in_jax(self):
        j = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = compute_metrics(j, t, "float32")
        assert m["jax_has_nan"] is True
        assert m["torch_has_nan"] is False

    def test_nan_in_torch(self):
        j = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        m = compute_metrics(j, t, "float32")
        assert m["jax_has_nan"] is False
        assert m["torch_has_nan"] is True

    def test_known_abs_error(self):
        j = np.array([1.0, 2.0], dtype=np.float32)
        t = np.array([1.5, 2.0], dtype=np.float32)
        m = compute_metrics(j, t, "float32")
        assert abs(m["max_abs_error"] - 0.5) < 1e-6
        assert abs(m["mean_abs_error"] - 0.25) < 1e-6

    def test_scalar_input(self):
        j = np.float32(1.0)
        t = np.float32(2.0)
        m = compute_metrics(np.array(j), np.array(t), "float32")
        assert m["max_abs_error"] == 1.0

    def test_bfloat16_metrics(self):
        dtype = ml_dtypes.bfloat16
        j = np.array([1.0], dtype=dtype)
        t = np.array([1.0], dtype=dtype)
        m = compute_metrics(j, t, "bfloat16")
        assert m["max_abs_error"] == 0.0

    def test_fp8_e4m3fn_metrics(self):
        dtype = ml_dtypes.float8_e4m3fn
        j = np.array([1.0], dtype=dtype)
        t = np.array([1.5], dtype=dtype)
        m = compute_metrics(j, t, "float8_e4m3fn")
        assert m["max_abs_error"] > 0.0

    def test_fp8_e5m2_metrics(self):
        dtype = ml_dtypes.float8_e5m2
        j = np.array([1.0], dtype=dtype)
        t = np.array([2.0], dtype=dtype)
        m = compute_metrics(j, t, "float8_e5m2")
        assert m["max_abs_error"] > 0.0

    def test_matrix_frobenius_error(self):
        j = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        t = np.array([[1.1, 0.0], [0.0, 1.0]], dtype=np.float32)
        m = compute_metrics(j, t, "float32")
        assert m["matrix_rel_fro_error"] > 0.0

    def test_frobenius_zero_for_identical(self):
        j = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        m = compute_metrics(j, j, "float32")
        assert m["matrix_rel_fro_error"] == 0.0

    def test_ulp_diff_nonzero_for_different_values(self):
        j = np.array([1.0], dtype=np.float32)
        # Next representable float32 after 1.0
        t = np.nextafter(np.float32(1.0), np.float32(2.0))
        t = np.array([t], dtype=np.float32)
        m = compute_metrics(j, t, "float32")
        assert m["max_ulp_diff"] > 0.0
        # Should be close to 1 ULP
        assert m["max_ulp_diff"] < 2.0

    def test_metrics_dict_keys(self):
        j = np.array([1.0], dtype=np.float32)
        m = compute_metrics(j, j, "float32")
        expected_keys = {
            "max_abs_error", "mean_abs_error", "max_rel_error", "mean_rel_error",
            "max_ulp_diff", "mean_ulp_diff", "matrix_rel_fro_error",
            "jax_has_nan", "torch_has_nan",
        }
        assert set(m.keys()) == expected_keys


class TestComputeAllClose:
    def test_identical_float_arrays(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert compute_all_close(a, a, "float32") is True

    def test_different_shapes(self):
        j = np.array([1.0, 2.0], dtype=np.float32)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert compute_all_close(j, t, "float32") is False

    def test_discrete_exact_match_required(self):
        j = np.array([1, 2, 3], dtype=np.int32)
        t = np.array([1, 2, 3], dtype=np.int32)
        assert compute_all_close(j, t, "float32") is True

    def test_discrete_mismatch_fails(self):
        j = np.array([1, 2, 3], dtype=np.int32)
        t = np.array([1, 2, 4], dtype=np.int32)
        assert compute_all_close(j, t, "float32") is False

    def test_bool_exact_match(self):
        j = np.array([True, False, True])
        t = np.array([True, False, True])
        assert compute_all_close(j, t, "float32") is True

    def test_bool_mismatch_fails(self):
        j = np.array([True, False, True])
        t = np.array([True, True, True])
        assert compute_all_close(j, t, "float32") is False

    def test_float_within_tolerance(self):
        j = np.array([1000.0], dtype=np.float32)
        t = np.array([1000.0 + 1e-7], dtype=np.float32)
        assert compute_all_close(j, t, "float32") is True

    def test_float_outside_tolerance(self):
        j = np.array([1.0], dtype=np.float32)
        t = np.array([2.0], dtype=np.float32)
        assert compute_all_close(j, t, "float32") is False

    def test_bfloat16_tolerance(self):
        dtype = ml_dtypes.bfloat16
        j = np.array([1.0], dtype=dtype)
        # bfloat16 has atol=1e-2, so 1.0 vs 1.005 should pass
        t = np.array([1.0 + 0.005], dtype=dtype)
        assert compute_all_close(j, t, "bfloat16") is True

    def test_fp8_large_tolerance(self):
        dtype = ml_dtypes.float8_e4m3fn
        j = np.array([1.0], dtype=dtype)
        t = np.array([1.0], dtype=dtype)
        # Same value should always pass
        assert compute_all_close(j, t, "float8_e4m3fn") is True

    def test_empty_arrays(self):
        j = np.array([], dtype=np.float32)
        t = np.array([], dtype=np.float32)
        assert compute_all_close(j, t, "float32") is True
