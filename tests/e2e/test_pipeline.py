"""End-to-end pipeline tests. Requires JAX + Torch (GPU optional)."""

import os
import pytest

from jax_op_diff.config import TestConfig
from jax_op_diff.core import RunFilters, PrecisionResult
from jax_op_diff import pipeline


# Use a small config for fast e2e tests
_SMALL_CONFIG = TestConfig(
    seed=42,
    jax_backend="cpu",
    torch_device="cpu",
    dtypes=("float32",),
    scalar_shapes=((),),
    vector_shapes=((8,),),
    matrix_shapes=((4, 4),),
    higher_dim_shapes=(),
    matmul_shapes=(((4, 8), (8, 4)),),
    batch_matmul_shapes=(),
    conv_shapes=(),
    linalg_shapes=((4, 4),),
    linalg_solve_shapes=(((4, 4), (4, 1)),),
)


@pytest.fixture
def small_config(tmp_path):
    return TestConfig(
        seed=_SMALL_CONFIG.seed,
        jax_backend=_SMALL_CONFIG.jax_backend,
        torch_device=_SMALL_CONFIG.torch_device,
        dtypes=_SMALL_CONFIG.dtypes,
        scalar_shapes=_SMALL_CONFIG.scalar_shapes,
        vector_shapes=_SMALL_CONFIG.vector_shapes,
        matrix_shapes=_SMALL_CONFIG.matrix_shapes,
        higher_dim_shapes=_SMALL_CONFIG.higher_dim_shapes,
        matmul_shapes=_SMALL_CONFIG.matmul_shapes,
        batch_matmul_shapes=_SMALL_CONFIG.batch_matmul_shapes,
        conv_shapes=_SMALL_CONFIG.conv_shapes,
        linalg_shapes=_SMALL_CONFIG.linalg_shapes,
        linalg_solve_shapes=_SMALL_CONFIG.linalg_solve_shapes,
        dump_dir=str(tmp_path / "dumps"),
        report_dir=str(tmp_path / "reports"),
    )


@pytest.fixture(autouse=True)
def _register_ops():
    """Ensure ops are registered."""
    import jax_op_diff.ops.all_ops  # noqa: F401


class TestRunCompare:
    def test_produces_results(self, small_config):
        filters = RunFilters(categories={"basic"})
        results = pipeline.run_compare(small_config, filters, dump=False)
        assert len(results) > 0
        assert all(isinstance(r, PrecisionResult) for r in results)

    def test_dump_creates_file(self, small_config):
        filters = RunFilters(categories={"basic"})
        pipeline.run_compare(small_config, filters, dump=True)
        dump_path = os.path.join(small_config.dump_dir, "dump.h5")
        assert os.path.exists(dump_path)


class TestRunJaxOnly:
    def test_produces_dump_stats(self, small_config):
        filters = RunFilters(categories={"basic"})
        stats = pipeline.run_jax_only(small_config, filters)
        assert stats.total > 0
        assert stats.dumped > 0
        assert stats.errors == 0


class TestRunReplay:
    def test_compare_then_replay(self, small_config):
        filters = RunFilters(categories={"basic"})
        # First: compare with dump
        pipeline.run_compare(small_config, filters, dump=True)
        # Then: replay from that dump
        results = pipeline.run_replay(small_config, filters, small_config.dump_dir)
        assert len(results) > 0
        assert all(isinstance(r, PrecisionResult) for r in results)

    def test_jax_only_then_replay(self, small_config):
        filters = RunFilters(categories={"basic"})
        # First: jax-only dump
        pipeline.run_jax_only(small_config, filters)
        # Then: replay
        results = pipeline.run_replay(small_config, filters, small_config.dump_dir)
        assert len(results) > 0


class TestRunJaxPrecision:
    def test_produces_results(self, small_config):
        filters = RunFilters(categories={"basic"})
        results = pipeline.run_jax_precision(small_config, filters, dump=False)
        assert len(results) > 0
        assert all(isinstance(r, PrecisionResult) for r in results)

    def test_cpu_vs_cpu_float32_all_close(self, small_config):
        """CPU fp32 ground truth vs CPU fp32 actual should be identical."""
        filters = RunFilters(categories={"basic"})
        results = pipeline.run_jax_precision(small_config, filters, dump=False)
        real = [r for r in results if not r.error_msg]
        assert len(real) > 0
        for r in real:
            assert r.all_close, f"{r.op_name} ({r.dtype}) not all_close"
            assert r.max_abs_error == 0.0

    def test_dump_creates_file(self, small_config):
        filters = RunFilters(categories={"basic"})
        pipeline.run_jax_precision(small_config, filters, dump=True)
        dump_path = os.path.join(small_config.dump_dir, "dump.h5")
        assert os.path.exists(dump_path)

    def test_dump_then_replay(self, small_config):
        filters = RunFilters(categories={"basic"})
        # First: jax-precision with dump
        original = pipeline.run_jax_precision(small_config, filters, dump=True)
        # Then: replay from that dump
        replayed = pipeline.run_jax_precision_replay(
            small_config, filters, small_config.dump_dir)
        assert len(replayed) > 0
        assert len(replayed) == len([r for r in original if not r.error_msg])

    def test_unsupported_dtype_no_crash(self, small_config):
        """FP8 on CPU should not crash — either succeeds (auto-upcast) or errors gracefully."""
        fp8_config = TestConfig(
            seed=small_config.seed,
            jax_backend="cpu",
            torch_device="cpu",
            dtypes=("float8_e4m3fn",),
            scalar_shapes=(),
            vector_shapes=((8,),),
            matrix_shapes=(),
            higher_dim_shapes=(),
            matmul_shapes=(),
            batch_matmul_shapes=(),
            conv_shapes=(),
            linalg_shapes=(),
            linalg_solve_shapes=(),
            dump_dir=str(small_config.dump_dir),
            report_dir=str(small_config.report_dir),
        )
        filters = RunFilters(op_names={"add"})
        results = pipeline.run_jax_precision(fp8_config, filters, dump=False)
        assert len(results) > 0
        assert all(isinstance(r, PrecisionResult) for r in results)
