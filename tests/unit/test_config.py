"""get_shapes_for_op inference logic tests."""

import pytest
from jax_op_diff.config import get_shapes_for_op, TestConfig
from jax_op_diff.op_registry import OpSpec, OpArity


def _make_op(**overrides) -> OpSpec:
    defaults = dict(
        name="test_op",
        category="test",
        arity=OpArity.UNARY,
        jax_fn=lambda x: x,
        shape_type="elementwise",
    )
    defaults.update(overrides)
    return OpSpec(**defaults)


class TestGetShapesForOp:
    config = TestConfig()

    def test_custom_shapes_highest_priority(self):
        op = _make_op(custom_shapes=((2, 2), (3, 3)))
        assert get_shapes_for_op(op, self.config) == [(2, 2), (3, 3)]

    def test_explicit_shape_type_overrides_arity(self):
        op = _make_op(arity=OpArity.UNARY, shape_type="linalg")
        shapes = get_shapes_for_op(op, self.config)
        assert shapes == list(self.config.linalg_shapes)

    def test_reduction_arity_infers_reduction_shapes(self):
        op = _make_op(arity=OpArity.REDUCTION, shape_type=None)
        shapes = get_shapes_for_op(op, self.config)
        expected = list(self.config.vector_shapes + self.config.matrix_shapes
                        + self.config.higher_dim_shapes)
        assert shapes == expected

    def test_conv_arity_infers_conv_shapes(self):
        op = _make_op(arity=OpArity.CONV, shape_type=None)
        shapes = get_shapes_for_op(op, self.config)
        assert shapes == list(self.config.conv_shapes)

    def test_fft_arity_infers_fft_shapes(self):
        op = _make_op(arity=OpArity.FFT, shape_type=None)
        shapes = get_shapes_for_op(op, self.config)
        assert shapes == list(self.config.vector_shapes)

    def test_elementwise_default(self):
        op = _make_op(shape_type="elementwise")
        shapes = get_shapes_for_op(op, self.config)
        expected = list(self.config.scalar_shapes + self.config.vector_shapes
                        + self.config.matrix_shapes + self.config.higher_dim_shapes)
        assert shapes == expected

    def test_all_shape_types_valid(self):
        for st in ("elementwise", "reduction", "matmul", "batch_matmul",
                    "conv", "fft", "linalg", "linalg_solve"):
            op = _make_op(shape_type=st)
            shapes = get_shapes_for_op(op, self.config)
            assert len(shapes) > 0, f"No shapes for shape_type={st}"
