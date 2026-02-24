"""OpSpec registration schema validation tests."""

import pytest
from jax_op_diff.op_registry import (
    OpSpec, OpArity, InputDomain, register,
    _REGISTRY, _validate_op_schema,
)


def _make_spec(**overrides) -> OpSpec:
    """Create a minimal valid OpSpec, with overrides."""
    defaults = dict(
        name="test_op",
        category="test",
        arity=OpArity.UNARY,
        jax_fn=lambda x: x,
        shape_type="elementwise",
    )
    defaults.update(overrides)
    return OpSpec(**defaults)


class TestSchemaValidation:
    def setup_method(self):
        """Save registry state before each test."""
        self._saved = list(_REGISTRY)

    def teardown_method(self):
        """Restore registry state after each test."""
        _REGISTRY.clear()
        _REGISTRY.extend(self._saved)

    def test_valid_spec_passes(self):
        register(_make_spec())

    def test_invalid_shape_type_fails(self):
        with pytest.raises(ValueError, match="unknown shape_type"):
            register(_make_spec(shape_type="elemntwise"))

    def test_invalid_dtype_in_supported_dtypes_fails(self):
        with pytest.raises(ValueError, match="unknown dtypes"):
            register(_make_spec(supported_dtypes=("fp16",)))

    def test_builder_without_torch_fn_fails(self):
        with pytest.raises(ValueError, match="torch_fn_builder.*torch_fn is None"):
            register(_make_spec(torch_fn_builder=lambda d: ((), {})))

    def test_adapter_without_torch_fn_fails(self):
        with pytest.raises(ValueError, match="torch_output_adapter.*torch_fn is None"):
            register(_make_spec(torch_output_adapter=lambda x, d: x))

    def test_input_keys_count_mismatch_fails(self):
        with pytest.raises(ValueError, match="torch_input_keys"):
            register(_make_spec(
                arity=OpArity.UNARY,
                torch_input_keys=("x", "y"),
            ))

    def test_valid_dtypes_pass(self):
        register(_make_spec(supported_dtypes=("float32", "bfloat16")))

    def test_none_shape_type_passes(self):
        # None shape_type is valid (deferred to get_shapes_for_op)
        register(_make_spec(shape_type=None))

    def test_torch_fn_with_builder_passes(self):
        register(_make_spec(
            torch_fn=lambda x: x,
            torch_fn_builder=lambda d: ((d["x"],), {}),
        ))
