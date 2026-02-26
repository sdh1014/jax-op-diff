import numpy as np

import jax_op_diff.ops.all_ops  # noqa: F401
from jax_op_diff.op_registry import generate_inputs, get_all_ops


def test_betainc_inputs_match_semantic_domains():
    op = next(op for op in get_all_ops() if op.name == "betainc")
    inputs = generate_inputs(op, (1024,), "float32", seed=42)

    a = inputs["lo"]
    b = inputs["x"]
    x = inputs["hi"]

    assert np.all(a > 0)
    assert np.all(b > 0)
    assert np.all(x >= 0)
    assert np.all(x <= 1)
