"""Global configuration, dtype definitions, and shared utilities."""

import dataclasses
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import torch
import ml_dtypes


# =============================================================================
# Test Configuration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TestConfig:
    seed: int = 42

    jax_backend: str = "gpu"
    torch_device: str = "cuda"

    dtypes: Tuple[str, ...] = ("float32", "bfloat16", "float8_e4m3fn", "float8_e5m2")

    # Elementwise shapes
    scalar_shapes: Tuple[tuple, ...] = ((),)
    vector_shapes: Tuple[tuple, ...] = ((8,), (16,), (1024,), (4096,))
    matrix_shapes: Tuple[tuple, ...] = ((4, 4), (8, 8), (128, 128), (512, 512))
    higher_dim_shapes: Tuple[tuple, ...] = ((2, 3, 64), (4, 8, 16, 32))

    # Matmul shapes: (lhs_shape, rhs_shape)
    matmul_shapes: Tuple[Tuple[tuple, tuple], ...] = (
        ((4, 8), (8, 4)),
        ((16, 32), (32, 16)),
        ((64, 128), (128, 64)),
        ((128, 256), (256, 128)),
    )
    batch_matmul_shapes: Tuple[Tuple[tuple, tuple], ...] = (
        ((2, 4, 8), (2, 8, 4)),
        ((4, 16, 32), (4, 32, 16)),
    )

    # Conv shapes: list of dicts
    conv_shapes: Tuple[dict, ...] = (
        {"input": (1, 1, 16), "kernel": (1, 1, 3), "strides": (1,), "padding": "SAME"},
        {
            "input": (1, 3, 32, 32),
            "kernel": (16, 3, 3, 3),
            "strides": (1, 1),
            "padding": "SAME",
        },
    )

    # Linalg shapes (square matrices for cholesky, eigh, svd, etc.)
    linalg_shapes: Tuple[tuple, ...] = ((4, 4), (8, 8), (32, 32), (64, 64))

    # Linalg solve shapes: (A_shape, b_shape)
    linalg_solve_shapes: Tuple[Tuple[tuple, tuple], ...] = (
        ((4, 4), (4, 1)),
        ((8, 8), (8, 4)),
        ((32, 32), (32, 8)),
        ((64, 64), (64, 16)),
    )

    # Output directories
    dump_dir: str = "output/dumps"
    report_dir: str = "output/reports"


DEFAULT_CONFIG = TestConfig()


# =============================================================================
# Dtype Definitions
# =============================================================================

DTYPE_MAP = {
    "float32": {
        "jax": jnp.float32,
        "torch": torch.float32,
        "numpy": np.float32,
    },
    "bfloat16": {
        "jax": jnp.bfloat16,
        "torch": torch.bfloat16,
        "numpy": ml_dtypes.bfloat16,
    },
    "float8_e4m3fn": {
        "jax": ml_dtypes.float8_e4m3fn,
        "torch": torch.float8_e4m3fn,
        "numpy": ml_dtypes.float8_e4m3fn,
    },
    "float8_e5m2": {
        "jax": ml_dtypes.float8_e5m2,
        "torch": torch.float8_e5m2,
        "numpy": ml_dtypes.float8_e5m2,
    },
}

NP_DTYPE_MAP = {
    "float32": np.float32,
    "bfloat16": ml_dtypes.bfloat16,
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
}

ATOL_MAP = {
    "float32": 1e-6,
    "bfloat16": 1e-2,
    "float8_e4m3fn": 0.125,
    "float8_e5m2": 0.25,
}


def is_fp8(dtype_key: str) -> bool:
    return dtype_key.startswith("float8")


def torch_supports_direct_compute(dtype_key: str) -> bool:
    """PyTorch does NOT support arithmetic on FP8 dtypes."""
    return not is_fp8(dtype_key)


def get_fp8_compute_dtype():
    """For FP8 ops in PyTorch, compute in this dtype then cast back."""
    return torch.float32


# =============================================================================
# Shared Utilities
# =============================================================================


def to_numpy(x):
    """Convert JAX array or PyTorch tensor to numpy."""
    if isinstance(x, jax.Array):
        return np.array(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_shape_str(shape) -> str:
    """Convert shape to a filesystem-safe string."""
    if isinstance(shape, dict):
        # conv shape
        inp = "x".join(map(str, shape["input"]))
        ker = "x".join(map(str, shape["kernel"]))
        return f"conv_{inp}_k{ker}"
    if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[0], tuple):
        # matmul pair: ((M,K), (K,N))
        lhs = "x".join(map(str, shape[0]))
        rhs = "x".join(map(str, shape[1]))
        return f"{lhs}__{rhs}"
    if isinstance(shape, tuple):
        return "x".join(map(str, shape)) if shape else "scalar"
    return str(shape)
