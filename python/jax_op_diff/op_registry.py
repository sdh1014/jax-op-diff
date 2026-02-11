"""Central operator registry with jax.lax -> torch mapping, and input generation."""

import dataclasses
import hashlib
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .config import NP_DTYPE_MAP


class OpArity(Enum):
    UNARY = "unary"
    BINARY = "binary"
    TERNARY = "ternary"
    REDUCTION = "reduction"
    MATMUL = "matmul"
    CONV = "conv"
    FFT = "fft"
    TYPE_CAST = "type_cast"


class InputDomain(Enum):
    """Constrains random input generation to valid domains."""
    REAL = "real"                 # any real number
    POSITIVE = "positive"         # (0, inf)
    UNIT = "unit"                 # (-1, 1)
    UNIT_CLOSED = "unit_closed"   # [0, 1]
    NON_ZERO = "non_zero"        # excludes zero
    SMALL_POSITIVE = "small_pos"  # (0, 10) -- for exp-family
    ABOVE_ONE = "above_one"       # [1, inf) -- for acosh
    COMPLEX = "complex"
    POSITIVE_DEFINITE = "positive_definite"  # SPD matrix for cholesky, eigh
    LOWER_TRIANGULAR = "lower_triangular"    # triangular system for triangular_solve


@dataclasses.dataclass
class AtenSpec:
    name: str
    overload: str = "default"


@dataclasses.dataclass
class OpSpec:
    name: str
    category: str
    arity: OpArity
    jax_fn: Callable
    torch_fn: Optional[Callable]
    input_domain: InputDomain = InputDomain.REAL
    shape_type: str = "elementwise"  # elementwise / reduction / matmul / batch_matmul / conv / fft
    supported_dtypes: Optional[Tuple[str, ...]] = None  # None = all default dtypes
    notes: str = ""
    torch_aten: Optional[AtenSpec] = None
    torch_input_keys: Optional[Tuple[str, ...]] = None
    torch_aten_builder: Optional[Callable[[dict], tuple[tuple, dict]]] = None
    torch_fn_builder: Optional[Callable[[dict], tuple[tuple, dict]]] = None
    torch_output_adapter: Optional[Callable[[Any, dict], Any]] = None


_REGISTRY: List[OpSpec] = []


def _default_torch_input_keys(arity: OpArity) -> Tuple[str, ...]:
    if arity in (OpArity.UNARY, OpArity.FFT, OpArity.TYPE_CAST, OpArity.REDUCTION):
        return ("x",)
    if arity in (OpArity.BINARY, OpArity.MATMUL):
        return ("x", "y")
    if arity == OpArity.TERNARY:
        return ("lo", "x", "hi")
    if arity == OpArity.CONV:
        return ("input", "kernel")
    return ()


def _default_torch_builder(arity: OpArity, input_keys: Tuple[str, ...]) -> Callable[[dict], tuple[tuple, dict]]:
    if arity == OpArity.REDUCTION:
        def build(call_inputs: dict) -> tuple[tuple, dict]:
            x = call_inputs["x"]
            axis = x.ndim - 1 if x.ndim > 0 else 0
            return (x, axis), {}

        return build

    def build(call_inputs: dict) -> tuple[tuple, dict]:
        return tuple(call_inputs[key] for key in input_keys), {}

    return build


def _identity_output(x: Any, _call_inputs: dict) -> Any:
    return x


def register(spec: OpSpec) -> OpSpec:
    if spec.torch_input_keys is None:
        spec.torch_input_keys = _default_torch_input_keys(spec.arity)
    if spec.torch_aten_builder is None:
        spec.torch_aten_builder = _default_torch_builder(spec.arity, spec.torch_input_keys)
    if spec.torch_fn_builder is None:
        spec.torch_fn_builder = _default_torch_builder(spec.arity, spec.torch_input_keys)
    if spec.torch_output_adapter is None:
        spec.torch_output_adapter = _identity_output

    _REGISTRY.append(spec)
    return spec




def op_spec(name: str, category: str, arity: OpArity, *,
            torch_fn: Optional[Callable] = None,
            input_domain: InputDomain = InputDomain.REAL,
            shape_type: str = "elementwise",
            supported_dtypes: Optional[Tuple[str, ...]] = None,
            notes: str = "",
            torch_aten: Optional[AtenSpec] = None,
            torch_input_keys: Optional[Tuple[str, ...]] = None,
            torch_aten_builder: Optional[Callable[[dict], tuple[tuple, dict]]] = None,
            torch_fn_builder: Optional[Callable[[dict], tuple[tuple, dict]]] = None,
            torch_output_adapter: Optional[Callable[[Any, dict], Any]] = None):
    """Decorator: register function as OpSpec jax_fn."""
    def decorator(jax_fn: Callable) -> OpSpec:
        return register(OpSpec(
            name=name,
            category=category,
            arity=arity,
            jax_fn=jax_fn,
            torch_fn=torch_fn,
            input_domain=input_domain,
            shape_type=shape_type,
            supported_dtypes=supported_dtypes,
            notes=notes,
            torch_aten=torch_aten,
            torch_input_keys=torch_input_keys,
            torch_aten_builder=torch_aten_builder,
            torch_fn_builder=torch_fn_builder,
            torch_output_adapter=torch_output_adapter,
        ))

    return decorator

def get_all_ops() -> List[OpSpec]:
    return list(_REGISTRY)


def get_ops_by_category(category: str) -> List[OpSpec]:
    return [op for op in _REGISTRY if op.category == category]


def get_categories() -> List[str]:
    seen = set()
    result = []
    for op in _REGISTRY:
        if op.category not in seen:
            seen.add(op.category)
            result.append(op.category)
    return result


# =============================================================================
# Deterministic input data generation
# =============================================================================


def _make_rng(seed: int, op_name: str, dtype_key: str, shape) -> np.random.RandomState:
    """Create a deterministic RNG from (seed, op_name, dtype, shape)."""
    key = f"{seed}|{op_name}|{dtype_key}|{shape}".encode("utf-8")
    h = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return np.random.RandomState(h)


def _raw(rng: np.random.RandomState, shape: tuple) -> np.ndarray:
    """Generate raw float64 random data."""
    if not shape:
        return np.float64(rng.randn())
    return rng.randn(*shape).astype(np.float64)


def _constrain(data: np.ndarray, domain: InputDomain) -> np.ndarray:
    """Constrain data to the valid input domain."""
    if domain == InputDomain.REAL:
        return data
    elif domain == InputDomain.POSITIVE:
        return np.abs(data) + 0.01
    elif domain == InputDomain.UNIT:
        return np.tanh(data)  # R -> (-1, 1)
    elif domain == InputDomain.UNIT_CLOSED:
        return 1.0 / (1.0 + np.exp(-data))  # sigmoid -> (0, 1)
    elif domain == InputDomain.NON_ZERO:
        return np.where(np.abs(data) < 0.01, np.where(data >= 0, 0.01, -0.01), data)
    elif domain == InputDomain.SMALL_POSITIVE:
        return np.abs(data).clip(0.01, 8.0)
    elif domain == InputDomain.ABOVE_ONE:
        return np.abs(data) + 1.0  # [1, inf)
    return data


def _cast(data: np.ndarray, dtype_key: str) -> np.ndarray:
    """Cast float64 data to target dtype."""
    return data.astype(NP_DTYPE_MAP[dtype_key])


def generate_inputs(op: OpSpec, shape, dtype_key: str, seed: int = 42) -> dict:
    """Generate deterministic inputs for a given operator.

    Returns dict with keys like 'x', 'y', 'z' as numpy arrays in target dtype.
    For matmul: shape is (lhs_shape, rhs_shape).
    For conv: shape is a dict with 'input', 'kernel', etc.
    """
    rng = _make_rng(seed, op.name, dtype_key, shape)
    domain = op.input_domain

    # Positive definite matrix: A @ A.T + scale * I
    if domain == InputDomain.POSITIVE_DEFINITE:
        n = shape[0]
        raw = _raw(rng, shape)
        spd = raw @ raw.T + np.eye(n) * n * 0.1
        return {"x": _cast(spd, dtype_key)}

    # Lower triangular system for triangular_solve (MATMUL arity)
    if domain == InputDomain.LOWER_TRIANGULAR:
        lhs_shape, rhs_shape = shape
        raw_a = _raw(rng, lhs_shape)
        a = np.tril(raw_a)
        np.fill_diagonal(a, np.abs(np.diag(a)) + 1.0)  # non-singular
        raw_b = _raw(rng, rhs_shape)
        return {"x": _cast(a, dtype_key), "y": _cast(raw_b, dtype_key)}

    # Complex inputs
    if domain == InputDomain.COMPLEX:
        if isinstance(shape, tuple) and len(shape) > 0 and not isinstance(shape[0], tuple):
            s = shape
        elif isinstance(shape, tuple) and len(shape) == 0:
            s = ()  # scalar
        else:
            s = (16,)  # fallback
        if s:
            real_part = rng.randn(*s).astype(np.float32)
            imag_part = rng.randn(*s).astype(np.float32)
        else:
            real_part = np.float32(rng.randn())
            imag_part = np.float32(rng.randn())
        x = real_part + 1j * imag_part
        return {"x": x}

    # Matmul shapes
    if op.arity == OpArity.MATMUL:
        lhs_shape, rhs_shape = shape
        raw_x = _constrain(_raw(rng, lhs_shape), domain)
        raw_y = _constrain(_raw(rng, rhs_shape), domain)
        return {"x": _cast(raw_x, dtype_key), "y": _cast(raw_y, dtype_key)}

    # Conv shapes
    if op.arity == OpArity.CONV:
        inp_shape = shape["input"]
        ker_shape = shape["kernel"]
        raw_inp = _constrain(_raw(rng, inp_shape), domain)
        raw_ker = _constrain(_raw(rng, ker_shape), domain)
        return {
            "input": _cast(raw_inp, dtype_key),
            "kernel": _cast(raw_ker, dtype_key),
            "strides": shape["strides"],
            "padding": shape["padding"],
        }

    # Standard elementwise / reduction / fft shapes
    if isinstance(shape, tuple) and shape and isinstance(shape[0], tuple):
        # Shouldn't happen for non-matmul, but handle gracefully
        shape = shape[0]

    if op.arity == OpArity.UNARY or op.arity == OpArity.FFT or op.arity == OpArity.TYPE_CAST:
        raw = _constrain(_raw(rng, shape), domain)
        return {"x": _cast(raw, dtype_key)}

    elif op.arity == OpArity.BINARY:
        raw_x = _constrain(_raw(rng, shape), domain)
        raw_y = _constrain(_raw(rng, shape), domain)
        return {"x": _cast(raw_x, dtype_key), "y": _cast(raw_y, dtype_key)}

    elif op.arity == OpArity.TERNARY:
        raw_a = _constrain(_raw(rng, shape), domain)
        raw_b = _constrain(_raw(rng, shape), domain)
        raw_c = _constrain(_raw(rng, shape), domain)
        stacked = np.sort(np.stack([raw_a, raw_b, raw_c], axis=0), axis=0)
        return {
            "lo": _cast(stacked[0], dtype_key),
            "x": _cast(stacked[1], dtype_key),
            "hi": _cast(stacked[2], dtype_key),
        }

    elif op.arity == OpArity.REDUCTION:
        raw = _constrain(_raw(rng, shape), domain)
        return {"x": _cast(raw, dtype_key)}

    return {}
