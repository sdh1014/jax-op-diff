"""All registered jax.lax operators for precision testing."""

import json
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import torch

from ..op_registry import AtenSpec, op_spec, OpArity, InputDomain, get_all_ops

# =============================================================================
# Basic operations
# =============================================================================

op_spec(
    "add",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("add", "Tensor"),
)(lax.add)
op_spec(
    "sub",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("sub", "Tensor"),
)(lax.sub)
op_spec(
    "mul",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("mul", "Tensor"),
)(lax.mul)
op_spec(
    "div",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("div", "Tensor"),
    input_domain=InputDomain.NON_ZERO,
)(lax.div)
op_spec(
    "rem",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("fmod", "Tensor"),
    input_domain=InputDomain.NON_ZERO,
    notes="jax.lax.rem = C-style truncation remainder; torch.fmod matches this",
)(lax.rem)
op_spec(
    "pow",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("pow", "Tensor_Tensor"),
    input_domain=InputDomain.POSITIVE,
)(lax.pow)
op_spec(
    "integer_pow",
    "basic",
    OpArity.UNARY,
    torch_fn=lambda x: torch.pow(x, 3),
    torch_aten=None,
    supported_dtypes=("float32",),
    notes="integer_pow with exponent=3",
)(lambda x: lax.integer_pow(x, 3))
op_spec(
    "max",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("maximum", "default"),
)(lax.max)
op_spec(
    "min",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("minimum", "default"),
)(lax.min)
op_spec(
    "nextafter",
    "basic",
    OpArity.BINARY,
    torch_aten=AtenSpec("nextafter", "default"),
    notes="Next representable floating-point value towards y",
)(lax.nextafter)

op_spec(
    "neg",
    "basic",
    OpArity.UNARY,
    torch_aten=AtenSpec("neg", "default"),
)(lax.neg)
op_spec(
    "abs",
    "basic",
    OpArity.UNARY,
    torch_aten=AtenSpec("abs", "default"),
)(lax.abs)
op_spec(
    "reciprocal",
    "basic",
    OpArity.UNARY,
    torch_aten=AtenSpec("reciprocal", "default"),
    input_domain=InputDomain.NON_ZERO,
)(lax.reciprocal)
op_spec(
    "square",
    "basic",
    OpArity.UNARY,
    torch_aten=AtenSpec("square", "default"),
)(lax.square)
op_spec(
    "sign",
    "basic",
    OpArity.UNARY,
    torch_aten=AtenSpec("sign", "default"),
)(lax.sign)

# =============================================================================
# Exponential and trigonometric
# =============================================================================

op_spec(
    "exp",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("exp", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.exp)
op_spec(
    "exp2",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("exp2", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.exp2)
op_spec(
    "expm1",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("expm1", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.expm1)
op_spec(
    "log",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("log", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.log)
op_spec(
    "log1p",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("log1p", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.log1p)
op_spec(
    "sqrt",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("sqrt", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.sqrt)
op_spec(
    "rsqrt",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("rsqrt", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.rsqrt)
op_spec(
    "cbrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=lambda x: torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0),
    torch_aten=None,
    input_domain=InputDomain.POSITIVE,
)(lax.cbrt)

op_spec(
    "sin",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("sin", "default"),
)(lax.sin)
op_spec(
    "cos",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("cos", "default"),
)(lax.cos)
op_spec(
    "tan",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("tan", "default"),
)(lax.tan)
op_spec(
    "asin",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("asin", "default"),
    input_domain=InputDomain.UNIT,
)(lax.asin)
op_spec(
    "acos",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("acos", "default"),
    input_domain=InputDomain.UNIT,
)(lax.acos)
op_spec(
    "atan",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("atan", "default"),
)(lax.atan)
op_spec(
    "atan2",
    "exp_trig",
    OpArity.BINARY,
    torch_aten=AtenSpec("atan2", "default"),
)(lax.atan2)

op_spec(
    "sinh",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("sinh", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.sinh)
op_spec(
    "cosh",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("cosh", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.cosh)
op_spec(
    "tanh",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("tanh", "default"),
)(lax.tanh)
op_spec(
    "asinh",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("asinh", "default"),
)(lax.asinh)
op_spec(
    "acosh",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("acosh", "default"),
    input_domain=InputDomain.ABOVE_ONE,
)(lax.acosh)
op_spec(
    "atanh",
    "exp_trig",
    OpArity.UNARY,
    torch_aten=AtenSpec("atanh", "default"),
    input_domain=InputDomain.UNIT,
)(lax.atanh)

# =============================================================================
# Normalization / reductions / cumulative
# =============================================================================

op_spec(
    "logistic",
    "normalization",
    OpArity.UNARY,
    torch_aten=AtenSpec("sigmoid", "default"),
    notes="logistic = sigmoid = 1/(1+exp(-x))",
)(lax.logistic)

op_spec(
    "reduce_sum",
    "normalization",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("sum", "dim_IntList"),
    shape_type="reduction",
)(lambda x, axis: lax.reduce_sum_p.bind(x, axes=(axis,)))
op_spec(
    "reduce_max",
    "normalization",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("amax", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.reduce_max_p.bind(x, axes=(axis,)))
op_spec(
    "reduce_min",
    "normalization",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("amin", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.reduce_min_p.bind(x, axes=(axis,)))
op_spec(
    "cumsum",
    "normalization",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("cumsum", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.cumsum(x, axis=axis))
op_spec(
    "cumprod",
    "normalization",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("cumprod", "default"),
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.cumprod(x, axis=axis))

# =============================================================================
# Activation functions
# =============================================================================

op_spec(
    "erf",
    "activation",
    OpArity.UNARY,
    torch_aten=AtenSpec("erf", "default"),
)(lax.erf)
op_spec(
    "erfc",
    "activation",
    OpArity.UNARY,
    torch_aten=AtenSpec("erfc", "default"),
)(lax.erfc)
op_spec(
    "erf_inv",
    "activation",
    OpArity.UNARY,
    torch_aten=AtenSpec("erfinv", "default"),
    input_domain=InputDomain.UNIT,
)(lax.erf_inv)

# =============================================================================
# Comparison operations
# =============================================================================

op_spec(
    "eq",
    "comparison",
    OpArity.BINARY,
    torch_aten=AtenSpec("eq", "Tensor"),
    notes="Returns bool",
)(lax.eq)
op_spec(
    "ne",
    "comparison",
    OpArity.BINARY,
    torch_aten=AtenSpec("ne", "Tensor"),
    notes="Returns bool",
)(lax.ne)
op_spec(
    "lt",
    "comparison",
    OpArity.BINARY,
    torch_aten=AtenSpec("lt", "Tensor"),
    notes="Returns bool",
)(lax.lt)
op_spec(
    "le",
    "comparison",
    OpArity.BINARY,
    torch_aten=AtenSpec("le", "Tensor"),
    notes="Returns bool",
)(lax.le)
op_spec(
    "gt",
    "comparison",
    OpArity.BINARY,
    torch_aten=AtenSpec("gt", "Tensor"),
    notes="Returns bool",
)(lax.gt)
op_spec(
    "ge",
    "comparison",
    OpArity.BINARY,
    torch_aten=AtenSpec("ge", "Tensor"),
    notes="Returns bool",
)(lax.ge)
op_spec(
    "clamp",
    "comparison",
    OpArity.TERNARY,
    torch_aten=AtenSpec("clamp", "Tensor"),
    notes="JAX: clamp(lo, x, hi); torch: clamp(x, min=lo, max=hi)",
)(lambda lo, x, hi: lax.clamp(lo, x, hi))

# =============================================================================
# Bitwise / shape / indexing operators
# =============================================================================

def _jax_i32(x):
    return lax.convert_element_type(x, jnp.int32)

def _torch_i32(x):
    return x.to(torch.int32)

def _jax_shift_amount(y):
    return jnp.mod(jnp.abs(_jax_i32(y)), 31)

def _torch_shift_amount(y):
    return torch.remainder(torch.abs(y.to(torch.int64)), 31)

def _torch_bitwise_and(x, y):
    return torch.bitwise_and(_torch_i32(x), _torch_i32(y))

def _torch_bitwise_not(x):
    return torch.bitwise_not(_torch_i32(x))

def _torch_bitwise_or(x, y):
    return torch.bitwise_or(_torch_i32(x), _torch_i32(y))

def _torch_bitwise_xor(x, y):
    return torch.bitwise_xor(_torch_i32(x), _torch_i32(y))

def _torch_shift_left(x, y):
    return torch.bitwise_left_shift(_torch_i32(x), _torch_shift_amount(y).to(torch.int32))

def _torch_shift_right_arithmetic(x, y):
    return torch.bitwise_right_shift(_torch_i32(x), _torch_shift_amount(y).to(torch.int32))

def _torch_shift_right_logical(x, y):
    x_i32 = _torch_i32(x)
    y_i64 = _torch_shift_amount(y)
    return ((x_i32.to(torch.int64) & 0xFFFFFFFF) >> y_i64).to(torch.int32)

def _torch_concatenate(x, y):
    return torch.cat((x, y), dim=x.ndim - 1)

def _torch_full_from_x(x):
    fill = torch.mean(x.float()).to(x.dtype)
    return torch.full(x.shape, fill, dtype=x.dtype, device=x.device)

def _torch_full_like_from_x(x):
    fill = torch.mean(x.float()).to(x.dtype)
    return torch.full_like(x, fill)

def _jax_gather_last(x, y):
    idx = jnp.mod(jnp.abs(_jax_i32(y)), x.shape[-1])
    return jnp.take_along_axis(x, idx, axis=-1)

def _torch_gather_last(x, y):
    idx = torch.remainder(torch.abs(y.to(torch.int64)), x.shape[-1])
    return torch.gather(x, dim=-1, index=idx)

def _jax_pad_one(x):
    config = [(1, 1, 0)] * x.ndim
    return lax.pad(x, jnp.array(0, dtype=x.dtype), config)

def _torch_pad_one(x):
    pads = [1, 1] * x.ndim
    return torch.nn.functional.pad(x, pad=pads, mode="constant", value=0)

def _torch_reshape_reverse(x):
    return torch.reshape(x, x.shape[::-1])

def _torch_rev_last(x):
    return torch.flip(x, dims=(-1,))

def _jax_scatter_index_flat(x, y):
    return jnp.mod(jnp.abs(_jax_i32(y.reshape(-1))), x.size)

def _torch_scatter_index_flat(x, y):
    return torch.remainder(torch.abs(y.reshape(-1).to(torch.int64)), x.numel())

def _jax_scatter_set_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].set(flat_y).reshape(x.shape)

def _torch_scatter_set_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _torch_scatter_index_flat(x, y)
    return flat_x.scatter(0, idx, flat_y).reshape_as(x)

def _jax_scatter_add_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].add(flat_y).reshape(x.shape)

def _torch_scatter_add_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _torch_scatter_index_flat(x, y)
    return flat_x.scatter_add(0, idx, flat_y).reshape_as(x)

def _jax_scatter_max_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].max(flat_y).reshape(x.shape)

def _torch_scatter_max_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _torch_scatter_index_flat(x, y)
    return flat_x.scatter_reduce(0, idx, flat_y, reduce="amax", include_self=True).reshape_as(x)

def _jax_scatter_min_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].min(flat_y).reshape(x.shape)

def _torch_scatter_min_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _torch_scatter_index_flat(x, y)
    return flat_x.scatter_reduce(0, idx, flat_y, reduce="amin", include_self=True).reshape_as(x)

def _jax_slice_last_half(x):
    k = max(1, x.shape[-1] // 2)
    starts = (0,) * x.ndim
    limits = (*x.shape[:-1], k)
    return lax.slice(x, start_indices=starts, limit_indices=limits)

def _torch_slice_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return x[..., :k]

def _jax_split_first_half(x):
    if x.shape[-1] <= 1:
        return x
    left = x.shape[-1] // 2
    right = x.shape[-1] - left
    return lax.split(x, sizes=(left, right), axis=-1)[0]

def _torch_split_first_half(x):
    if x.shape[-1] <= 1:
        return x
    left = x.shape[-1] // 2
    right = x.shape[-1] - left
    return torch.split(x, [left, right], dim=-1)[0]

def _torch_squeeze_noop(x):
    return torch.squeeze(torch.unsqueeze(x, 0), 0)

def _jax_tile_leading(x):
    reps = (2,) + (1,) * (x.ndim - 1)
    return lax.tile(x, reps)

def _torch_tile_leading(x):
    reps = (2,) + (1,) * (x.ndim - 1)
    return torch.tile(x, reps)

def _torch_top_k_values(x):
    return torch.topk(x, k=min(5, x.shape[-1]), dim=-1).values

def _jax_transpose_reverse(x):
    perm = tuple(range(x.ndim - 1, -1, -1))
    return lax.transpose(x, perm)

def _torch_transpose_reverse(x):
    perm = tuple(range(x.ndim - 1, -1, -1))
    return torch.permute(x, perm)

op_spec(
    "bitwise_and",
    "bitwise",
    OpArity.BINARY,
    torch_aten=AtenSpec("bitwise_and", "Tensor"),
    supported_dtypes=("float32",),
)(lambda x, y: lax.bitwise_and(_jax_i32(x), _jax_i32(y)))
op_spec(
    "bitwise_not",
    "bitwise",
    OpArity.UNARY,
    torch_aten=AtenSpec("bitwise_not", "default"),
    supported_dtypes=("float32",),
)(lambda x: lax.bitwise_not(_jax_i32(x)))
op_spec(
    "bitwise_or",
    "bitwise",
    OpArity.BINARY,
    torch_aten=AtenSpec("bitwise_or", "Tensor"),
    supported_dtypes=("float32",),
)(lambda x, y: lax.bitwise_or(_jax_i32(x), _jax_i32(y)))
op_spec(
    "bitwise_xor",
    "bitwise",
    OpArity.BINARY,
    torch_aten=AtenSpec("bitwise_xor", "Tensor"),
    supported_dtypes=("float32",),
)(lambda x, y: lax.bitwise_xor(_jax_i32(x), _jax_i32(y)))
op_spec(
    "shift_left",
    "bitwise",
    OpArity.BINARY,
    torch_aten=AtenSpec("bitwise_left_shift", "Tensor"),
    supported_dtypes=("float32",),
)(lambda x, y: lax.shift_left(_jax_i32(x), _jax_shift_amount(y)))
op_spec(
    "shift_right_arithmetic",
    "bitwise",
    OpArity.BINARY,
    torch_aten=AtenSpec("bitwise_right_shift", "Tensor"),
    supported_dtypes=("float32",),
)(lambda x, y: lax.shift_right_arithmetic(_jax_i32(x), _jax_shift_amount(y)))
op_spec(
    "shift_right_logical",
    "bitwise",
    OpArity.BINARY,
    torch_fn=_torch_shift_right_logical,
    supported_dtypes=("float32",),
)(lambda x, y: lax.shift_right_logical(_jax_i32(x), _jax_shift_amount(y)))

op_spec(
    "concatenate",
    "shape",
    OpArity.BINARY,
    torch_aten=AtenSpec("cat", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, y: lax.concatenate((x, y), dimension=x.ndim - 1))
op_spec(
    "full",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("full", "default"),
    supported_dtypes=("float32",),
)(lambda x: lax.full(x.shape, jnp.mean(x, dtype=jnp.float32), x.dtype))
op_spec(
    "full_like",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("full_like", "default"),
    supported_dtypes=("float32",),
)(lambda x: lax.full_like(x, jnp.mean(x, dtype=jnp.float32)))
op_spec(
    "reshape",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("reshape", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.reshape(x, x.shape[::-1]))
op_spec(
    "rev",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("flip", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.rev(x, dimensions=(x.ndim - 1,)))
op_spec(
    "split",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("split", "sizes"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_split_first_half)
op_spec(
    "squeeze",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("squeeze", "dim"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.squeeze(lax.expand_dims(x, (0,)), dimensions=(0,)))
op_spec(
    "tile",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("tile", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_tile_leading)
op_spec(
    "transpose",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("permute", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_transpose_reverse)

op_spec(
    "gather",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_gather_last,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_gather_last)
op_spec(
    "pad",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("pad", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_pad_one)
op_spec(
    "scatter",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_set_flat,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_set_flat)
op_spec(
    "scatter_add",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_add_flat,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_add_flat)
op_spec(
    "scatter_max",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_max_flat,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_max_flat)
op_spec(
    "scatter_min",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_min_flat,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_min_flat)
op_spec(
    "slice",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_slice_last_half)
op_spec(
    "top_k",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("topk", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.top_k(x, k=min(5, x.shape[-1]))[0])

# =============================================================================
# Additional direct ATen mappings from manifest
# =============================================================================

def _torch_broadcast_leading2(x):
    return torch.expand(x, (2,) + tuple(x.shape))

def _jax_broadcast_in_dim_leading2(x):
    target = (2,) + tuple(x.shape)
    dims = tuple(range(1, x.ndim + 1))
    return lax.broadcast_in_dim(x, shape=target, broadcast_dimensions=dims)

def _torch_broadcast_in_dim_leading2(x):
    target = (2,) + tuple(x.shape)
    if x.ndim == 0:
        return torch.expand(x, target)
    return torch.unsqueeze(x, 0).expand(target)

def _torch_broadcast_to_rank_plus1(x):
    return torch.unsqueeze(x, 0)

def _torch_broadcasted_iota_last(x):
    base = torch.arange(0, x.shape[-1], dtype=x.dtype, device=x.device)
    if x.ndim == 1:
        return base
    return base.reshape((1,) * (x.ndim - 1) + (x.shape[-1],)).expand(x.shape)

def _jax_collapse_first2(x):
    if x.ndim < 2:
        return x
    return lax.collapse(x, start_dimension=0, stop_dimension=2)

def _torch_collapse_first2(x):
    if x.ndim < 2:
        return x
    return torch.reshape(x, (x.shape[0] * x.shape[1],) + tuple(x.shape[2:]))

def _jax_dynamic_index_in_dim_last(x):
    idx = jnp.array(x.shape[-1] // 2, dtype=jnp.int32)
    return lax.dynamic_index_in_dim(x, idx, axis=-1, keepdims=False)

def _torch_dynamic_index_in_dim_last(x):
    return torch.select(x, -1, x.shape[-1] // 2)

def _jax_dynamic_slice_last_half(x):
    k = max(1, x.shape[-1] // 2)
    starts = tuple(jnp.array(0, dtype=jnp.int32) for _ in range(x.ndim))
    sizes = tuple(x.shape[:-1]) + (k,)
    return lax.dynamic_slice(x, starts, sizes)

def _torch_dynamic_slice_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return x[..., :k]

def _jax_dynamic_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return lax.dynamic_slice_in_dim(x, jnp.array(0, dtype=jnp.int32), k, axis=-1)

def _torch_dynamic_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return x[..., :k]

def _jax_dynamic_update_slice_prefix(x, y):
    k = max(1, x.shape[-1] // 2)
    update = y[..., :k]
    starts = (0,) * x.ndim
    return lax.dynamic_update_slice(x, update, starts)

def _torch_dynamic_update_slice_prefix(x, y):
    k = max(1, x.shape[-1] // 2)
    out = x.clone()
    out[..., :k] = y[..., :k]
    return out

def _jax_dynamic_update_slice_in_dim_prefix(x, y):
    k = max(1, x.shape[-1] // 2)
    update = y[..., :k]
    return lax.dynamic_update_slice_in_dim(x, update, 0, axis=-1)

def _torch_dynamic_update_slice_in_dim_prefix(x, y):
    k = max(1, x.shape[-1] // 2)
    out = x.clone()
    out[..., :k] = y[..., :k]
    return out

def _jax_dynamic_update_index_in_dim_last(x, y):
    return lax.dynamic_update_index_in_dim(x, y[..., 0], 0, axis=-1)

def _torch_dynamic_update_index_in_dim_last(x, y):
    out = x.clone()
    out[..., 0] = y[..., 0]
    return out

def _torch_expand_dims_front(x):
    return torch.unsqueeze(x, 0)

def _torch_index_in_dim_last(x):
    return torch.select(x, -1, x.shape[-1] // 2)

def _jax_index_take_last_two(x):
    idx = jnp.array([[0, x.shape[-1] - 1]], dtype=jnp.int32)
    return lax.index_take(x, idx, axes=(x.ndim - 1,))

def _torch_index_take_last_two(x):
    idx = torch.tensor([0, x.shape[-1] - 1], dtype=torch.int64, device=x.device)
    return torch.index_select(x, x.ndim - 1, idx).movedim(-1, 0)

def _torch_iota_last_dim(x):
    return torch.arange(0, x.shape[-1], dtype=x.dtype, device=x.device)

def _jax_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return lax.slice_in_dim(x, 0, k, stride=1, axis=-1)

def _torch_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return x[..., :k]

def _torch_sort_key_val_keys(x):
    return torch.sort(x, dim=-1, stable=True).values

def _jax_scatter_mul_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].multiply(flat_y).reshape(x.shape)

def _torch_scatter_mul_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _torch_scatter_index_flat(x, y)
    return flat_x.scatter_reduce(0, idx, flat_y, reduce="prod", include_self=True).reshape_as(x)

def _jax_scatter_sub_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].add(-flat_y).reshape(x.shape)

def _torch_scatter_sub_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _torch_scatter_index_flat(x, y)
    return flat_x.scatter_add(0, idx, -flat_y).reshape_as(x)

def _torch_reduce_and_last(x, axis):
    return torch.all(x > 0, dim=axis)

def _torch_reduce_or_last(x, axis):
    return torch.any(x > 0, dim=axis)

op_spec(
    "approx_max_k",
    "reduction",
    OpArity.UNARY,
    torch_fn=None,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="TODO: no strict 1:1 ATen mapping",
)(lambda x: lax.approx_max_k(x, k=min(5, x.shape[-1]))[0])
op_spec(
    "approx_min_k",
    "reduction",
    OpArity.UNARY,
    torch_fn=None,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="TODO: no strict 1:1 ATen mapping",
)(lambda x: lax.approx_min_k(x, k=min(5, x.shape[-1]))[0])
op_spec(
    "broadcast",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("expand", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.broadcast(x, sizes=(2,)))
op_spec(
    "broadcast_in_dim",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_broadcast_in_dim_leading2,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_broadcast_in_dim_leading2)
op_spec(
    "broadcast_to_rank",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("unsqueeze", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.broadcast_to_rank(x, x.ndim + 1))
op_spec(
    "broadcasted_iota",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("arange", "start_step"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.broadcasted_iota(x.dtype, shape=x.shape, dimension=x.ndim - 1))
op_spec(
    "collapse",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("reshape", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_collapse_first2)
op_spec(
    "expand_dims",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("unsqueeze", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.expand_dims(x, (0,)))
op_spec(
    "iota",
    "shape",
    OpArity.UNARY,
    torch_aten=AtenSpec("arange", "start_step"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.iota(x.dtype, x.shape[-1]))

op_spec(
    "dynamic_index_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("select", "int"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_index_in_dim_last)
op_spec(
    "dynamic_slice",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_slice_last_half)
op_spec(
    "dynamic_slice_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_slice_in_dim_last_half)
op_spec(
    "dynamic_update_slice",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_slice_prefix,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_update_slice_prefix)
op_spec(
    "dynamic_update_slice_in_dim",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_slice_in_dim_prefix,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_update_slice_in_dim_prefix)
op_spec(
    "dynamic_update_index_in_dim",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_index_in_dim_last,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(
    _jax_dynamic_update_index_in_dim_last
)
op_spec(
    "index_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("select", "int"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.index_in_dim(x, x.shape[-1] // 2, axis=-1, keepdims=False))
op_spec(
    "index_take",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_index_take_last_two,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_index_take_last_two)
op_spec(
    "slice_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_slice_in_dim_last_half)
op_spec(
    "sort_key_val",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_sort_key_val_keys,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.sort_key_val(x, -x, dimension=-1, is_stable=True)[0])
op_spec(
    "scatter_mul",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_mul_flat,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_mul_flat)
op_spec(
    "scatter_sub",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_sub_flat,
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_sub_flat)

op_spec(
    "reduce_and",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("all", "dim"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, axis: lax.reduce_and(x > 0, axes=(axis,)))
op_spec(
    "reduce_or",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("any", "dim"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, axis: lax.reduce_or(x > 0, axes=(axis,)))
op_spec(
    "reduce_xor",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.remainder((x > 0).to(torch.int32).sum(dim=axis), 2).to(torch.bool),
    torch_aten=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="Reduce xor on predicate (x > 0)",
)(lambda x, axis: lax.reduce_xor(x > 0, axes=(axis,)))

# =============================================================================
# Rounding operations
# =============================================================================

op_spec(
    "floor",
    "rounding",
    OpArity.UNARY,
    torch_aten=AtenSpec("floor", "default"),
)(lax.floor)
op_spec(
    "ceil",
    "rounding",
    OpArity.UNARY,
    torch_aten=AtenSpec("ceil", "default"),
)(lax.ceil)
op_spec(
    "round",
    "rounding",
    OpArity.UNARY,
    torch_aten=AtenSpec("round", "default"),
    notes="Both use half-to-even (banker's rounding)",
)(lambda x: lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN))
op_spec(
    "is_finite",
    "rounding",
    OpArity.UNARY,
    torch_aten=AtenSpec("isfinite", "default"),
    notes="Returns bool",
)(lax.is_finite)

# =============================================================================
# Complex number operations
# =============================================================================

op_spec(
    "complex",
    "complex",
    OpArity.BINARY,
    torch_aten=AtenSpec("complex", "default"),
    supported_dtypes=("float32",),
    notes="Constructs complex from real and imag parts",
)(lax.complex)
op_spec(
    "conj",
    "complex",
    OpArity.UNARY,
    torch_aten=AtenSpec("conj", "default"),
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
    notes="torch.conj returns lazy view; need resolve_conj() for numpy()",
)(lax.conj)
op_spec(
    "real",
    "complex",
    OpArity.UNARY,
    torch_aten=AtenSpec("real", "default"),
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
)(lax.real)
op_spec(
    "imag",
    "complex",
    OpArity.UNARY,
    torch_aten=AtenSpec("imag", "default"),
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
)(lax.imag)

# =============================================================================
# Additional reductions (sort, argmax, argmin, top_k)
# =============================================================================

op_spec(
    "reduce_prod",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("prod", "dim_int"),
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.reduce_prod_p.bind(x, axes=(axis,)))
op_spec(
    "cummax",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("cummax", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.cummax(x, axis=axis))
op_spec(
    "cummin",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("cummin", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.cummin(x, axis=axis))
op_spec(
    "cumlogsumexp",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("logcumsumexp", "default"),
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.cumlogsumexp(x, axis=axis))

op_spec(
    "sort",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("sort", "default"),
    shape_type="reduction",
    notes="Compares sorted values along axis",
)(lambda x, axis: lax.sort(x, dimension=axis))
op_spec(
    "argmax",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("argmax", "default"),
    shape_type="reduction",
    notes="Returns int indices; abs error = 0 means frameworks agree",
)(lambda x, axis: lax.argmax(x, axis, jnp.int32))
op_spec(
    "argmin",
    "reduction",
    OpArity.REDUCTION,
    torch_aten=AtenSpec("argmin", "default"),
    shape_type="reduction",
    notes="Returns int indices; abs error = 0 means frameworks agree",
)(lambda x, axis: lax.argmin(x, axis, jnp.int32))

# =============================================================================
# Special mathematical functions
# =============================================================================

op_spec(
    "lgamma",
    "special",
    OpArity.UNARY,
    torch_aten=AtenSpec("lgamma", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.lgamma)
op_spec(
    "digamma",
    "special",
    OpArity.UNARY,
    torch_aten=AtenSpec("digamma", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.digamma)
op_spec(
    "bessel_i0e",
    "special",
    OpArity.UNARY,
    torch_aten=AtenSpec("special_i0e", "default"),
)(lax.bessel_i0e)
op_spec(
    "bessel_i1e",
    "special",
    OpArity.UNARY,
    torch_aten=AtenSpec("special_i1e", "default"),
)(lax.bessel_i1e)
op_spec(
    "igamma",
    "special",
    OpArity.BINARY,
    torch_aten=AtenSpec("special_gammainc", "default"),
    input_domain=InputDomain.POSITIVE,
    notes="JAX igamma(a,x) = torch.special.gammainc(a,x)",
)(lax.igamma)
op_spec(
    "igammac",
    "special",
    OpArity.BINARY,
    torch_aten=AtenSpec("special_gammaincc", "default"),
    input_domain=InputDomain.POSITIVE,
    notes="JAX igammac(a,x) = torch.special.gammaincc(a,x)",
)(lax.igammac)
op_spec(
    "betainc",
    "special",
    OpArity.TERNARY,
    torch_fn=None,
    torch_aten=None,
    input_domain=InputDomain.POSITIVE,
    notes="TODO: No PyTorch equivalent",
)(lax.betainc)

# =============================================================================
# Type conversion operations
# =============================================================================

op_spec(
    "dot",
    "linalg",
    OpArity.MATMUL,
    torch_aten=AtenSpec("matmul", "default"),
    shape_type="matmul",
    supported_dtypes=("float32", "bfloat16"),
)(lax.dot)
op_spec(
    "batch_matmul",
    "linalg",
    OpArity.MATMUL,
    torch_aten=AtenSpec("bmm", "default"),
    shape_type="batch_matmul",
    supported_dtypes=("float32", "bfloat16"),
)(lax.batch_matmul)

# --- Cholesky decomposition ---
op_spec(
    "cholesky",
    "linalg",
    OpArity.UNARY,
    torch_aten=AtenSpec("linalg_cholesky", "default"),
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Input: positive definite matrix",
)(lambda x: jax.lax.linalg.cholesky(x))

op_spec(
    "triangular_solve",
    "linalg",
    OpArity.MATMUL,
    torch_aten=AtenSpec("linalg_solve_triangular", "default"),
    input_domain=InputDomain.LOWER_TRIANGULAR,
    shape_type="linalg_solve",
    supported_dtypes=("float32",),
    notes="Solve A @ x = b where A is lower triangular",
)(lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True))

def _torch_linalg_qr_q(x):
    return torch.linalg.qr(x).Q

def _torch_linalg_svd_u(x):
    return torch.linalg.svd(x, full_matrices=False).U

def _torch_linalg_eigh_vecs(x):
    return torch.linalg.eigh(x).eigenvectors

def _torch_linalg_eig_vecs(x):
    return torch.linalg.eig(x).eigenvectors

def _jax_linalg_eig_values_sorted(x):
    eigvals = jax.lax.linalg.eig(
        x,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=True,
    )[0]
    _, eigvals = jax.lax.sort_key_val(jax.lax.imag(eigvals), eigvals, dimension=-1)
    _, eigvals = jax.lax.sort_key_val(jax.lax.real(eigvals), eigvals, dimension=-1)
    return eigvals

def _jax_linalg_householder_product(x):
    if x.shape[-1] < 2:
        return x
    a = x[..., :-1]
    taus = jnp.ones((a.shape[-1],), dtype=x.dtype)
    return jax.lax.linalg.householder_product(a, taus)

def _torch_linalg_householder_product(x):
    if x.shape[-1] < 2:
        return x
    a = x[..., :-1]
    taus = torch.ones((a.shape[-1],), dtype=x.dtype, device=x.device)
    return torch.linalg.householder_product(a, taus)

def _torch_linalg_lu_matrix(x):
    return torch.linalg.lu(x).P

def _jax_lu_pivots_to_perm(x):
    _, _, piv = jax.lax.linalg.lu(x)
    return jax.lax.linalg.lu_pivots_to_permutation(piv, permutation_size=x.shape[-1])

def _torch_lu_pivots_to_perm(x):
    _, piv, _ = torch.linalg.lu_factor_ex(x)
    n = x.shape[-1]
    batch_shape = piv.shape[:-1]
    perm = torch.arange(n, device=x.device, dtype=torch.int64)
    if batch_shape:
        perm = perm.expand(*batch_shape, n).clone()
    else:
        perm = perm.clone()
    piv0 = piv.to(torch.int64) - 1
    for i in range(n):
        j = piv0[..., i]
        pi = perm[..., i].clone()
        pj = torch.gather(perm, -1, j.unsqueeze(-1)).squeeze(-1)
        perm[..., i] = pj
        perm.scatter_(-1, j.unsqueeze(-1), pi.unsqueeze(-1))
    return perm.to(torch.int32)

op_spec(
    "eig",
    "linalg",
    OpArity.UNARY,
    torch_aten=AtenSpec("linalg_eig", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
)(_jax_linalg_eig_values_sorted)
op_spec(
    "eigh",
    "linalg",
    OpArity.UNARY,
    torch_aten=AtenSpec("_linalg_eigh", "default"),
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: jax.lax.linalg.eigh(x)[0])
op_spec(
    "householder_product",
    "linalg",
    OpArity.UNARY,
    torch_aten=AtenSpec("linalg_householder_product", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
)(_jax_linalg_householder_product)
op_spec(
    "lu",
    "linalg",
    OpArity.UNARY,
    torch_fn=lambda x: torch.linalg.lu_factor(x)[0],
    torch_aten=None,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: jax.lax.linalg.lu(x)[0])
op_spec(
    "lu_pivots_to_permutation",
    "linalg",
    OpArity.UNARY,
    torch_fn=None,
    torch_aten=None,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="TODO: no strict 1:1 ATen mapping",
)(_jax_lu_pivots_to_perm)
op_spec(
    "qr",
    "linalg",
    OpArity.UNARY,
    torch_aten=AtenSpec("linalg_qr", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: lax.abs(jax.lax.linalg.qr(x)[0]))
op_spec(
    "svd",
    "linalg",
    OpArity.UNARY,
    torch_aten=AtenSpec("_linalg_svd", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: jax.lax.linalg.svd(x, full_matrices=False)[1])
op_spec(
    "zeta",
    "special",
    OpArity.BINARY,
    torch_aten=AtenSpec("special_zeta", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, y: lax.zeta(lax.abs(x) + 2.0, lax.abs(y) + 1.0))

# --- Convolution ---
def _jax_conv(inp, ker, strides, padding):
    """Wrapper for jax.lax.conv with NCHW layout."""
    ndim = inp.ndim - 2
    if ndim == 1:
        dn = lax.conv_dimension_numbers(inp.shape, ker.shape, ('NCH', 'OIH', 'NCH'))
    else:
        dn = lax.conv_dimension_numbers(inp.shape, ker.shape, ('NCHW', 'OIHW', 'NCHW'))
    return lax.conv_general_dilated(inp, ker, strides, padding, dimension_numbers=dn)

def _torch_conv(inp, ker):
    """Wrapper: NCHW layout for torch."""
    ndim = inp.ndim - 2
    if ndim == 1:
        return torch.nn.functional.conv1d(inp, ker, padding='same')
    else:
        return torch.nn.functional.conv2d(inp, ker, padding='same')

op_spec(
    "conv_general_dilated",
    "linalg",
    OpArity.CONV,
    torch_aten=AtenSpec("convolution", "default"),
    shape_type="conv",
    supported_dtypes=("float32",),
    notes="Both use NCHW layout; JAX via dimension_numbers, torch via F.conv",
)(_jax_conv)

# =============================================================================
# FFT operations
# =============================================================================

op_spec(
    "fft",
    "fft",
    OpArity.FFT,
    torch_aten=AtenSpec("fft_fft", "default"),
    input_domain=InputDomain.COMPLEX,
    shape_type="fft",
    supported_dtypes=("float32",),
    notes="Complex-to-complex FFT",
)(lambda x: lax.fft(x, fft_type="FFT", fft_lengths=(x.shape[-1],)))

op_spec(
    "convert_element_type",
    "type_cast",
    OpArity.TYPE_CAST,
    torch_fn=lambda x: x.to(torch.bfloat16),
    torch_aten=None,
    supported_dtypes=("float32",),
    notes="Cast float32 to bfloat16",
)(lambda x: lax.convert_element_type(x, jnp.bfloat16))
op_spec(
    "bitcast_convert_type",
    "type_cast",
    OpArity.TYPE_CAST,
    torch_fn=lambda x: x.view(torch.int32),
    torch_aten=None,
    supported_dtypes=("float32",),
    notes="Bitcast float32 to int32",
)(lambda x: lax.bitcast_convert_type(x, jnp.int32))

# =============================================================================
# Remaining manifest ops (no torch mapping — registered as MISSING placeholders)
# =============================================================================

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "docs" / "jax_lax_operators.json"
_MANIFEST_GROUPS = json.loads(_MANIFEST_PATH.read_text())["operators"]

def _missing_lax_op(op_name):
    def _fn(*_args, **_kwargs):
        raise RuntimeError(f"jax.lax symbol is unavailable in installed version: {op_name}")

    return _fn

def _register_remaining_ops():
    """Register remaining manifest ops as MISSING placeholders (torch_fn=None)."""
    existing = {op.name for op in get_all_ops()}

    for name, group in (
        [(n, "general") for n in _MANIFEST_GROUPS["general_operators"]]
        + [(n, "sharding") for n in _MANIFEST_GROUPS["sharding_related_operators"]]
        + [(n, "linalg") for n in _MANIFEST_GROUPS["linear_algebra_operators"]]
    ):
        if name in existing:
            continue

        if group == "linalg" and hasattr(lax.linalg, name):
            jax_fn = getattr(lax.linalg, name)
            notes = "No torch mapping"
        elif hasattr(lax, name):
            jax_fn = getattr(lax, name)
            notes = "No torch mapping"
        else:
            jax_fn = _missing_lax_op(name)
            notes = "No torch mapping; jax.lax symbol unavailable in installed version"

        op_spec(
            name,
            "unmapped",
            OpArity.UNARY,
            torch_fn=None,
            torch_aten=None,
            notes="TODO: no strict 1:1 ATen mapping",
        )(jax_fn)

_register_remaining_ops()

def _ensure_todo_notes_for_unmapped():
    for op in get_all_ops():
        if op.torch_fn is None and op.torch_aten is None:
            note = (op.notes or "").strip()
            if note.startswith("TODO:"):
                continue
            op.notes = f"TODO: {note}" if note else "TODO: no strict 1:1 ATen mapping"

_ensure_todo_notes_for_unmapped()

# =============================================================================
# Torch call plans (ATen/torch signature and output adaptation)
# =============================================================================

_TORCH_ATEN_PLAN_REGISTRY = {}
_TORCH_FN_PLAN_REGISTRY = {}
_TORCH_OUTPUT_ADAPTER_REGISTRY = {}

def aten_plan(*op_names):
    def decorator(fn):
        for op_name in op_names:
            _TORCH_ATEN_PLAN_REGISTRY[op_name] = fn
        return fn

    return decorator

def torch_fn_plan(*op_names):
    def decorator(fn):
        for op_name in op_names:
            _TORCH_FN_PLAN_REGISTRY[op_name] = fn
        return fn

    return decorator

def output_adapter(*op_names):
    def decorator(fn):
        for op_name in op_names:
            _TORCH_OUTPUT_ADAPTER_REGISTRY[op_name] = fn
        return fn

    return decorator

def _axis_last(x):
    return x.ndim - 1 if x.ndim > 0 else 0

@output_adapter("cummax", "cummin", "sort")
def _adapt_first(output, _inputs):
    if isinstance(output, tuple):
        return output[0]
    return output

@output_adapter("top_k")
def _adapt_tuple_first(output, _inputs):
    if isinstance(output, tuple):
        return output[0]
    return output

@output_adapter("qr")
def _adapt_tuple_first_abs(output, _inputs):
    out = output[0] if isinstance(output, tuple) else output
    return torch.abs(out)

@output_adapter("svd")
def _adapt_svd_singular_values(output, _inputs):
    return output[1] if isinstance(output, tuple) else output

@output_adapter("eig")
def _adapt_eig_values_sorted(output, _inputs):
    eigvals = output[0] if isinstance(output, tuple) else output
    idx_imag = torch.argsort(torch.imag(eigvals), dim=-1, stable=True)
    eigvals = torch.gather(eigvals, -1, idx_imag)
    idx_real = torch.argsort(torch.real(eigvals), dim=-1, stable=True)
    return torch.gather(eigvals, -1, idx_real)

@output_adapter("eigh")
def _adapt_tuple_second(output, _inputs):
    if isinstance(output, tuple):
        return output[1]
    return output

@output_adapter("split")
def _adapt_split_first(output, _inputs):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

@output_adapter("scatter", "scatter_add", "scatter_max")
def _adapt_scatter_reshape(output, inputs):
    return output.reshape_as(inputs["x"])

@output_adapter("broadcasted_iota")
def _adapt_broadcasted_iota(output, inputs):
    x = inputs["x"]
    if x.ndim == 1:
        return output
    return output.reshape((1,) * (x.ndim - 1) + (x.shape[-1],)).expand(x.shape)

@output_adapter("householder_product")
def _adapt_householder_product(output, inputs):
    x = inputs["x"]
    if x.shape[-1] < 2:
        return x
    return output

@output_adapter("conj")
def _adapt_conj_resolved(output, _inputs):
    return output.resolve_conj() if hasattr(output, "resolve_conj") else output

@aten_plan("reduce_sum", "reduce_max", "reduce_min")
def _build_reduce_dim_intlist(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, [axis], False), {}

@aten_plan("reduce_prod", "sort", "argmax", "argmin")
def _build_reduce_dim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, axis, False), {}

@aten_plan("reduce_and", "reduce_or")
def _build_reduce_bool_dim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x > 0, axis, False), {}

@aten_plan("cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp")
def _build_reduce_dim_no_keepdim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, axis), {}

@aten_plan("zeta")
def _build_zeta_shifted(inputs):
    x = inputs["x"]
    y = inputs["y"]
    return (torch.abs(x) + 2.0, torch.abs(y) + 1.0), {}

@aten_plan("clamp")
def _build_clamp(inputs):
    return (inputs["x"], inputs["lo"], inputs["hi"]), {}

@aten_plan("bitwise_and", "bitwise_or", "bitwise_xor")
def _build_bitwise_binary_i32(inputs):
    return (_torch_i32(inputs["x"]), _torch_i32(inputs["y"])), {}

@aten_plan("bitwise_not")
def _build_bitwise_not_i32(inputs):
    return (_torch_i32(inputs["x"]),), {}

@aten_plan("shift_left", "shift_right_arithmetic")
def _build_shift_i32(inputs):
    x = _torch_i32(inputs["x"])
    y = _torch_shift_amount(inputs["y"]).to(torch.int32)
    return (x, y), {}

@aten_plan("concatenate")
def _build_concatenate(inputs):
    x = inputs["x"]
    return ((x, inputs["y"]), x.ndim - 1), {}

@aten_plan("full")
def _build_full_from_x(inputs):
    x = inputs["x"]
    fill = torch.mean(x.float()).item()
    return (tuple(x.shape), fill), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
    }

@aten_plan("full_like")
def _build_full_like_from_x(inputs):
    x = inputs["x"]
    fill = torch.mean(x.float()).item()
    return (x, fill), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
        "memory_format": None,
    }

@aten_plan("reshape")
def _build_reshape_reverse(inputs):
    x = inputs["x"]
    return (x, x.shape[::-1]), {}

@aten_plan("collapse")
def _build_collapse_first2(inputs):
    x = inputs["x"]
    if x.ndim < 2:
        target = tuple(x.shape)
    else:
        target = (x.shape[0] * x.shape[1],) + tuple(x.shape[2:])
    return (x, target), {}

@aten_plan("rev")
def _build_rev_last(inputs):
    x = inputs["x"]
    return (x, [x.ndim - 1]), {}

@aten_plan("split")
def _build_split_first_half(inputs):
    x = inputs["x"]
    left = max(1, x.shape[-1] // 2)
    right = x.shape[-1] - left
    return (x, [left, right], -1), {}

@aten_plan("squeeze")
def _build_squeeze_noop(inputs):
    x = inputs["x"]
    return (torch.unsqueeze(x, 0), 0), {}

@aten_plan("expand_dims")
def _build_expand_dims_front(inputs):
    x = inputs["x"]
    return (x, 0), {}

@aten_plan("tile")
def _build_tile_leading(inputs):
    x = inputs["x"]
    reps = (2,) + (1,) * (x.ndim - 1)
    return (x, reps), {}

@aten_plan("transpose")
def _build_transpose_reverse(inputs):
    x = inputs["x"]
    perm = tuple(range(x.ndim - 1, -1, -1))
    return (x, perm), {}

@aten_plan("gather")
def _build_gather_last(inputs):
    x = inputs["x"]
    y = inputs["y"]
    idx = torch.remainder(torch.abs(y.to(torch.int64)), x.shape[-1])
    return (x, x.ndim - 1, idx), {"sparse_grad": False}

@aten_plan("pad")
def _build_pad_one(inputs):
    x = inputs["x"]
    pads = [1, 1] * x.ndim
    return (x, pads, "constant", 0), {}

@aten_plan("scatter")
def _build_scatter_set(inputs):
    x = inputs["x"]
    y = inputs["y"]
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = torch.remainder(torch.abs(y.reshape(-1).to(torch.int64)), x.numel())
    return (flat_x, 0, idx, flat_y), {}

@aten_plan("scatter_add")
def _build_scatter_add(inputs):
    x = inputs["x"]
    y = inputs["y"]
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = torch.remainder(torch.abs(y.reshape(-1).to(torch.int64)), x.numel())
    return (flat_x, 0, idx, flat_y), {}

@aten_plan("scatter_max")
def _build_scatter_max(inputs):
    x = inputs["x"]
    y = inputs["y"]
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = torch.remainder(torch.abs(y.reshape(-1).to(torch.int64)), x.numel())
    return (flat_x, 0, idx, flat_y, "amax"), {"include_self": True}

@aten_plan("slice")
def _build_slice_last_half(inputs):
    x = inputs["x"]
    k = max(1, x.shape[-1] // 2)
    return (x, x.ndim - 1, 0, k, 1), {}

@aten_plan("dynamic_slice", "dynamic_slice_in_dim", "slice_in_dim")
def _build_slice_like_last_half(inputs):
    x = inputs["x"]
    k = max(1, x.shape[-1] // 2)
    return (x, x.ndim - 1, 0, k, 1), {}

@aten_plan("dynamic_index_in_dim", "index_in_dim")
def _build_select_last_middle(inputs):
    x = inputs["x"]
    return (x, x.ndim - 1, x.shape[-1] // 2), {}

@aten_plan("top_k")
def _build_top_k_values(inputs):
    x = inputs["x"]
    return (x, min(5, x.shape[-1]), x.ndim - 1, True, True), {}

@aten_plan("broadcast")
def _build_broadcast_leading2(inputs):
    x = inputs["x"]
    return (x, (2,) + tuple(x.shape)), {}

@aten_plan("broadcast_to_rank")
def _build_broadcast_to_rank_plus1(inputs):
    x = inputs["x"]
    return (x, 0), {}

@aten_plan("broadcasted_iota")
def _build_broadcasted_iota(inputs):
    x = inputs["x"]
    return (0, x.shape[-1], 1), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
    }

@aten_plan("iota")
def _build_iota_last_dim(inputs):
    x = inputs["x"]
    return (0, x.shape[-1], 1), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
    }

@aten_plan("householder_product")
def _build_householder_product(inputs):
    x = inputs["x"]
    a = x[..., :-1]
    taus = torch.ones((a.shape[-1],), dtype=x.dtype, device=x.device)
    return (a, taus), {}

@aten_plan("triangular_solve")
def _build_triangular_solve(inputs):
    return (inputs["x"], inputs["y"]), {
        "upper": False,
        "left": True,
        "unitriangular": False,
    }

@aten_plan("conv_general_dilated")
def _build_convolution(inputs):
    inp = inputs["input"]
    ker = inputs["kernel"]
    ndim = inp.ndim - 2
    strides = tuple(int(s) for s in inputs["strides"])
    if inputs["padding"] == "SAME":
        padding = tuple(int(ker.shape[2 + i] // 2) for i in range(ndim))
    else:
        padding = tuple(int(p) for p in inputs["padding"])
    dilation = (1,) * ndim
    output_padding = (0,) * ndim
    return (inp, ker, None, strides, padding, dilation, False, output_padding, 1), {}

@aten_plan("fft")
def _build_fft_like(inputs):
    return (inputs["x"], None, -1, None), {}

def _configure_torch_call_plans():
    for op in get_all_ops():
        if op.name in _TORCH_ATEN_PLAN_REGISTRY:
            op.torch_aten_builder = _TORCH_ATEN_PLAN_REGISTRY[op.name]
        if op.name in _TORCH_FN_PLAN_REGISTRY:
            op.torch_fn_builder = _TORCH_FN_PLAN_REGISTRY[op.name]
        if op.name in _TORCH_OUTPUT_ADAPTER_REGISTRY:
            op.torch_output_adapter = _TORCH_OUTPUT_ADAPTER_REGISTRY[op.name]

_configure_torch_call_plans()
