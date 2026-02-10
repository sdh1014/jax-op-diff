"""All registered jax.lax operators for precision testing."""

import json
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special as jsp
import torch

from ..op_registry import AtenSpec, op_spec, OpArity, InputDomain, get_all_ops

# =============================================================================
# Basic operations
# =============================================================================

op_spec(
    "add",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.add,
    torch_aten=AtenSpec("add", "Tensor"),
)(lax.add)
op_spec(
    "sub",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.sub,
    torch_aten=AtenSpec("sub", "Tensor"),
)(lax.sub)
op_spec(
    "mul",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.mul,
    torch_aten=AtenSpec("mul", "Tensor"),
)(lax.mul)
op_spec(
    "div",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.div,
    torch_aten=AtenSpec("div", "Tensor"),
    input_domain=InputDomain.NON_ZERO,
)(lax.div)
op_spec(
    "rem",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.fmod,
    torch_aten=AtenSpec("fmod", "Tensor"),
    input_domain=InputDomain.NON_ZERO,
    notes="jax.lax.rem = C-style truncation remainder; torch.fmod matches this",
)(lax.rem)
op_spec(
    "pow",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.pow,
    torch_aten=AtenSpec("pow", "Tensor_Tensor"),
    input_domain=InputDomain.POSITIVE,
)(lax.pow)
op_spec(
    "max",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.maximum,
    torch_aten=AtenSpec("maximum", "default"),
)(lax.max)
op_spec(
    "min",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.minimum,
    torch_aten=AtenSpec("minimum", "default"),
)(lax.min)
op_spec(
    "nextafter",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.nextafter,
    torch_aten=AtenSpec("nextafter", "default"),
    notes="Next representable floating-point value towards y",
)(lax.nextafter)

op_spec(
    "neg",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.neg,
    torch_aten=AtenSpec("neg", "default"),
)(lax.neg)
op_spec(
    "abs",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.abs,
    torch_aten=AtenSpec("abs", "default"),
)(lax.abs)
op_spec(
    "reciprocal",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.reciprocal,
    torch_aten=AtenSpec("reciprocal", "default"),
    input_domain=InputDomain.NON_ZERO,
)(lax.reciprocal)
op_spec(
    "square",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.square,
    torch_aten=AtenSpec("square", "default"),
)(lax.square)
op_spec(
    "sign",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.sign,
    torch_aten=AtenSpec("sign", "default"),
)(lax.sign)
op_spec(
    "integer_pow_2",
    "basic",
    OpArity.UNARY,
    torch_fn=lambda x: torch.pow(x, 2),
    torch_aten=AtenSpec("pow", "Tensor_Scalar"),
    notes="integer_pow with exponent=2",
)(lambda x: lax.integer_pow(x, 2))
op_spec(
    "integer_pow_3",
    "basic",
    OpArity.UNARY,
    torch_fn=lambda x: torch.pow(x, 3),
    torch_aten=AtenSpec("pow", "Tensor_Scalar"),
    notes="integer_pow with exponent=3",
)(lambda x: lax.integer_pow(x, 3))

# =============================================================================
# Exponential and trigonometric
# =============================================================================

op_spec(
    "exp",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.exp,
    torch_aten=AtenSpec("exp", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.exp)
op_spec(
    "exp2",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.exp2,
    torch_aten=AtenSpec("exp2", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.exp2)
op_spec(
    "expm1",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.expm1,
    torch_aten=AtenSpec("expm1", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.expm1)
op_spec(
    "log",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.log,
    torch_aten=AtenSpec("log", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.log)
op_spec(
    "log1p",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.log1p,
    torch_aten=AtenSpec("log1p", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.log1p)
op_spec(
    "sqrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.sqrt,
    torch_aten=AtenSpec("sqrt", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.sqrt)
op_spec(
    "rsqrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.rsqrt,
    torch_aten=AtenSpec("rsqrt", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.rsqrt)
op_spec(
    "cbrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=lambda x: torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0),
    input_domain=InputDomain.POSITIVE,
    notes="PyTorch lacks native cbrt; using sign(x)*pow(|x|, 1/3)",
)(lax.cbrt)

op_spec(
    "sin",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.sin,
    torch_aten=AtenSpec("sin", "default"),
)(lax.sin)
op_spec(
    "cos",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.cos,
    torch_aten=AtenSpec("cos", "default"),
)(lax.cos)
op_spec(
    "tan",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.tan,
    torch_aten=AtenSpec("tan", "default"),
)(lax.tan)
op_spec(
    "asin",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.asin,
    torch_aten=AtenSpec("asin", "default"),
    input_domain=InputDomain.UNIT,
)(lax.asin)
op_spec(
    "acos",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.acos,
    torch_aten=AtenSpec("acos", "default"),
    input_domain=InputDomain.UNIT,
)(lax.acos)
op_spec(
    "atan",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.atan,
    torch_aten=AtenSpec("atan", "default"),
)(lax.atan)
op_spec(
    "atan2",
    "exp_trig",
    OpArity.BINARY,
    torch_fn=torch.atan2,
    torch_aten=AtenSpec("atan2", "default"),
)(lax.atan2)

op_spec(
    "sinh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.sinh,
    torch_aten=AtenSpec("sinh", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.sinh)
op_spec(
    "cosh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.cosh,
    torch_aten=AtenSpec("cosh", "default"),
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.cosh)
op_spec(
    "tanh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.tanh,
    torch_aten=AtenSpec("tanh", "default"),
)(lax.tanh)
op_spec(
    "asinh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.asinh,
    torch_aten=AtenSpec("asinh", "default"),
)(lax.asinh)
op_spec(
    "acosh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.acosh,
    torch_aten=AtenSpec("acosh", "default"),
    input_domain=InputDomain.ABOVE_ONE,
)(lax.acosh)
op_spec(
    "atanh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.atanh,
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
    torch_fn=torch.sigmoid,
    torch_aten=AtenSpec("sigmoid", "default"),
    notes="logistic = sigmoid = 1/(1+exp(-x))",
)(lax.logistic)

op_spec(
    "reduce_sum",
    "normalization",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.sum(x, dim=axis),
    torch_aten=AtenSpec("sum", "dim_IntList"),
    shape_type="reduction",
)(lambda x, axis: jnp.sum(x, axis=axis))
op_spec(
    "reduce_max",
    "normalization",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.max(x, dim=axis).values,
    torch_aten=AtenSpec("amax", "default"),
    shape_type="reduction",
)(lambda x, axis: jnp.max(x, axis=axis))
op_spec(
    "reduce_min",
    "normalization",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.min(x, dim=axis).values,
    torch_aten=AtenSpec("amin", "default"),
    shape_type="reduction",
)(lambda x, axis: jnp.min(x, axis=axis))
op_spec(
    "cumsum",
    "normalization",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.cumsum(x, dim=axis),
    torch_aten=AtenSpec("cumsum", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.cumsum(x, axis=axis))
op_spec(
    "cumprod",
    "normalization",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.cumprod(x, dim=axis),
    torch_aten=AtenSpec("cumprod", "default"),
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.cumprod(x, axis=axis))

# =============================================================================
# Activation functions
# =============================================================================

op_spec(
    "sigmoid",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.sigmoid,
    torch_aten=AtenSpec("sigmoid", "default"),
    notes="Same as logistic; primary activation function",
)(lax.logistic)
op_spec(
    "tanh_act",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.tanh,
    torch_aten=AtenSpec("tanh", "default"),
    notes="tanh as activation function",
)(lax.tanh)
op_spec(
    "erf",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.erf,
    torch_aten=AtenSpec("erf", "default"),
)(lax.erf)
op_spec(
    "erfc",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.erfc,
    torch_aten=AtenSpec("erfc", "default"),
)(lax.erfc)
op_spec(
    "erf_inv",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.erfinv,
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
    torch_fn=torch.eq,
    torch_aten=AtenSpec("eq", "Tensor"),
    notes="Returns bool",
)(lax.eq)
op_spec(
    "ne",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ne,
    torch_aten=AtenSpec("ne", "Tensor"),
    notes="Returns bool",
)(lax.ne)
op_spec(
    "lt",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.lt,
    torch_aten=AtenSpec("lt", "Tensor"),
    notes="Returns bool",
)(lax.lt)
op_spec(
    "le",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.le,
    torch_aten=AtenSpec("le", "Tensor"),
    notes="Returns bool",
)(lax.le)
op_spec(
    "gt",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.gt,
    torch_aten=AtenSpec("gt", "Tensor"),
    notes="Returns bool",
)(lax.gt)
op_spec(
    "ge",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ge,
    torch_aten=AtenSpec("ge", "Tensor"),
    notes="Returns bool",
)(lax.ge)
op_spec(
    "clamp",
    "comparison",
    OpArity.TERNARY,
    torch_fn=lambda lo, x, hi: torch.clamp(x, min=lo, max=hi),
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


def _jax_bitwise_and(x, y):
    return lax.bitwise_and(_jax_i32(x), _jax_i32(y))


def _torch_bitwise_and(x, y):
    return torch.bitwise_and(_torch_i32(x), _torch_i32(y))


def _jax_bitwise_not(x):
    return lax.bitwise_not(_jax_i32(x))


def _torch_bitwise_not(x):
    return torch.bitwise_not(_torch_i32(x))


def _jax_bitwise_or(x, y):
    return lax.bitwise_or(_jax_i32(x), _jax_i32(y))


def _torch_bitwise_or(x, y):
    return torch.bitwise_or(_torch_i32(x), _torch_i32(y))


def _jax_bitwise_xor(x, y):
    return lax.bitwise_xor(_jax_i32(x), _jax_i32(y))


def _torch_bitwise_xor(x, y):
    return torch.bitwise_xor(_torch_i32(x), _torch_i32(y))


def _jax_shift_left(x, y):
    return lax.shift_left(_jax_i32(x), _jax_shift_amount(y))


def _torch_shift_left(x, y):
    return torch.bitwise_left_shift(_torch_i32(x), _torch_shift_amount(y).to(torch.int32))


def _jax_shift_right_arithmetic(x, y):
    return lax.shift_right_arithmetic(_jax_i32(x), _jax_shift_amount(y))


def _torch_shift_right_arithmetic(x, y):
    return torch.bitwise_right_shift(_torch_i32(x), _torch_shift_amount(y).to(torch.int32))


def _jax_shift_right_logical(x, y):
    return lax.shift_right_logical(_jax_i32(x), _jax_shift_amount(y))


def _torch_shift_right_logical(x, y):
    x_i32 = _torch_i32(x)
    y_i64 = _torch_shift_amount(y)
    return ((x_i32.to(torch.int64) & 0xFFFFFFFF) >> y_i64).to(torch.int32)


def _jax_concatenate(x, y):
    return lax.concatenate((x, y), dimension=x.ndim - 1)


def _torch_concatenate(x, y):
    return torch.cat((x, y), dim=x.ndim - 1)


def _jax_full_from_x(x):
    return lax.full(x.shape, jnp.mean(x, dtype=jnp.float32), x.dtype)


def _torch_full_from_x(x):
    fill = torch.mean(x.float()).to(x.dtype)
    return torch.full(x.shape, fill, dtype=x.dtype, device=x.device)


def _jax_full_like_from_x(x):
    return lax.full_like(x, jnp.mean(x, dtype=jnp.float32))


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


def _jax_reshape_reverse(x):
    return lax.reshape(x, x.shape[::-1])


def _torch_reshape_reverse(x):
    return torch.reshape(x, x.shape[::-1])


def _jax_rev_last(x):
    return lax.rev(x, dimensions=(x.ndim - 1,))


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


def _jax_squeeze_noop(x):
    return lax.squeeze(lax.expand_dims(x, (0,)), dimensions=(0,))


def _torch_squeeze_noop(x):
    return torch.squeeze(torch.unsqueeze(x, 0), 0)


def _jax_tile_leading(x):
    reps = (2,) + (1,) * (x.ndim - 1)
    return lax.tile(x, reps)


def _torch_tile_leading(x):
    reps = (2,) + (1,) * (x.ndim - 1)
    return torch.tile(x, reps)


def _jax_top_k_values(x):
    return lax.top_k(x, k=min(5, x.shape[-1]))[0]


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
    torch_fn=_torch_bitwise_and,
    torch_aten=AtenSpec("bitwise_and", "Tensor"),
    supported_dtypes=("float32",),
)(_jax_bitwise_and)
op_spec(
    "bitwise_not",
    "bitwise",
    OpArity.UNARY,
    torch_fn=_torch_bitwise_not,
    torch_aten=AtenSpec("bitwise_not", "default"),
    supported_dtypes=("float32",),
)(_jax_bitwise_not)
op_spec(
    "bitwise_or",
    "bitwise",
    OpArity.BINARY,
    torch_fn=_torch_bitwise_or,
    torch_aten=AtenSpec("bitwise_or", "Tensor"),
    supported_dtypes=("float32",),
)(_jax_bitwise_or)
op_spec(
    "bitwise_xor",
    "bitwise",
    OpArity.BINARY,
    torch_fn=_torch_bitwise_xor,
    torch_aten=AtenSpec("bitwise_xor", "Tensor"),
    supported_dtypes=("float32",),
)(_jax_bitwise_xor)
op_spec(
    "shift_left",
    "bitwise",
    OpArity.BINARY,
    torch_fn=_torch_shift_left,
    torch_aten=AtenSpec("bitwise_left_shift", "Tensor"),
    supported_dtypes=("float32",),
)(_jax_shift_left)
op_spec(
    "shift_right_arithmetic",
    "bitwise",
    OpArity.BINARY,
    torch_fn=_torch_shift_right_arithmetic,
    torch_aten=AtenSpec("bitwise_right_shift", "Tensor"),
    supported_dtypes=("float32",),
)(_jax_shift_right_arithmetic)
op_spec(
    "shift_right_logical",
    "bitwise",
    OpArity.BINARY,
    torch_fn=_torch_shift_right_logical,
    torch_aten=AtenSpec("bitwise_right_shift", "Tensor"),
    supported_dtypes=("float32",),
)(_jax_shift_right_logical)

op_spec(
    "concatenate",
    "shape",
    OpArity.BINARY,
    torch_fn=_torch_concatenate,
    torch_aten=AtenSpec("cat", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_concatenate)
op_spec(
    "full",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_full_from_x,
    torch_aten=AtenSpec("full", "default"),
    supported_dtypes=("float32",),
)(_jax_full_from_x)
op_spec(
    "full_like",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_full_like_from_x,
    torch_aten=AtenSpec("full_like", "default"),
    supported_dtypes=("float32",),
)(_jax_full_like_from_x)
op_spec(
    "reshape",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_reshape_reverse,
    torch_aten=AtenSpec("reshape", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_reshape_reverse)
op_spec(
    "rev",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_rev_last,
    torch_aten=AtenSpec("flip", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_rev_last)
op_spec(
    "split",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_split_first_half,
    torch_aten=AtenSpec("split", "sizes"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_split_first_half)
op_spec(
    "squeeze",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_squeeze_noop,
    torch_aten=AtenSpec("squeeze", "dim"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_squeeze_noop)
op_spec(
    "tile",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_tile_leading,
    torch_aten=AtenSpec("tile", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_tile_leading)
op_spec(
    "transpose",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_transpose_reverse,
    torch_aten=AtenSpec("permute", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_transpose_reverse)

op_spec(
    "gather",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_gather_last,
    torch_aten=AtenSpec("gather", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_gather_last)
op_spec(
    "pad",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_pad_one,
    torch_aten=AtenSpec("pad", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_pad_one)
op_spec(
    "scatter",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_set_flat,
    torch_aten=AtenSpec("scatter", "src"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_set_flat)
op_spec(
    "scatter_add",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_add_flat,
    torch_aten=AtenSpec("scatter_add", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_add_flat)
op_spec(
    "scatter_max",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_max_flat,
    torch_aten=AtenSpec("scatter_reduce", "two"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_max_flat)
op_spec(
    "scatter_min",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_min_flat,
    torch_aten=AtenSpec("scatter_reduce", "two"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_min_flat)
op_spec(
    "slice",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_slice_last_half,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_slice_last_half)
op_spec(
    "top_k",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_top_k_values,
    torch_aten=AtenSpec("topk", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_top_k_values)

# =============================================================================
# Additional direct ATen mappings from manifest
# =============================================================================


def _jax_approx_max_k_values(x):
    return lax.approx_max_k(x, k=min(5, x.shape[-1]))[0]


def _torch_approx_max_k_values(x):
    return torch.topk(x, k=min(5, x.shape[-1]), dim=-1, largest=True, sorted=True).values


def _jax_approx_min_k_values(x):
    return lax.approx_min_k(x, k=min(5, x.shape[-1]))[0]


def _torch_approx_min_k_values(x):
    return torch.topk(x, k=min(5, x.shape[-1]), dim=-1, largest=False, sorted=True).values


def _jax_broadcast_leading2(x):
    return lax.broadcast(x, sizes=(2,))


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


def _jax_broadcast_to_rank_plus1(x):
    return lax.broadcast_to_rank(x, x.ndim + 1)


def _torch_broadcast_to_rank_plus1(x):
    return torch.unsqueeze(x, 0)


def _jax_broadcasted_iota_last(x):
    return lax.broadcasted_iota(x.dtype, shape=x.shape, dimension=x.ndim - 1)


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


def _jax_expand_dims_front(x):
    return lax.expand_dims(x, (0,))


def _torch_expand_dims_front(x):
    return torch.unsqueeze(x, 0)


def _jax_index_in_dim_last(x):
    return lax.index_in_dim(x, x.shape[-1] // 2, axis=-1, keepdims=False)


def _torch_index_in_dim_last(x):
    return torch.select(x, -1, x.shape[-1] // 2)


def _jax_index_take_last_two(x):
    idx = jnp.array([[0, x.shape[-1] - 1]], dtype=jnp.int32)
    return lax.index_take(x, idx, axes=(x.ndim - 1,))


def _torch_index_take_last_two(x):
    idx = torch.tensor([0, x.shape[-1] - 1], dtype=torch.int64, device=x.device)
    return torch.index_select(x, x.ndim - 1, idx).movedim(-1, 0)


def _jax_iota_last_dim(x):
    return lax.iota(x.dtype, x.shape[-1])


def _torch_iota_last_dim(x):
    return torch.arange(0, x.shape[-1], dtype=x.dtype, device=x.device)


def _jax_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return lax.slice_in_dim(x, 0, k, stride=1, axis=-1)


def _torch_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return x[..., :k]


def _jax_sort_key_val_keys(x):
    return lax.sort_key_val(x, -x, dimension=-1, is_stable=True)[0]


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


def _jax_reduce_and_last(x, axis):
    return lax.reduce_and(x > 0, axes=(axis,))


def _torch_reduce_and_last(x, axis):
    return torch.all(x > 0, dim=axis)


def _jax_reduce_or_last(x, axis):
    return lax.reduce_or(x > 0, axes=(axis,))


def _torch_reduce_or_last(x, axis):
    return torch.any(x > 0, dim=axis)


op_spec(
    "approx_max_k",
    "reduction",
    OpArity.UNARY,
    torch_fn=_torch_approx_max_k_values,
    torch_aten=AtenSpec("topk", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="Top-k approximation values only",
)(_jax_approx_max_k_values)
op_spec(
    "approx_min_k",
    "reduction",
    OpArity.UNARY,
    torch_fn=_torch_approx_min_k_values,
    torch_aten=AtenSpec("topk", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="Bottom-k approximation values only",
)(_jax_approx_min_k_values)
op_spec(
    "broadcast",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_broadcast_leading2,
    torch_aten=AtenSpec("expand", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_broadcast_leading2)
op_spec(
    "broadcast_in_dim",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_broadcast_in_dim_leading2,
    torch_aten=AtenSpec("expand", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_broadcast_in_dim_leading2)
op_spec(
    "broadcast_to_rank",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_broadcast_to_rank_plus1,
    torch_aten=AtenSpec("unsqueeze", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_broadcast_to_rank_plus1)
op_spec(
    "broadcasted_iota",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_broadcasted_iota_last,
    torch_aten=AtenSpec("arange", "start_step"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_broadcasted_iota_last)
op_spec(
    "collapse",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_collapse_first2,
    torch_aten=AtenSpec("reshape", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_collapse_first2)
op_spec(
    "expand_dims",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_expand_dims_front,
    torch_aten=AtenSpec("unsqueeze", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_expand_dims_front)
op_spec(
    "iota",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_iota_last_dim,
    torch_aten=AtenSpec("arange", "start_step"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_iota_last_dim)

op_spec(
    "dynamic_index_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_dynamic_index_in_dim_last,
    torch_aten=AtenSpec("select", "int"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_index_in_dim_last)
op_spec(
    "dynamic_slice",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_dynamic_slice_last_half,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_slice_last_half)
op_spec(
    "dynamic_slice_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_dynamic_slice_in_dim_last_half,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_slice_in_dim_last_half)
op_spec(
    "dynamic_update_slice",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_slice_prefix,
    torch_aten=AtenSpec("slice_scatter", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_update_slice_prefix)
op_spec(
    "dynamic_update_slice_in_dim",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_slice_in_dim_prefix,
    torch_aten=AtenSpec("slice_scatter", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_update_slice_in_dim_prefix)
op_spec(
    "index_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_index_in_dim_last,
    torch_aten=AtenSpec("select", "int"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_index_in_dim_last)
op_spec(
    "index_take",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_index_take_last_two,
    torch_aten=AtenSpec("index_select", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_index_take_last_two)
op_spec(
    "slice_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_slice_in_dim_last_half,
    torch_aten=AtenSpec("slice", "Tensor"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_slice_in_dim_last_half)
op_spec(
    "sort_key_val",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_sort_key_val_keys,
    torch_aten=AtenSpec("sort", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="Compare sorted keys only",
)(_jax_sort_key_val_keys)
op_spec(
    "scatter_mul",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_mul_flat,
    torch_aten=AtenSpec("scatter_reduce", "two"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_mul_flat)
op_spec(
    "scatter_sub",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_scatter_sub_flat,
    torch_aten=AtenSpec("scatter_add", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_scatter_sub_flat)

op_spec(
    "reduce_and",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=_torch_reduce_and_last,
    torch_aten=AtenSpec("all", "dim"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_reduce_and_last)
op_spec(
    "reduce_or",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=_torch_reduce_or_last,
    torch_aten=AtenSpec("any", "dim"),
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_reduce_or_last)

# =============================================================================
# Rounding operations
# =============================================================================

op_spec(
    "floor",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.floor,
    torch_aten=AtenSpec("floor", "default"),
)(lax.floor)
op_spec(
    "ceil",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.ceil,
    torch_aten=AtenSpec("ceil", "default"),
)(lax.ceil)
op_spec(
    "round",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.round,
    torch_aten=AtenSpec("round", "default"),
    notes="Both use half-to-even (banker's rounding)",
)(lambda x: lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN))
op_spec(
    "is_finite",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.isfinite,
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
    torch_fn=torch.complex,
    torch_aten=AtenSpec("complex", "default"),
    supported_dtypes=("float32",),
    notes="Constructs complex from real and imag parts",
)(lax.complex)
op_spec(
    "conj",
    "complex",
    OpArity.UNARY,
    torch_fn=lambda x: torch.conj(x).resolve_conj(),
    torch_aten=AtenSpec("conj", "default"),
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
    notes="torch.conj returns lazy view; need resolve_conj() for numpy()",
)(lax.conj)
op_spec(
    "real",
    "complex",
    OpArity.UNARY,
    torch_fn=torch.real,
    torch_aten=AtenSpec("real", "default"),
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
)(lax.real)
op_spec(
    "imag",
    "complex",
    OpArity.UNARY,
    torch_fn=torch.imag,
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
    torch_fn=lambda x, axis: torch.prod(x, dim=axis),
    torch_aten=AtenSpec("prod", "dim_int"),
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: jnp.prod(x, axis=axis))
op_spec(
    "cummax",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.cummax(x, dim=axis).values,
    torch_aten=AtenSpec("cummax", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.cummax(x, axis=axis))
op_spec(
    "cummin",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.cummin(x, dim=axis).values,
    torch_aten=AtenSpec("cummin", "default"),
    shape_type="reduction",
)(lambda x, axis: lax.cummin(x, axis=axis))
op_spec(
    "cumlogsumexp",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.logcumsumexp(x, dim=axis),
    torch_aten=AtenSpec("logcumsumexp", "default"),
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.cumlogsumexp(x, axis=axis))

op_spec(
    "sort",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.sort(x, dim=axis).values,
    torch_aten=AtenSpec("sort", "default"),
    shape_type="reduction",
    notes="Compares sorted values along axis",
)(lambda x, axis: lax.sort(x, dimension=axis))
op_spec(
    "argmax",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.argmax(x, dim=axis),
    torch_aten=AtenSpec("argmax", "default"),
    shape_type="reduction",
    notes="Returns int indices; abs error = 0 means frameworks agree",
)(lambda x, axis: jnp.argmax(x, axis=axis))
op_spec(
    "argmin",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.argmin(x, dim=axis),
    torch_aten=AtenSpec("argmin", "default"),
    shape_type="reduction",
    notes="Returns int indices; abs error = 0 means frameworks agree",
)(lambda x, axis: jnp.argmin(x, axis=axis))
op_spec(
    "top_k_values",
    "reduction",
    OpArity.UNARY,
    torch_fn=lambda x: torch.topk(x, k=min(5, x.shape[-1]), dim=-1).values,
    torch_aten=AtenSpec("topk", "default"),
    shape_type="reduction",
    notes="Top-k values along last axis (k=min(5, size))",
)(lambda x: lax.top_k(x, k=min(5, x.shape[-1]))[0])

# =============================================================================
# Special mathematical functions
# =============================================================================

op_spec(
    "lgamma",
    "special",
    OpArity.UNARY,
    torch_fn=torch.lgamma,
    torch_aten=AtenSpec("lgamma", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.lgamma)
op_spec(
    "digamma",
    "special",
    OpArity.UNARY,
    torch_fn=torch.digamma,
    torch_aten=AtenSpec("digamma", "default"),
    input_domain=InputDomain.POSITIVE,
)(lax.digamma)
op_spec(
    "polygamma_1",
    "special",
    OpArity.UNARY,
    torch_fn=lambda x: torch.special.polygamma(1, x),
    torch_aten=AtenSpec("special_polygamma", "default"),
    input_domain=InputDomain.POSITIVE,
    notes="Trigamma function: polygamma(1, x)",
)(lambda x: jsp.polygamma(1, x))
op_spec(
    "bessel_i0e",
    "special",
    OpArity.UNARY,
    torch_fn=torch.special.i0e,
    torch_aten=AtenSpec("special_i0e", "default"),
)(lax.bessel_i0e)
op_spec(
    "bessel_i1e",
    "special",
    OpArity.UNARY,
    torch_fn=torch.special.i1e,
    torch_aten=AtenSpec("special_i1e", "default"),
)(lax.bessel_i1e)
op_spec(
    "igamma",
    "special",
    OpArity.BINARY,
    torch_fn=torch.special.gammainc,
    torch_aten=AtenSpec("special_gammainc", "default"),
    input_domain=InputDomain.POSITIVE,
    notes="JAX igamma(a,x) = torch.special.gammainc(a,x)",
)(lax.igamma)
op_spec(
    "igammac",
    "special",
    OpArity.BINARY,
    torch_fn=torch.special.gammaincc,
    torch_aten=AtenSpec("special_gammaincc", "default"),
    input_domain=InputDomain.POSITIVE,
    notes="JAX igammac(a,x) = torch.special.gammaincc(a,x)",
)(lax.igammac)
op_spec(
    "betainc",
    "special",
    OpArity.TERNARY,
    torch_fn=None,
    input_domain=InputDomain.POSITIVE,
    notes="No PyTorch equivalent",
)(lax.betainc)

# =============================================================================
# Type conversion operations
# =============================================================================

op_spec(
    "convert_f32_to_bf16",
    "type_ops",
    OpArity.TYPE_CAST,
    torch_fn=lambda x: x.to(torch.bfloat16),
    torch_aten=AtenSpec("to", "dtype"),
    supported_dtypes=("float32",),
    notes="Cast float32 -> bfloat16",
)(lambda x: lax.convert_element_type(x, jnp.bfloat16))
op_spec(
    "convert_bf16_to_f32",
    "type_ops",
    OpArity.TYPE_CAST,
    torch_fn=lambda x: x.to(torch.float32),
    torch_aten=AtenSpec("to", "dtype"),
    supported_dtypes=("bfloat16",),
    notes="Cast bfloat16 -> float32",
)(lambda x: lax.convert_element_type(x, jnp.float32))
op_spec(
    "reduce_precision_fp16",
    "type_ops",
    OpArity.TYPE_CAST,
    torch_fn=None,
    supported_dtypes=("float32",),
    notes="Simulates FP16 precision; no PyTorch equivalent",
)(lambda x: lax.reduce_precision(x, exponent_bits=5, mantissa_bits=10))
op_spec(
    "bitcast_f32_to_i32",
    "type_ops",
    OpArity.TYPE_CAST,
    torch_fn=lambda x: x.view(torch.int32),
    torch_aten=AtenSpec("view", "dtype"),
    supported_dtypes=("float32",),
    notes="Reinterpret float32 bits as int32",
)(lambda x: lax.bitcast_convert_type(x, jnp.int32))

# =============================================================================
# Linear algebra operations
# =============================================================================

op_spec(
    "dot",
    "linalg",
    OpArity.MATMUL,
    torch_fn=torch.matmul,
    torch_aten=AtenSpec("matmul", "default"),
    shape_type="matmul",
    supported_dtypes=("float32", "bfloat16"),
)(lax.dot)
op_spec(
    "batch_matmul",
    "linalg",
    OpArity.MATMUL,
    torch_fn=torch.bmm,
    torch_aten=AtenSpec("bmm", "default"),
    shape_type="batch_matmul",
    supported_dtypes=("float32", "bfloat16"),
)(lax.batch_matmul)

# --- Cholesky decomposition ---
op_spec(
    "cholesky",
    "linalg",
    OpArity.UNARY,
    torch_fn=lambda x: torch.linalg.cholesky(x),
    torch_aten=AtenSpec("linalg_cholesky", "default"),
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Input: positive definite matrix",
)(lambda x: jax.lax.linalg.cholesky(x))

# --- Eigenvalues of symmetric matrices ---
op_spec(
    "eigh_eigenvalues",
    "linalg",
    OpArity.UNARY,
    torch_fn=lambda x: torch.linalg.eigh(x).eigenvalues,
    torch_aten=AtenSpec("linalg_eigh", "default"),
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Eigenvalues only (eigenvectors have sign ambiguity)",
)(# JAX eigh returns (eigenvectors, eigenvalues) — opposite of numpy
                lambda x: jax.lax.linalg.eigh(x)[1])

# --- Singular values ---
op_spec(
    "svd_singular_values",
    "linalg",
    OpArity.UNARY,
    torch_fn=lambda x: torch.linalg.svdvals(x),
    torch_aten=AtenSpec("linalg_svdvals", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Singular values only (U/V have sign ambiguity)",
)(lambda x: jax.lax.linalg.svd(x, full_matrices=False)[1])

# --- QR: compare R with sign normalization ---
def _jax_qr_r(x):
    _, r = jax.lax.linalg.qr(x)
    signs = jnp.sign(jnp.diag(r))
    signs = jnp.where(signs == 0, 1.0, signs)
    return r * signs[:, None]

def _torch_qr_r(x):
    _, r = torch.linalg.qr(x)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return r * signs[:, None]

op_spec(
    "qr_r",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_qr_r,
    torch_aten=AtenSpec("linalg_qr", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="R factor with diagonal sign normalization for uniqueness",
)(_jax_qr_r)

# --- Triangular solve ---
op_spec(
    "triangular_solve",
    "linalg",
    OpArity.MATMUL,
    torch_fn=lambda a, b: torch.linalg.solve_triangular(a, b, upper=False),
    torch_aten=AtenSpec("linalg_solve_triangular", "default"),
    input_domain=InputDomain.LOWER_TRIANGULAR,
    shape_type="linalg_solve",
    supported_dtypes=("float32",),
    notes="Solve A @ x = b where A is lower triangular",
)(lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True))



def _jax_linalg_qr_q(x):
    return jax.lax.linalg.qr(x)[0]


def _torch_linalg_qr_q(x):
    return torch.linalg.qr(x).Q


def _jax_linalg_svd_u(x):
    return jax.lax.linalg.svd(x, full_matrices=False)[0]


def _torch_linalg_svd_u(x):
    return torch.linalg.svd(x, full_matrices=False).U


def _jax_linalg_eigh_vecs(x):
    return jax.lax.linalg.eigh(x)[0]


def _torch_linalg_eigh_vecs(x):
    return torch.linalg.eigh(x).eigenvectors


def _jax_linalg_eig_vecs(x):
    return jax.lax.linalg.eig(x, compute_left_eigenvectors=False, compute_right_eigenvectors=True)[1]


def _torch_linalg_eig_vecs(x):
    return torch.linalg.eig(x).eigenvectors


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


def _jax_linalg_lu_matrix(x):
    return jax.lax.linalg.lu(x)[0]


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


def _jax_zeta(x, y):
    return lax.zeta(jnp.abs(x) + 2.0, jnp.abs(y) + 1.0)


def _torch_zeta(x, y):
    return torch.special.zeta(torch.abs(x) + 2.0, torch.abs(y) + 1.0)


op_spec(
    "eig",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_linalg_eig_vecs,
    torch_aten=AtenSpec("linalg_eig", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Right eigenvectors only",
)(_jax_linalg_eig_vecs)
op_spec(
    "eigh",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_linalg_eigh_vecs,
    torch_aten=AtenSpec("linalg_eigh", "default"),
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Eigenvectors only",
)(_jax_linalg_eigh_vecs)
op_spec(
    "householder_product",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_linalg_householder_product,
    torch_aten=AtenSpec("linalg_householder_product", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Householder product from synthetic taus",
)(_jax_linalg_householder_product)
op_spec(
    "lu",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_linalg_lu_matrix,
    torch_aten=AtenSpec("linalg_lu", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Compares P matrix",
)(_jax_linalg_lu_matrix)
op_spec(
    "lu_pivots_to_permutation",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_lu_pivots_to_perm,
    torch_aten=AtenSpec("linalg_lu_factor_ex", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Permutation from LU pivots",
)(_jax_lu_pivots_to_perm)
op_spec(
    "qr",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_linalg_qr_q,
    torch_aten=AtenSpec("linalg_qr", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Q factor",
)(_jax_linalg_qr_q)
op_spec(
    "svd",
    "linalg",
    OpArity.UNARY,
    torch_fn=_torch_linalg_svd_u,
    torch_aten=AtenSpec("linalg_svd", "default"),
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="U matrix",
)(_jax_linalg_svd_u)
op_spec(
    "zeta",
    "special",
    OpArity.BINARY,
    torch_fn=_torch_zeta,
    torch_aten=AtenSpec("special_zeta", "default"),
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="Hurwitz zeta with shifted positive domain",
)(_jax_zeta)


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
    torch_fn=_torch_conv,
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
    torch_fn=lambda x: torch.fft.fft(x),
    torch_aten=AtenSpec("fft_fft", "default"),
    input_domain=InputDomain.COMPLEX,
    shape_type="fft",
    supported_dtypes=("float32",),
    notes="Complex-to-complex FFT",
)(lambda x: lax.fft(x, fft_type="FFT", fft_lengths=(x.shape[-1],)))
op_spec(
    "ifft",
    "fft",
    OpArity.FFT,
    torch_fn=lambda x: torch.fft.ifft(x),
    torch_aten=AtenSpec("fft_ifft", "default"),
    input_domain=InputDomain.COMPLEX,
    shape_type="fft",
    supported_dtypes=("float32",),
    notes="Complex-to-complex inverse FFT",
)(lambda x: lax.fft(x, fft_type="IFFT", fft_lengths=(x.shape[-1],)))


# =============================================================================
# Remaining manifest ops (no torch mapping — registered as MISSING placeholders)
# =============================================================================

# Ops already covered by explicit registrations above (variant names).
_ALREADY_COVERED = {
    "convert_element_type", "integer_pow", "reduce_precision",
    "bitcast_convert_type", "nextafter", "polygamma", "sort",
    "argmax", "argmin",
}


def _register_remaining_ops():
    """Register remaining manifest ops as MISSING placeholders (torch_fn=None)."""
    manifest_path = Path(__file__).resolve().parents[3] / "docs" / "jax_lax_operators.json"
    groups = json.loads(manifest_path.read_text())["operators"]
    existing = {op.name for op in get_all_ops()}

    for name, group in (
        [(n, "general") for n in groups["general_operators"]]
        + [(n, "sharding") for n in groups["sharding_related_operators"]]
        + [(n, "linalg") for n in groups["linear_algebra_operators"]]
    ):
        if name in existing or name in _ALREADY_COVERED:
            continue
        if group == "linalg":
            if not hasattr(lax.linalg, name):
                continue
            jax_fn = getattr(lax.linalg, name)
        else:
            if not hasattr(lax, name):
                continue
            jax_fn = getattr(lax, name)

        op_spec(
            name,
            "unmapped",
            OpArity.UNARY,
            torch_fn=None,
            notes="No torch mapping",
        )(jax_fn)


_register_remaining_ops()


# =============================================================================
# Strict 1:1 mapping enforcement
# =============================================================================

# Operators below currently rely on composed semantics, partial outputs,
# or transformed inputs; they are disabled in strict one-to-one mode.
_STRICT_NON_1TO1_OPS = {
    "approx_max_k", "approx_min_k",
    "bitwise_and", "bitwise_not", "bitwise_or", "bitwise_xor",
    "shift_left", "shift_right_arithmetic", "shift_right_logical",
    "dynamic_index_in_dim", "dynamic_slice", "dynamic_slice_in_dim",
    "dynamic_update_slice", "dynamic_update_slice_in_dim",
    "index_in_dim", "index_take", "slice_in_dim",
    "sort_key_val",
    "top_k", "top_k_values",
    "reduce_and", "reduce_or",
    "full", "full_like",
    "eig", "eigh", "qr", "svd", "lu", "householder_product",
    "lu_pivots_to_permutation",
    "zeta",
}


def _disable_strict_non_1to1_mappings():
    seen_aten: dict[tuple[str, str], str] = {}

    for op in get_all_ops():
        if op.name in _STRICT_NON_1TO1_OPS:
            op.torch_fn = None
            op.torch_aten = None
            op.notes = "No strict 1:1 ATen mapping"
            continue

        if op.torch_aten is None:
            continue

        key = (op.torch_aten.name, op.torch_aten.overload)
        if key in seen_aten:
            op.torch_fn = None
            op.torch_aten = None
            op.notes = f"No strict 1:1 ATen mapping (duplicate of {seen_aten[key]})"
            continue

        seen_aten[key] = op.name


_disable_strict_non_1to1_mappings()


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


@output_adapter("eigh_eigenvalues")
def _adapt_eigh_eigenvalues(output, _inputs):
    if isinstance(output, tuple):
        return output[0]
    return output


@output_adapter("qr_r")
def _adapt_qr_r(output, _inputs):
    if isinstance(output, tuple):
        r = output[1]
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return r * signs[:, None]
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


@output_adapter("conj")
def _adapt_conj_resolved(output, _inputs):
    return output.resolve_conj() if hasattr(output, "resolve_conj") else output


@aten_plan("integer_pow_2")
def _build_integer_pow_2(inputs):
    return (inputs["x"], 2), {}


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


@aten_plan("cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp")
def _build_reduce_dim_no_keepdim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, axis), {}


@aten_plan("clamp")
def _build_clamp(inputs):
    return (inputs["x"], inputs["lo"], inputs["hi"]), {}


@aten_plan("concatenate")
def _build_concatenate(inputs):
    x = inputs["x"]
    return ((x, inputs["y"]), x.ndim - 1), {}


@aten_plan("reshape")
def _build_reshape_reverse(inputs):
    x = inputs["x"]
    return (x, x.shape[::-1]), {}


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


@aten_plan("polygamma_1")
def _build_polygamma_1(inputs):
    return (1, inputs["x"]), {}


@aten_plan("convert_f32_to_bf16")
def _build_convert_f32_to_bf16(inputs):
    x = inputs["x"]
    return (x, torch.bfloat16, False, False, None), {}


@aten_plan("bitcast_f32_to_i32")
def _build_bitcast_f32_to_i32(inputs):
    x = inputs["x"]
    return (x, torch.int32), {}


@aten_plan("eigh_eigenvalues")
def _build_eigh_eigenvalues(inputs):
    return (inputs["x"], "L"), {}


@aten_plan("qr_r")
def _build_qr_reduced(inputs):
    return (inputs["x"], "reduced"), {}


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


@aten_plan("fft", "ifft")
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
