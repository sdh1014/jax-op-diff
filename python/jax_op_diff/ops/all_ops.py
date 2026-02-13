"""All registered jax.lax operators for precision testing."""

import json
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import torch

from ..op_registry import op_spec, OpArity, InputDomain, get_all_ops

# =============================================================================
# Basic operations
# =============================================================================

op_spec(
    "add",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.add.Tensor,
)(lax.add)
op_spec(
    "sub",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.sub.Tensor,
)(lax.sub)
op_spec(
    "mul",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.mul.Tensor,
)(lax.mul)
op_spec(
    "div",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.div.Tensor,
    input_domain=InputDomain.NON_ZERO,
)(lax.div)
op_spec(
    "rem",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.fmod.Tensor,
    input_domain=InputDomain.NON_ZERO,
    notes="jax.lax.rem = C-style truncation remainder; torch.fmod matches this",
)(lax.rem)
op_spec(
    "pow",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.pow.Tensor_Tensor,
    input_domain=InputDomain.POSITIVE,
)(lax.pow)
op_spec(
    "integer_pow",
    "basic",
    OpArity.UNARY,
    torch_fn=lambda x: torch.pow(x, 3),
    supported_dtypes=("float32",),
    notes="integer_pow with exponent=3",
)(lambda x: lax.integer_pow(x, 3))
op_spec(
    "max",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.maximum.default,
)(lax.max)
op_spec(
    "min",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.minimum.default,
)(lax.min)
op_spec(
    "nextafter",
    "basic",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.nextafter.default,
    notes="Next representable floating-point value towards y",
)(lax.nextafter)

op_spec(
    "neg",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.neg.default,
)(lax.neg)
op_spec(
    "abs",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.abs.default,
)(lax.abs)
op_spec(
    "reciprocal",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.reciprocal.default,
    input_domain=InputDomain.NON_ZERO,
)(lax.reciprocal)
op_spec(
    "square",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.square.default,
)(lax.square)
op_spec(
    "sign",
    "basic",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.sign.default,
)(lax.sign)

# =============================================================================
# Exponential and trigonometric
# =============================================================================

op_spec(
    "exp",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.exp.default,
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.exp)
op_spec(
    "exp2",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.exp2.default,
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.exp2)
op_spec(
    "expm1",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.expm1.default,
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.expm1)
op_spec(
    "log",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.log.default,
    input_domain=InputDomain.POSITIVE,
)(lax.log)
op_spec(
    "log1p",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.log1p.default,
    input_domain=InputDomain.POSITIVE,
)(lax.log1p)
op_spec(
    "sqrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.sqrt.default,
    input_domain=InputDomain.POSITIVE,
)(lax.sqrt)
op_spec(
    "rsqrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.rsqrt.default,
    input_domain=InputDomain.POSITIVE,
)(lax.rsqrt)
op_spec(
    "cbrt",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=None,
    input_domain=InputDomain.POSITIVE,
    notes="TODO: no strict 1:1 ATen mapping",
)(lax.cbrt)

op_spec(
    "sin",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.sin.default,
)(lax.sin)
op_spec(
    "cos",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.cos.default,
)(lax.cos)
op_spec(
    "tan",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.tan.default,
)(lax.tan)
op_spec(
    "asin",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.asin.default,
    input_domain=InputDomain.UNIT,
)(lax.asin)
op_spec(
    "acos",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.acos.default,
    input_domain=InputDomain.UNIT,
)(lax.acos)
op_spec(
    "atan",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.atan.default,
)(lax.atan)
op_spec(
    "atan2",
    "exp_trig",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.atan2.default,
)(lax.atan2)

op_spec(
    "sinh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.sinh.default,
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.sinh)
op_spec(
    "cosh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.cosh.default,
    input_domain=InputDomain.SMALL_POSITIVE,
)(lax.cosh)
op_spec(
    "tanh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.tanh.default,
)(lax.tanh)
op_spec(
    "asinh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.asinh.default,
)(lax.asinh)
op_spec(
    "acosh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.acosh.default,
    input_domain=InputDomain.ABOVE_ONE,
)(lax.acosh)
op_spec(
    "atanh",
    "exp_trig",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.atanh.default,
    input_domain=InputDomain.UNIT,
)(lax.atanh)

# =============================================================================
# Normalization / reductions / cumulative
# =============================================================================

op_spec(
    "logistic",
    "normalization",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.sigmoid.default,
    notes="logistic = sigmoid = 1/(1+exp(-x))",
)(lax.logistic)

op_spec(
    "reduce_sum",
    "normalization",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_intlist(inputs),
    torch_fn=torch.ops.aten.sum.dim_IntList,
    shape_type="reduction",
)(lambda x, axis: lax.reduce_sum_p.bind(x, axes=(axis,)))
op_spec(
    "reduce_max",
    "normalization",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_intlist(inputs),
    torch_fn=torch.ops.aten.amax.default,
    shape_type="reduction",
)(lambda x, axis: lax.reduce_max_p.bind(x, axes=(axis,)))
op_spec(
    "reduce_min",
    "normalization",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_intlist(inputs),
    torch_fn=torch.ops.aten.amin.default,
    shape_type="reduction",
)(lambda x, axis: lax.reduce_min_p.bind(x, axes=(axis,)))
op_spec(
    "cumsum",
    "normalization",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_no_keepdim(inputs),
    torch_fn=torch.ops.aten.cumsum.default,
    shape_type="reduction",
)(lambda x, axis: lax.cumsum(x, axis=axis))
op_spec(
    "cumprod",
    "normalization",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_no_keepdim(inputs),
    torch_fn=torch.ops.aten.cumprod.default,
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
    torch_fn=torch.ops.aten.erf.default,
)(lax.erf)
op_spec(
    "erfc",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.erfc.default,
)(lax.erfc)
op_spec(
    "erf_inv",
    "activation",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.erfinv.default,
    input_domain=InputDomain.UNIT,
)(lax.erf_inv)

# =============================================================================
# Comparison operations
# =============================================================================

op_spec(
    "eq",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.eq.Tensor,
    notes="Returns bool",
)(lax.eq)
op_spec(
    "ne",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.ne.Tensor,
    notes="Returns bool",
)(lax.ne)
op_spec(
    "lt",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.lt.Tensor,
    notes="Returns bool",
)(lax.lt)
op_spec(
    "le",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.le.Tensor,
    notes="Returns bool",
)(lax.le)
op_spec(
    "gt",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.gt.Tensor,
    notes="Returns bool",
)(lax.gt)
op_spec(
    "ge",
    "comparison",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.ge.Tensor,
    notes="Returns bool",
)(lax.ge)
op_spec(
    "clamp",
    "comparison",
    OpArity.TERNARY,
    torch_fn_builder=lambda inputs: _build_clamp(inputs),
    torch_fn=torch.ops.aten.clamp.Tensor,
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

def _torch_shift_right_logical(x, y):
    x_i32 = _torch_i32(x)
    y_i64 = _torch_shift_amount(y)
    return ((x_i32.to(torch.int64) & 0xFFFFFFFF) >> y_i64).to(torch.int32)

def _jax_gather_last(x, y):
    idx = jnp.mod(jnp.abs(_jax_i32(y)), x.shape[-1])
    return jnp.take_along_axis(x, idx, axis=-1)

def _torch_gather_last(x, y):
    idx = torch.remainder(torch.abs(y.to(torch.int64)), x.shape[-1])
    return torch.gather(x, dim=-1, index=idx)

def _jax_pad_one(x):
    config = [(1, 1, 0)] * x.ndim
    return lax.pad(x, jnp.array(0, dtype=x.dtype), config)

def _jax_scatter_index_flat(x, y):
    return jnp.mod(jnp.abs(_jax_i32(y.reshape(-1))), x.size)

def _jax_scatter_set_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].set(flat_y).reshape(x.shape)

def _jax_scatter_add_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].add(flat_y).reshape(x.shape)

def _jax_scatter_max_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].max(flat_y).reshape(x.shape)

def _jax_scatter_min_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].min(flat_y).reshape(x.shape)

def _jax_slice_last_half(x):
    k = max(1, x.shape[-1] // 2)
    starts = (0,) * x.ndim
    limits = (*x.shape[:-1], k)
    return lax.slice(x, start_indices=starts, limit_indices=limits)

def _jax_split_first_half(x):
    if x.shape[-1] <= 1:
        return x
    left = x.shape[-1] // 2
    right = x.shape[-1] - left
    return lax.split(x, sizes=(left, right), axis=-1)[0]

def _jax_tile_leading(x):
    reps = (2,) + (1,) * (x.ndim - 1)
    return lax.tile(x, reps)

def _jax_transpose_reverse(x):
    perm = tuple(range(x.ndim - 1, -1, -1))
    return lax.transpose(x, perm)

op_spec(
    "bitwise_and",
    "bitwise",
    OpArity.BINARY,
    torch_fn_builder=lambda inputs: _build_bitwise_binary_i32(inputs),
    torch_fn=torch.ops.aten.bitwise_and.Tensor,
    supported_dtypes=("float32",),
)(lambda x, y: lax.bitwise_and(_jax_i32(x), _jax_i32(y)))
op_spec(
    "bitwise_not",
    "bitwise",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_bitwise_not_i32(inputs),
    torch_fn=torch.ops.aten.bitwise_not.default,
    supported_dtypes=("float32",),
)(lambda x: lax.bitwise_not(_jax_i32(x)))
op_spec(
    "bitwise_or",
    "bitwise",
    OpArity.BINARY,
    torch_fn_builder=lambda inputs: _build_bitwise_binary_i32(inputs),
    torch_fn=torch.ops.aten.bitwise_or.Tensor,
    supported_dtypes=("float32",),
)(lambda x, y: lax.bitwise_or(_jax_i32(x), _jax_i32(y)))
op_spec(
    "bitwise_xor",
    "bitwise",
    OpArity.BINARY,
    torch_fn_builder=lambda inputs: _build_bitwise_binary_i32(inputs),
    torch_fn=torch.ops.aten.bitwise_xor.Tensor,
    supported_dtypes=("float32",),
)(lambda x, y: lax.bitwise_xor(_jax_i32(x), _jax_i32(y)))
op_spec(
    "shift_left",
    "bitwise",
    OpArity.BINARY,
    torch_fn_builder=lambda inputs: _build_shift_i32(inputs),
    torch_fn=torch.ops.aten.bitwise_left_shift.Tensor,
    supported_dtypes=("float32",),
)(lambda x, y: lax.shift_left(_jax_i32(x), _jax_shift_amount(y)))
op_spec(
    "shift_right_arithmetic",
    "bitwise",
    OpArity.BINARY,
    torch_fn_builder=lambda inputs: _build_shift_i32(inputs),
    torch_fn=torch.ops.aten.bitwise_right_shift.Tensor,
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
    torch_fn_builder=lambda inputs: _build_concatenate(inputs),
    torch_fn=torch.ops.aten.cat.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, y: lax.concatenate((x, y), dimension=x.ndim - 1))
op_spec(
    "full",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_full_from_x(inputs),
    torch_fn=torch.ops.aten.full.default,
    supported_dtypes=("float32",),
)(lambda x: lax.full(x.shape, jnp.mean(x, dtype=jnp.float32), x.dtype))
op_spec(
    "full_like",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_full_like_from_x(inputs),
    torch_fn=torch.ops.aten.full_like.default,
    supported_dtypes=("float32",),
)(lambda x: lax.full_like(x, jnp.mean(x, dtype=jnp.float32)))
op_spec(
    "reshape",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_reshape_reverse(inputs),
    torch_fn=torch.ops.aten.reshape.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.reshape(x, x.shape[::-1]))
op_spec(
    "rev",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_rev_last(inputs),
    torch_fn=torch.ops.aten.flip.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.rev(x, dimensions=(x.ndim - 1,)))
op_spec(
    "split",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_split_first_half(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_split_first(output, inputs),
    torch_fn=torch.ops.aten.split.sizes,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_split_first_half)
op_spec(
    "squeeze",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_squeeze_noop(inputs),
    torch_fn=torch.ops.aten.squeeze.dim,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.squeeze(lax.expand_dims(x, (0,)), dimensions=(0,)))
op_spec(
    "tile",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_tile_leading(inputs),
    torch_fn=torch.ops.aten.tile.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_tile_leading)
op_spec(
    "transpose",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_transpose_reverse(inputs),
    torch_fn=torch.ops.aten.permute.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_transpose_reverse)

op_spec(
    "gather",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_gather_last,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_gather_last)
op_spec(
    "pad",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_pad_one(inputs),
    torch_fn=torch.ops.aten.pad.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_pad_one)
op_spec(
    "scatter",
    "indexing",
    OpArity.BINARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes=(
        "TODO: jax.lax.scatter set semantics (conflicting indices update order) "
        "is not strict 1:1 with current PyTorch path."
    ),
)(_jax_scatter_set_flat)
op_spec(
    "scatter_add",
    "indexing",
    OpArity.BINARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes=(
        "TODO: jax.lax.scatter_add semantics with conflicting indices is not "
        "strict 1:1 with current PyTorch path."
    ),
)(_jax_scatter_add_flat)
op_spec(
    "scatter_max",
    "indexing",
    OpArity.BINARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes=(
        "TODO: jax.lax.scatter_max semantics with conflicting indices is not "
        "strict 1:1 with current PyTorch path."
    ),
)(_jax_scatter_max_flat)
op_spec(
    "scatter_min",
    "indexing",
    OpArity.BINARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes=(
        "TODO: jax.lax.scatter_min semantics with conflicting indices is not "
        "strict 1:1 with current PyTorch path."
    ),
)(_jax_scatter_min_flat)
op_spec(
    "slice",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_slice_last_half(inputs),
    torch_fn=torch.ops.aten.slice.Tensor,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_slice_last_half)
op_spec(
    "top_k",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_top_k_values(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_tuple_first(output, inputs),
    torch_fn=torch.ops.aten.topk.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.top_k(x, k=min(5, x.shape[-1]))[0])

# =============================================================================
# Additional direct ATen mappings from manifest
# =============================================================================

def _jax_broadcast_in_dim_leading2(x):
    target = (2,) + tuple(x.shape)
    dims = tuple(range(1, x.ndim + 1))
    return lax.broadcast_in_dim(x, shape=target, broadcast_dimensions=dims)

def _torch_broadcast_in_dim_leading2(x):
    target = (2,) + tuple(x.shape)
    if x.ndim == 0:
        return torch.expand(x, target)
    return torch.unsqueeze(x, 0).expand(target)

def _jax_collapse_first2(x):
    if x.ndim < 2:
        return x
    return lax.collapse(x, start_dimension=0, stop_dimension=2)

def _jax_dynamic_index_in_dim_last(x):
    idx = jnp.array(x.shape[-1] // 2, dtype=jnp.int32)
    return lax.dynamic_index_in_dim(x, idx, axis=-1, keepdims=False)

def _jax_dynamic_slice_last_half(x):
    k = max(1, x.shape[-1] // 2)
    starts = tuple(jnp.array(0, dtype=jnp.int32) for _ in range(x.ndim))
    sizes = tuple(x.shape[:-1]) + (k,)
    return lax.dynamic_slice(x, starts, sizes)

def _jax_dynamic_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return lax.dynamic_slice_in_dim(x, jnp.array(0, dtype=jnp.int32), k, axis=-1)

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

def _jax_index_take_last_two(x):
    idx = jnp.array([[0, x.shape[-1] - 1]], dtype=jnp.int32)
    return lax.index_take(x, idx, axes=(x.ndim - 1,))

def _torch_index_take_last_two(x):
    idx = torch.tensor([0, x.shape[-1] - 1], dtype=torch.int64, device=x.device)
    return torch.index_select(x, x.ndim - 1, idx).movedim(-1, 0)

def _jax_slice_in_dim_last_half(x):
    k = max(1, x.shape[-1] // 2)
    return lax.slice_in_dim(x, 0, k, stride=1, axis=-1)

def _torch_sort_key_val_keys(x):
    return torch.sort(x, dim=-1, stable=True).values

def _jax_scatter_mul_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].multiply(flat_y).reshape(x.shape)

def _jax_scatter_sub_flat(x, y):
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    idx = _jax_scatter_index_flat(x, y)
    return flat_x.at[idx].add(-flat_y).reshape(x.shape)

op_spec(
    "approx_max_k",
    "reduction",
    OpArity.UNARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="TODO: no strict 1:1 ATen mapping",
)(lambda x: lax.approx_max_k(x, k=min(5, x.shape[-1]))[0])
op_spec(
    "approx_min_k",
    "reduction",
    OpArity.UNARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes="TODO: no strict 1:1 ATen mapping",
)(lambda x: lax.approx_min_k(x, k=min(5, x.shape[-1]))[0])
op_spec(
    "broadcast",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_broadcast_leading2(inputs),
    torch_fn=torch.ops.aten.expand.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.broadcast(x, sizes=(2,)))
op_spec(
    "broadcast_in_dim",
    "shape",
    OpArity.UNARY,
    torch_fn=_torch_broadcast_in_dim_leading2,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_broadcast_in_dim_leading2)
op_spec(
    "broadcast_to_rank",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_broadcast_to_rank_plus1(inputs),
    torch_fn=torch.ops.aten.unsqueeze.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.broadcast_to_rank(x, x.ndim + 1))
op_spec(
    "broadcasted_iota",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_broadcasted_iota(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_broadcasted_iota(output, inputs),
    torch_fn=torch.ops.aten.arange.start_step,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.broadcasted_iota(x.dtype, shape=x.shape, dimension=x.ndim - 1))
op_spec(
    "collapse",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_collapse_first2(inputs),
    torch_fn=torch.ops.aten.reshape.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_collapse_first2)
op_spec(
    "expand_dims",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_expand_dims_front(inputs),
    torch_fn=torch.ops.aten.unsqueeze.default,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.expand_dims(x, (0,)))
op_spec(
    "iota",
    "shape",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_iota_last_dim(inputs),
    torch_fn=torch.ops.aten.arange.start_step,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.iota(x.dtype, x.shape[-1]))

op_spec(
    "dynamic_index_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_select_last_middle(inputs),
    torch_fn=torch.ops.aten.select.int,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_index_in_dim_last)
op_spec(
    "dynamic_slice",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_slice_like_last_half(inputs),
    torch_fn=torch.ops.aten.slice.Tensor,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_slice_last_half)
op_spec(
    "dynamic_slice_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_slice_like_last_half(inputs),
    torch_fn=torch.ops.aten.slice.Tensor,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_slice_in_dim_last_half)
op_spec(
    "dynamic_update_slice",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_slice_prefix,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_update_slice_prefix)
op_spec(
    "dynamic_update_slice_in_dim",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_slice_in_dim_prefix,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_dynamic_update_slice_in_dim_prefix)
op_spec(
    "dynamic_update_index_in_dim",
    "indexing",
    OpArity.BINARY,
    torch_fn=_torch_dynamic_update_index_in_dim_last,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(
    _jax_dynamic_update_index_in_dim_last
)
op_spec(
    "index_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_select_last_middle(inputs),
    torch_fn=torch.ops.aten.select.int,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.index_in_dim(x, x.shape[-1] // 2, axis=-1, keepdims=False))
op_spec(
    "index_take",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_index_take_last_two,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_index_take_last_two)
op_spec(
    "slice_in_dim",
    "indexing",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_slice_like_last_half(inputs),
    torch_fn=torch.ops.aten.slice.Tensor,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(_jax_slice_in_dim_last_half)
op_spec(
    "sort_key_val",
    "indexing",
    OpArity.UNARY,
    torch_fn=_torch_sort_key_val_keys,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x: lax.sort_key_val(x, -x, dimension=-1, is_stable=True)[0])
op_spec(
    "scatter_mul",
    "indexing",
    OpArity.BINARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes=(
        "TODO: jax.lax.scatter_mul semantics with conflicting indices is not "
        "strict 1:1 with current PyTorch path."
    ),
)(_jax_scatter_mul_flat)
op_spec(
    "scatter_sub",
    "indexing",
    OpArity.BINARY,
    torch_fn=None,
    shape_type="reduction",
    supported_dtypes=("float32",),
    notes=(
        "TODO: jax.lax.scatter_sub semantics with conflicting indices is not "
        "strict 1:1 with current PyTorch path."
    ),
)(_jax_scatter_sub_flat)

op_spec(
    "reduce_and",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_bool_dim(inputs),
    torch_fn=torch.ops.aten.all.dim,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, axis: lax.reduce_and(x > 0, axes=(axis,)))
op_spec(
    "reduce_or",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_bool_dim(inputs),
    torch_fn=torch.ops.aten.any.dim,
    shape_type="reduction",
    supported_dtypes=("float32",),
)(lambda x, axis: lax.reduce_or(x > 0, axes=(axis,)))
op_spec(
    "reduce_xor",
    "reduction",
    OpArity.REDUCTION,
    torch_fn=lambda x, axis: torch.remainder((x > 0).to(torch.int32).sum(dim=axis), 2).to(torch.bool),
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
    torch_fn=torch.ops.aten.floor.default,
)(lax.floor)
op_spec(
    "ceil",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.ceil.default,
)(lax.ceil)
op_spec(
    "round",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.round.default,
    notes="Both use half-to-even (banker's rounding)",
)(lambda x: lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN))
op_spec(
    "is_finite",
    "rounding",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.isfinite.default,
    notes="Returns bool",
)(lax.is_finite)

# =============================================================================
# Complex number operations
# =============================================================================

op_spec(
    "complex",
    "complex",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.complex.default,
    supported_dtypes=("float32",),
    notes="Constructs complex from real and imag parts",
)(lax.complex)
op_spec(
    "conj",
    "complex",
    OpArity.UNARY,
    torch_output_adapter=lambda output, inputs: _adapt_conj_resolved(output, inputs),
    torch_fn=torch.ops.aten.conj.default,
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
    notes="torch.conj returns lazy view; need resolve_conj() for numpy()",
)(lax.conj)
op_spec(
    "real",
    "complex",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.real.default,
    input_domain=InputDomain.COMPLEX,
    supported_dtypes=("float32",),
)(lax.real)
op_spec(
    "imag",
    "complex",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.imag.default,
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
    torch_fn_builder=lambda inputs: _build_reduce_dim(inputs),
    torch_fn=torch.ops.aten.prod.dim_int,
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.reduce_prod_p.bind(x, axes=(axis,)))
op_spec(
    "cummax",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_no_keepdim(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_first(output, inputs),
    torch_fn=torch.ops.aten.cummax.default,
    shape_type="reduction",
)(lambda x, axis: lax.cummax(x, axis=axis))
op_spec(
    "cummin",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_no_keepdim(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_first(output, inputs),
    torch_fn=torch.ops.aten.cummin.default,
    shape_type="reduction",
)(lambda x, axis: lax.cummin(x, axis=axis))
op_spec(
    "cumlogsumexp",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim_no_keepdim(inputs),
    torch_fn=torch.ops.aten.logcumsumexp.default,
    shape_type="reduction",
    input_domain=InputDomain.SMALL_POSITIVE,
)(lambda x, axis: lax.cumlogsumexp(x, axis=axis))

op_spec(
    "sort",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_first(output, inputs),
    torch_fn=torch.ops.aten.sort.default,
    shape_type="reduction",
    notes="Compares sorted values along axis",
)(lambda x, axis: lax.sort(x, dimension=axis))
op_spec(
    "argmax",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim(inputs),
    torch_fn=torch.ops.aten.argmax.default,
    shape_type="reduction",
    notes="Returns int indices; abs error = 0 means frameworks agree",
)(lambda x, axis: lax.argmax(x, axis, jnp.int32))
op_spec(
    "argmin",
    "reduction",
    OpArity.REDUCTION,
    torch_fn_builder=lambda inputs: _build_reduce_dim(inputs),
    torch_fn=torch.ops.aten.argmin.default,
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
    torch_fn=torch.ops.aten.lgamma.default,
    input_domain=InputDomain.POSITIVE,
)(lax.lgamma)
op_spec(
    "digamma",
    "special",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.digamma.default,
    input_domain=InputDomain.POSITIVE,
)(lax.digamma)
op_spec(
    "bessel_i0e",
    "special",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.special_i0e.default,
)(lax.bessel_i0e)
op_spec(
    "bessel_i1e",
    "special",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.special_i1e.default,
)(lax.bessel_i1e)
op_spec(
    "igamma",
    "special",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.special_gammainc.default,
    input_domain=InputDomain.POSITIVE,
    notes="JAX igamma(a,x) = torch.special.gammainc(a,x)",
)(lax.igamma)
op_spec(
    "igammac",
    "special",
    OpArity.BINARY,
    torch_fn=torch.ops.aten.special_gammaincc.default,
    input_domain=InputDomain.POSITIVE,
    notes="JAX igammac(a,x) = torch.special.gammaincc(a,x)",
)(lax.igammac)
op_spec(
    "betainc",
    "special",
    OpArity.TERNARY,
    torch_fn=None,
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
    torch_fn=torch.ops.aten.matmul.default,
    shape_type="matmul",
    supported_dtypes=("float32", "bfloat16"),
)(lax.dot)
op_spec(
    "batch_matmul",
    "linalg",
    OpArity.MATMUL,
    torch_fn=torch.ops.aten.bmm.default,
    shape_type="batch_matmul",
    supported_dtypes=("float32", "bfloat16"),
)(lax.batch_matmul)

# --- Cholesky decomposition ---
op_spec(
    "cholesky",
    "linalg",
    OpArity.UNARY,
    torch_fn=torch.ops.aten.linalg_cholesky.default,
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="Input: positive definite matrix",
)(lambda x: jax.lax.linalg.cholesky(x))

op_spec(
    "triangular_solve",
    "linalg",
    OpArity.MATMUL,
    torch_fn_builder=lambda inputs: _build_triangular_solve(inputs),
    torch_fn=torch.ops.aten.linalg_solve_triangular.default,
    input_domain=InputDomain.LOWER_TRIANGULAR,
    shape_type="linalg_solve",
    supported_dtypes=("float32",),
    notes="Solve A @ x = b where A is lower triangular",
)(lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True))

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

def _jax_lu_pivots_to_perm(x):
    _, _, piv = jax.lax.linalg.lu(x)
    return jax.lax.linalg.lu_pivots_to_permutation(piv, permutation_size=x.shape[-1])

op_spec(
    "eig",
    "linalg",
    OpArity.UNARY,
    torch_output_adapter=lambda output, inputs: _adapt_eig_values_sorted(output, inputs),
    torch_fn=torch.ops.aten.linalg_eig.default,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(_jax_linalg_eig_values_sorted)
op_spec(
    "eigh",
    "linalg",
    OpArity.UNARY,
    torch_output_adapter=lambda output, inputs: _adapt_tuple_second(output, inputs),
    torch_fn=torch.ops.aten._linalg_eigh.default,
    input_domain=InputDomain.POSITIVE_DEFINITE,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: jax.lax.linalg.eigh(x)[0])
op_spec(
    "householder_product",
    "linalg",
    OpArity.UNARY,
    torch_fn_builder=lambda inputs: _build_householder_product(inputs),
    torch_output_adapter=lambda output, inputs: _adapt_householder_product(output, inputs),
    torch_fn=torch.ops.aten.linalg_householder_product.default,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(_jax_linalg_householder_product)
op_spec(
    "lu",
    "linalg",
    OpArity.UNARY,
    torch_fn=lambda x: torch.linalg.lu_factor(x)[0],
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: jax.lax.linalg.lu(x)[0])
op_spec(
    "lu_pivots_to_permutation",
    "linalg",
    OpArity.UNARY,
    torch_fn=None,
    shape_type="linalg",
    supported_dtypes=("float32",),
    notes="TODO: no strict 1:1 ATen mapping",
)(_jax_lu_pivots_to_perm)
op_spec(
    "qr",
    "linalg",
    OpArity.UNARY,
    torch_output_adapter=lambda output, inputs: _adapt_tuple_first_abs(output, inputs),
    torch_fn=torch.ops.aten.linalg_qr.default,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: lax.abs(jax.lax.linalg.qr(x)[0]))
op_spec(
    "svd",
    "linalg",
    OpArity.UNARY,
    torch_output_adapter=lambda output, inputs: _adapt_svd_singular_values(output, inputs),
    torch_fn=torch.ops.aten._linalg_svd.default,
    shape_type="linalg",
    supported_dtypes=("float32",),
)(lambda x: jax.lax.linalg.svd(x, full_matrices=False)[1])
op_spec(
    "zeta",
    "special",
    OpArity.BINARY,
    torch_fn_builder=lambda inputs: _build_zeta_shifted(inputs),
    torch_fn=torch.ops.aten.special_zeta.default,
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

op_spec(
    "conv_general_dilated",
    "linalg",
    OpArity.CONV,
    torch_fn_builder=lambda inputs: _build_convolution(inputs),
    torch_fn=torch.ops.aten.convolution.default,
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
    torch_fn_builder=lambda inputs: _build_fft_like(inputs),
    torch_fn=torch.ops.aten.fft_fft.default,
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
    supported_dtypes=("float32",),
    notes="Cast float32 to bfloat16",
)(lambda x: lax.convert_element_type(x, jnp.bfloat16))
op_spec(
    "bitcast_convert_type",
    "type_cast",
    OpArity.TYPE_CAST,
    torch_fn=lambda x: x.view(torch.int32),
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
            notes="TODO: no strict 1:1 ATen mapping",
        )(jax_fn)

_register_remaining_ops()

def _ensure_todo_notes_for_unmapped():
    for op in get_all_ops():
        if op.torch_fn is None:
            note = (op.notes or "").strip()
            if note.startswith("TODO:"):
                continue
            op.notes = f"TODO: {note}" if note else "TODO: no strict 1:1 ATen mapping"

_ensure_todo_notes_for_unmapped()

# =============================================================================
# Torch call plans (signature and output adaptation)
# =============================================================================

def _axis_last(x):
    return x.ndim - 1 if x.ndim > 0 else 0

def _adapt_first(output, _inputs):
    if isinstance(output, tuple):
        return output[0]
    return output

def _adapt_tuple_first(output, _inputs):
    if isinstance(output, tuple):
        return output[0]
    return output

def _adapt_tuple_first_abs(output, _inputs):
    out = output[0] if isinstance(output, tuple) else output
    return torch.abs(out)

def _adapt_svd_singular_values(output, _inputs):
    return output[1] if isinstance(output, tuple) else output

def _adapt_eig_values_sorted(output, _inputs):
    eigvals = output[0] if isinstance(output, tuple) else output
    idx_imag = torch.argsort(torch.imag(eigvals), dim=-1, stable=True)
    eigvals = torch.gather(eigvals, -1, idx_imag)
    idx_real = torch.argsort(torch.real(eigvals), dim=-1, stable=True)
    return torch.gather(eigvals, -1, idx_real)

def _adapt_tuple_second(output, _inputs):
    if isinstance(output, tuple):
        return output[1]
    return output

def _adapt_split_first(output, _inputs):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def _adapt_broadcasted_iota(output, inputs):
    x = inputs["x"]
    if x.ndim == 1:
        return output
    return output.reshape((1,) * (x.ndim - 1) + (x.shape[-1],)).expand(x.shape)

def _adapt_householder_product(output, inputs):
    x = inputs["x"]
    if x.shape[-1] < 2:
        return x
    return output

def _adapt_conj_resolved(output, _inputs):
    return output.resolve_conj() if hasattr(output, "resolve_conj") else output

def _build_reduce_dim_intlist(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, [axis], False), {}

def _build_reduce_dim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, axis, False), {}

def _build_reduce_bool_dim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x > 0, axis, False), {}

def _build_reduce_dim_no_keepdim(inputs):
    x = inputs["x"]
    axis = _axis_last(x)
    return (x, axis), {}

def _build_zeta_shifted(inputs):
    x = inputs["x"]
    y = inputs["y"]
    return (torch.abs(x) + 2.0, torch.abs(y) + 1.0), {}

def _build_clamp(inputs):
    return (inputs["x"], inputs["lo"], inputs["hi"]), {}

def _build_bitwise_binary_i32(inputs):
    return (_torch_i32(inputs["x"]), _torch_i32(inputs["y"])), {}

def _build_bitwise_not_i32(inputs):
    return (_torch_i32(inputs["x"]),), {}

def _build_shift_i32(inputs):
    x = _torch_i32(inputs["x"])
    y = _torch_shift_amount(inputs["y"]).to(torch.int32)
    return (x, y), {}

def _build_concatenate(inputs):
    x = inputs["x"]
    return ((x, inputs["y"]), x.ndim - 1), {}

def _build_full_from_x(inputs):
    x = inputs["x"]
    fill = torch.mean(x.float()).item()
    return (tuple(x.shape), fill), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
    }

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

def _build_reshape_reverse(inputs):
    x = inputs["x"]
    return (x, x.shape[::-1]), {}

def _build_collapse_first2(inputs):
    x = inputs["x"]
    if x.ndim < 2:
        target = tuple(x.shape)
    else:
        target = (x.shape[0] * x.shape[1],) + tuple(x.shape[2:])
    return (x, target), {}

def _build_rev_last(inputs):
    x = inputs["x"]
    return (x, [x.ndim - 1]), {}

def _build_split_first_half(inputs):
    x = inputs["x"]
    left = max(1, x.shape[-1] // 2)
    right = x.shape[-1] - left
    return (x, [left, right], -1), {}

def _build_squeeze_noop(inputs):
    x = inputs["x"]
    return (torch.unsqueeze(x, 0), 0), {}

def _build_expand_dims_front(inputs):
    x = inputs["x"]
    return (x, 0), {}

def _build_tile_leading(inputs):
    x = inputs["x"]
    reps = (2,) + (1,) * (x.ndim - 1)
    return (x, reps), {}

def _build_transpose_reverse(inputs):
    x = inputs["x"]
    perm = tuple(range(x.ndim - 1, -1, -1))
    return (x, perm), {}

def _build_pad_one(inputs):
    x = inputs["x"]
    pads = [1, 1] * x.ndim
    return (x, pads, "constant", 0), {}

def _build_slice_last_half(inputs):
    x = inputs["x"]
    k = max(1, x.shape[-1] // 2)
    return (x, x.ndim - 1, 0, k, 1), {}

def _build_slice_like_last_half(inputs):
    x = inputs["x"]
    k = max(1, x.shape[-1] // 2)
    return (x, x.ndim - 1, 0, k, 1), {}

def _build_select_last_middle(inputs):
    x = inputs["x"]
    return (x, x.ndim - 1, x.shape[-1] // 2), {}

def _build_top_k_values(inputs):
    x = inputs["x"]
    return (x, min(5, x.shape[-1]), x.ndim - 1, True, True), {}

def _build_broadcast_leading2(inputs):
    x = inputs["x"]
    return (x, (2,) + tuple(x.shape)), {}

def _build_broadcast_to_rank_plus1(inputs):
    x = inputs["x"]
    return (x, 0), {}

def _build_broadcasted_iota(inputs):
    x = inputs["x"]
    return (0, x.shape[-1], 1), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
    }

def _build_iota_last_dim(inputs):
    x = inputs["x"]
    return (0, x.shape[-1], 1), {
        "dtype": x.dtype,
        "layout": None,
        "device": x.device,
        "pin_memory": False,
    }

def _build_householder_product(inputs):
    x = inputs["x"]
    a = x[..., :-1]
    taus = torch.ones((a.shape[-1],), dtype=x.dtype, device=x.device)
    return (a, taus), {}

def _build_triangular_solve(inputs):
    return (inputs["x"], inputs["y"]), {
        "upper": False,
        "left": True,
        "unitriangular": False,
    }

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

def _build_fft_like(inputs):
    return (inputs["x"], None, -1, None), {}
