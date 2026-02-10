"""All registered jax.lax operators for precision testing."""

import json
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special as jsp
import torch
import torch.nn.functional as F

from ..op_registry import register, OpSpec, OpArity, InputDomain, get_all_ops

# =============================================================================
# Basic operations
# =============================================================================

register(OpSpec("add", "basic", OpArity.BINARY, lax.add, torch.add))
register(OpSpec("sub", "basic", OpArity.BINARY, lax.sub, torch.sub))
register(OpSpec("mul", "basic", OpArity.BINARY, lax.mul, torch.mul))
register(OpSpec("div", "basic", OpArity.BINARY, lax.div, torch.div,
                input_domain=InputDomain.NON_ZERO))
register(OpSpec("rem", "basic", OpArity.BINARY, lax.rem, torch.fmod,
                input_domain=InputDomain.NON_ZERO,
                notes="jax.lax.rem = C-style truncation remainder; torch.fmod matches this"))
register(OpSpec("pow", "basic", OpArity.BINARY, lax.pow, torch.pow,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("max", "basic", OpArity.BINARY, lax.max, torch.maximum))
register(OpSpec("min", "basic", OpArity.BINARY, lax.min, torch.minimum))
register(OpSpec("nextafter", "basic", OpArity.BINARY, lax.nextafter, torch.nextafter,
                notes="Next representable floating-point value towards y"))

register(OpSpec("neg", "basic", OpArity.UNARY, lax.neg, torch.neg))
register(OpSpec("abs", "basic", OpArity.UNARY, lax.abs, torch.abs))
register(OpSpec("reciprocal", "basic", OpArity.UNARY, lax.reciprocal, torch.reciprocal,
                input_domain=InputDomain.NON_ZERO))
register(OpSpec("square", "basic", OpArity.UNARY, lax.square, torch.square))
register(OpSpec("sign", "basic", OpArity.UNARY, lax.sign, torch.sign))
register(OpSpec("integer_pow_2", "basic", OpArity.UNARY,
                lambda x: lax.integer_pow(x, 2),
                lambda x: torch.pow(x, 2),
                notes="integer_pow with exponent=2"))
register(OpSpec("integer_pow_3", "basic", OpArity.UNARY,
                lambda x: lax.integer_pow(x, 3),
                lambda x: torch.pow(x, 3),
                notes="integer_pow with exponent=3"))

# =============================================================================
# Exponential and trigonometric
# =============================================================================

register(OpSpec("exp", "exp_trig", OpArity.UNARY, lax.exp, torch.exp,
                input_domain=InputDomain.SMALL_POSITIVE))
register(OpSpec("exp2", "exp_trig", OpArity.UNARY, lax.exp2, torch.exp2,
                input_domain=InputDomain.SMALL_POSITIVE))
register(OpSpec("expm1", "exp_trig", OpArity.UNARY, lax.expm1, torch.expm1,
                input_domain=InputDomain.SMALL_POSITIVE))
register(OpSpec("log", "exp_trig", OpArity.UNARY, lax.log, torch.log,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("log1p", "exp_trig", OpArity.UNARY, lax.log1p, torch.log1p,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("sqrt", "exp_trig", OpArity.UNARY, lax.sqrt, torch.sqrt,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("rsqrt", "exp_trig", OpArity.UNARY, lax.rsqrt, torch.rsqrt,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("cbrt", "exp_trig", OpArity.UNARY, lax.cbrt,
                lambda x: torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0),
                input_domain=InputDomain.POSITIVE,
                notes="PyTorch lacks native cbrt; using sign(x)*pow(|x|, 1/3)"))

register(OpSpec("sin", "exp_trig", OpArity.UNARY, lax.sin, torch.sin))
register(OpSpec("cos", "exp_trig", OpArity.UNARY, lax.cos, torch.cos))
register(OpSpec("tan", "exp_trig", OpArity.UNARY, lax.tan, torch.tan))
register(OpSpec("asin", "exp_trig", OpArity.UNARY, lax.asin, torch.asin,
                input_domain=InputDomain.UNIT))
register(OpSpec("acos", "exp_trig", OpArity.UNARY, lax.acos, torch.acos,
                input_domain=InputDomain.UNIT))
register(OpSpec("atan", "exp_trig", OpArity.UNARY, lax.atan, torch.atan))
register(OpSpec("atan2", "exp_trig", OpArity.BINARY, lax.atan2, torch.atan2))

register(OpSpec("sinh", "exp_trig", OpArity.UNARY, lax.sinh, torch.sinh,
                input_domain=InputDomain.SMALL_POSITIVE))
register(OpSpec("cosh", "exp_trig", OpArity.UNARY, lax.cosh, torch.cosh,
                input_domain=InputDomain.SMALL_POSITIVE))
register(OpSpec("tanh", "exp_trig", OpArity.UNARY, lax.tanh, torch.tanh))
register(OpSpec("asinh", "exp_trig", OpArity.UNARY, lax.asinh, torch.asinh))
register(OpSpec("acosh", "exp_trig", OpArity.UNARY, lax.acosh, torch.acosh,
                input_domain=InputDomain.ABOVE_ONE))
register(OpSpec("atanh", "exp_trig", OpArity.UNARY, lax.atanh, torch.atanh,
                input_domain=InputDomain.UNIT))

# =============================================================================
# Normalization / reductions / cumulative
# =============================================================================

register(OpSpec("logistic", "normalization", OpArity.UNARY, lax.logistic, torch.sigmoid,
                notes="logistic = sigmoid = 1/(1+exp(-x))"))

register(OpSpec("reduce_sum", "normalization", OpArity.REDUCTION,
                lambda x, axis: jnp.sum(x, axis=axis),
                lambda x, axis: torch.sum(x, dim=axis),
                shape_type="reduction"))
register(OpSpec("reduce_max", "normalization", OpArity.REDUCTION,
                lambda x, axis: jnp.max(x, axis=axis),
                lambda x, axis: torch.max(x, dim=axis).values,
                shape_type="reduction"))
register(OpSpec("reduce_min", "normalization", OpArity.REDUCTION,
                lambda x, axis: jnp.min(x, axis=axis),
                lambda x, axis: torch.min(x, dim=axis).values,
                shape_type="reduction"))
register(OpSpec("cumsum", "normalization", OpArity.REDUCTION,
                lambda x, axis: lax.cumsum(x, axis=axis),
                lambda x, axis: torch.cumsum(x, dim=axis),
                shape_type="reduction"))
register(OpSpec("cumprod", "normalization", OpArity.REDUCTION,
                lambda x, axis: lax.cumprod(x, axis=axis),
                lambda x, axis: torch.cumprod(x, dim=axis),
                shape_type="reduction",
                input_domain=InputDomain.SMALL_POSITIVE))

# =============================================================================
# Activation functions
# =============================================================================

register(OpSpec("sigmoid", "activation", OpArity.UNARY, lax.logistic, torch.sigmoid,
                notes="Same as logistic; primary activation function"))
register(OpSpec("tanh_act", "activation", OpArity.UNARY, lax.tanh, torch.tanh,
                notes="tanh as activation function"))
register(OpSpec("erf", "activation", OpArity.UNARY, lax.erf, torch.erf))
register(OpSpec("erfc", "activation", OpArity.UNARY, lax.erfc, torch.erfc))
register(OpSpec("erf_inv", "activation", OpArity.UNARY, lax.erf_inv, torch.erfinv,
                input_domain=InputDomain.UNIT))

# =============================================================================
# Comparison operations
# =============================================================================

register(OpSpec("eq", "comparison", OpArity.BINARY, lax.eq, torch.eq,
                notes="Returns bool"))
register(OpSpec("ne", "comparison", OpArity.BINARY, lax.ne, torch.ne,
                notes="Returns bool"))
register(OpSpec("lt", "comparison", OpArity.BINARY, lax.lt, torch.lt,
                notes="Returns bool"))
register(OpSpec("le", "comparison", OpArity.BINARY, lax.le, torch.le,
                notes="Returns bool"))
register(OpSpec("gt", "comparison", OpArity.BINARY, lax.gt, torch.gt,
                notes="Returns bool"))
register(OpSpec("ge", "comparison", OpArity.BINARY, lax.ge, torch.ge,
                notes="Returns bool"))
register(OpSpec("clamp", "comparison", OpArity.TERNARY,
                lambda lo, x, hi: lax.clamp(lo, x, hi),
                lambda lo, x, hi: torch.clamp(x, min=lo, max=hi),
                notes="JAX: clamp(lo, x, hi); torch: clamp(x, min=lo, max=hi)"))

# =============================================================================
# Rounding operations
# =============================================================================

register(OpSpec("floor", "rounding", OpArity.UNARY, lax.floor, torch.floor))
register(OpSpec("ceil", "rounding", OpArity.UNARY, lax.ceil, torch.ceil))
register(OpSpec("round", "rounding", OpArity.UNARY,
                lambda x: lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN),
                torch.round,
                notes="Both use half-to-even (banker's rounding)"))
register(OpSpec("is_finite", "rounding", OpArity.UNARY, lax.is_finite, torch.isfinite,
                notes="Returns bool"))

# =============================================================================
# Complex number operations
# =============================================================================

register(OpSpec("complex", "complex", OpArity.BINARY,
                lax.complex, torch.complex,
                supported_dtypes=("float32",),
                notes="Constructs complex from real and imag parts"))
register(OpSpec("conj", "complex", OpArity.UNARY, lax.conj,
                lambda x: torch.conj(x).resolve_conj(),
                input_domain=InputDomain.COMPLEX,
                supported_dtypes=("float32",),
                notes="torch.conj returns lazy view; need resolve_conj() for numpy()"))
register(OpSpec("real", "complex", OpArity.UNARY, lax.real, torch.real,
                input_domain=InputDomain.COMPLEX,
                supported_dtypes=("float32",)))
register(OpSpec("imag", "complex", OpArity.UNARY, lax.imag, torch.imag,
                input_domain=InputDomain.COMPLEX,
                supported_dtypes=("float32",)))

# =============================================================================
# Additional reductions (sort, argmax, argmin, top_k)
# =============================================================================

register(OpSpec("reduce_prod", "reduction", OpArity.REDUCTION,
                lambda x, axis: jnp.prod(x, axis=axis),
                lambda x, axis: torch.prod(x, dim=axis),
                shape_type="reduction",
                input_domain=InputDomain.SMALL_POSITIVE))
register(OpSpec("cummax", "reduction", OpArity.REDUCTION,
                lambda x, axis: lax.cummax(x, axis=axis),
                lambda x, axis: torch.cummax(x, dim=axis).values,
                shape_type="reduction"))
register(OpSpec("cummin", "reduction", OpArity.REDUCTION,
                lambda x, axis: lax.cummin(x, axis=axis),
                lambda x, axis: torch.cummin(x, dim=axis).values,
                shape_type="reduction"))
register(OpSpec("cumlogsumexp", "reduction", OpArity.REDUCTION,
                lambda x, axis: lax.cumlogsumexp(x, axis=axis),
                lambda x, axis: torch.logcumsumexp(x, dim=axis),
                shape_type="reduction",
                input_domain=InputDomain.SMALL_POSITIVE))

register(OpSpec("sort", "reduction", OpArity.REDUCTION,
                lambda x, axis: lax.sort(x, dimension=axis),
                lambda x, axis: torch.sort(x, dim=axis).values,
                shape_type="reduction",
                notes="Compares sorted values along axis"))
register(OpSpec("argmax", "reduction", OpArity.REDUCTION,
                lambda x, axis: jnp.argmax(x, axis=axis),
                lambda x, axis: torch.argmax(x, dim=axis),
                shape_type="reduction",
                notes="Returns int indices; abs error = 0 means frameworks agree"))
register(OpSpec("argmin", "reduction", OpArity.REDUCTION,
                lambda x, axis: jnp.argmin(x, axis=axis),
                lambda x, axis: torch.argmin(x, dim=axis),
                shape_type="reduction",
                notes="Returns int indices; abs error = 0 means frameworks agree"))
register(OpSpec("top_k_values", "reduction", OpArity.UNARY,
                lambda x: lax.top_k(x, k=min(5, x.shape[-1]))[0],
                lambda x: torch.topk(x, k=min(5, x.shape[-1]), dim=-1).values,
                shape_type="reduction",
                notes="Top-k values along last axis (k=min(5, size))"))

# =============================================================================
# Special mathematical functions
# =============================================================================

register(OpSpec("lgamma", "special", OpArity.UNARY, lax.lgamma, torch.lgamma,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("digamma", "special", OpArity.UNARY, lax.digamma, torch.digamma,
                input_domain=InputDomain.POSITIVE))
register(OpSpec("polygamma_1", "special", OpArity.UNARY,
                lambda x: jsp.polygamma(1, x),
                lambda x: torch.special.polygamma(1, x),
                input_domain=InputDomain.POSITIVE,
                notes="Trigamma function: polygamma(1, x)"))
register(OpSpec("bessel_i0e", "special", OpArity.UNARY, lax.bessel_i0e, torch.special.i0e))
register(OpSpec("bessel_i1e", "special", OpArity.UNARY, lax.bessel_i1e, torch.special.i1e))
register(OpSpec("igamma", "special", OpArity.BINARY, lax.igamma, torch.special.gammainc,
                input_domain=InputDomain.POSITIVE,
                notes="JAX igamma(a,x) = torch.special.gammainc(a,x)"))
register(OpSpec("igammac", "special", OpArity.BINARY, lax.igammac, torch.special.gammaincc,
                input_domain=InputDomain.POSITIVE,
                notes="JAX igammac(a,x) = torch.special.gammaincc(a,x)"))
register(OpSpec("betainc", "special", OpArity.TERNARY, lax.betainc, None,
                input_domain=InputDomain.POSITIVE,
                notes="No PyTorch equivalent"))

# =============================================================================
# Type conversion operations
# =============================================================================

register(OpSpec("convert_f32_to_bf16", "type_ops", OpArity.TYPE_CAST,
                lambda x: lax.convert_element_type(x, jnp.bfloat16),
                lambda x: x.to(torch.bfloat16),
                supported_dtypes=("float32",),
                notes="Cast float32 -> bfloat16"))
register(OpSpec("convert_bf16_to_f32", "type_ops", OpArity.TYPE_CAST,
                lambda x: lax.convert_element_type(x, jnp.float32),
                lambda x: x.to(torch.float32),
                supported_dtypes=("bfloat16",),
                notes="Cast bfloat16 -> float32"))
register(OpSpec("reduce_precision_fp16", "type_ops", OpArity.TYPE_CAST,
                lambda x: lax.reduce_precision(x, exponent_bits=5, mantissa_bits=10),
                None,
                supported_dtypes=("float32",),
                notes="Simulates FP16 precision; no PyTorch equivalent"))
register(OpSpec("bitcast_f32_to_i32", "type_ops", OpArity.TYPE_CAST,
                lambda x: lax.bitcast_convert_type(x, jnp.int32),
                lambda x: x.view(torch.int32),
                supported_dtypes=("float32",),
                notes="Reinterpret float32 bits as int32"))

# =============================================================================
# Linear algebra operations
# =============================================================================

register(OpSpec("dot", "linalg", OpArity.MATMUL,
                lax.dot, torch.matmul,
                shape_type="matmul",
                supported_dtypes=("float32", "bfloat16")))
register(OpSpec("batch_matmul", "linalg", OpArity.MATMUL,
                lax.batch_matmul, torch.bmm,
                shape_type="batch_matmul",
                supported_dtypes=("float32", "bfloat16")))

# --- Cholesky decomposition ---
register(OpSpec("cholesky", "linalg", OpArity.UNARY,
                lambda x: jax.lax.linalg.cholesky(x),
                lambda x: torch.linalg.cholesky(x),
                input_domain=InputDomain.POSITIVE_DEFINITE,
                shape_type="linalg",
                supported_dtypes=("float32",),
                notes="Input: positive definite matrix"))

# --- Eigenvalues of symmetric matrices ---
register(OpSpec("eigh_eigenvalues", "linalg", OpArity.UNARY,
                # JAX eigh returns (eigenvectors, eigenvalues) — opposite of numpy
                lambda x: jax.lax.linalg.eigh(x)[1],
                lambda x: torch.linalg.eigh(x).eigenvalues,
                input_domain=InputDomain.POSITIVE_DEFINITE,
                shape_type="linalg",
                supported_dtypes=("float32",),
                notes="Eigenvalues only (eigenvectors have sign ambiguity)"))

# --- Singular values ---
register(OpSpec("svd_singular_values", "linalg", OpArity.UNARY,
                lambda x: jax.lax.linalg.svd(x, full_matrices=False)[1],
                lambda x: torch.linalg.svdvals(x),
                shape_type="linalg",
                supported_dtypes=("float32",),
                notes="Singular values only (U/V have sign ambiguity)"))

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

register(OpSpec("qr_r", "linalg", OpArity.UNARY,
                _jax_qr_r, _torch_qr_r,
                shape_type="linalg",
                supported_dtypes=("float32",),
                notes="R factor with diagonal sign normalization for uniqueness"))

# --- Triangular solve ---
register(OpSpec("triangular_solve", "linalg", OpArity.MATMUL,
                lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True),
                lambda a, b: torch.linalg.solve_triangular(a, b, upper=False),
                input_domain=InputDomain.LOWER_TRIANGULAR,
                shape_type="linalg_solve",
                supported_dtypes=("float32",),
                notes="Solve A @ x = b where A is lower triangular"))


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
        return F.conv1d(inp, ker, padding='same')
    else:
        return F.conv2d(inp, ker, padding='same')


register(OpSpec("conv_general_dilated", "linalg", OpArity.CONV,
                _jax_conv, _torch_conv,
                shape_type="conv",
                supported_dtypes=("float32",),
                notes="Both use NCHW layout; JAX via dimension_numbers, torch via F.conv"))

# =============================================================================
# FFT operations
# =============================================================================

register(OpSpec("fft", "fft", OpArity.FFT,
                lambda x: lax.fft(x, fft_type="FFT", fft_lengths=(x.shape[-1],)),
                lambda x: torch.fft.fft(x),
                input_domain=InputDomain.COMPLEX,
                shape_type="fft",
                supported_dtypes=("float32",),
                notes="Complex-to-complex FFT"))
register(OpSpec("ifft", "fft", OpArity.FFT,
                lambda x: lax.fft(x, fft_type="IFFT", fft_lengths=(x.shape[-1],)),
                lambda x: torch.fft.ifft(x),
                input_domain=InputDomain.COMPLEX,
                shape_type="fft",
                supported_dtypes=("float32",),
                notes="Complex-to-complex inverse FFT"))


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
    manifest_path = Path(__file__).resolve().parent.parent / "docs" / "jax_lax_operators.json"
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

        register(OpSpec(
            name=name,
            category="unmapped",
            arity=OpArity.UNARY,
            jax_fn=jax_fn,
            torch_fn=None,
            notes="No torch mapping",
        ))


_register_remaining_ops()
