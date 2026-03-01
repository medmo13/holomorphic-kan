"""
Utility Functions for CVKAN
=============================

Includes:
  - Complex weight initializers (Kaiming, Glorot adapted for ℂ)
  - Wirtinger calculus helpers (∂f/∂z*, ∂f/∂z)
  - Data helpers (to_complex, to_real_imag, batch_iter)
  - Normalization helpers for complex data
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable, Iterator, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------

def complex_glorot_uniform(key: jax.Array, shape: tuple, dtype=jnp.complex64) -> jnp.ndarray:
    """
    Complex Glorot (Xavier) uniform initialization.

    Scales by sqrt(2 / (fan_in + fan_out)) following the real-valued formula,
    then generates independent real and imaginary parts with that scale.

    Args:
        key:   JAX PRNG key.
        shape: Weight tensor shape (fan_in, fan_out, ...).
        dtype: Complex dtype.

    Returns:
        Complex weight array of given shape.
    """
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    k1, k2 = jax.random.split(key)
    re = jax.random.uniform(k1, shape, minval=-limit, maxval=limit)
    im = jax.random.uniform(k2, shape, minval=-limit, maxval=limit)
    return (re + 1j * im).astype(dtype)


def complex_kaiming_uniform(
    key: jax.Array,
    shape: tuple,
    mode: str = "fan_in",
    dtype=jnp.complex64,
) -> jnp.ndarray:
    """
    Complex Kaiming (He) uniform initialization.

    Scales by sqrt(2 / fan) to account for ReLU-type activations.

    Args:
        key:   JAX PRNG key.
        shape: Weight tensor shape.
        mode:  'fan_in' or 'fan_out'.
        dtype: Complex dtype.

    Returns:
        Complex weight array of given shape.
    """
    fan = shape[0] if mode == "fan_in" else (shape[1] if len(shape) > 1 else shape[0])
    limit = jnp.sqrt(6.0 / fan)
    k1, k2 = jax.random.split(key)
    re = jax.random.uniform(k1, shape, minval=-limit, maxval=limit)
    im = jax.random.uniform(k2, shape, minval=-limit, maxval=limit)
    return (re + 1j * im).astype(dtype)


def complex_normal(
    key: jax.Array,
    shape: tuple,
    std: float = 0.02,
    dtype=jnp.complex64,
) -> jnp.ndarray:
    """Normal initialization for complex weights."""
    k1, k2 = jax.random.split(key)
    re = jax.random.normal(k1, shape) * std
    im = jax.random.normal(k2, shape) * std
    return (re + 1j * im).astype(dtype)


# ---------------------------------------------------------------------------
# Wirtinger Calculus
# ---------------------------------------------------------------------------

def wirtinger_gradient(
    f: Callable,
    z: jnp.ndarray,
    conjugate: bool = False,
) -> jnp.ndarray:
    """
    Compute the Wirtinger derivative ∂f/∂z̄ or ∂f/∂z.

    For a real-valued loss L(z) = L(x + iy):
        ∂L/∂z̄ = (1/2)(∂L/∂x + i·∂L/∂y)   (conjugate Wirtinger, used in optimization)
        ∂L/∂z  = (1/2)(∂L/∂x - i·∂L/∂y)

    JAX computes gradients w.r.t. real inputs, so we use the Wirtinger
    formulas by treating z as (x, y) = (Re z, Im z).

    Args:
        f:          Function f: ℂ^n → ℝ (or ℂ, handled via re/im decomposition).
        z:          Complex input array.
        conjugate:  If True, return ∂f/∂z̄ (default); else ∂f/∂z.

    Returns:
        Complex array of same shape as z.
    """
    def real_wrapper(x_re, x_im):
        return jnp.real(f((x_re + 1j * x_im).astype(jnp.complex64)))

    grad_re, grad_im = jax.grad(real_wrapper, argnums=(0, 1))(z.real, z.imag)

    if conjugate:
        # ∂f/∂z̄ = (1/2)(∂f/∂x + i * ∂f/∂y)
        return 0.5 * (grad_re + 1j * grad_im)
    else:
        # ∂f/∂z = (1/2)(∂f/∂x - i * ∂f/∂y)
        return 0.5 * (grad_re - 1j * grad_im)


def wirtinger_value_and_grad(
    f: Callable,
    z: jnp.ndarray,
    conjugate: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute f(z) and its Wirtinger gradient simultaneously.

    Returns:
        (f_value, gradient)
    """
    val = f(z)
    grad = wirtinger_gradient(f, z, conjugate=conjugate)
    return val, grad


# ---------------------------------------------------------------------------
# Data Conversion Helpers
# ---------------------------------------------------------------------------

def to_complex(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Combine real and imaginary arrays into a complex array.

    Args:
        x: Real part array.
        y: Imaginary part array, same shape as x.

    Returns:
        Complex array x + iy.
    """
    return x.astype(jnp.float32) + 1j * y.astype(jnp.float32)


def to_real_imag(z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split a complex array into real and imaginary parts.

    Args:
        z: Complex input array.

    Returns:
        (real_part, imag_part) tuple of float arrays.
    """
    return z.real, z.imag


def interleave_real_imag(z: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten complex array to interleaved real/imaginary float vector.

    Shape (..., n) -> (..., 2n), with layout [re_0, im_0, re_1, im_1, ...].

    Args:
        z: Complex array of shape (..., n).

    Returns:
        Float array of shape (..., 2n).
    """
    re = z.real
    im = z.imag
    return jnp.concatenate([re, im], axis=-1)


def complex_from_interleaved(x: jnp.ndarray) -> jnp.ndarray:
    """
    Convert interleaved real/imaginary float vector back to complex.

    Shape (..., 2n) -> (..., n).

    Args:
        x: Float array of shape (..., 2n).

    Returns:
        Complex array of shape (..., n).
    """
    n = x.shape[-1] // 2
    return x[..., :n] + 1j * x[..., n:]


# ---------------------------------------------------------------------------
# Complex Normalization
# ---------------------------------------------------------------------------

def complex_layer_norm(z: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """
    Layer normalization over the last axis for complex inputs.

    Normalizes each sample by its complex mean and standard deviation.

    Args:
        z:   Complex array of shape (batch, features).
        eps: Numerical stability epsilon.

    Returns:
        Normalized complex array.
    """
    mean = jnp.mean(z, axis=-1, keepdims=True)
    centered = z - mean
    var = jnp.mean(jnp.abs(centered) ** 2, axis=-1, keepdims=True)
    return centered / jnp.sqrt(var + eps)


def complex_batch_norm_stats(z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute batch mean and variance for complex inputs.

    Args:
        z: Complex array of shape (batch, features).

    Returns:
        (mean, variance) where mean is complex, variance is real.
    """
    mean = jnp.mean(z, axis=0)
    var  = jnp.mean(jnp.abs(z - mean) ** 2, axis=0)
    return mean, var


# ---------------------------------------------------------------------------
# Batch Iteration
# ---------------------------------------------------------------------------

def batch_iter(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    key: jax.Array,
    drop_last: bool = True,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Yield shuffled mini-batches from (x, y).

    Args:
        x:          Input array (numpy or jax).
        y:          Target array.
        batch_size: Number of samples per batch.
        key:        JAX PRNG key for shuffling.
        drop_last:  If True, drop the last incomplete batch.

    Yields:
        (x_batch, y_batch) tuples as JAX arrays.
    """
    n = len(x)
    idx = jax.random.permutation(key, n)
    for start in range(0, n, batch_size):
        end = start + batch_size
        if drop_last and end > n:
            break
        batch_idx = idx[start:end]
        yield jnp.array(x[batch_idx]), jnp.array(y[batch_idx])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def complex_mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error for complex arrays (uses |error|^2)."""
    diff = y_pred - y_true
    return jnp.mean(jnp.abs(diff) ** 2)


def complex_mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error for complex arrays (uses |error|)."""
    return jnp.mean(jnp.abs(y_pred - y_true))


def complex_r2(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    R² score adapted for complex outputs.

    Uses total power: R² = 1 - Σ|e|² / Σ|y - ȳ|²
    """
    ss_res = jnp.sum(jnp.abs(y_true - y_pred) ** 2)
    ss_tot = jnp.sum(jnp.abs(y_true - jnp.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)
