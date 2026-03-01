"""
Complex-Valued Activation Functions
=====================================

Provides a library of activation functions suitable for complex-valued
neural networks, including:

  - ModReLU       : |z| activation with learned bias (Arjovsky et al., 2016)
  - zReLU         : zero outside the first quadrant (Guberman, 2016)
  - CReLU         : ReLU applied to real and imaginary parts separately
  - ComplexSplineActivation : learnable spline on each edge (KAN-style)
  - SplitComplexActivation  : apply any real fn to Re and Im independently
  - HolomorphicActivation   : polynomial activation preserving holomorphicity
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap
from typing import Callable, Optional
import equinox as eqx


# ---------------------------------------------------------------------------
# Simple closed-form activations
# ---------------------------------------------------------------------------

def ModReLU(z: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    ModReLU activation: ReLU(|z| + b) * z / |z|

    Preserves phase, only gates magnitude.

    Args:
        z:    Complex input array.
        bias: Learnable scalar bias (real).

    Returns:
        Complex array of same shape as z.
    """
    magnitude = jnp.abs(z)
    activated_mag = jax.nn.relu(magnitude + bias)
    # Avoid division by zero
    phase = z / (magnitude + 1e-8)
    return activated_mag * phase


def zReLU(z: jnp.ndarray) -> jnp.ndarray:
    """
    zReLU: pass z only if both Re(z) > 0 and Im(z) > 0 (first quadrant).

    Args:
        z: Complex input array.

    Returns:
        Complex array, zeroed outside the first quadrant.
    """
    mask = (z.real > 0) & (z.imag > 0)
    return jnp.where(mask, z, 0.0 + 0.0j)


def CReLU(z: jnp.ndarray) -> jnp.ndarray:
    """
    Complex ReLU: ReLU applied independently to real and imaginary parts.

    Args:
        z: Complex input array.

    Returns:
        Complex array with non-negative real and imaginary parts.
    """
    return jax.nn.relu(z.real) + 1j * jax.nn.relu(z.imag)


def complex_sigmoid(z: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid applied independently to Re and Im."""
    return jax.nn.sigmoid(z.real) + 1j * jax.nn.sigmoid(z.imag)


def complex_tanh(z: jnp.ndarray) -> jnp.ndarray:
    """Complex hyperbolic tangent (holomorphic)."""
    return jnp.tanh(z)


def complex_gelu(z: jnp.ndarray) -> jnp.ndarray:
    """GELU applied to real and imaginary parts separately."""
    return jax.nn.gelu(z.real) + 1j * jax.nn.gelu(z.imag)


# ---------------------------------------------------------------------------
# Learnable activations (Equinox modules)
# ---------------------------------------------------------------------------

class ComplexSplineActivation(eqx.Module):
    """
    Learnable complex activation function for a single KAN edge.

    Implements the KAN activation as a weighted sum of basis functions:

        φ(z) = Σ_k  c_k * B_k(z)   +   w_b * silu(z)

    where B_k are complex B-spline (or other) basis functions and
    c_k, w_b are learnable complex-valued parameters.

    Args:
        n_basis:    Number of basis functions.
        basis_fn:   Callable (z, n_basis, ...) -> (n_basis,) complex array.
        init_noise: Std-dev for random weight initialization.
    """
    coeffs:   jnp.ndarray   # shape: (n_basis,) complex
    w_base:   jnp.ndarray   # scalar complex: weight for residual SiLU
    n_basis:  int            = eqx.field(static=True)
    basis_fn: Callable       = eqx.field(static=True)
    basis_kw: dict           = eqx.field(static=True)

    def __init__(
        self,
        n_basis: int,
        basis_fn: Callable,
        basis_kw: Optional[dict] = None,
        *,
        key: jax.Array,
    ):
        self.n_basis  = n_basis
        self.basis_fn = basis_fn
        self.basis_kw = basis_kw or {}

        k1, k2 = jax.random.split(key)
        scale = 1.0 / jnp.sqrt(n_basis)
        re = jax.random.normal(k1, (n_basis,)) * scale
        im = jax.random.normal(k2, (n_basis,)) * scale
        self.coeffs = re + 1j * im
        self.w_base = jnp.array(1.0 + 0.0j)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate spline activation at scalar z.

        Args:
            z: Complex scalar.

        Returns:
            Complex scalar output.
        """
        B = self.basis_fn(z[None], n_basis=self.n_basis, **self.basis_kw)[0]
        spline_out = jnp.dot(self.coeffs, B)
        # Residual connection via SiLU (analytic continuation)
        silu_z = z * jax.nn.sigmoid(z.real)  # split-SiLU for stability
        return spline_out + self.w_base * silu_z


class SplitComplexActivation(eqx.Module):
    """
    Apply a real-valued activation independently to Re(z) and Im(z).

    This is a common approach in complex networks where preserving
    full holomorphicity is not required (e.g., classification).

    Args:
        real_fn: A real → real activation function (e.g., jax.nn.relu).
    """
    real_fn: Callable = eqx.field(static=True)

    def __init__(self, real_fn: Callable = jax.nn.gelu):
        self.real_fn = real_fn

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.real_fn(z.real) + 1j * self.real_fn(z.imag)


class HolomorphicActivation(eqx.Module):
    """
    Holomorphic activation via a learnable polynomial in z.

    Implements: φ(z) = Σ_{k=0}^{K} a_k * z^k

    where a_k are learnable complex coefficients. This is holomorphic
    by construction and satisfies the Cauchy-Riemann equations exactly.

    Args:
        degree: Maximum polynomial degree K.
    """
    coeffs: jnp.ndarray    # shape: (degree+1,) complex
    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 4, *, key: jax.Array):
        self.degree = degree
        k1, k2 = jax.random.split(key)
        scale = 1.0 / jnp.sqrt(degree + 1)
        re = jax.random.normal(k1, (degree + 1,)) * scale
        im = jax.random.normal(k2, (degree + 1,)) * scale
        self.coeffs = (re + 1j * im).astype(jnp.complex64)
        # Initialize close to identity: a_1 ≈ 1, rest ≈ 0
        self.coeffs = self.coeffs.at[1].set(1.0 + 0.0j)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate polynomial activation."""
        out = jnp.zeros_like(z)
        z_pow = jnp.ones_like(z)
        for k in range(self.degree + 1):
            out = out + self.coeffs[k] * z_pow
            z_pow = z_pow * z
        return out
