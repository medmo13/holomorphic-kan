"""
Basis Functions for Complex-Valued KANs
=========================================

Provides complex-domain basis expansions used as the learnable
univariate functions inside each KAN edge:

  - B-spline basis  (real grid lifted to complex via split or holomorphic)
  - Chebyshev basis (first-kind, analytically continued to ℂ)
  - Fourier basis   (naturally complex)

All functions operate on JAX arrays and are fully JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_complex(z: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
    """Clamp real and imaginary parts independently."""
    re = jnp.clip(z.real, low, high)
    im = jnp.clip(z.imag, low, high)
    return re + 1j * im


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------

def _bspline_basis_1d(t: jnp.ndarray, knots: jnp.ndarray, degree: int) -> jnp.ndarray:
    """
    Evaluate B-spline basis functions at scalar t.

    Uses the Cox-de Boor recursion, supporting complex t values via
    analytic continuation (all arithmetic stays in ℂ).

    Args:
        t:      Scalar (possibly complex) evaluation point.
        knots:  1-D array of knot positions (real).
        degree: Spline degree (k), yielding (len(knots) - k - 1) basis fns.

    Returns:
        Array of shape (n_basis,) with complex values.
    """
    n = len(knots) - degree - 1  # number of basis functions

    # Degree-0: indicator functions (made differentiable via soft step)
    # For complex t, we use real part for interval membership
    t_re = t.real if jnp.iscomplexobj(t) else t

    # Base case: B_{i,0}(t) = 1 if knot[i] <= t < knot[i+1], else 0
    # Use soft indicator for gradient flow
    eps = 1e-8

    def soft_indicator(lo, hi, x):
        left  = jax.nn.sigmoid((x - lo) / eps)
        right = jax.nn.sigmoid((hi - x) / eps)
        return left * right

    basis = jnp.array(
        [soft_indicator(knots[i], knots[i + 1], t_re) for i in range(len(knots) - 1)],
        dtype=jnp.complex64 if jnp.iscomplexobj(t) else jnp.float32,
    )

    # De Boor recursion
    for d in range(1, degree + 1):
        new_basis = []
        for i in range(len(knots) - d - 1):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            c1 = jnp.where(denom1 != 0, (t - knots[i]) / (denom1 + eps), 0.0)
            c2 = jnp.where(denom2 != 0, (knots[i + d + 1] - t) / (denom2 + eps), 0.0)
            new_basis.append(c1 * basis[i] + c2 * basis[i + 1])
        basis = jnp.stack(new_basis)

    return basis  # shape: (n,)


def complex_bspline_basis(
    z: jnp.ndarray,
    n_basis: int = 8,
    degree: int = 3,
    domain: tuple[float, float] = (-1.0, 1.0),
    mode: str = "split",
) -> jnp.ndarray:
    """
    Compute complex B-spline basis for input array z.

    Args:
        z:        Input array of shape (...,) with complex dtype.
        n_basis:  Number of basis functions per component.
        degree:   B-spline degree.
        domain:   (low, high) for the knot grid.
        mode:     'split'       – apply real basis to Re(z) and Im(z) separately,
                                  output shape (..., 2*n_basis)
                  'holomorphic' – analytically continue into ℂ,
                                  output shape (..., n_basis)

    Returns:
        Basis activations of shape (..., n_basis) or (..., 2*n_basis).
    """
    n_knots = n_basis + degree + 1
    knots = jnp.linspace(domain[0], domain[1], n_knots)
    # Pad knots to ensure clamped boundary
    knots = jnp.concatenate([
        jnp.full(degree, domain[0]),
        jnp.linspace(domain[0], domain[1], n_knots - 2 * degree),
        jnp.full(degree, domain[1]),
    ])

    _basis_fn = partial(_bspline_basis_1d, knots=knots, degree=degree)

    flat = z.reshape(-1)

    if mode == "split":
        def _eval_split(zi):
            re_basis = _basis_fn(zi.real.astype(jnp.float32))
            im_basis = _basis_fn(zi.imag.astype(jnp.float32))
            return jnp.concatenate([re_basis, im_basis]).astype(jnp.complex64)
        basis = vmap(_eval_split)(flat)
        return basis.reshape(*z.shape, 2 * n_basis)

    elif mode == "holomorphic":
        def _eval_holo(zi):
            return _basis_fn(zi.astype(jnp.complex64))
        basis = vmap(_eval_holo)(flat)
        return basis.reshape(*z.shape, n_basis)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'split' or 'holomorphic'.")


# ---------------------------------------------------------------------------
# Chebyshev basis
# ---------------------------------------------------------------------------

def complex_chebyshev_basis(
    z: jnp.ndarray,
    degree: int = 8,
    mode: str = "split",
) -> jnp.ndarray:
    """
    Chebyshev polynomial basis T_0(z), T_1(z), ..., T_{degree}(z).

    The three-term recurrence T_{n+1}(z) = 2z*T_n(z) - T_{n-1}(z)
    is valid over ℂ, enabling holomorphic evaluation.

    Args:
        z:      Input array of complex dtype.
        degree: Maximum Chebyshev degree (inclusive).
        mode:   'split' or 'holomorphic' (see complex_bspline_basis).

    Returns:
        Basis array of shape (..., degree+1) or (..., 2*(degree+1)).
    """
    def _cheb_1d(x):
        """Evaluate Chebyshev polynomials via recurrence."""
        T = [jnp.ones_like(x), x]
        for _ in range(2, degree + 1):
            T.append(2.0 * x * T[-1] - T[-2])
        return jnp.stack(T, axis=-1)  # shape: (degree+1,)

    if mode == "split":
        re_basis = _cheb_1d(jnp.tanh(z.real))   # map to [-1,1]
        im_basis = _cheb_1d(jnp.tanh(z.imag))
        return jnp.concatenate([re_basis, im_basis], axis=-1).astype(jnp.complex64)

    elif mode == "holomorphic":
        # Normalize z to approximately unit disk
        z_norm = z / (jnp.abs(z).max() + 1e-8)
        return _cheb_1d(z_norm).astype(jnp.complex64)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

def complex_fourier_basis(
    z: jnp.ndarray,
    n_freqs: int = 8,
    mode: str = "split",
) -> jnp.ndarray:
    """
    Fourier feature basis: [e^{i*k*z} for k in 0..n_freqs-1].

    This is naturally complex and holomorphic. In split mode,
    real and imaginary parts of z are treated as independent 1-D signals.

    Args:
        z:       Input array of complex dtype.
        n_freqs: Number of frequency components.
        mode:    'split' or 'holomorphic'.

    Returns:
        Basis array of shape (..., n_freqs) or (..., 2*n_freqs).
    """
    freqs = jnp.arange(n_freqs, dtype=jnp.float32)

    if mode == "split":
        re_basis = jnp.exp(1j * jnp.pi * jnp.outer(z.real.ravel(), freqs))
        im_basis = jnp.exp(1j * jnp.pi * jnp.outer(z.imag.ravel(), freqs))
        basis = jnp.concatenate([re_basis, im_basis], axis=-1)
        return basis.reshape(*z.shape, 2 * n_freqs)

    elif mode == "holomorphic":
        # e^{i*k*z} for complex z -> e^{i*k*(a+ib)} = e^{-k*b} * e^{i*k*a}
        flat = z.reshape(-1)
        basis = jnp.exp(1j * jnp.pi * jnp.outer(flat, freqs))
        return basis.reshape(*z.shape, n_freqs)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")
