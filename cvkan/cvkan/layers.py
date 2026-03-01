"""
Complex-Valued KAN Layers
===========================

Implements the core KAN layer(s) using the Kolmogorov-Arnold representation:

    KAN Layer: f(x)_j = Σ_i  φ_{i,j}(x_i)

where each φ_{i,j} is a learnable univariate complex-valued function
represented by a ComplexSplineActivation (or other basis activation).

Two layer variants are provided:
  - CVKANLayer  : Full KAN layer with one learnable function per (in, out) pair.
  - CVKANDense  : Linear complex layer (standard dense) as a drop-in comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from typing import Callable, Optional, Literal

from cvkan.activations import ComplexSplineActivation, SplitComplexActivation
from cvkan.basis import complex_bspline_basis, complex_chebyshev_basis, complex_fourier_basis
from cvkan.utils import complex_glorot_uniform


# ---------------------------------------------------------------------------
# Helper: build basis function from string name
# ---------------------------------------------------------------------------

_BASIS_REGISTRY = {
    "bspline":    complex_bspline_basis,
    "chebyshev":  complex_chebyshev_basis,
    "fourier":    complex_fourier_basis,
}


def _resolve_basis(name: str) -> Callable:
    if name not in _BASIS_REGISTRY:
        raise ValueError(
            f"Unknown basis '{name}'. Choose from: {list(_BASIS_REGISTRY.keys())}"
        )
    return _BASIS_REGISTRY[name]


# ---------------------------------------------------------------------------
# CVKANLayer
# ---------------------------------------------------------------------------

class CVKANLayer(eqx.Module):
    """
    A single complex-valued KAN layer.

    Each input-output pair (i, j) has its own learnable activation
    φ_{i,j}: ℂ → ℂ, parameterized as a complex spline.

    The output is:
        y_j = Σ_{i=0}^{in_features-1}  φ_{i,j}(x_i)

    This mirrors the original KAN formulation but extended to ℂ.

    Args:
        in_features:  Number of complex input features.
        out_features: Number of complex output features.
        n_basis:      Number of basis functions per activation.
        basis:        Basis type: 'bspline', 'chebyshev', or 'fourier'.
        basis_mode:   'split' or 'holomorphic' (see basis module).
        basis_kw:     Additional keyword arguments forwarded to basis_fn.
        use_bias:     Whether to add a learnable complex bias.
        key:          JAX PRNG key.
    """
    activations: list          # (in_features * out_features,) ComplexSplineActivation
    bias:        Optional[jnp.ndarray]
    in_features:  int  = eqx.field(static=True)
    out_features: int  = eqx.field(static=True)
    n_basis:      int  = eqx.field(static=True)

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        n_basis:      int = 8,
        basis:        str = "bspline",
        basis_mode:   str = "split",
        basis_kw:     Optional[dict] = None,
        use_bias:     bool = True,
        *,
        key: jax.Array,
    ):
        self.in_features  = in_features
        self.out_features = out_features
        self.n_basis      = n_basis

        basis_fn = _resolve_basis(basis)
        # Adjust n_basis for split mode (uses 2x basis per component)
        effective_n_basis = n_basis * 2 if basis_mode == "split" else n_basis
        kw = {"mode": basis_mode, **(basis_kw or {})}

        # Create one activation per (in, out) pair
        n_activations = in_features * out_features
        keys = jax.random.split(key, n_activations + 1)
        self.activations = [
            ComplexSplineActivation(
                n_basis   = effective_n_basis,
                basis_fn  = basis_fn,
                basis_kw  = kw,
                key       = keys[k],
            )
            for k in range(n_activations)
        ]

        if use_bias:
            self.bias = jnp.zeros(out_features, dtype=jnp.complex64)
        else:
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Complex input of shape (in_features,) or (batch, in_features).

        Returns:
            Complex output of shape (out_features,) or (batch, out_features).
        """
        batched = x.ndim == 2
        if not batched:
            x = x[None]  # (1, in_features)

        batch_size = x.shape[0]

        # Apply each activation φ_{i,j} to x_i
        # activations indexed as [i * out_features + j]
        out = jnp.zeros((batch_size, self.out_features), dtype=jnp.complex64)

        for i in range(self.in_features):
            for j in range(self.out_features):
                act = self.activations[i * self.out_features + j]
                # Apply activation to each sample in batch
                phi_vals = vmap(act)(x[:, i])  # (batch,)
                out = out.at[:, j].add(phi_vals)

        if self.bias is not None:
            out = out + self.bias[None, :]

        return out if batched else out[0]

    def l1_regularization(self) -> jnp.ndarray:
        """
        Compute L1 regularization on activation magnitudes.

        Encourages sparsity in the learned activation network,
        consistent with the original KAN training procedure.

        Returns:
            Scalar real regularization loss.
        """
        total = jnp.zeros(())
        for act in self.activations:
            total = total + jnp.mean(jnp.abs(act.coeffs))
        return total / len(self.activations)

    def entropy_regularization(self) -> jnp.ndarray:
        """
        Entropy-based regularization to encourage sparse activations.

        Computes normalized entropy of |c_k| distribution per activation.

        Returns:
            Scalar real regularization loss.
        """
        total = jnp.zeros(())
        for act in self.activations:
            probs = jnp.abs(act.coeffs)
            probs = probs / (probs.sum() + 1e-8)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
            total = total + entropy
        return total / len(self.activations)


# ---------------------------------------------------------------------------
# CVKANDense  (standard complex linear layer for comparison/hybrid use)
# ---------------------------------------------------------------------------

class CVKANDense(eqx.Module):
    """
    Complex-valued dense (linear) layer.

    Computes: y = x @ W^T + b  in ℂ.

    This can be used as:
      - A baseline comparison to CVKANLayer.
      - A hybrid layer interleaved with KAN layers.
      - A projection layer at the input/output of a CVKAN network.

    Args:
        in_features:  Number of complex input features.
        out_features: Number of complex output features.
        use_bias:     Whether to include a learnable complex bias.
        init:         Weight initialization: 'glorot' or 'kaiming'.
        key:          JAX PRNG key.
    """
    weight: jnp.ndarray   # shape: (out_features, in_features) complex
    bias:   Optional[jnp.ndarray]

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        use_bias:     bool = True,
        init:         str  = "glorot",
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        shape = (out_features, in_features)

        if init == "glorot":
            from cvkan.utils import complex_glorot_uniform
            self.weight = complex_glorot_uniform(k1, shape)
        elif init == "kaiming":
            from cvkan.utils import complex_kaiming_uniform
            self.weight = complex_kaiming_uniform(k1, shape)
        else:
            raise ValueError(f"Unknown init '{init}'.")

        self.bias = jnp.zeros(out_features, dtype=jnp.complex64) if use_bias else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Complex input of shape (..., in_features).

        Returns:
            Complex output of shape (..., out_features).
        """
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
