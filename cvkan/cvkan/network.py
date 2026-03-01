"""
Complex-Valued KAN Network
============================

Stacks multiple CVKANLayer instances into a deep network with optional:
  - Intermediate normalization
  - Residual connections
  - Mixed KAN + Dense architectures
  - Configurable output modes (complex, split-real, magnitude)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, List, Literal, Union

from cvkan.layers import CVKANLayer, CVKANDense
from cvkan.activations import (
    ComplexSplineActivation,
    SplitComplexActivation,
    ModReLU,
    CReLU,
)
from cvkan.utils import complex_layer_norm


# ---------------------------------------------------------------------------
# CVKAN Network
# ---------------------------------------------------------------------------

class CVKAN(eqx.Module):
    """
    Complex-Valued Kolmogorov-Arnold Network.

    A multi-layer KAN where each layer applies learnable complex
    univariate functions on each input-output edge.

    Architecture:
        Input → [CVKANLayer_0 → Norm? → ...] → Output

    Args:
        layer_sizes:   List of integers defining the network width at each
                       layer boundary, e.g. [2, 4, 4, 1] gives 3 KAN layers.
        n_basis:       Number of basis functions per activation edge.
        basis:         Basis type: 'bspline', 'chebyshev', or 'fourier'.
        basis_mode:    'split' or 'holomorphic'.
        use_bias:      Include learnable complex bias in each layer.
        use_norm:      Apply complex layer norm between layers.
        residual:      Add residual connections (requires matching widths).
        output_mode:   How to produce final output:
                         'complex'   – return complex array as-is
                         'real'      – take real part only
                         'magnitude' – take absolute value
                         'split'     – concatenate real and imaginary parts
        key:           JAX PRNG key.

    Example::

        import jax, jax.numpy as jnp
        from cvkan import CVKAN

        key = jax.random.PRNGKey(0)
        model = CVKAN(layer_sizes=[4, 8, 8, 2], n_basis=6, key=key)

        x = jax.random.normal(key, (32, 4)) + 1j * jax.random.normal(key, (32, 4))
        y = jax.vmap(model)(x)   # shape: (32, 2) complex
    """

    layers:       list
    use_norm:     bool  = eqx.field(static=True)
    residual:     bool  = eqx.field(static=True)
    output_mode:  str   = eqx.field(static=True)
    layer_sizes:  list  = eqx.field(static=True)

    def __init__(
        self,
        layer_sizes:  List[int],
        n_basis:      int  = 8,
        basis:        str  = "bspline",
        basis_mode:   str  = "split",
        use_bias:     bool = True,
        use_norm:     bool = False,
        residual:     bool = False,
        output_mode:  str  = "complex",
        *,
        key: jax.Array,
    ):
        assert len(layer_sizes) >= 2, "Need at least input and output size."
        assert output_mode in ("complex", "real", "magnitude", "split"), \
            f"Unknown output_mode '{output_mode}'."

        self.layer_sizes = layer_sizes
        self.use_norm    = use_norm
        self.residual    = residual
        self.output_mode = output_mode

        n_layers = len(layer_sizes) - 1
        keys = jax.random.split(key, n_layers)

        self.layers = [
            CVKANLayer(
                in_features  = layer_sizes[i],
                out_features = layer_sizes[i + 1],
                n_basis      = n_basis,
                basis        = basis,
                basis_mode   = basis_mode,
                use_bias     = use_bias,
                key          = keys[i],
            )
            for i in range(n_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Complex input of shape (in_features,) or (batch, in_features).

        Returns:
            Output according to output_mode.
        """
        h = x.astype(jnp.complex64)

        for i, layer in enumerate(self.layers):
            h_new = layer(h)

            # Optional residual connection (if dimensions match)
            if self.residual and h.shape == h_new.shape:
                h_new = h_new + h

            # Optional normalization (skip on last layer)
            if self.use_norm and i < len(self.layers) - 1:
                h_new = complex_layer_norm(h_new) if h_new.ndim == 2 else h_new

            h = h_new

        return self._apply_output_mode(h)

    def _apply_output_mode(self, z: jnp.ndarray) -> jnp.ndarray:
        if self.output_mode == "complex":
            return z
        elif self.output_mode == "real":
            return z.real
        elif self.output_mode == "magnitude":
            return jnp.abs(z)
        elif self.output_mode == "split":
            return jnp.concatenate([z.real, z.imag], axis=-1)
        else:
            raise ValueError(f"Unknown output_mode '{self.output_mode}'.")

    def regularization_loss(
        self,
        lambda_l1:      float = 1e-4,
        lambda_entropy: float = 1e-4,
    ) -> jnp.ndarray:
        """
        Total regularization loss across all KAN layers.

        Combines L1 sparsity and entropy regularization as in the
        original KAN paper, adapted for complex activations.

        Args:
            lambda_l1:      L1 regularization coefficient.
            lambda_entropy: Entropy regularization coefficient.

        Returns:
            Scalar regularization loss.
        """
        reg = jnp.zeros(())
        for layer in self.layers:
            if isinstance(layer, CVKANLayer):
                reg = reg + lambda_l1 * layer.l1_regularization()
                reg = reg + lambda_entropy * layer.entropy_regularization()
        return reg

    def count_parameters(self) -> int:
        """Count total number of (real) learnable parameters."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(
            2 * l.size if jnp.iscomplexobj(l) else l.size
            for l in leaves
        )

    def summary(self) -> str:
        """Return a human-readable summary of the network architecture."""
        lines = [
            "=" * 60,
            f"  Complex-Valued KAN  (CVKAN)",
            "=" * 60,
            f"  Layer sizes  : {self.layer_sizes}",
            f"  Output mode  : {self.output_mode}",
            f"  Layer norm   : {self.use_norm}",
            f"  Residual     : {self.residual}",
            f"  Parameters   : {self.count_parameters():,}",
            "-" * 60,
        ]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CVKANLayer):
                lines.append(
                    f"  Layer {i}: CVKANLayer "
                    f"({layer.in_features} → {layer.out_features}), "
                    f"n_basis={layer.n_basis}, "
                    f"activations={len(layer.activations)}"
                )
            elif isinstance(layer, CVKANDense):
                w = layer.weight
                lines.append(
                    f"  Layer {i}: CVKANDense "
                    f"({w.shape[1]} → {w.shape[0]})"
                )
        lines.append("=" * 60)
        return "\n".join(lines)
