"""
CVKAN: Complex-Valued Kolmogorov-Arnold Networks in JAX
=======================================================

A JAX-based library for training Kolmogorov-Arnold Networks (KANs)
with complex-valued inputs, outputs, and learnable activation functions.

Key features:
  - Complex-valued B-spline basis functions
  - Wirtinger calculus for complex gradient computation
  - Holomorphic and split-complex activation modes
  - Modular layer architecture compatible with Flax/Optax
"""

from cvkan.layers import CVKANLayer, CVKANDense
from cvkan.network import CVKAN
from cvkan.activations import (
    ComplexSplineActivation,
    SplitComplexActivation,
    HolomorphicActivation,
    ModReLU,
    zReLU,
    CReLU,
)
from cvkan.basis import (
    complex_bspline_basis,
    complex_chebyshev_basis,
    complex_fourier_basis,
)
from cvkan.utils import (
    complex_kaiming_uniform,
    complex_glorot_uniform,
    wirtinger_gradient,
    to_complex,
    to_real_imag,
)
from cvkan.trainer import CVKANTrainer

__version__ = "0.1.0"
__author__  = "CVKAN Contributors"

__all__ = [
    # Layers
    "CVKANLayer",
    "CVKANDense",
    # Network
    "CVKAN",
    # Activations
    "ComplexSplineActivation",
    "SplitComplexActivation",
    "HolomorphicActivation",
    "ModReLU",
    "zReLU",
    "CReLU",
    # Basis functions
    "complex_bspline_basis",
    "complex_chebyshev_basis",
    "complex_fourier_basis",
    # Utils
    "complex_kaiming_uniform",
    "complex_glorot_uniform",
    "wirtinger_gradient",
    "to_complex",
    "to_real_imag",
    # Trainer
    "CVKANTrainer",
]
