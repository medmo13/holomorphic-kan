# CVKAN — Complex-Valued Kolmogorov-Arnold Networks in JAX

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/backend-JAX-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**CVKAN** is a JAX-based library for training **Kolmogorov-Arnold Networks (KANs)** with **complex-valued** inputs, outputs, and learnable activation functions. It extends the original KAN architecture into the complex domain using:

- **Wirtinger calculus** for principled complex gradient descent
- **Complex B-spline, Chebyshev, and Fourier basis functions**
- **Holomorphic and split-complex activation modes**
- Full **JIT compilation** and **vmap** compatibility via Equinox + Optax

---

## Background

### Kolmogorov-Arnold Networks (KANs)

KANs replace fixed nonlinear activations (like ReLU) with **learnable univariate functions** on every edge of the network graph. Based on the Kolmogorov-Arnold representation theorem, they offer high interpretability and function approximation power. Each layer computes:

$$\mathbf{y}_j = \sum_{i} \varphi_{i,j}(x_i)$$

where each $\varphi_{i,j}$ is a learnable function parameterized by spline coefficients.

### Complex Extension

Extending KANs to $\mathbb{C}$ enables direct processing of complex signals (RF, audio STFT, optical fields, quantum states) without splitting real/imaginary parts artificially. CVKAN supports:

| Mode | Description |
|---|---|
| `split` | Apply real basis to $\text{Re}(z)$ and $\text{Im}(z)$ independently |
| `holomorphic` | Analytically continue the basis functions into $\mathbb{C}$ |

Gradients are computed using **Wirtinger derivatives**:

$$\frac{\partial L}{\partial \bar{z}} = \frac{1}{2}\left(\frac{\partial L}{\partial x} + i\frac{\partial L}{\partial y}\right)$$

---

## Installation

```bash
git clone https://github.com/yourusername/cvkan
cd cvkan
pip install -e ".[dev]"
```

For GPU support:
```bash
pip install -e ".[gpu]"
```

**Dependencies:** `jax`, `equinox`, `optax`, `numpy`

---

## Quick Start

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cvkan import CVKAN, CVKANTrainer

# 1. Create a complex KAN: 4 inputs → hidden layers → 2 outputs
key = jax.random.PRNGKey(0)
model = CVKAN(
    layer_sizes = [4, 16, 8, 2],
    n_basis     = 8,           # spline basis functions per edge
    basis       = "bspline",   # 'bspline', 'chebyshev', or 'fourier'
    basis_mode  = "split",     # 'split' or 'holomorphic'
    use_norm    = True,        # complex layer normalization
    output_mode = "complex",   # 'complex', 'real', 'magnitude', 'split'
    key         = key,
)

print(model.summary())

# 2. Generate complex data
n = 1000
x = np.random.randn(n, 4) + 1j * np.random.randn(n, 4)
y = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)

# 3. Train
trainer = CVKANTrainer(
    model,
    optimizer      = optax.adam(1e-3),
    lambda_l1      = 1e-4,      # sparsity regularization
    lambda_entropy = 1e-4,      # entropy regularization
)

history = trainer.fit(
    x_train        = x[:800],
    y_train        = y[:800],
    x_val          = x[800:],
    y_val          = y[800:],
    epochs         = 100,
    batch_size     = 32,
    early_stopping = 15,
)

# 4. Evaluate
metrics = trainer.evaluate(x[800:], y[800:])
print(f"R² = {metrics['r2']:.4f} | MAE = {metrics['mae']:.4e}")

# 5. Predict
y_pred = trainer.predict(x[:5])   # shape: (5, 2) complex
```

---

## API Reference

### Basis Functions (`cvkan.basis`)

All basis functions accept complex arrays and return complex basis evaluations:

```python
from cvkan.basis import complex_bspline_basis, complex_chebyshev_basis, complex_fourier_basis

z = jnp.array([0.5 + 0.3j, -0.2 + 0.8j])

# B-spline (split mode): shape (2, 2*n_basis)
B = complex_bspline_basis(z, n_basis=8, degree=3, mode="split")

# Chebyshev: shape (2, degree+1)
C = complex_chebyshev_basis(z, degree=8, mode="holomorphic")

# Fourier: shape (2, n_freqs)
F = complex_fourier_basis(z, n_freqs=8, mode="split")
```

### Activations (`cvkan.activations`)

```python
from cvkan.activations import ModReLU, zReLU, CReLU, ComplexSplineActivation, HolomorphicActivation

z = jnp.array([-0.5 + 1.2j])

# Magnitude-gated ReLU (learnable bias)
out = ModReLU(z, bias=jnp.array(-0.3))

# First-quadrant gate
out = zReLU(z)   # → 0+0j (negative real part)

# Per-component ReLU
out = CReLU(z)   # → 0+1.2j

# Learnable spline activation (used inside KAN layers)
act = ComplexSplineActivation(n_basis=8, basis_fn=complex_bspline_basis, key=key)
out = act(z[0])  # scalar in, scalar out

# Holomorphic polynomial activation
poly_act = HolomorphicActivation(degree=4, key=key)
out = poly_act(z[0])
```

### Layers (`cvkan.layers`)

```python
from cvkan.layers import CVKANLayer, CVKANDense

# KAN layer: one learnable function per (in, out) pair
layer = CVKANLayer(in_features=4, out_features=8, n_basis=6, key=key)
x = jnp.ones((32, 4), dtype=jnp.complex64)
y = layer(x)   # (32, 8)

# Dense complex linear layer (for baselines or hybrid nets)
dense = CVKANDense(in_features=4, out_features=8, key=key)
y = dense(x)   # (32, 8)

# Regularization losses
l1  = layer.l1_regularization()
ent = layer.entropy_regularization()
```

### Network (`cvkan.network`)

```python
from cvkan import CVKAN

model = CVKAN(
    layer_sizes = [2, 8, 8, 1],
    n_basis     = 8,
    basis       = "bspline",      # 'bspline' | 'chebyshev' | 'fourier'
    basis_mode  = "split",        # 'split' | 'holomorphic'
    use_bias    = True,
    use_norm    = True,           # complex layer normalization
    residual    = False,          # residual connections
    output_mode = "complex",      # 'complex' | 'real' | 'magnitude' | 'split'
    key         = key,
)

# Single sample (unbatched)
x = jnp.ones(2, dtype=jnp.complex64)
y = model(x)

# Batched via vmap
x_batch = jnp.ones((16, 2), dtype=jnp.complex64)
y_batch = jax.vmap(model)(x_batch)

# Network info
print(model.summary())
print(f"Parameters: {model.count_parameters():,}")
```

### Trainer (`cvkan.trainer`)

```python
from cvkan import CVKANTrainer

trainer = CVKANTrainer(
    model,
    optimizer      = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
    lambda_l1      = 1e-4,
    lambda_entropy = 1e-4,
)

# Train
history = trainer.fit(
    x_train, y_train,
    epochs=100, batch_size=64,
    x_val=x_val, y_val=y_val,
    early_stopping=20,
    verbose=True,
)

# Predict and evaluate
y_pred  = trainer.predict(x_test)
metrics = trainer.evaluate(x_test, y_test)  # {'mse', 'mae', 'r2'}

# Save/load
trainer.save("model.eqx")
trainer.load("model.eqx")
```

### Utilities (`cvkan.utils`)

```python
from cvkan.utils import (
    complex_glorot_uniform,   # Complex Glorot weight init
    complex_kaiming_uniform,  # Complex Kaiming weight init
    wirtinger_gradient,       # ∂f/∂z or ∂f/∂z̄
    to_complex,               # (re, im) → complex
    to_real_imag,             # complex → (re, im)
    complex_mse,              # |ŷ - y|² mean
    complex_mae,              # |ŷ - y| mean
    complex_r2,               # R² score for complex outputs
    complex_layer_norm,       # Layer normalization for complex arrays
    batch_iter,               # Shuffled mini-batch generator
)
```

---

## Examples

| File | Description |
|---|---|
| `examples/01_complex_function_fitting.py` | Approximate `f(z) = z² + sin(z)/(1+\|z\|)` |
| `examples/02_iq_classification.py` | RF modulation recognition on IQ signals |

---

## Architecture Design Decisions

### Why Equinox?
Equinox treats PyTrees as first-class objects and handles complex arrays naturally within JAX's transform system (jit, vmap, grad). It's lighter than Flax and avoids the boilerplate of Haiku while supporting stateful-free functional design.

### Wirtinger vs. Real Gradients
JAX's autodiff computes Wirtinger (holomorphic) gradients by default when using complex arrays, equivalent to ∂L/∂z̄. This is the correct update direction for minimizing real-valued loss over complex parameters, consistent with economic Cauchy-Riemann gradient descent.

### Split vs. Holomorphic Mode
- **Split mode** treats Re(z) and Im(z) as independent real signals — more stable numerically, works well for arbitrary complex data.
- **Holomorphic mode** evaluates basis functions at complex arguments via analytic continuation — theoretically richer but can have larger imaginary parts in basis evaluations.

---

## Citation

If you use CVKAN in your research, please cite:

```bibtex
@software{cvkan2024,
  title  = {CVKAN: Complex-Valued Kolmogorov-Arnold Networks in JAX},
  year   = {2024},
  url    = {https://github.com/yourusername/cvkan},
}
```

Original KAN paper:
```bibtex
@article{liu2024kan,
  title   = {KAN: Kolmogorov-Arnold Networks},
  author  = {Liu, Ziming and others},
  journal = {arXiv:2404.19756},
  year    = {2024},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
