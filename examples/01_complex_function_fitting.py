"""
Example 1: Fitting a Complex-Valued Function
===============================================

Demonstrates training a CVKAN to approximate the complex function:

    f(z) = z² + sin(z) / (1 + |z|)

This function is meromorphic (analytic except at z = -1, |z|-wise),
making it a good test for holomorphic basis modes.

Run with:
    python examples/01_complex_function_fitting.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import sys, os

# Make the package importable from parent directory during development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cvkan import CVKAN, CVKANTrainer


# ---------------------------------------------------------------------------
# 1. Generate dataset
# ---------------------------------------------------------------------------

def target_fn(z: np.ndarray) -> np.ndarray:
    """Complex target function to approximate."""
    return z**2 + np.sin(z) / (1.0 + np.abs(z) + 1e-8)


key = jax.random.PRNGKey(42)
n_train, n_val = 2000, 400

# Sample z in the unit square of ℂ
key, k1, k2, k3, k4 = jax.random.split(key, 5)
x_train = (
    jax.random.uniform(k1, (n_train,), minval=-2.0, maxval=2.0) +
    1j * jax.random.uniform(k2, (n_train,), minval=-2.0, maxval=2.0)
)
x_val = (
    jax.random.uniform(k3, (n_val,), minval=-2.0, maxval=2.0) +
    1j * jax.random.uniform(k4, (n_val,), minval=-2.0, maxval=2.0)
)

y_train = target_fn(np.array(x_train))
y_val   = target_fn(np.array(x_val))

# Reshape to (n, 1) for network compatibility
x_train = x_train[:, None]
x_val   = x_val[:, None]
y_train = y_train[:, None]
y_val   = y_val[:, None]

print(f"Training set: {x_train.shape}, dtype={x_train.dtype}")
print(f"Target range: |y| in [{np.abs(y_train).min():.3f}, {np.abs(y_train).max():.3f}]")


# ---------------------------------------------------------------------------
# 2. Build the model
# ---------------------------------------------------------------------------

key, model_key = jax.random.split(key)

model = CVKAN(
    layer_sizes = [1, 8, 8, 1],   # 1 complex input → hidden → 1 complex output
    n_basis     = 6,
    basis       = "bspline",
    basis_mode  = "split",        # split Re/Im for stability
    use_bias    = True,
    use_norm    = True,
    output_mode = "complex",
    key         = model_key,
)

print("\n" + model.summary())


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(3e-3),
)

trainer = CVKANTrainer(
    model,
    optimizer      = optimizer,
    lambda_l1      = 1e-4,
    lambda_entropy = 1e-4,
)

history = trainer.fit(
    x_train        = np.array(x_train),
    y_train        = np.array(y_train),
    epochs         = 200,
    batch_size     = 64,
    x_val          = np.array(x_val),
    y_val          = np.array(y_val),
    verbose        = True,
    early_stopping = 30,
)


# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------

metrics = trainer.evaluate(np.array(x_val), np.array(y_val))
print(f"\nFinal Validation Metrics:")
print(f"  MSE : {metrics['mse']:.4e}")
print(f"  MAE : {metrics['mae']:.4e}")
print(f"  R²  : {metrics['r2']:.4f}")

# Sample predictions
y_pred = trainer.predict(np.array(x_val[:5]))
y_true = np.array(y_val[:5])
print("\nSample predictions vs targets:")
for i in range(5):
    print(
        f"  pred={y_pred[i, 0]:.4f}  "
        f"true={y_true[i, 0]:.4f}  "
        f"error={abs(y_pred[i, 0] - y_true[i, 0]):.4f}"
    )
