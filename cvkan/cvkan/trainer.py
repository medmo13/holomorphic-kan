"""
CVKAN Trainer
===============

A high-level training loop compatible with Optax optimizers.
Handles:
  - JIT-compiled complex gradient computation
  - Wirtinger-aware parameter updates
  - Learning rate scheduling
  - Metric tracking (loss, R², MAE)
  - Early stopping
  - Checkpoint saving/loading

Usage::

    from cvkan import CVKAN, CVKANTrainer
    import optax

    model   = CVKAN(layer_sizes=[4, 8, 2], key=jax.random.PRNGKey(0))
    trainer = CVKANTrainer(model, optimizer=optax.adam(1e-3))
    history = trainer.fit(x_train, y_train, epochs=100, batch_size=32)
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from cvkan.network import CVKAN
from cvkan.utils import complex_mse, complex_mae, complex_r2, batch_iter


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(model: CVKAN, x: jnp.ndarray, y: jnp.ndarray, **reg_kw) -> jnp.ndarray:
    """MSE + optional KAN regularization."""
    y_pred = jax.vmap(model)(x)
    data_loss = complex_mse(y_pred, y.astype(jnp.complex64))
    reg_loss  = model.regularization_loss(**reg_kw) if reg_kw else jnp.zeros(())
    return data_loss + reg_loss


def mae_loss(model: CVKAN, x: jnp.ndarray, y: jnp.ndarray, **reg_kw) -> jnp.ndarray:
    """MAE + optional KAN regularization."""
    y_pred = jax.vmap(model)(x)
    data_loss = complex_mae(y_pred, y.astype(jnp.complex64))
    reg_loss  = model.regularization_loss(**reg_kw) if reg_kw else jnp.zeros(())
    return data_loss + reg_loss


# ---------------------------------------------------------------------------
# Single training step (JIT-compiled)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _train_step(
    model:       CVKAN,
    opt_state:   optax.OptState,
    optimizer:   optax.GradientTransformation,
    x:           jnp.ndarray,
    y:           jnp.ndarray,
    loss_fn:     Callable,
) -> Tuple[CVKAN, optax.OptState, jnp.ndarray]:
    """
    One gradient update step.

    Equinox's filter_grad correctly handles complex arrays by treating
    their real and imaginary parts as separate parameters, which is
    equivalent to Wirtinger gradient descent on the conjugate variable.

    Args:
        model:     Current model (Equinox module).
        opt_state: Current optimizer state.
        optimizer: Optax optimizer.
        x:         Batch inputs.
        y:         Batch targets.
        loss_fn:   Scalar loss function (model, x, y) -> scalar.

    Returns:
        (updated_model, updated_opt_state, loss_value)
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss_val


@eqx.filter_jit
def _eval_step(
    model: CVKAN,
    x:     jnp.ndarray,
    y:     jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute evaluation metrics without gradient tracking."""
    y_pred = jax.vmap(model)(x)
    y_true = y.astype(jnp.complex64)
    return {
        "mse": complex_mse(y_pred, y_true),
        "mae": complex_mae(y_pred, y_true),
        "r2":  complex_r2(y_pred, y_true),
    }


# ---------------------------------------------------------------------------
# CVKANTrainer
# ---------------------------------------------------------------------------

class CVKANTrainer:
    """
    High-level training manager for CVKAN models.

    Args:
        model:          CVKAN network to train.
        optimizer:      Optax optimizer (default: adam with lr=1e-3).
        loss_fn:        Loss function (model, x, y) -> scalar.
                        Defaults to MSE + KAN regularization.
        lambda_l1:      L1 regularization coefficient.
        lambda_entropy: Entropy regularization coefficient.

    Example::

        trainer = CVKANTrainer(
            model,
            optimizer=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(1e-3),
            ),
        )
        history = trainer.fit(x_train, y_train, epochs=50)
        print(history["val_r2"])
    """

    def __init__(
        self,
        model:          CVKAN,
        optimizer:      Optional[optax.GradientTransformation] = None,
        loss_fn:        Optional[Callable] = None,
        lambda_l1:      float = 1e-4,
        lambda_entropy: float = 1e-4,
    ):
        self.model     = model
        self.optimizer = optimizer or optax.adam(1e-3)
        self.lambda_l1 = lambda_l1
        self.lambda_en = lambda_entropy

        if loss_fn is None:
            # Closure over regularization strengths
            def _default_loss(m, x, y):
                return mse_loss(m, x, y, lambda_l1=lambda_l1, lambda_entropy=lambda_entropy)
            self.loss_fn = _default_loss
        else:
            self.loss_fn = loss_fn

        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_mse":    [], "val_mae":  [], "val_r2": [],
            "epoch_time": [],
        }

    def fit(
        self,
        x_train:        np.ndarray,
        y_train:        np.ndarray,
        epochs:         int = 100,
        batch_size:     int = 32,
        x_val:          Optional[np.ndarray] = None,
        y_val:          Optional[np.ndarray] = None,
        verbose:        bool = True,
        early_stopping: Optional[int] = None,
        key:            jax.Array = jax.random.PRNGKey(42),
    ) -> Dict[str, List[float]]:
        """
        Train the CVKAN model.

        Args:
            x_train:        Training inputs, complex array (n, in_features).
            y_train:        Training targets, complex array (n, out_features).
            epochs:         Number of training epochs.
            batch_size:     Mini-batch size.
            x_val:          Optional validation inputs.
            y_val:          Optional validation targets.
            verbose:        Print training progress.
            early_stopping: Stop if val_loss doesn't improve for this many epochs.
            key:            PRNG key for shuffling.

        Returns:
            Training history dictionary.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            key, subkey = jax.random.split(key)

            # --- Training loop ---
            epoch_losses = []
            for xb, yb in batch_iter(x_train, y_train, batch_size, subkey):
                self.model, self.opt_state, loss_val = _train_step(
                    self.model, self.opt_state, self.optimizer,
                    xb, yb, self.loss_fn,
                )
                epoch_losses.append(float(loss_val))

            train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(train_loss)

            # --- Validation ---
            val_metrics: Dict[str, float] = {}
            if x_val is not None and y_val is not None:
                vm = _eval_step(
                    self.model,
                    jnp.array(x_val),
                    jnp.array(y_val),
                )
                val_metrics = {k: float(v) for k, v in vm.items()}
                self.history["val_loss"].append(val_metrics["mse"])
                self.history["val_mse"].append(val_metrics["mse"])
                self.history["val_mae"].append(val_metrics["mae"])
                self.history["val_r2"].append(val_metrics["r2"])
            else:
                self.history["val_loss"].append(train_loss)

            epoch_time = time.time() - t0
            self.history["epoch_time"].append(epoch_time)

            # --- Logging ---
            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                msg = f"Epoch {epoch:4d}/{epochs} | train_loss={train_loss:.4e}"
                if val_metrics:
                    msg += (
                        f" | val_mse={val_metrics['mse']:.4e}"
                        f" | val_r2={val_metrics['r2']:.4f}"
                    )
                msg += f" | {epoch_time:.2f}s"
                print(msg)

            # --- Early stopping ---
            if early_stopping is not None:
                current_val = self.history["val_loss"][-1]
                if current_val < best_val_loss - 1e-6:
                    best_val_loss = current_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}.")
                        break

        return self.history

    def predict(self, x: np.ndarray) -> jnp.ndarray:
        """Run inference on x and return predictions."""
        return jax.vmap(self.model)(jnp.array(x))

    def evaluate(
        self, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics on a dataset."""
        metrics = _eval_step(self.model, jnp.array(x), jnp.array(y))
        return {k: float(v) for k, v in metrics.items()}

    def save(self, path: str) -> None:
        """
        Save model weights to disk using Equinox's serialization.

        Args:
            path: File path ending in '.eqx'.
        """
        eqx.tree_serialise_leaves(path, self.model)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model weights from disk.

        Args:
            path: File path to the saved .eqx file.
        """
        self.model = eqx.tree_deserialise_leaves(path, self.model)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        print(f"Model loaded from {path}")
