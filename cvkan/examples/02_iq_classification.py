"""
Example 2: IQ Signal Classification with CVKAN
=================================================

Classifies quadrature (IQ) signals — naturally complex-valued — using
a CVKAN. This mimics RF modulation recognition tasks common in
communications engineering (AMR, radar, sonar).

Synthetic dataset: 4-class modulation problem
  - Class 0: QPSK  → phases in {±45°, ±135°}
  - Class 1: BPSK  → phases in {0°, 180°}
  - Class 2: 8PSK  → phases in multiples of 45°
  - Class 3: QAM16 → mixed amplitude + phase

Run with:
    python examples/02_iq_classification.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cvkan import CVKAN, CVKANTrainer


# ---------------------------------------------------------------------------
# 1. Synthetic IQ dataset
# ---------------------------------------------------------------------------

def make_iq_dataset(n_per_class: int = 500, snr_db: float = 15.0, seed: int = 0):
    """Generate synthetic IQ samples for 4 modulation classes."""
    rng = np.random.default_rng(seed)
    sigma = 10 ** (-snr_db / 20)

    samples, labels = [], []

    for cls in range(4):
        if cls == 0:   # QPSK: 4 constellation points
            angles = rng.choice([45, 135, 225, 315], n_per_class) * np.pi / 180
            r = np.ones(n_per_class)
        elif cls == 1:  # BPSK: 2 constellation points
            angles = rng.choice([0, 180], n_per_class) * np.pi / 180
            r = np.ones(n_per_class)
        elif cls == 2:  # 8PSK: 8 constellation points
            angles = rng.choice(range(0, 360, 45), n_per_class) * np.pi / 180
            r = np.ones(n_per_class)
        else:            # QAM16: 16 mixed-amplitude points
            real_pts = rng.choice([-3, -1, 1, 3], n_per_class)
            imag_pts = rng.choice([-3, -1, 1, 3], n_per_class)
            s = (real_pts + 1j * imag_pts) / 3.0  # normalize
            noise = rng.normal(0, sigma, n_per_class) + 1j * rng.normal(0, sigma, n_per_class)
            samples.append(s + noise)
            labels.append(np.full(n_per_class, cls))
            continue

        # Add AWGN
        noise = rng.normal(0, sigma, n_per_class) + 1j * rng.normal(0, sigma, n_per_class)
        s = r * np.exp(1j * angles) + noise
        samples.append(s)
        labels.append(np.full(n_per_class, cls))

    x = np.concatenate(samples).astype(np.complex64)
    y = np.concatenate(labels).astype(np.int32)

    # Shuffle
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


print("Generating IQ dataset...")
x_all, y_all = make_iq_dataset(n_per_class=600, snr_db=12.0)

# Train/val split
n_train = 1800
x_train, x_val = x_all[:n_train, None], x_all[n_train:, None]
y_train, y_val = y_all[:n_train], y_all[n_train:]

print(f"  Train: {x_train.shape}, Val: {x_val.shape}")
print(f"  Classes: {np.bincount(y_train)}")

# One-hot encode targets as complex (imaginary part = 0)
n_classes = 4
y_train_oh = np.eye(n_classes)[y_train].astype(np.complex64)
y_val_oh   = np.eye(n_classes)[y_val].astype(np.complex64)


# ---------------------------------------------------------------------------
# 2. Build classifier model
# ---------------------------------------------------------------------------

key = jax.random.PRNGKey(7)
key, model_key = jax.random.split(key)

model = CVKAN(
    layer_sizes = [1, 16, 8, n_classes],
    n_basis     = 8,
    basis       = "chebyshev",
    basis_mode  = "split",
    use_bias    = True,
    use_norm    = True,
    output_mode = "magnitude",   # |z| → softmax for classification
    key         = model_key,
)

print("\n" + model.summary())


# ---------------------------------------------------------------------------
# 3. Custom classification loss
# ---------------------------------------------------------------------------

def softmax_cross_entropy(model, x, y_oh, lambda_l1=1e-4, lambda_entropy=1e-4):
    """Softmax CE on magnitude outputs."""
    logits = jax.vmap(model)(x.astype(jnp.complex64))   # (batch, n_classes) real
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce_loss = -jnp.mean(jnp.sum(y_oh.real * log_probs, axis=-1))
    reg     = model.regularization_loss(lambda_l1, lambda_entropy)
    return ce_loss + reg


optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=5e-3,
        warmup_steps=100, decay_steps=2000,
    ) |> optax.scale_by_learning_rate,  # type: ignore[operator]
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(3e-3),
)

trainer = CVKANTrainer(model, optimizer=optimizer, loss_fn=softmax_cross_entropy)

history = trainer.fit(
    x_train        = x_train,
    y_train        = y_train_oh,
    epochs         = 150,
    batch_size     = 64,
    x_val          = x_val,
    y_val          = y_val_oh,
    verbose        = True,
    early_stopping = 20,
    key            = key,
)

# ---------------------------------------------------------------------------
# 4. Accuracy
# ---------------------------------------------------------------------------

logits_val = np.array(trainer.predict(x_val))
preds      = np.argmax(logits_val, axis=-1)
accuracy   = (preds == y_val).mean() * 100

print(f"\nValidation Accuracy: {accuracy:.2f}%")
print("Per-class accuracy:")
for c in range(n_classes):
    mask = y_val == c
    acc_c = (preds[mask] == c).mean() * 100
    names = ["QPSK", "BPSK", "8PSK", "QAM16"]
    print(f"  {names[c]:6s}: {acc_c:.1f}%")
