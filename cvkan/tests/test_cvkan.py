"""
Unit Tests for CVKAN
======================

Tests cover:
  - Basis function shapes and complex dtypes
  - Layer forward pass dimensions
  - Network forward pass
  - Regularization values (non-negative)
  - Wirtinger gradient correctness
  - Trainer basic training loop
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Ensure reproducibility
jax.config.update("jax_enable_x64", False)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cvkan.basis import complex_bspline_basis, complex_chebyshev_basis, complex_fourier_basis
from cvkan.layers import CVKANLayer, CVKANDense
from cvkan.network import CVKAN
from cvkan.activations import ComplexSplineActivation, ModReLU, zReLU, CReLU
from cvkan.utils import (
    to_complex, to_real_imag, wirtinger_gradient,
    complex_mse, complex_mae, complex_r2,
    complex_glorot_uniform, complex_kaiming_uniform,
)
from cvkan.trainer import CVKANTrainer


KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

class TestBasisFunctions:
    def test_bspline_split_shape(self):
        z = jnp.array([0.5 + 0.3j, -0.2 + 0.1j])
        out = complex_bspline_basis(z, n_basis=4, degree=2, mode="split")
        assert out.shape == (2, 8), f"Expected (2, 8), got {out.shape}"
        assert jnp.iscomplexobj(out)

    def test_bspline_holomorphic_shape(self):
        z = jnp.array([0.5 + 0.3j])
        out = complex_bspline_basis(z, n_basis=4, degree=2, mode="holomorphic")
        assert out.shape == (1, 4)

    def test_chebyshev_split(self):
        z = jnp.ones((3,), dtype=jnp.complex64) * (0.5 + 0.2j)
        out = complex_chebyshev_basis(z, degree=5, mode="split")
        assert out.shape == (3, 12)

    def test_fourier_split(self):
        z = jnp.ones((4,), dtype=jnp.complex64)
        out = complex_fourier_basis(z, n_freqs=6, mode="split")
        assert out.shape == (4, 12)

    def test_fourier_holomorphic(self):
        z = jnp.ones((4,), dtype=jnp.complex64)
        out = complex_fourier_basis(z, n_freqs=6, mode="holomorphic")
        assert out.shape == (4, 6)

    def test_basis_no_nan(self):
        z = jnp.array([0.0 + 0.0j, 1.0 + 1.0j, -1.0 - 1.0j])
        for fn in [complex_bspline_basis, complex_chebyshev_basis, complex_fourier_basis]:
            out = fn(z)
            assert not jnp.any(jnp.isnan(jnp.abs(out))), f"NaN in {fn.__name__} output"


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class TestActivations:
    def test_modrelu(self):
        z = jnp.array([1.0 + 1.0j, -0.5 + 0.5j])
        bias = jnp.array(-0.5)
        out = ModReLU(z, bias)
        assert out.shape == z.shape
        assert jnp.iscomplexobj(out)

    def test_zrelu_first_quadrant(self):
        z = jnp.array([1.0 + 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, -1.0 - 1.0j])
        out = zReLU(z)
        assert out[0] != 0
        assert out[1] == 0
        assert out[2] == 0
        assert out[3] == 0

    def test_crelu(self):
        z = jnp.array([-1.0 + 2.0j, 3.0 - 4.0j])
        out = CReLU(z)
        assert out[0].real == 0.0
        assert out[0].imag == 2.0
        assert out[1].real == 3.0
        assert out[1].imag == 0.0

    def test_complex_spline_activation(self):
        from cvkan.basis import complex_bspline_basis
        act = ComplexSplineActivation(
            n_basis=8, basis_fn=complex_bspline_basis,
            basis_kw={"mode": "split"}, key=KEY
        )
        z = jnp.array(0.3 + 0.2j)
        out = act(z)
        assert out.shape == ()
        assert jnp.iscomplexobj(out)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class TestLayers:
    def test_cvkan_layer_unbatched(self):
        layer = CVKANLayer(in_features=3, out_features=4, n_basis=4, key=KEY)
        x = jnp.ones(3, dtype=jnp.complex64)
        out = layer(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_cvkan_layer_batched(self):
        layer = CVKANLayer(in_features=3, out_features=4, n_basis=4, key=KEY)
        x = jnp.ones((8, 3), dtype=jnp.complex64)
        out = layer(x)
        assert out.shape == (8, 4)

    def test_cvkan_dense_batched(self):
        layer = CVKANDense(in_features=5, out_features=3, key=KEY)
        x = jnp.ones((10, 5), dtype=jnp.complex64)
        out = layer(x)
        assert out.shape == (10, 3)

    def test_l1_regularization_positive(self):
        layer = CVKANLayer(in_features=2, out_features=2, n_basis=4, key=KEY)
        reg = layer.l1_regularization()
        assert float(reg) >= 0.0

    def test_entropy_regularization_positive(self):
        layer = CVKANLayer(in_features=2, out_features=2, n_basis=4, key=KEY)
        reg = layer.entropy_regularization()
        assert float(reg) >= 0.0


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class TestNetwork:
    def _build_model(self, output_mode="complex"):
        return CVKAN(
            layer_sizes=[2, 4, 3],
            n_basis=4,
            output_mode=output_mode,
            key=KEY,
        )

    def test_complex_output(self):
        model = self._build_model("complex")
        x = jnp.ones((5, 2), dtype=jnp.complex64)
        out = jax.vmap(model)(x)
        assert out.shape == (5, 3)
        assert jnp.iscomplexobj(out)

    def test_real_output(self):
        model = self._build_model("real")
        x = jnp.ones((5, 2), dtype=jnp.complex64)
        out = jax.vmap(model)(x)
        assert out.shape == (5, 3)
        assert not jnp.iscomplexobj(out)

    def test_magnitude_output(self):
        model = self._build_model("magnitude")
        x = jnp.ones((5, 2), dtype=jnp.complex64)
        out = jax.vmap(model)(x)
        assert out.shape == (5, 3)
        assert jnp.all(out >= 0)

    def test_split_output(self):
        model = self._build_model("split")
        x = jnp.ones((5, 2), dtype=jnp.complex64)
        out = jax.vmap(model)(x)
        assert out.shape == (5, 6)  # 3 complex -> 6 real

    def test_no_nan(self):
        model = self._build_model("complex")
        x = jnp.ones((4, 2), dtype=jnp.complex64) * (0.3 + 0.7j)
        out = jax.vmap(model)(x)
        assert not jnp.any(jnp.isnan(jnp.abs(out)))

    def test_regularization_loss(self):
        model = self._build_model()
        reg = model.regularization_loss()
        assert float(reg) >= 0.0

    def test_count_parameters(self):
        model = self._build_model()
        n_params = model.count_parameters()
        assert n_params > 0
        print(f"\n  Parameter count: {n_params}")


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_to_complex(self):
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        z = to_complex(x, y)
        assert jnp.all(z.real == x)
        assert jnp.all(z.imag == y)

    def test_to_real_imag(self):
        z = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])
        re, im = to_real_imag(z)
        assert jnp.all(re == jnp.array([1.0, 3.0]))
        assert jnp.all(im == jnp.array([2.0, 4.0]))

    def test_complex_mse_zero(self):
        y = jnp.array([1.0 + 1.0j, 2.0 + 0.0j])
        loss = complex_mse(y, y)
        assert jnp.isclose(loss, 0.0, atol=1e-6)

    def test_complex_r2_perfect(self):
        y = jnp.array([1.0 + 1.0j, 2.0 + 0.0j, -1.0 + 0.5j])
        r2 = complex_r2(y, y)
        assert jnp.isclose(r2, 1.0, atol=1e-5)

    def test_glorot_uniform_shape(self):
        w = complex_glorot_uniform(KEY, (4, 6))
        assert w.shape == (4, 6)
        assert jnp.iscomplexobj(w)

    def test_kaiming_uniform_shape(self):
        w = complex_kaiming_uniform(KEY, (8, 4))
        assert w.shape == (8, 4)
        assert jnp.iscomplexobj(w)

    def test_wirtinger_gradient(self):
        """Wirtinger gradient of |z|² should be z̄ (conj Wirtinger = z)."""
        f = lambda z: jnp.sum(jnp.abs(z) ** 2)
        z = jnp.array([1.0 + 2.0j, -1.0 + 0.5j])
        grad = wirtinger_gradient(f, z, conjugate=True)
        # ∂|z|²/∂z̄ = z
        expected = z
        assert jnp.allclose(grad.real, expected.real, atol=1e-4)
        assert jnp.allclose(grad.imag, expected.imag, atol=1e-4)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_basic_training_step(self):
        import numpy as np
        import optax

        key = jax.random.PRNGKey(1)
        model = CVKAN(layer_sizes=[2, 4, 1], n_basis=4, key=key)
        trainer = CVKANTrainer(model, optimizer=optax.adam(1e-3))

        x = np.random.randn(20, 2).astype(np.float32) + 1j * np.random.randn(20, 2).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32) + 1j * np.random.randn(20, 1).astype(np.float32)

        history = trainer.fit(x, y, epochs=5, batch_size=10, verbose=False)
        assert len(history["train_loss"]) == 5
        assert all(np.isfinite(history["train_loss"]))

    def test_predict_shape(self):
        import numpy as np
        key = jax.random.PRNGKey(2)
        model = CVKAN(layer_sizes=[3, 4, 2], n_basis=4, key=key)
        trainer = CVKANTrainer(model)

        x = np.ones((10, 3), dtype=np.complex64)
        out = trainer.predict(x)
        assert out.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
