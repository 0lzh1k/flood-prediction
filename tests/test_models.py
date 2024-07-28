import numpy as np
import torch

from models.model import train_gbm, UNet


def test_train_gbm_predict_proba():
    # Two-feature dataset to mirror app usage
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = train_gbm(X, y)

    X_test = rng.normal(size=(10, 2))
    proba = model.predict_proba(X_test)
    assert proba.shape == (10, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    # Ensure probabilities sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_unet_forward_and_gradients():
    model = UNet()
    x = torch.randn(1, 1, 64, 64)
    y = torch.rand(1, 1, 64, 64)

    criterion = torch.nn.BCEWithLogitsLoss()
    out = model(x)
    assert out.shape == x.shape
    loss = criterion(out, y)
    loss.backward()

    # Check a couple of parameters received gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
