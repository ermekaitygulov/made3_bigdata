import pytest
import numpy as np
from linear_regression import NumpyLinReg as np_lr


@pytest.fixture
def test_data():
    weights = np.array([[1.5], [0.3], [-0.7]])
    X = np.random.normal(size=(1000, 3))
    y = X @ weights
    yield X, y


def test_linalg_core_init_weight_correct(test_data):
    X, y = test_data
    weights, bias = np_lr.init_weights(X)
    assert weights.shape[0] == X.shape[1]
    assert bias == 0
    assert np.allclose(weights, np.zeros_like(weights))


def test_linalg_core_compute_grads_toward_target(test_data):
    X, y = test_data

    weights_bias = np_lr.init_weights(X)
    grad_w_b = np_lr.compute_grad(X, y, *weights_bias)
    weights_bias = np_lr.sgd_step(*grad_w_b, *weights_bias, X.shape[0])
    first_pred = np_lr.predict(X, *weights_bias)

    grad_w_b = np_lr.compute_grad(X, y, *weights_bias)
    np_lr.sgd_step(*grad_w_b, *weights_bias, X.shape[0])
    second_pred = np_lr.predict(X, *weights_bias)

    first_dist = np.abs(first_pred - y).sum()
    second_dist = np.abs(second_pred - y).sum()
    assert first_dist > second_dist


def test_linalg_core_steps_scale_with_lambda(test_data):
    X, y = test_data

    weights_bias = np_lr.init_weights(X)
    big_grad_w, big_grad_b = np_lr.compute_grad(X, y, *weights_bias, lr=0.1)
    small_grad_w, small_grad_b = np_lr.compute_grad(X, y, *weights_bias, lr=0.01)
    assert np.linalg.norm(big_grad_w) > np.linalg.norm(small_grad_w)
    assert np.linalg.norm(big_grad_b) > np.linalg.norm(small_grad_b)

