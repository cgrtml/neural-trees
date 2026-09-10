"""Tests for how the torch-backed estimators pick a device."""
import numpy as np
import pytest
import torch
from sklearn.datasets import load_iris

from neural_trees import GALNetwork, HierarchicalMixtureOfExperts, SoftDecisionTree
from neural_trees._validation import resolve_device

ESTIMATORS = [
    lambda **kw: SoftDecisionTree(depth=2, max_epochs=3, random_state=0, **kw),
    lambda **kw: HierarchicalMixtureOfExperts(depth=1, max_epochs=3, random_state=0, **kw),
    lambda **kw: GALNetwork(max_epochs=5, random_state=0, **kw),
]
IDS = ["SoftDecisionTree", "HierarchicalMixtureOfExperts", "GALNetwork"]


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


def test_auto_resolves_to_something_torch_accepts():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}
    # Whatever it picked has to actually be usable.
    torch.zeros(2, device=device)


def test_auto_prefers_an_accelerator_when_one_exists():
    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_explicit_device_strings_pass_through():
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.parametrize("bad", ["bogus", "cpu:not-a-number", 3.5])
def test_unusable_device_fails_with_the_parameter_named(bad):
    with pytest.raises(ValueError, match="device must be"):
        resolve_device(bad)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_fit_records_the_resolved_device(make_estimator, iris):
    X, y = iris
    estimator = make_estimator(device="auto").fit(X, y)

    assert isinstance(estimator.device_, torch.device)
    assert estimator.device_ == resolve_device("auto")
    assert estimator.predict(X).shape == y.shape


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_prediction_does_not_re_resolve_the_device(make_estimator, iris):
    """
    Predict must use what fit resolved, not re-read the parameter. Re-resolving
    would send a model trained on an accelerator through a CPU forward pass, or
    fail outright if the machine's accelerator appeared or vanished in between.
    """
    X, y = iris
    estimator = make_estimator(device="auto").fit(X, y)
    estimator.device = "definitely-not-a-device"

    proba = estimator.predict_proba(X)
    assert proba.shape[0] == len(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_moe_predicts_on_cpu_whatever_device_it_trained_on(iris):
    """
    The float64 copy that makes predictions order-independent is pinned to the
    CPU: MPS has no float64 at all, and CUDA's is slow.
    """
    X, y = iris
    moe = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=3, random_state=0, device="auto"
    ).fit(X, y)

    assert next(moe.model_double_.parameters()).device.type == "cpu"
    assert next(moe.model_double_.parameters()).dtype == torch.float64
    np.testing.assert_allclose(
        moe.predict_proba(X[:20]), moe.predict_proba(X[:20][::-1])[::-1], rtol=1e-9
    )


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_bad_device_is_rejected_at_fit(make_estimator, iris):
    X, y = iris
    with pytest.raises(ValueError, match="device must be"):
        make_estimator(device="bogus").fit(X, y)
