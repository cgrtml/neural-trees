"""The import error must name PyTorch instead of a bare ModuleNotFoundError."""
import importlib
import sys

import pytest

TORCH_BACKED_MODULES = [
    "neural_trees.decision_trees.soft_decision_tree",
    "neural_trees.mixture_of_experts.hierarchical_moe",
    "neural_trees.classical.multilayer_perceptron",
]


@pytest.mark.parametrize("module_name", TORCH_BACKED_MODULES)
def test_missing_torch_raises_actionable_import_error(module_name, monkeypatch):
    # Setting the entry to None makes `import torch` fail the way it would on a
    # machine without PyTorch installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(ImportError, match="pip install torch"):
        importlib.import_module(module_name)

    # Leave a clean module table for the rest of the session.
    sys.modules.pop(module_name, None)


NUMPY_ONLY = [
    "HardDecisionTree",
    "HardRoutedExperts",
    "MultivariateDecisionTree",
    "NaiveBayesClassifier",
    "OmnivariateDecisionTree",
    "WeightedKNN",
    "combined_5x2cv_f_test",
    "mcnemar_test",
    "paired_t_test",
]
TORCH_BACKED = [
    "GALNetwork",
    "HierarchicalMixtureOfExperts",
    "SoftDecisionTree",
]


@pytest.fixture
def without_torch(monkeypatch):
    """Import the package fresh, in an interpreter where torch is unusable."""
    import importlib

    for name in [m for m in sys.modules if m.startswith("neural_trees")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch.utils.data"):
        monkeypatch.setitem(sys.modules, name, None)

    package = importlib.import_module("neural_trees")
    yield package
    for name in [m for m in sys.modules if m.startswith("neural_trees")]:
        sys.modules.pop(name, None)


def test_package_imports_without_torch(without_torch):
    """
    Nothing in the numpy half needs PyTorch, and it should not be dragged in by
    the package import. A browser running Pyodide is the case that prompted
    this: torch cannot go there, the rest can.
    """
    assert without_torch.__version__


@pytest.mark.parametrize("name", NUMPY_ONLY)
def test_numpy_estimators_are_usable_without_torch(without_torch, name):
    assert hasattr(without_torch, name)


def test_a_numpy_estimator_actually_fits_without_torch(without_torch):
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    knn = without_torch.WeightedKNN().fit(X, y)
    assert knn.score(X, y) > 0.9


@pytest.mark.parametrize("name", TORCH_BACKED)
def test_torch_backed_names_fail_only_when_touched(without_torch, name):
    """Import succeeds; the error arrives when the estimator is asked for."""
    with pytest.raises(ImportError, match="pip install torch"):
        getattr(without_torch, name)


def test_unknown_attribute_still_raises_attribute_error(without_torch):
    with pytest.raises(AttributeError, match="has no attribute"):
        without_torch.NoSuchEstimator


def test_dir_lists_the_lazy_names_too(without_torch):
    listed = dir(without_torch)
    for name in TORCH_BACKED + NUMPY_ONLY:
        assert name in listed


@pytest.mark.parametrize("name", TORCH_BACKED + NUMPY_ONLY)
def test_everything_resolves_when_torch_is_present(name):
    import neural_trees

    assert getattr(neural_trees, name) is not None
