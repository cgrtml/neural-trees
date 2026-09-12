"""
neural-trees: soft decision trees, hierarchical mixtures of experts,
constructive networks and classifier comparison tests, with a PyTorch backend.

The implementations start from published algorithms and depart from them where
this library makes its own design choices; see the Citation section of the
README for both.

PyTorch is imported lazily. Importing this package, and using the estimators
that are pure numpy, works in an environment without torch installed; the
torch-backed estimators raise an actionable ImportError when they are first
touched rather than at import time. That keeps the numpy-only half usable
where torch cannot go, a browser via Pyodide being the case that prompted it.
"""

from typing import TYPE_CHECKING, Any

from neural_trees.classical.k_nearest_neighbors import WeightedKNN
from neural_trees.classical.naive_bayes import NaiveBayesClassifier
from neural_trees.decision_trees.hard_tree import HardDecisionTree
from neural_trees.decision_trees.multivariate_tree import MultivariateDecisionTree
from neural_trees.decision_trees.omnivariate_tree import OmnivariateDecisionTree
from neural_trees.mixture_of_experts.hard_router import HardRoutedExperts
from neural_trees.statistical_tests.classifier_comparison import (
    combined_5x2cv_f_test,
    mcnemar_test,
    paired_t_test,
)

__version__ = "0.6.1"
__author__ = "Cagri Temel"

# name -> module that defines it, for the estimators that need torch
_LAZY = {
    "GALNetwork": "neural_trees.classical.multilayer_perceptron",
    "HierarchicalMixtureOfExperts": "neural_trees.mixture_of_experts.hierarchical_moe",
    "SoftDecisionTree": "neural_trees.decision_trees.soft_decision_tree",
}

if TYPE_CHECKING:  # so type checkers and IDEs still see them
    from neural_trees.classical.multilayer_perceptron import GALNetwork
    from neural_trees.decision_trees.soft_decision_tree import SoftDecisionTree
    from neural_trees.mixture_of_experts.hierarchical_moe import (
        HierarchicalMixtureOfExperts,
    )


def __getattr__(name: str) -> Any:
    """Import a torch-backed estimator the first time it is asked for."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    attribute = getattr(importlib.import_module(module_path), name)
    globals()[name] = attribute  # cache, so this runs once
    return attribute


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "GALNetwork",
    "HardDecisionTree",
    "HardRoutedExperts",
    "HierarchicalMixtureOfExperts",
    "MultivariateDecisionTree",
    "NaiveBayesClassifier",
    "OmnivariateDecisionTree",
    "SoftDecisionTree",
    "WeightedKNN",
    "combined_5x2cv_f_test",
    "mcnemar_test",
    "paired_t_test",
]
