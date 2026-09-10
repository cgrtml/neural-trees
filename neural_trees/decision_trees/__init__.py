"""
Decision tree implementations.

`SoftDecisionTree` needs PyTorch and is imported on first use, so this package
can be imported without it. The multivariate and omnivariate trees, and the
hard export, are pure numpy.
"""

from typing import TYPE_CHECKING, Any

from .hard_tree import HardDecisionTree
from .multivariate_tree import MultivariateDecisionTree
from .omnivariate_tree import OmnivariateDecisionTree

if TYPE_CHECKING:
    from .soft_decision_tree import SoftDecisionTree


def __getattr__(name: str) -> Any:
    if name == "SoftDecisionTree":
        from .soft_decision_tree import SoftDecisionTree

        globals()[name] = SoftDecisionTree
        return SoftDecisionTree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HardDecisionTree",
    "MultivariateDecisionTree",
    "OmnivariateDecisionTree",
    "SoftDecisionTree",
]
