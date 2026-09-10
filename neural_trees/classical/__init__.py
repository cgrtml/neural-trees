"""
Classical algorithms from "Introduction to Machine Learning" (Alpaydın, 2020).
Clean, well-documented implementations for educational purposes.

`GALNetwork` needs PyTorch and is imported on first use, so this package can be
imported without it.
"""

from typing import TYPE_CHECKING, Any

from .k_nearest_neighbors import WeightedKNN
from .naive_bayes import NaiveBayesClassifier

if TYPE_CHECKING:
    from .multilayer_perceptron import GALNetwork


def __getattr__(name: str) -> Any:
    if name == "GALNetwork":
        from .multilayer_perceptron import GALNetwork

        globals()[name] = GALNetwork
        return GALNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GALNetwork", "NaiveBayesClassifier", "WeightedKNN"]
