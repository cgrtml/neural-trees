"""
neural-trees: Implementations of algorithms from Prof. Dr. Ethem Alpaydın's
research papers and textbook "Introduction to Machine Learning" (MIT Press).

Reference:
    Alpaydın, E. (2020). Introduction to Machine Learning (4th ed.). MIT Press.
"""

__version__ = "0.4.0"
__author__ = "Cagri Temel"

from neural_trees.classical.k_nearest_neighbors import WeightedKNN
from neural_trees.classical.multilayer_perceptron import GALNetwork
from neural_trees.classical.naive_bayes import NaiveBayesClassifier
from neural_trees.decision_trees.multivariate_tree import MultivariateDecisionTree
from neural_trees.decision_trees.omnivariate_tree import OmnivariateDecisionTree
from neural_trees.decision_trees.soft_decision_tree import SoftDecisionTree
from neural_trees.mixture_of_experts.hierarchical_moe import (
    HierarchicalMixtureOfExperts,
)
from neural_trees.statistical_tests.classifier_comparison import (
    combined_5x2cv_f_test,
    mcnemar_test,
    paired_t_test,
)

__all__ = [
    "GALNetwork",
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
