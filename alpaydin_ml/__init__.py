"""
alpaydin-ml: Implementations of algorithms from Prof. Dr. Ethem Alpaydın's
research papers and textbook "Introduction to Machine Learning" (MIT Press).

Reference:
    Alpaydın, E. (2020). Introduction to Machine Learning (4th ed.). MIT Press.
    https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/
"""

__version__ = "0.1.0"
__author__ = "Community Contributors"
__paper_author__ = "Prof. Dr. Ethem Alpaydın"

from alpaydin_ml.decision_trees.soft_decision_tree import SoftDecisionTree
from alpaydin_ml.decision_trees.omnivariate_tree import OmnivariateDecisionTree
from alpaydin_ml.statistical_tests.classifier_comparison import (
    combined_5x2cv_f_test,
    mcnemar_test,
    paired_t_test,
)
from alpaydin_ml.mixture_of_experts.hierarchical_moe import HierarchicalMixtureOfExperts

__all__ = [
    "SoftDecisionTree",
    "OmnivariateDecisionTree",
    "HierarchicalMixtureOfExperts",
    "combined_5x2cv_f_test",
    "mcnemar_test",
    "paired_t_test",
]
