"""Tests for the Naive Bayes classifier and its three likelihoods."""
import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split

from neural_trees import NaiveBayesClassifier


@pytest.fixture(scope="module")
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


def test_gaussian_fit_predict(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    nb = NaiveBayesClassifier().fit(X_train, y_train)

    preds = nb.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset(set(np.unique(y_train)))
    assert nb.score(X_test, y_test) > 0.85


def test_bernoulli_likelihood_on_binarized_features(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    threshold = X_train.mean(axis=0)
    nb = NaiveBayesClassifier(likelihood="bernoulli").fit(
        (X_train > threshold).astype(float), y_train
    )

    score = nb.score((X_test > threshold).astype(float), y_test)
    majority = np.bincount(y_test).max() / len(y_test)
    assert score > majority


def test_multinomial_likelihood_on_count_features():
    X, y = load_digits(return_X_y=True)  # pixel intensities are counts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    nb = NaiveBayesClassifier(likelihood="multinomial").fit(X_train, y_train)
    assert nb.score(X_test, y_test) > 0.8


@pytest.mark.parametrize("likelihood", ["gaussian", "bernoulli", "multinomial"])
def test_every_likelihood_returns_a_proper_distribution(likelihood, iris_split):
    X_train, X_test, y_train, _ = iris_split
    if likelihood != "gaussian":
        threshold = X_train.mean(axis=0)
        X_train = (X_train > threshold).astype(float)
        X_test = (X_test > threshold).astype(float)

    nb = NaiveBayesClassifier(likelihood=likelihood).fit(X_train, y_train)
    proba = nb.predict_proba(X_test)
    log_proba = nb.predict_log_proba(X_test)

    assert proba.shape == (len(X_test), 3)
    assert (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.exp(log_proba), proba, atol=1e-12)


def test_class_priors_match_label_frequencies():
    X, y = load_iris(return_X_y=True)
    y = np.where(y == 2, 1, y)  # make the classes deliberately unbalanced
    nb = NaiveBayesClassifier().fit(X, y)

    expected = np.log(np.bincount(y) / len(y))
    np.testing.assert_allclose(nb.class_log_prior_, expected)


def test_zero_variance_feature_does_not_produce_nan():
    """A feature constant within a class would divide by zero without smoothing."""
    X, y = load_iris(return_X_y=True)
    X = np.column_stack([X, np.ones(len(X))])  # constant column

    nb = NaiveBayesClassifier().fit(X, y)
    proba = nb.predict_proba(X)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_unknown_likelihood_is_rejected_at_fit():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="likelihood must be"):
        NaiveBayesClassifier(likelihood="poisson").fit(X, y)
