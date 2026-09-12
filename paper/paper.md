---
title: 'neural-trees: soft decision trees, mixtures of experts and classifier comparison tests for Python'
tags:
  - Python
  - machine learning
  - decision trees
  - interpretability
  - scikit-learn
  - PyTorch
authors:
  - name: Cagri Temel
    orcid: 0009-0003-3359-6939
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 11 September 2026
bibliography: paper.bib
---

# Summary

`neural-trees` is a scikit-learn compatible library of differentiable and
constructive classifiers: soft decision trees, multivariate and omnivariate
trees, hierarchical mixtures of experts, a grow-and-prune network, and the
statistical tests used to decide whether one classifier is genuinely better
than another. Models train with a PyTorch backend and behave as ordinary
estimators, so they compose with `Pipeline`, `GridSearchCV` and the rest of the
scikit-learn ecosystem.

The library is aimed at two uses. The first is teaching and research on tree
structured models, where the interesting object is the split itself: a soft
tree replaces a hard threshold with a sigmoid gate, which makes the boundary
smooth and the whole tree trainable by gradient descent. The second is honest
model comparison. The package ships Alpaydın's combined 5x2cv F test alongside
McNemar's test and the paired t-test, because an accuracy difference on one
split says little until you know how much that number moves when only the fold
assignment changes.

# Statement of need

Soft decision trees, omnivariate trees and hierarchical mixtures of experts are
standard material in machine learning courses and are cited regularly, but
working implementations with a familiar API are scarce. A practitioner who
wants to compare a soft tree against CART on their own data typically has to
reimplement the model from the paper. The reimplementation is then unverified,
which is the problem this library ran into itself.

An earlier release of this package shipped four models that did not work, and
none of them announced it:

- `HierarchicalMixtureOfExperts` could not take a single gradient step. Child
  weights were written in place into one preallocated tensor, which autograd
  had saved for the multiplication backward, so every call to `backward()`
  raised.
- `OmnivariateDecisionTree` computed its node classifier's decision, discarded
  it, and always descended into the right child. Every sample reached the same
  leaf. Five-fold accuracy at depth three was 0.000 on Iris.
- `GALNetwork` took one full-batch gradient step per epoch, so growth and
  pruning decisions were made on a network that had barely moved from its
  initialization. It scored 0.333 on three well separated blobs, which is
  chance for three classes.
- Condensed nearest neighbour made a single pass over the training set, leaving
  a prototype store that did not classify that set correctly.

All four were found by writing tests, not by reading the code. The library had
21 tests when the work began and one model under test; it has 283 now, and
every classifier is checked against `sklearn.utils.estimator_checks`. Four of
the seven pass all 55 checks. The three that accept `sample_weight` are held to
63 checks instead and pass 62, failing only the one that requires weighting a
sample to be bit-identical to repeating it, which no stochastic mini-batch
learner can satisfy.

# State of the field

`scikit-learn` provides CART and ensembles of it, but nothing differentiable.
Deep learning frameworks provide the autograd machinery but no estimator API
and no tree models. Research code accompanying the original papers, where it
exists, is generally unmaintained and does not follow the estimator contract.
`neural-trees` occupies the gap: the models of the tree literature, behind the
interface practitioners already use.

# Software design

The implementations begin from published algorithms and depart from them where
measurement justified it. Each deviation is documented in the module docstring
with the numbers behind it. Three are worth recording here.

**Units fitted to the residual.** `GALNetwork` added hidden units with random
weights, so a new unit perturbed every logit on arrival and then had to be
trained from noise. Fitting the unit to the residual error of the frozen
network first, in the manner of cascade-correlation [@fahlman1990cascade], and
installing it with zero outgoing weights, makes growth non-destructive. Over
three seeds of five-fold cross-validation on Iris this moved accuracy from
0.938 to 0.956 while reducing the network from 17.4 hidden units to 6.6.

**Growth decided per leaf.** `SoftDecisionTree` can grow one leaf at a time,
splitting the leaf carrying the most expected error, which is the rule of
İrsoy, Yıldız and Alpaydın [@irsoy2012soft]. On a synthetic problem with 800
samples and 20 features it reached 0.885 using 3.7 splits, against 0.832 for a
fixed depth-6 tree using 63.

**A deepening that must not preserve the function exactly.** Deepening a soft
tree by splitting every leaf into two children that inherit the parent's class
distribution leaves the mixture unchanged, which looks like the safe way to
grow. It is not: with identical children the mixture does not depend on the new
gate at all, so the gate's gradient is exactly zero and the children receive
identical gradients forever. The level is dead weight. Growing that way reached
0.753 on Iris and 0.754 on Wine, against 0.958 and 0.977 for trees of the same
depth trained from scratch.
A small perturbation of the new leaf distributions breaks the symmetry, and a
regression test asserts the zero gradient exists without it.

Trained soft models export to plain numpy. `to_hard_tree()` reads each gate as
a hard decision and prints the resulting rules; on Wine the export agrees with
the model it came from on 99.8% of held-out samples on average, with 97.1% in
the worst fold, and predicts about four times faster. The export reports its
agreement rather than assuming it, because a mixture over leaves is not a
single path.

# Research impact statement

The library makes two things routine that are otherwise laborious. Comparing a
differentiable tree against CART on a new dataset becomes a `Pipeline` and a
`cross_val_score`. Deciding whether the resulting difference means anything
becomes a call to `combined_5x2cv_f_test` [@alpaydin1999combined] rather than a
judgement about two numbers. The benchmark table in the repository is generated
by a script in the repository, so its claims can be re-run rather than trusted.
The version measured for every figure quoted here is archived at
[10.5281/zenodo.22718898](https://doi.org/10.5281/zenodo.22718898).

A secondary contribution is negative and measured. Several design choices that
sound obviously correct are shown not to be: exactly function-preserving
deepening prevents learning, node-level statistical significance does not
compose into tree-level accuracy, and carrying optimizer momentum across
architecture changes, which a plausible hypothesis blamed for a growth policy's
failure, changes nothing. Each is documented with the measurement that settled
it.

# AI usage disclosure

Substantial portions of the 0.2.0 through 0.6.0 development cycle, including
the diagnosis of the four defects described above, the test suite and much of
the prose in this paper, were produced with the assistance of a large language
model used as a pair programmer. Every empirical claim in this paper and in the
repository documentation was produced by running the code and recording the
output; no number here was reported without being measured. The author reviewed
and is responsible for all of it.

# Acknowledgements

The algorithms implemented here come from the work of Ethem Alpaydın, Olcay
Taner Yıldız and Oğuzhan İrsoy. Thanks to the external contributors whose pull
requests are part of these releases: @snoopuppy582, @aribaskagan and
@yunaremaia.

# References
