"""
Explaining one prediction of a soft decision tree.

A soft tree's prediction is a mixture over leaves, so "the path the sample
took" is not literally defined: every sample reaches every leaf with some
probability. What can be said exactly is which leaf received most of the
sample's probability mass, how each gate on the way to that leaf leaned and
why, and which single-feature change would have produced a different class.
This module says exactly that and nothing more.

Three deliberate limits, so the output is not mistaken for something it is
not:

* The feature attribution is the size of each feature's term in the gates
  along the dominant path, weighted by how much of the sample reached each
  gate. It is a faithful reading of a linear gate. It is not SHAP and makes
  no game-theoretic claim.
* The counterfactual is a single-feature change that flips a gate on the
  dominant path, and it is reported only if re-predicting the changed sample
  actually changes the class. If no such change exists among the gates on the
  path, the field is None rather than a guess.
* Everything is in the model's input units. If the model was fitted on
  standardised features, so are the numbers here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class GateStep:
    """One gate on the dominant path."""

    node: int
    went: str                      # "left" or "right"
    probability: float             # probability the sample went the way it went
    terms: List[Dict[str, float]]  # largest |weight * value| terms, signed
    arrival: float                 # probability mass that reached this gate


@dataclass
class Counterfactual:
    feature: str
    index: int
    from_value: float
    to_value: float
    new_class: Any
    new_probability: float


@dataclass
class Explanation:
    predicted_class: Any
    probabilities: Dict[Any, float]
    leaf: int
    leaf_probability: float
    leaf_distribution: Dict[Any, float]
    path: List[GateStep]
    attributions: Dict[str, float]
    counterfactual: Optional[Counterfactual]
    feature_names: List[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("feature_names", None)
        return d

    def to_text(self, decimals: int = 3) -> str:
        f = f"{{:+.{decimals}f}}"
        lines = [
            f"predicted {self.predicted_class!r} with probability "
            f"{self.probabilities[self.predicted_class]:.{decimals}f}",
            f"dominant leaf {self.leaf} received {self.leaf_probability:.{decimals}f} "
            f"of the sample's mass; its distribution is "
            + ", ".join(f"{k!r}: {v:.{decimals}f}" for k, v in self.leaf_distribution.items()),
            "path:",
        ]
        for s in self.path:
            terms = " ".join(f"{f.format(t['contribution'])}[{t['feature']}]" for t in s.terms)
            lines.append(f"  gate {s.node}: went {s.went} with p={s.probability:.{decimals}f}"
                         f"  ({terms})")
        top = sorted(self.attributions.items(), key=lambda kv: -kv[1])[:5]
        lines.append("largest contributions: " + ", ".join(f"{k} {v:.{decimals}f}" for k, v in top))
        if self.counterfactual is None:
            lines.append("counterfactual: no single-feature change on this path flips the class")
        else:
            c = self.counterfactual
            lines.append(
                f"counterfactual: set {c.feature} from {c.from_value:.{decimals}f} to "
                f"{c.to_value:.{decimals}f} and the prediction becomes {c.new_class!r} "
                f"(p={c.new_probability:.{decimals}f})")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _parent(node: int) -> int:
    return (node - 1) // 2


def explain_soft_tree(
    model,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    max_terms: int = 3,
    counterfactual: bool = True,
) -> List[Explanation]:
    """
    Build an :class:`Explanation` for every row of `X`.

    `model` is a fitted :class:`~neural_trees.SoftDecisionTree`; `X` has
    already been validated and cast by the caller.
    """
    import torch
    import torch.nn.functional as F

    m = model.model_
    n_features = model.n_features_in_
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    feature_names = list(feature_names)
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries, expected {n_features}")
    classes = list(model.classes_)

    m.eval()
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=model.device_)
    with torch.no_grad():
        logits = m.gate_logits(X_t).cpu().numpy()                   # (n, n_internal)
        gate_p = 1.0 / (1.0 + np.exp(-logits))                       # P(right)
        log_mu_bottom, _, terminal = m._walk(X_t)
        node_dist = F.softmax(m.node_logits, dim=1).cpu().numpy()    # (n_nodes, K)
        weights = m.gates.weight.detach().cpu().numpy()              # (n_internal, p)
        biases = m.gates.bias.detach().cpu().numpy()
        # arrival probability of every acting leaf, per sample
        leaf_idx = np.concatenate([idx.cpu().numpy() for _, idx in terminal])
        leaf_mu = np.concatenate([lm.exp().cpu().numpy() for lm, _ in terminal], axis=1)
        # arrival probability at every internal node: product of gate choices
        # down from the root, computed from the path itself below
    proba_all = model.predict_proba(X)

    out: List[Explanation] = []
    for i in range(X_t.shape[0]):
        x = np.asarray(X[i], dtype=np.float64)
        proba = proba_all[i]
        pred = int(np.argmax(proba))
        best = int(np.argmax(leaf_mu[i]))
        leaf = int(leaf_idx[best])

        # dominant path: climb from the leaf to the root
        chain = []
        node = leaf
        while node > 0:
            par = _parent(node)
            chain.append((par, "left" if node == 2 * par + 1 else "right"))
            node = par
        chain.reverse()

        path: List[GateStep] = []
        attributions = np.zeros(n_features)
        arrival = 1.0
        for gnode, went in chain:
            p_right = float(gate_p[i, gnode])
            p_went = p_right if went == "right" else 1.0 - p_right
            contrib = weights[gnode] * x
            order = np.argsort(-np.abs(contrib))[:max_terms]
            terms = [{"feature": feature_names[j], "weight": float(weights[gnode, j]),
                      "value": float(x[j]), "contribution": float(contrib[j])} for j in order]
            path.append(GateStep(node=int(gnode), went=went, probability=p_went,
                                 terms=terms, arrival=arrival))
            attributions += arrival * np.abs(contrib)
            arrival *= p_went

        cf: Optional[Counterfactual] = None
        if counterfactual and chain:
            best_delta = np.inf
            for gnode, _ in chain:
                w, b = weights[gnode], float(biases[gnode])
                z = float(w @ x + b)
                for j in range(n_features):
                    if w[j] == 0.0:
                        continue
                    # move feature j just past the gate's zero crossing
                    new_val = x[j] - z / w[j] - np.sign(w[j]) * np.sign(z) * 1e-3
                    delta = abs(new_val - x[j])
                    if delta >= best_delta:
                        continue
                    x_cf = x.copy()
                    x_cf[j] = new_val
                    p_cf = model.predict_proba(x_cf.reshape(1, -1))[0]
                    k = int(np.argmax(p_cf))
                    if k != pred:
                        best_delta = delta
                        cf = Counterfactual(feature=feature_names[j], index=j,
                                            from_value=float(x[j]), to_value=float(new_val),
                                            new_class=classes[k], new_probability=float(p_cf[k]))

        out.append(Explanation(
            predicted_class=classes[pred],
            probabilities={classes[k]: float(proba[k]) for k in range(len(classes))},
            leaf=leaf,
            leaf_probability=float(leaf_mu[i, best]),
            leaf_distribution={classes[k]: float(node_dist[leaf, k]) for k in range(len(classes))},
            path=path,
            attributions={feature_names[j]: float(attributions[j]) for j in range(n_features)},
            counterfactual=cf,
            feature_names=feature_names,
        ))
    return out
