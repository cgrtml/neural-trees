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
