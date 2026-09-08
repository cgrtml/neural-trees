"""
The Streamlit playground is not importable in a test (it runs Streamlit calls
at module level), but it can still be checked against the library API.

What breaks in practice is not the UI: it is `app.py` quietly falling out of
step with the estimators, because nothing here exercises it. These tests parse
the file and verify that every neural-trees symbol it names, and every keyword
it passes to one of our estimators, still exists.
"""
import ast
import inspect
from pathlib import Path

import pytest

import neural_trees

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
OUR_ESTIMATORS = {
    name for name in neural_trees.__all__ if isinstance(getattr(neural_trees, name), type)
}


@pytest.fixture(scope="module")
def app_tree():
    if not APP_PATH.exists():
        pytest.skip("app.py is not present")
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_app_parses(app_tree):
    assert app_tree.body


def test_imported_symbols_still_exist(app_tree):
    imported = []
    for node in ast.walk(app_tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("neural_trees"):
            module = __import__(node.module, fromlist=["_"])
            for alias in node.names:
                imported.append((node.module, alias.name, hasattr(module, alias.name)))

    assert imported, "app.py no longer imports anything from neural_trees"
    missing = [f"{mod}.{name}" for mod, name, ok in imported if not ok]
    assert not missing, f"app.py imports symbols that no longer exist: {missing}"


def test_constructor_keywords_are_real_parameters(app_tree):
    """A renamed or removed parameter would raise only when a user clicks that model."""
    problems = []
    for node in ast.walk(app_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in OUR_ESTIMATORS:
            continue
        estimator = getattr(neural_trees, node.func.id)
        valid = set(inspect.signature(estimator.__init__).parameters) - {"self"}
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in valid:
                problems.append(f"{node.func.id}(..., {keyword.arg}=...)")

    assert not problems, f"app.py passes parameters the estimators no longer accept: {problems}"
