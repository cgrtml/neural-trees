"""
The Streamlit playground is not importable in a test (it runs Streamlit calls
at module level), but it can still be checked against the library API.

What breaks in practice is not the UI: it is the playground quietly falling
out of step with the estimators, because nothing here exercises it. These
tests parse `app.py`, every page in `views/` and the registry in
`playground/`, and verify that every neural-trees symbol they name, and every
keyword they pass to one of our estimators, still exists.
"""
import ast
import inspect
from pathlib import Path

import pytest

import neural_trees

ROOT = Path(__file__).resolve().parent.parent
APP_FILES = [ROOT / "app.py", *sorted((ROOT / "views").glob("*.py")), *sorted((ROOT / "playground").glob("*.py"))]
OUR_ESTIMATORS = {
    name for name in neural_trees.__all__ if isinstance(getattr(neural_trees, name), type)
}


@pytest.fixture(scope="module")
def app_tree():
    """One module whose body is every playground file's body, so a walk sees all of them."""
    files = [f for f in APP_FILES if f.exists()]
    if not files:
        pytest.skip("the playground is not present")
    body = []
    for f in files:
        body.extend(ast.parse(f.read_text(encoding="utf-8"), filename=str(f)).body)
    return ast.Module(body=body, type_ignores=[])


def test_app_parses(app_tree):
    assert app_tree.body
    assert len([f for f in APP_FILES if f.exists()]) >= 7


def test_imported_symbols_still_exist(app_tree):
    imported = []
    for node in ast.walk(app_tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("neural_trees"):
            module = __import__(node.module, fromlist=["_"])
            for alias in node.names:
                imported.append((node.module, alias.name, hasattr(module, alias.name)))

    assert imported, "the playground no longer imports anything from neural_trees"
    missing = [f"{mod}.{name}" for mod, name, ok in imported if not ok]
    assert not missing, f"the playground imports symbols that no longer exist: {missing}"


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

    assert not problems, f"the playground passes parameters the estimators no longer accept: {problems}"
