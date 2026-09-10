# Contributing to neural-trees

Thank you for your interest in contributing to neural-trees. This guide is
written to be friendly to first-time contributors. If anything is unclear,
open an issue and ask.

## Quick start

1. **Fork** this repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/neural-trees.git
   cd neural-trees
   ```
3. **Install** in editable mode with the development extras:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your change:
   ```bash
   git checkout -b your-branch-name
   ```
5. **Make your change**, **commit**, and **push** to your fork:
   ```bash
   git add -A
   git commit -m "Short description of your change"
   git push origin your-branch-name
   ```
6. Open a **pull request** from your fork to `cgrtml/neural-trees:main`.

## What to work on

If this is your first contribution, look at the issues tagged
[`good first issue`](https://github.com/cgrtml/neural-trees/labels/good%20first%20issue).
These are intentionally small in scope (10 to 20 minutes of work) and do
not require deep knowledge of the codebase.

Common kinds of contributions:

- **Documentation.** Improve a docstring, fix a typo, add a usage example,
  or expand the README with a section that helped you understand the
  library.
- **Examples.** Add a runnable script in `examples/` that shows a specific
  use case (using neural-trees inside a scikit-learn `Pipeline`, comparing
  classifiers with the 5x2cv F-test, plotting a decision boundary).
- **Tests.** Add a unit test that covers an edge case, a reproducibility
  check, or a previously untested method.
- **Small features.** Add input validation, improve an error message, add
  type hints, support a newer Python version.

If you have a larger idea, open an issue first to discuss before writing
the code.

## Pull request checklist

Before opening your PR, please make sure:

- [ ] Your branch is based on the latest `main`.
- [ ] You have run the tests locally: `pytest tests/`.
- [ ] Coverage has not dropped below **95%**, which CI enforces:
      `pytest tests/ --cov=neural_trees --cov-fail-under=95`. It currently sits
      at 97%, so there is a little room, but new code should come with tests.
- [ ] `ruff check .` passes. CI pins `ruff==0.16.6`.
- [ ] `mypy` passes. The package ships a `py.typed` marker, so its annotations
      are what type checkers see in downstream code.
- [ ] If you changed a model, the notebooks still run:
      `python scripts/run_notebooks.py`. They are committed with their outputs,
      so a model change can silently turn a printed number into a wrong one.
- [ ] Code style is consistent with the existing files (PEP 8, four-space
      indentation, plain ASCII in source).
- [ ] Public functions and classes have docstrings.
- [ ] The PR title and description explain *what* changed and *why*.

## Style notes

- The codebase uses **plain ASCII** in source files. No em-dashes, no
  smart quotes. This avoids encoding issues in environments that ship
  with older locales.
- Public APIs follow the **scikit-learn estimator interface**: `fit`,
  `predict`, `predict_proba`, `score`. If your change touches an
  estimator, please keep this contract intact.
- The PyTorch backend is optional from the user's perspective. Imports
  should fail with a clear message if PyTorch is missing, not silently.

## How reviews work

I (Cagri) review every pull request myself. For small, well-scoped PRs
the typical turnaround is 24 to 48 hours. Larger PRs may take longer
and may go through a round or two of revision.

If your PR has been open for more than a week without a response, feel
free to tag me in a comment.

## Code of conduct

Be kind. Be specific. Assume good faith. If a comment would not pass
the test of being read aloud to a stranger, do not post it.

## Workshop and event contributions

If you are contributing as part of a workshop, course, or hackathon
(for example, the WSU Data and Analytics Breakout on May 15, 2026),
please mention the event in your PR description. Your name will be
added to the **Contributors** section of the README once your PR is
merged.

## Questions

Open an issue with the `question` label, or reach me at
[cagritemel@ieee.org](mailto:cagritemel@ieee.org).
