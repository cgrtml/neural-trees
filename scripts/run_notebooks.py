"""
Execute every notebook and fail on the first cell error.

The notebooks are committed with their outputs, which is what makes them
readable on GitHub, and also what makes them go stale silently: nothing catches
it when a model change turns a printed number into a wrong one. Running them in
CI turns that into a build failure.

Notebooks are executed in memory. Outputs are not written back, so CI never
produces a diff.

Run with:
    python scripts/run_notebooks.py
"""
import sys
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
TIMEOUT_SECONDS = 1800


def main() -> int:
    warnings.filterwarnings("ignore")
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {NOTEBOOK_DIR}")
        return 1

    failures = []
    for path in notebooks:
        print(f"Executing {path.name} ... ", end="", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=TIMEOUT_SECONDS,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        )
        try:
            client.execute()
        except Exception as exc:
            print("FAILED")
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
        else:
            print("ok")

    for name, message in failures:
        print(f"\n--- {name} ---\n{message}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
