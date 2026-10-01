"""
One line of usage numbers for neural-trees, appended to a local log.

Run it on a schedule or by hand; compare lines two weeks apart. It reads:
GitHub stars and the 14-day traffic (needs `gh auth` with push rights),
PyPI downloads for the last 30 days without mirrors (pypistats.org), and
the playground's HTTP status. Streamlit's viewer count has no API; type
it in with --viewers after reading it from the app's Analytics panel.

    python scripts/usage_snapshot.py --viewers 16
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import date
from pathlib import Path

LOG = Path.home() / "neural-trees-usage.jsonl"


def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else {}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "neural-trees-usage"}), timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewers", type=int, default=None, help="Streamlit 'unique viewers' read from the Analytics panel")
    args = ap.parse_args()
    repo = gh("repos/cgrtml/neural-trees")
    views = gh("repos/cgrtml/neural-trees/traffic/views")
    clones = gh("repos/cgrtml/neural-trees/traffic/clones")
    pypi, pypi_error = {}, None
    for attempt in range(3):            # pypistats rate-limits aggressively; wait and retry
        try:
            pypi = get("https://pypistats.org/api/packages/neural-trees/recent?mirrors=false")["data"]
            break
        except Exception as e:  # noqa: BLE001
            pypi_error = str(e)[:80]
            time.sleep(20)
    # a sleeping Streamlit app answers 303 to its wake-up page, a live one 200; both mean it exists
    app = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "https://neural-trees.streamlit.app"],
                         capture_output=True, text=True).stdout or None
    row = {
        "date": date.today().isoformat(),
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "open_issues": repo.get("open_issues_count"),
        "views_14d": views.get("count"), "unique_visitors_14d": views.get("uniques"),
        "clones_14d": clones.get("count"), "unique_cloners_14d": clones.get("uniques"),
        "pypi_last_week": pypi.get("last_week"), "pypi_last_month": pypi.get("last_month"),
        "pypi_error": pypi_error if not pypi else None,
        "app_http": app, "streamlit_unique_viewers": args.viewers,
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    print(f"appended to {LOG}")


if __name__ == "__main__":
    main()
