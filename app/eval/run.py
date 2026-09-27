"""Runnable evaluation entry point.

Usage:
    python -m app.eval.run

Runs the three-way comparison, prints the table, saves the chart under
app/eval/outputs/, and writes results.md at the repository root. Works with no
API cost in the default mock mode; set LLM_PROVIDER=groq for real LLM scoring.
"""

from __future__ import annotations

import os

from app.eval.harness import (
    build_table,
    run_evaluation,
    save_chart,
    write_results_md,
)

# Repository root = two levels up from this file (app/eval/run.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    results = run_evaluation()

    table = build_table(results)
    print("\n" + table + "\n")

    chart_path = save_chart(results)
    print(f"Chart saved to: {chart_path}")

    md_path = os.path.join(_REPO_ROOT, "results.md")
    write_results_md(results, table, chart_path, md_path)
    print(f"Results written to: {md_path}")


if __name__ == "__main__":
    main()
