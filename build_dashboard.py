#!/usr/bin/env python3
"""Rebuild the complete ranking-exploration dashboard from one source workbook."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "input" / "source_of_truth_merged.xlsx"
PREPARED = ROOT / "data" / "data_event_results" / "source_of_truth_jpar_input.csv"
CALCULATION = ROOT / "data" / "data_jpar_v2" / "source_of_truth_calculation_df.csv"
RANKING_DIR = ROOT / "data" / "ranking_diagnostics"
EXTERNAL_DIR = ROOT / "data" / "colleague_systems"
OUTPUT = ROOT / "output" / "interactive_rank_comparison.html"


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing sole source input: {SOURCE}")

    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    CALCULATION.parent.mkdir(parents=True, exist_ok=True)
    RANKING_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    run(
        "code/prepare_master_for_jpar.py",
        "--input", "input/source_of_truth_merged.xlsx",
        "--output", "data/data_event_results/source_of_truth_jpar_input.csv",
    )
    run("code/jpar_pipeline.py", "--config", "config/dashboard_config.json")
    run(
        "code/colleague_methods/build_outputs.py",
        "--input", "data/data_jpar_v2/source_of_truth_calculation_df.csv",
        "--output-dir", "data/colleague_systems",
    )
    run(
        "code/build_ranking_diagnostics_report.py",
        "--input", "data/data_jpar_v2/source_of_truth_calculation_df.csv",
        "--output-dir", "data/ranking_diagnostics",
    )
    run(
        "code/build_interactive_rank_table.py",
        "--input", "data/ranking_diagnostics/master_ranking_systems.csv",
        "--calculation-df", "data/data_jpar_v2/source_of_truth_calculation_df.csv",
        "--external-systems-dir", "data/colleague_systems",
        "--output", "output/interactive_rank_comparison.html",
    )

    print(f"\nDashboard rebuilt successfully:\n{OUTPUT}")


if __name__ == "__main__":
    main()
