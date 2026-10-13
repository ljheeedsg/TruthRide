"""Generate the existing per-dataset figure set from saved experiment results.

This script only runs plotting programs. It does not run experiments or edit
result JSON files. Figures already present in a dataset result directory are
copied to a one-time backup folder before being regenerated.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = ("beijing", "rome", "chicago")

PLOT_SCRIPTS = (
    ("experiment2_Fig. 1：Quality-Value Function.py", ()),
    ("experiment2_Fig. 2：每轮平均数据质量.py", ()),
    ("experiment2_Fig. 3：累计平均数据质量.py", ()),
    ("experiment2_Fig. 4：累计 RMSE.py", ("--dataset",)),
    ("experiment2_Fig. 5：每轮RMSE.py", ("--dataset",)),
    ("experiment2_Fig. 8：三类工人数量变化趋势.py", ()),
    ("experiment2_Fig. 10：Cumulative Platform Utility.py", ()),
    ("experiment2_Fig. 11：Per-Round Platform Utility.py", ()),
    ("experiment2_Fig. 14：Cost-Utility Bubble Plot.py", ()),
    ("experiment2_Fig. 15：Seed Trusted Worker Retention Rate.py", ()),
    ("experiment2_Fig. 16：Trusted Worker Pool Size.py", ()),
    ("experiment2_Fig. 17：Active Worker Pool Size.py", ()),
    ("experiment2_Fig. 18：Worker Participation Length.py", ()),
    ("experiment2_Fig. 19补充：最终留存与可信池规模.py", ()),
    ("experiment2_Fig. 33-34：任务对比.py", ()),
)

AGGREGATE_INPUTS = (
    "experiment2_random_longrun_round_results.json",
    "experiment2_epsilon_first_longrun_round_results.json",
    "experiment2_cmab_longrun_round_results.json",
    "experiment2_cmab_trust_round_results.json",
    "experiment2_cmab_trust_pgrd_round_results.json",
    "experiment2_cmab_trust_pgrd_lgsc_round_results.json",
)

ALL_RUN_INPUTS = (
    "experiment2_random_longrun_round_results_all_runs.json",
    "experiment2_epsilon_first_longrun_round_results_all_runs.json",
    "experiment2_cmab_longrun_round_results_all_runs.json",
    "experiment2_cmab_trust_round_results_all_runs.json",
    "experiment2_cmab_trust_pgrd_round_results_all_runs.json",
    "experiment2_cmab_trust_pgrd_lgsc_round_results_all_runs.json",
)


def preflight(dataset: str) -> list[Path]:
    results_dir = PROJECT_DIR / "results" / dataset
    worker_options = PROJECT_DIR / "data" / dataset / "experiment2_worker_options.json"
    required = [results_dir / name for name in AGGREGATE_INPUTS]
    required.extend(results_dir / name for name in ALL_RUN_INPUTS)
    required.append(worker_options)
    required.extend(PROJECT_DIR / script for script, _ in PLOT_SCRIPTS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset}: missing required plotting inputs/scripts:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return required


def backup_existing_figures(results_dir: Path) -> Path | None:
    existing = [
        path
        for path in results_dir.iterdir()
        if path.is_file()
        and path.name.startswith("experiment2_Fig")
        and path.suffix.lower() in {".png", ".pdf", ".csv"}
    ]
    if not existing:
        return None

    backup_dir = results_dir / "backup_before_batch_figures"
    backup_dir.mkdir(exist_ok=True)
    for source in existing:
        destination = backup_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    return backup_dir


def run_dataset(dataset: str, dry_run: bool = False) -> None:
    preflight(dataset)
    results_dir = PROJECT_DIR / "results" / dataset
    backup_dir = None if dry_run else backup_existing_figures(results_dir)
    if backup_dir:
        print(f"[{dataset}] Existing figure files copied to: {backup_dir}")

    env = os.environ.copy()
    env["TRUTHRIDE_DATASET"] = dataset
    print(f"[{dataset}] Plot output directory: {results_dir}")
    for script_name, extra_args in PLOT_SCRIPTS:
        command = [sys.executable, str(PROJECT_DIR / script_name)]
        command.extend(extra_args)
        if extra_args:
            command.append(dataset)
        print(f"[{dataset}] {' '.join(command)}")
        if not dry_run:
            subprocess.run(command, cwd=results_dir, env=env, check=True)

    if not dry_run:
        image_files = [
            path for path in results_dir.iterdir()
            if path.is_file() and path.name.startswith("experiment2_Fig")
            and path.suffix.lower() in {".png", ".pdf"}
        ]
        print(f"[{dataset}] Finished; found {len(image_files)} Fig image files (PNG/PDF).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the existing figure scripts from saved dataset results."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
        help="Datasets to plot (default: beijing rome chicago).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print commands without generating figures.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        run_dataset(dataset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
