"""
One-click reproduction pipeline.

For each dataset it runs:
  Step 1 (worker trajectories) -> Step 2 (tasks) -> Step 3 (worker options)
and writes everything into data/<dataset>/.

Usage:
  python run_pipeline.py --datasets beijing rome chicago
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ["beijing", "rome", "chicago"]


def run(cmd, env):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env=env)


def build_dataset(dataset, trusted_ratio=None, malicious_ratio=None, out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "data" / dataset
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TRUTHRIDE_DATASET=dataset, TRUTHRIDE_BASEMAP="0", MPLBACKEND="Agg")

    step1 = [
        sys.executable, "experiment2_第1步产生车数据.py",
        "--output-seg", out / "experiment2_vehicle.csv",
        "--output-plot", out / "experiment2_grid_partition.png",
        "--summary-file", out / "experiment2_vehicle_summary.json",
        "--all-runs-summary-file", out / "experiment2_vehicle_summary_all_runs.json",
        "--runs", 1,
    ]
    if trusted_ratio is not None:
        step1 += ["--trusted-ratio", trusted_ratio]
    if malicious_ratio is not None:
        step1 += ["--malicious-ratio", malicious_ratio]
    run(step1, env)

    run([
        sys.executable, "experiment2_第2步产生任务数据.py",
        "--vehicle-file", out / "experiment2_vehicle.csv",
        "--vehicle-summary-file", out / "experiment2_vehicle_summary.json",
        "--task-csv", out / "experiment2_tasks.csv",
        "--task-json", out / "experiment2_task_segments.json",
        "--plot-file", out / "experiment2_tasks_distribution.png",
        "--summary-file", out / "experiment2_tasks_summary.json",
        "--all-runs-summary-file", out / "experiment2_tasks_summary_all_runs.json",
        "--runs", 1,
    ], env)

    run([
        sys.executable, "experiment2_第3步产生工人可选项.py",
        "--vehicle-file", out / "experiment2_vehicle.csv",
        "--task-file", out / "experiment2_tasks.csv",
        "--task-summary-file", out / "experiment2_tasks_summary.json",
        "--output-json", out / "experiment2_worker_options.json",
        "--summary-file", out / "experiment2_worker_options_summary.json",
        "--all-runs-summary-file", out / "experiment2_worker_options_summary_all_runs.json",
        "--runs", 1,
    ], env)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        build_dataset(dataset)


if __name__ == "__main__":
    main()
