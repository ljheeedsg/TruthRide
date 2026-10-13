"""Run the four Beijing parameter-sensitivity experiments and draw their figures.

All generated bundles, figures, and CSV summaries are written to
results/beijing/sensitivity. No E1/main-experiment results are overwritten.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results" / "beijing" / "sensitivity"
WORKER_OPTIONS = PROJECT_DIR / "data" / "beijing" / "experiment2_worker_options.json"
GROUPS = ("malicious", "validation_grid", "membership_fee", "sunk_threshold")


def preserve_previous_outputs() -> Path | None:
    if not OUTPUT_DIR.exists():
        return None
    previous = [
        path for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name.startswith("experiment2_")
    ]
    if not previous:
        return None

    backup_dir = OUTPUT_DIR / "backup_before_parameter_sensitivity_rerun"
    backup_dir.mkdir(exist_ok=True)
    for source in previous:
        destination = backup_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    return backup_dir


def load_experiment_modules():
    os.environ["TRUTHRIDE_DATASET"] = "beijing"
    sys.path.insert(0, str(PROJECT_DIR))
    os.chdir(OUTPUT_DIR)

    config = importlib.import_module("experiment2_重构_配置")
    importlib.reload(config)
    config.BASE_CONFIG["WORKER_OPTIONS_FILE"] = str(WORKER_OPTIONS)

    main_module = importlib.import_module("experiment2_重构_main")
    importlib.reload(main_module)

    rmse_utils = importlib.import_module("experiment2_rmse_plot_utils")
    rmse_utils.WORKER_OPTIONS_FILE = str(WORKER_OPTIONS)

    modules = {
        "malicious": importlib.import_module("experiment2_恶意比例鲁棒性_common"),
        "validation_grid": importlib.import_module("experiment2_验证网格数敏感性_common"),
        "membership_fee": importlib.import_module("experiment2_验证会员费F_m_common"),
        "sunk_threshold": importlib.import_module("experiment2_沉没阈值敏感性_common"),
    }

    # These three experiments use the existing Beijing worker population to
    # identify the seed trusted workers. Keep that file read-only and explicit.
    for name in ("validation_grid", "membership_fee", "sunk_threshold"):
        modules[name].WORKER_OPTIONS_FILE = str(WORKER_OPTIONS)

    # The malicious-ratio experiment invokes the data-generation scripts. Resolve
    # those script paths from the project even though outputs go to sensitivity/.
    malicious = modules["malicious"]
    def run_command_from_project(args):
        command = list(args)
        if len(command) > 1:
            script_path = Path(command[1])
            if not script_path.is_absolute():
                command[1] = str(PROJECT_DIR / script_path)
        print("\n>>>", " ".join(str(arg) for arg in command), flush=True)
        return subprocess.run([str(arg) for arg in command], cwd=PROJECT_DIR, check=True)

    malicious.run_command = run_command_from_project
    return modules


def run_group(name: str, module, runs: int) -> None:
    print(f"\n===== Beijing sensitivity: {name} ({runs} runs/value) =====", flush=True)
    if name == "malicious":
        bundle = module.run_full_experiment(runs_per_ratio=runs)
    else:
        bundle = module.run_full_experiment(runs_per_value=runs)
    module.plot_all_figures(bundle)
    print(f"[{name}] bundle saved: {module.RESULTS_BUNDLE_FILE}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and plot the four Beijing parameter-sensitivity experiments."
    )
    parser.add_argument(
        "--groups", nargs="+", choices=GROUPS, default=list(GROUPS),
        help="Sensitivity groups to run (default: all four).",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Runs per parameter value (default: 10; matches the existing settings).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List groups without running experiments.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.dry_run:
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Runs per parameter value: {args.runs}")
        print("Groups: " + ", ".join(args.groups))
        return

    if not WORKER_OPTIONS.is_file():
        raise FileNotFoundError(f"Beijing worker options not found: {WORKER_OPTIONS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = preserve_previous_outputs()
    if backup_dir:
        print(f"Previous sensitivity outputs copied to: {backup_dir}", flush=True)

    modules = load_experiment_modules()
    for name in args.groups:
        run_group(name, modules[name], args.runs)

    image_count = sum(
        1 for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}
    )
    print(f"\nDone. PNG/PDF files in {OUTPUT_DIR}: {image_count}", flush=True)


if __name__ == "__main__":
    main()
