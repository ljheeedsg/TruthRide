import csv
import json
import math
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment2_paper_plot_style import FIG2_SAVE_PAD
from experiment2_rmse_plot_utils import (
    build_worker_indexes,
    compute_task_error_records,
)


# ============================================================
# Fig. 4: Cumulative RMSE
#
# Statistics:
# - use true checkpoint cumulative RMSE values
# - x-axis checkpoints: 20, 40, 60, 80, 100, 120, 140, 144
# - y-value at checkpoint r:
#       cumulative RMSE at round r
# - if round r is missing, use the nearest previous available round
#
# Visualization:
# - grouped bars only
# - aligned with current Fig.3 grouped-bar style
# ============================================================

SERIES_CONFIG = [
    {
        "label": "Random",
        "csv_key": "random",
        "path": "experiment2_random_longrun_round_results.json",
    },
    {
        "label": "Explore-First",
        "csv_key": "epsilon_first",
        "path": "experiment2_epsilon_first_longrun_round_results.json",
    },
    {
        "label": "UCB-Greedy",
        "csv_key": "cmab",
        "path": "experiment2_cmab_longrun_round_results.json",
    },
    {
        "label": "Trust-Aware",
        "csv_key": "cmab_trust",
        "path": "experiment2_cmab_trust_round_results.json",
    },
    {
        "label": "Membership-Aware",
        "csv_key": "cmab_trust_pgrd",
        "path": "experiment2_cmab_trust_pgrd_round_results.json",
    },
    {
        "label": "TruthRide",
        "csv_key": "truthride",
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results.json",
    },
]


OUTPUT_PNG = "experiment2_Fig. 4：累计 RMSE.png"
OUTPUT_PDF = "experiment2_Fig. 4：累计 RMSE.pdf"
OUTPUT_CSV = "experiment2_Fig. 4：累计 RMSE.csv"

CHECKPOINTS = [20, 40, 60, 80, 100, 120, 140, 144]

FIGSIZE = (4.18, 2.70)
SAVE_PAD = FIG2_SAVE_PAD

BAR_TOTAL_WIDTH = 0.82

YLIM_MODE = "manual"
YLIM = (0.0, 0.40)
YTICKS = [0.0, 0.1, 0.2, 0.3, 0.4]

LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.015, 0.985)
LEGEND_NCOL = 2

COLOR_CYCLE = ["#00A6FB", "#FF9800", "#2FBF71", "#FF3B30", "#9C27B0", "#00B8A9"]
HATCH_CYCLE = ["//", "\\\\", "xx", "--", "///", "++"]


def load_round_results(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pooled_cumulative_rmse(all_runs, task_detail_map):
    """Cumulative task-level SSE/count across all seeds, with no smoothing.

    Every covered task/run is one sample; executed_by reports are averaged
    before the error is squared. Empty rounds add no samples. Before the
    first covered task the result is NaN, not a zero-error observation.
    """
    if not all_runs:
        raise ValueError("Saved all-runs results are required")
    totals_by_round = {}
    expected_rounds = None
    seeds = set()
    for run in all_runs:
        seed = run["seed"]
        if seed in seeds:
            raise ValueError(f"Duplicate seed {seed} in all-runs results")
        seeds.add(seed)
        seen_rounds = set()
        for record in run["round_results"]:
            round_id = int(record["round_id"])
            if round_id in seen_rounds:
                raise ValueError(f"Duplicate round {round_id} in seed {seed}")
            seen_rounds.add(round_id)
            errors = compute_task_error_records(record, task_detail_map)
            total = totals_by_round.setdefault(round_id, {"squared_error": 0.0, "task_count": 0})
            total["squared_error"] += math.fsum(item["squared_error"] for item in errors)
            total["task_count"] += len(errors)
        if expected_rounds is None:
            expected_rounds = seen_rounds
        elif seen_rounds != expected_rounds:
            raise ValueError(f"Round IDs differ across seeds (seed {seed})")

    rounds = sorted(totals_by_round)
    if not rounds or rounds != list(range(1, rounds[-1] + 1)):
        raise ValueError("Expected every saved round from round 1, including empty rounds")
    values, cumulative_records = [], []
    cumulative_squared_error = 0.0
    cumulative_task_count = 0
    for round_id in rounds:
        total = totals_by_round[round_id]
        cumulative_squared_error += total["squared_error"]
        cumulative_task_count += total["task_count"]
        value = (
            math.sqrt(cumulative_squared_error / cumulative_task_count)
            if cumulative_task_count else math.nan
        )
        values.append(round(value, 4))
        cumulative_records.append({
            "round_id": round_id,
            "current_round_task_count": total["task_count"],
            "cumulative_task_count": cumulative_task_count,
            "cumulative_squared_error": cumulative_squared_error,
            "cumulative_rmse": value,
        })
    return rounds, values, cumulative_records


def extract_checkpoint_values(rounds, values, checkpoints):
    round_to_value = {
        int(round_id): float(value)
        for round_id, value in zip(rounds, values)
    }

    available_rounds = sorted(round_to_value.keys())
    checkpoint_values = []

    for checkpoint in checkpoints:
        if checkpoint in round_to_value:
            checkpoint_values.append(round(round_to_value[checkpoint], 4))
            continue

        previous_rounds = [round_id for round_id in available_rounds if round_id <= checkpoint]
        if previous_rounds:
            nearest_round = previous_rounds[-1]
            checkpoint_values.append(round(round_to_value[nearest_round], 4))
        else:
            checkpoint_values.append(np.nan)

    return checkpoint_values


def save_checkpoint_plot_data_csv(checkpoints, values_dict):
    headers = ["checkpoint_round"] + [config["csv_key"] for config in SERIES_CONFIG]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, checkpoint in enumerate(checkpoints):
            row = [checkpoint]
            for config in SERIES_CONFIG:
                row.append(values_dict[config["label"]][idx])
            writer.writerow(row)


def set_figure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
        }
    )


def plot_grouped_bars(ax, checkpoints, values_dict):
    x = np.arange(len(checkpoints))
    n_series = len(SERIES_CONFIG)
    bar_width = BAR_TOTAL_WIDTH / max(1, n_series)

    for idx, config in enumerate(SERIES_CONFIG):
        values = values_dict[config["label"]]
        offset = (idx - (n_series - 1) / 2.0) * bar_width

        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            label=config["label"],
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            hatch=HATCH_CYCLE[idx % len(HATCH_CYCLE)],
            edgecolor="#111111",
            linewidth=0.82,
            alpha=0.95,
            zorder=3 + idx,
        )

    ax.set_xlabel("Checkpoint Round")
    ax.set_ylabel("Cumulative RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels([str(checkpoint) for checkpoint in checkpoints])

    if YLIM_MODE == "manual":
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)

    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.50,
        color="#b5b5b5",
        alpha=0.30,
    )
    ax.grid(False, axis="x")
    ax.grid(False, which="minor")

    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.80)
    ax.spines["bottom"].set_linewidth(0.80)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=0.75,
        pad=2.0,
        color="#444444",
    )

    leg = ax.legend(
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        ncol=LEGEND_NCOL,
        frameon=True,
        fancybox=False,
        framealpha=0.76,
        facecolor="white",
        edgecolor="#666666",
        handlelength=1.45,
        handletextpad=0.28,
        columnspacing=0.62,
        labelspacing=0.18,
        borderpad=0.24,
        borderaxespad=0.0,
    )
    leg.get_frame().set_linewidth(0.45)


def plot_figure(dataset=None):
    set_figure_style()

    project_dir = Path(__file__).resolve().parent
    dataset = dataset or os.environ.get("TRUTHRIDE_DATASET", "beijing")
    results_dir = project_dir / "results" / dataset
    workers = load_round_results(project_dir / "data" / dataset / "experiment2_worker_options.json")
    _, task_detail_map, _ = build_worker_indexes(workers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    checkpoint_values_dict = {}

    for config in SERIES_CONFIG:
        all_runs_path = results_dir / (Path(config["path"]).stem + "_all_runs.json")
        all_runs = load_round_results(all_runs_path)
        if {run["seed"] for run in all_runs} != set(range(3, 13)):
            raise ValueError(f"Expected main-experiment seeds 3-12: {all_runs_path}")
        rounds, cumulative_rmse, _ = pooled_cumulative_rmse(all_runs, task_detail_map)

        checkpoint_values_dict[config["label"]] = extract_checkpoint_values(
            rounds,
            cumulative_rmse,
            CHECKPOINTS,
        )

    save_checkpoint_plot_data_csv(CHECKPOINTS, checkpoint_values_dict)
    plot_grouped_bars(ax, CHECKPOINTS, checkpoint_values_dict)

    fig.subplots_adjust(
        left=0.155,
        right=0.985,
        bottom=0.17,
        top=0.965,
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.environ.get("TRUTHRIDE_DATASET", "beijing"), choices=("beijing", "rome", "chicago"))
    args = parser.parse_args()
    plot_figure(dataset=args.dataset)
