import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment2_paper_plot_style import FIG2_SAVE_PAD


# ============================================================
# Fig. 3: Cumulative Average Data Quality
#
# Statistics:
# - use true checkpoint cumulative values
# - x-axis checkpoints: 20, 40, 60, 80, 100, 120, 140, 144
# - y-value at checkpoint r:
#       cumulative_avg_quality at round r
# - if round r is missing, use the nearest previous available round
#
# Visualization:
# - grouped bars only
# - keep the current parameter-study-like grouped-bar style
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


OUTPUT_PNG = "experiment2_fig03_cumulative_average_data_quality.png"
OUTPUT_PDF = "experiment2_fig03_cumulative_average_data_quality.pdf"
OUTPUT_CSV = "experiment2_fig03_cumulative_average_data_quality.csv"

METRIC_KEY = "cumulative_avg_quality"
CHECKPOINTS = [20, 40, 60, 80, 100, 120, 140, 144]

FIGSIZE = (4.18, 2.70)
SAVE_PAD = FIG2_SAVE_PAD

BAR_TOTAL_WIDTH = 0.82

YLIM_MODE = "manual"
YLIM = (0.45, 0.83)
YTICKS = [0.5, 0.6, 0.7, 0.8]

LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.015, 0.985)
LEGEND_NCOL = 2

COLOR_CYCLE = ["#00A6FB", "#FF9800", "#2FBF71", "#FF3B30", "#9C27B0", "#00B8A9"]
HATCH_CYCLE = ["//", "\\\\", "xx", "--", "///", "++"]


def load_round_results(path):
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    return [
        item for item in records
        if int(item.get("num_tasks", 0)) > 0
    ]


def extract_checkpoint_values(records, checkpoints, metric_key):
    round_to_value = {
        int(item["round_id"]): float(item[metric_key])
        for item in records
        if metric_key in item
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
    ax.set_ylabel("Cumulative Average Quality")

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


def plot_figure():
    set_figure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    checkpoint_values_dict = {}

    for config in SERIES_CONFIG:
        records = load_round_results(config["path"])
        checkpoint_values_dict[config["label"]] = extract_checkpoint_values(
            records=records,
            checkpoints=CHECKPOINTS,
            metric_key=METRIC_KEY,
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
    plot_figure()
