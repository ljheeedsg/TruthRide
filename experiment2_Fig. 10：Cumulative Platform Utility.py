import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment2_paper_plot_style import FIG2_SAVE_PAD


# ============================================================
# Fig. 10: Cumulative Platform Utility
#
# Statistics:
# - use true checkpoint cumulative values
# - x-axis checkpoints: 20, 40, 60, 80, 100, 120, 140, 144
# - y-value at checkpoint r:
#       cumulative_platform_utility accumulated from Round 1 to Round r
# - if round r is missing, use the nearest previous available round
#
# Visualization:
# - grouped bars only
# - y-axis starts from 0, so bars are attached to the x-axis
# - consistent with Fig.3 / Fig.4 checkpoint cumulative bar charts
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


OUTPUT_PNG = "experiment2_Fig. 10：Cumulative Platform Utility.png"
OUTPUT_PDF = "experiment2_Fig. 10：Cumulative Platform Utility.pdf"
OUTPUT_CSV = "experiment2_Fig. 10：Cumulative Platform Utility.csv"

METRIC_KEY = "cumulative_platform_utility"

CHECKPOINTS = [20, 40, 60, 80, 100, 120, 140, 144]

FIGSIZE = (4.18, 2.70)
SAVE_PAD = FIG2_SAVE_PAD

BAR_TOTAL_WIDTH = 0.82

# Plot scale:
# Original utility values are large, so the figure displays values in 1e4 units.
# The CSV still saves the original unscaled values.
PLOT_SCALE = 1e4

# Important:
# Set y-axis lower bound to 0 so bars are attached to the x-axis.
YLIM_MODE = "manual"
YLIM = (0.0, 7.2)
YTICKS = [0, 1, 2, 3, 4, 5, 6, 7]

LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.015, 0.985)
LEGEND_NCOL = 2

COLOR_CYCLE = [
    "#00A6FB",  # Random
    "#FF9800",  # Explore-First
    "#2FBF71",  # UCB-Greedy
    "#FF3B30",  # Trust-Aware
    "#9C27B0",  # Membership-Aware
    "#00B8A9",  # TruthRide
]

HATCH_CYCLE = ["//", "\\\\", "xx", "--", "///", "++"]


def load_round_results(path):
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    return [
        item for item in records
        if int(item.get("num_tasks", 0)) > 0
    ]


def extract_checkpoint_values(records, checkpoints, metric_key):
    """
    Extract cumulative metric values at selected checkpoints.

    For checkpoint r, use the record whose round_id == r.
    If exact round r is missing, use the nearest previous available round.
    """
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

        previous_rounds = [
            round_id for round_id in available_rounds
            if round_id <= checkpoint
        ]

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
            "hatch.linewidth": 0.48,
        }
    )


def plot_grouped_bars(ax, checkpoints, values_dict):
    x = np.arange(len(checkpoints))
    n_series = len(SERIES_CONFIG)
    bar_width = BAR_TOTAL_WIDTH / max(1, n_series)

    all_scaled_values = []

    for idx, config in enumerate(SERIES_CONFIG):
        label = config["label"]
        raw_values = values_dict[label]
        values = [value / PLOT_SCALE for value in raw_values]

        all_scaled_values.extend(values)

        is_truthride = label == "TruthRide"
        offset = (idx - (n_series - 1) / 2.0) * bar_width

        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            label=label,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            hatch=HATCH_CYCLE[idx % len(HATCH_CYCLE)],
            edgecolor="#111111",
            linewidth=0.92 if is_truthride else 0.82,
            alpha=0.98 if is_truthride else 0.95,
            zorder=7 if is_truthride else 3 + idx,
        )

    ax.set_xlabel("Checkpoint Round")
    ax.set_ylabel(r"Cumulative Platform Utility ($\times 10^4$)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(checkpoint) for checkpoint in checkpoints])

    if YLIM_MODE == "manual" and YLIM is not None:
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
    elif all_scaled_values:
        # Auto mode with fixed lower bound at 0.
        # This prevents bars from visually floating above the x-axis.
        y_max = max(all_scaled_values)
        upper = y_max * 1.10 if y_max > 0 else 1.0
        ax.set_ylim(0.0, upper)

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

    plot_grouped_bars(
        ax=ax,
        checkpoints=CHECKPOINTS,
        values_dict=checkpoint_values_dict,
    )

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