import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Fig. 11: Per-Round Platform Utility
# Style:
# - 20-round interval average line chart
# - no raw 144-round dense line
# - white background, in-panel legend, no title
# ============================================================


# -----------------------------
# Basic plotting parameters
# -----------------------------
OUTPUT_PNG = "experiment2_Fig. 11：Per-Round Platform Utility.png"
OUTPUT_PDF = "experiment2_Fig. 11：Per-Round Platform Utility.pdf"
OUTPUT_CSV = "experiment2_Fig. 11：Per-Round Platform Utility.csv"

METRIC_KEY = "platform_utility"

# First smooth the per-round utility using a trailing moving average.
SMOOTH_WINDOW = 5

# Then aggregate the smoothed values into 20-round intervals.
INTERVAL_SIZE = 20

FIGSIZE = (4.18, 2.70)
SAVE_PAD = 0.014

LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.02, 0.98)
LEGEND_NCOL = 2
LEGEND_FONT_SIZE = 6.7
LEGEND_HANDLE_LENGTH = 1.50
LEGEND_HANDLE_TEXT_PAD = 0.28
LEGEND_COLUMN_SPACING = 0.62

TRUTHRIDE_LABEL = "TruthRide"


# -----------------------------
# Algorithm styles
# -----------------------------
SERIES_CONFIG = [
    {
        "label": "Random",
        "csv_key": "random",
        "path": "experiment2_random_longrun_round_results.json",
        "color": "#8e7aa8",
        "marker": "o",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.72,
        "zorder": 3,
    },
    {
        "label": "Explore-First",
        "csv_key": "epsilon_first",
        "path": "experiment2_epsilon_first_longrun_round_results.json",
        "color": "#ff7f0e",
        "marker": "D",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.72,
        "zorder": 3,
    },
    {
        "label": "UCB-Greedy",
        "csv_key": "cmab",
        "path": "experiment2_cmab_longrun_round_results.json",
        "color": "#2ca02c",
        "marker": "^",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    {
        "label": "Trust-Aware",
        "csv_key": "cmab_trust",
        "path": "experiment2_cmab_trust_round_results.json",
        "color": "#1f77b4",
        "marker": "s",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    {
        "label": "Membership-Aware",
        "csv_key": "cmab_trust_pgrd",
        "path": "experiment2_cmab_trust_pgrd_round_results.json",
        "color": "#8c564b",
        "marker": "v",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    {
        "label": "TruthRide",
        "csv_key": "truthride",
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results.json",
        "color": "#b22222",
        "marker": "o",
        "linewidth": 2.35,
        "markersize": 5.0,
        "alpha": 1.0,
        "zorder": 7,
    },
]


def load_round_results(path):
    """
    Load per-round results and remove invalid rounds with zero tasks.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    return [
        item for item in records
        if int(item.get("num_tasks", 0)) > 0
    ]


def moving_average(values, window):
    """
    Trailing moving average.

    For round t, it only uses current and previous values:

        y_smooth(t) = average[y(t-window+1), ..., y(t)]

    This avoids using future data.
    """
    if window <= 1 or len(values) <= 2:
        return [round(float(value), 4) for value in values]

    smoothed = []

    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        end = idx + 1
        smoothed.append(round(sum(values[start:end]) / (end - start), 4))

    return smoothed


def compute_interval_means(rounds, values, interval_size):
    """
    Aggregate smoothed per-round values into fixed-size round intervals.

    Example:
        interval_size = 20

        Rounds 1–20     -> one average value
        Rounds 21–40    -> one average value
        Rounds 41–60    -> one average value
        ...

    The final interval is truncated by the actual maximum round.
    For example, if max_round = 144, the last label is 141–144.
    """
    if not rounds or not values:
        return [], []

    paired = sorted(zip(rounds, values), key=lambda item: item[0])

    min_round = paired[0][0]
    max_round = paired[-1][0]

    interval_labels = []
    interval_means = []

    start = min_round

    while start <= max_round:
        end = min(start + interval_size - 1, max_round)

        interval_values = [
            value
            for round_id, value in paired
            if start <= round_id <= end
        ]

        if interval_values:
            interval_labels.append(f"{start}–{end}")
            interval_means.append(
                round(float(sum(interval_values) / len(interval_values)), 4)
            )

        start += interval_size

    return interval_labels, interval_means


def save_interval_plot_data_csv(interval_labels, interval_values_dict, output_csv):
    """
    Save plotted interval-level values to CSV.
    """
    headers = ["round_interval"] + [config["csv_key"] for config in SERIES_CONFIG]

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, interval_label in enumerate(interval_labels):
            row = [interval_label]

            for config in SERIES_CONFIG:
                label = config["label"]
                values = interval_values_dict.get(label, [])
                row.append(values[idx] if idx < len(values) else "")

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
            "legend.fontsize": 8.0,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
        }
    )


def plot_interval_average_lines(ax, interval_labels, interval_values_dict):
    """
    Plot one interval-average line for each algorithm.
    """
    x_positions = list(range(len(interval_labels)))

    legend_handles = []
    all_values = [
        value
        for values in interval_values_dict.values()
        for value in values
    ]

    y_min = min(all_values)
    y_max = max(all_values)
    y_span = max(y_max - y_min, 1.0)

    for config in SERIES_CONFIG:
        label = config["label"]
        values = interval_values_dict[label]

        color = config["color"]
        marker = config["marker"]
        is_truthride = label == TRUTHRIDE_LABEL

        ax.plot(
            x_positions,
            values,
            color=color,
            linewidth=config["linewidth"],
            marker=marker,
            markersize=config["markersize"],
            markerfacecolor=color,
            markeredgecolor="#111111" if is_truthride else color,
            markeredgewidth=0.82 if is_truthride else 0.60,
            alpha=config["alpha"],
            zorder=config["zorder"],
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=color,
                linewidth=config["linewidth"],
                marker=marker,
                markersize=config["markersize"],
                markerfacecolor=color,
                markeredgecolor="#111111" if is_truthride else color,
                markeredgewidth=0.82 if is_truthride else 0.60,
                alpha=config["alpha"],
            )
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(interval_labels)

    ax.set_xlabel("Round interval")
    ax.set_ylabel("Average Per-Round Platform Utility")

    # Clean Fig.11-like background.
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="#b5b5b5",
        alpha=0.28,
    )
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    # Light axes: remove top and right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=0.75,
        pad=2.0,
        labelsize=8.0,
    )

    # If platform utility has both positive and negative values,
    # draw a zero-profit reference line.
    if y_min < 0 < y_max:
        ax.axhline(
            0.0,
            color="#808080",
            linewidth=0.7,
            alpha=0.85,
            zorder=1,
        )

    ax.set_ylim(
        y_min - 0.05 * y_span,
        y_max + 0.10 * y_span,
    )

    leg = ax.legend(
        handles=legend_handles,
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        ncol=min(LEGEND_NCOL, len(legend_handles)),
        frameon=True,
        fancybox=False,
        framealpha=0.76,
        facecolor="white",
        edgecolor="#666666",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=0.18,
        borderpad=0.24,
        borderaxespad=0.0,
    )
    leg.get_frame().set_linewidth(0.45)

    return leg


def plot_figure():
    set_figure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    interval_labels = None
    interval_values_dict = {}

    for config in SERIES_CONFIG:
        records = load_round_results(config["path"])

        x_full = [int(item["round_id"]) for item in records]
        raw_y = [float(item[METRIC_KEY]) for item in records]

        # Step 1: smooth per-round platform utility.
        y_smooth = moving_average(raw_y, SMOOTH_WINDOW)

        # Step 2: compute 20-round interval means.
        current_interval_labels, interval_means = compute_interval_means(
            x_full,
            y_smooth,
            INTERVAL_SIZE,
        )

        if interval_labels is None:
            interval_labels = current_interval_labels

        interval_values_dict[config["label"]] = interval_means

    plot_interval_average_lines(
        ax,
        interval_labels or [],
        interval_values_dict,
    )

    ax.set_title("")

    fig.subplots_adjust(
        left=0.155,
        right=0.985,
        bottom=0.17,
        top=0.965,
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)

    save_interval_plot_data_csv(
        interval_labels or [],
        interval_values_dict,
        OUTPUT_CSV,
    )

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()