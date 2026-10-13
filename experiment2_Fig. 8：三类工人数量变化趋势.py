import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# ============================================================
# Fig. 8: Evolution of Three Worker States
#
# Style:
# - clean line chart with soft background bands
# - straight line segments, no curved interpolation
# - trailing moving average
# - sampled points connected by straight line segments
# - white background
# - light y-grid only
# ============================================================

ROUND_RESULTS_FILE = "experiment2_cmab_trust_pgrd_lgsc_round_results.json"

OUTPUT_PNG = "experiment2_Fig. 8：三类工人数量变化趋势.png"
OUTPUT_PDF = "experiment2_Fig. 8：三类工人数量变化趋势.pdf"
OUTPUT_CSV = "experiment2_Fig. 8：三类工人数量变化趋势.csv"

SMOOTH_WINDOW = 10
POINT_EVERY = 10

FIGSIZE = (4.18, 2.70)
SAVE_PAD = 0.014

LEGEND_LOC = "upper right"
LEGEND_BBOX_TO_ANCHOR = (0.985, 0.985)
LEGEND_NCOL = 1
LEGEND_FONT_SIZE = 6.7
LEGEND_HANDLE_LENGTH = 1.50
LEGEND_HANDLE_TEXT_PAD = 0.28
LEGEND_COLUMN_SPACING = 0.62

YLIM = (0, 250)
YTICKS = [0, 50, 100, 150, 200, 250]

# Soft visual band behind each curve.
# This is only for visual emphasis, not a statistical confidence interval.
BAND_ALPHA = 0.12
BAND_WIDTH_RATIO = 0.085  # 5.5% of y-axis span


SERIES_CONFIG = [
    {
        "label": "Trusted",
        "csv_key": "trusted_count",
        "metric_key": "trusted_count",
        "color": "#2ca02c",
        "marker": "o",
        "linewidth": 2.05,
        "markersize": 4.8,
        "alpha": 1.0,
        "zorder": 7,
    },
    {
        "label": "Unknown",
        "csv_key": "unknown_count",
        "metric_key": "unknown_count",
        "color": "#ff7f0e",
        "marker": "s",
        "linewidth": 1.45,
        "markersize": 4.0,
        "alpha": 0.88,
        "zorder": 5,
    },
    {
        "label": "Malicious",
        "csv_key": "malicious_count",
        "metric_key": "malicious_count",
        "color": "#d62728",
        "marker": "^",
        "linewidth": 1.45,
        "markersize": 4.0,
        "alpha": 0.88,
        "zorder": 5,
    },
]


def load_round_results(path):
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

    This avoids using future information.
    """
    if window <= 1 or len(values) <= 2:
        return [round(float(value), 4) for value in values]

    smoothed = []

    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        end = idx + 1
        smoothed.append(round(sum(values[start:end]) / (end - start), 4))

    return smoothed


def sample_points(x, y, step):
    """
    Sample points for plotting.

    The CSV still saves the full smoothed data.
    The figure only draws sampled points connected by straight line segments.
    """
    if step <= 1:
        return x, y

    x_sampled = x[::step]
    y_sampled = y[::step]

    if x_sampled[-1] != x[-1]:
        x_sampled.append(x[-1])
        y_sampled.append(y[-1])

    return x_sampled, y_sampled


def save_plot_data_csv(rounds, series_rows):
    headers = ["round"] + [config["csv_key"] for config in SERIES_CONFIG]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for round_id in rounds:
            row = [round_id]

            for values in series_rows:
                row.append(values.get(round_id, ""))

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


def add_soft_band(ax, x_values, y_values, color, zorder):
    """
    Add a soft background band behind a line.

    Note:
    This band is a visual aid only. It is not a confidence interval.
    """
    y_min, y_max = YLIM
    band_width = (y_max - y_min) * BAND_WIDTH_RATIO

    y_array = np.array(y_values, dtype=float)
    lower = np.maximum(y_min, y_array - band_width)
    upper = np.minimum(y_max, y_array + band_width)

    ax.fill_between(
        x_values,
        lower,
        upper,
        color=color,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=zorder,
    )


def plot_figure():
    set_figure_style()

    round_results = load_round_results(ROUND_RESULTS_FILE)
    rounds = [int(item["round_id"]) for item in round_results]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    csv_series_rows = []
    legend_handles = []

    for config in SERIES_CONFIG:
        raw_values = [
            float(item[config["metric_key"]])
            for item in round_results
        ]

        values = moving_average(raw_values, SMOOTH_WINDOW)

        csv_series_rows.append(dict(zip(rounds, values)))

        x_plot, y_plot = sample_points(rounds, values, POINT_EVERY)

        is_primary = config["label"] == "Trusted"

        # Soft background band first.
        add_soft_band(
            ax=ax,
            x_values=x_plot,
            y_values=y_plot,
            color=config["color"],
            zorder=config["zorder"] - 2,
        )

        # Straight line segments.
        ax.plot(
            x_plot,
            y_plot,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linestyle="-",
            drawstyle="default",
            linewidth=config["linewidth"],
            markersize=config["markersize"],
            markerfacecolor=config["color"],
            markeredgecolor="#111111" if is_primary else config["color"],
            markeredgewidth=0.82 if is_primary else 0.60,
            alpha=config["alpha"],
            zorder=config["zorder"],
            solid_capstyle="butt",
            solid_joinstyle="miter",
            antialiased=True,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=config["label"],
                color=config["color"],
                linestyle="-",
                linewidth=config["linewidth"],
                marker=config["marker"],
                markersize=config["markersize"],
                markerfacecolor=config["color"],
                markeredgecolor="#111111" if is_primary else config["color"],
                markeredgewidth=0.82 if is_primary else 0.60,
                alpha=config["alpha"],
            )
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Number of Workers")

    ax.set_xlim(0, 144)
    ax.set_ylim(*YLIM)

    ax.set_xticks(range(0, 141, 20))
    ax.set_yticks(YTICKS)

    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.50,
        color="#b5b5b5",
        alpha=0.28,
    )
    ax.grid(False, which="minor")
    ax.grid(False, axis="x")

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
        labelsize=8.0,
        color="#444444",
    )

    leg = ax.legend(
        handles=legend_handles,
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        ncol=LEGEND_NCOL,
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

    fig.subplots_adjust(
        left=0.155,
        right=0.985,
        bottom=0.17,
        top=0.965,
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)

    save_plot_data_csv(rounds, csv_series_rows)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()