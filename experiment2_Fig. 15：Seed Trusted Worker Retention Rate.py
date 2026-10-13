import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# ============================================================
# Fig. 15: Seed Trusted Worker Retention Rate
#
# Style:
# - clean line chart with wide angular background bands
# - sampled points connected by straight line segments
# - no smoothing
# - background bands are visual aids only, NOT confidence intervals
# ============================================================

WORKER_OPTIONS_FILE = "experiment2_worker_options.json"

OUTPUT_PNG = "experiment2_Fig. 15：Seed Trusted Worker Retention Rate.png"
OUTPUT_PDF = "experiment2_Fig. 15：Seed Trusted Worker Retention Rate.pdf"
OUTPUT_CSV = "experiment2_Fig. 15：Seed Trusted Worker Retention Rate.csv"

FIGSIZE = (4.18, 2.70)
SAVE_PAD = 0.012

# Keep original retention curve unchanged.
SMOOTH_WINDOW = 1

# Larger interval makes the line and band more angular.
POINT_EVERY = 10

LEGEND_LOC = "lower left"
LEGEND_BBOX_TO_ANCHOR = (0.02, 0.03)
LEGEND_NCOL = 1
LEGEND_FONT_SIZE = 6.7
LEGEND_HANDLE_LENGTH = 1.50
LEGEND_HANDLE_TEXT_PAD = 0.28
LEGEND_COLUMN_SPACING = 0.62

YLIM = (0.0, 1.02)
YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Wide but still light band.
# Wider band + sampled points = polygon-like angular background.
BAND_ALPHA = 0.12
BAND_WIDTH_RATIO = 0.085


SERIES_CONFIG = [
    {
        "label": "Trust-Aware",
        "csv_key": "trust_aware",
        "path": "experiment2_cmab_trust_round_results_all_runs.json",
        "color": "#1f77b4",
        "marker": "s",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.86,
        "zorder": 4,
    },
    {
        "label": "Membership-Aware",
        "csv_key": "membership_aware",
        "path": "experiment2_cmab_trust_pgrd_round_results_all_runs.json",
        "color": "#8c564b",
        "marker": "v",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.86,
        "zorder": 4,
    },
    {
        "label": "TruthRide",
        "csv_key": "truthride",
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results_all_runs.json",
        "color": "#b22222",
        "marker": "o",
        "linewidth": 2.10,
        "markersize": 4.8,
        "alpha": 1.0,
        "zorder": 7,
    },
]


def load_seed_trusted_worker_ids(path):
    with Path(path).open("r", encoding="utf-8") as f:
        worker_options = json.load(f)

    return {
        int(worker["worker_id"])
        for worker in worker_options.values()
        if worker.get("init_category") == "trusted"
    }


def load_all_runs(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values, window):
    """
    Trailing moving average.

    With SMOOTH_WINDOW = 1, this keeps the original retention curve unchanged.
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

    The CSV still saves the full curve.
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


def compute_seed_retention_curve(all_runs_payload, seed_trusted_ids):
    seed_count = max(1, len(seed_trusted_ids))
    per_run_curves = []
    round_axis = None

    for run_payload in all_runs_payload:
        round_results = [
            item for item in run_payload["round_results"]
            if int(item.get("num_tasks", 0)) > 0
        ]

        rounds = [int(item["round_id"]) for item in round_results]

        if round_axis is None:
            round_axis = rounds

        left_seed_ids = set()
        retention_values = []

        for item in round_results:
            left_ids = {
                int(worker_id)
                for worker_id in item.get("left_worker_ids_this_round", [])
            }

            left_seed_ids.update(left_ids & seed_trusted_ids)
            retained = seed_count - len(left_seed_ids)
            retention_values.append(round(retained / seed_count, 4))

        per_run_curves.append(retention_values)

    if not per_run_curves:
        return [], []

    averaged_curve = []

    for idx in range(len(per_run_curves[0])):
        averaged_curve.append(
            round(sum(curve[idx] for curve in per_run_curves) / len(per_run_curves), 4)
        )

    return round_axis, averaged_curve


def save_plot_data_csv(rounds, series_rows):
    headers = ["round"] + [config["csv_key"] for config in SERIES_CONFIG]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, round_id in enumerate(rounds):
            row = [round_id]

            for values in series_rows:
                row.append(values[idx])

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


def add_angular_soft_band(ax, x_values, y_values, color, zorder):
    """
    Add a wide angular background band behind a sampled line.

    The band follows sampled points rather than every round,
    so the band edges look polygon-like and angular.
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
        antialiased=False,
    )


def plot_figure():
    set_figure_style()

    dataset = os.environ.get("TRUTHRIDE_DATASET", "beijing")
    worker_options_file = Path(__file__).resolve().parent / "data" / dataset / WORKER_OPTIONS_FILE
    seed_trusted_ids = load_seed_trusted_worker_ids(worker_options_file)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    csv_rounds = None
    csv_series_rows = []
    legend_handles = []

    for config in SERIES_CONFIG:
        all_runs_payload = load_all_runs(config["path"])

        rounds, raw_values = compute_seed_retention_curve(
            all_runs_payload=all_runs_payload,
            seed_trusted_ids=seed_trusted_ids,
        )

        values = moving_average(raw_values, SMOOTH_WINDOW)

        if csv_rounds is None:
            csv_rounds = rounds

        csv_series_rows.append(values)

        x_plot, y_plot = sample_points(rounds, values, POINT_EVERY)

        is_truthride = config["label"] == "TruthRide"

        # The band uses sampled points, so it has angular polygon edges.
        add_angular_soft_band(
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
            markeredgecolor="#111111" if is_truthride else config["color"],
            markeredgewidth=0.82 if is_truthride else 0.60,
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
                markeredgecolor="#111111" if is_truthride else config["color"],
                markeredgewidth=0.82 if is_truthride else 0.60,
                alpha=config["alpha"],
            )
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Seed Trusted Retention")

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

    save_plot_data_csv(csv_rounds, csv_series_rows)

    print(f"Seed trusted worker count: {len(seed_trusted_ids)}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()
