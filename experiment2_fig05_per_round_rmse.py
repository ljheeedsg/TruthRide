import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiment2_rmse_plot_utils import (
    SERIES_CONFIG_RMSE,
    build_worker_indexes,
    compute_rmse_series,
    load_round_results,
    load_workers,
)


# ============================================================
# Fig. 5: Per-Round RMSE
#
# Style aligned with Fig. 2:
# - per-round RMSE with trailing moving average
# - sampled points connected by straight line segments
# - white background
# - light y-grid only
# - no top/right spines
# - in-panel legend
# - TruthRide highlighted
# ============================================================

OUTPUT_PNG = "experiment2_fig05_per_round_rmse.png"
OUTPUT_PDF = "experiment2_fig05_per_round_rmse.pdf"
OUTPUT_CSV = "experiment2_fig05_per_round_rmse.csv"

# Same display logic as Fig.2.
SMOOTH_WINDOW = 10
POINT_EVERY = 10

FIGSIZE = (4.18, 2.70)
SAVE_PAD = 0.014

YLIM_MODE = "manual"
YLIM = (0.0, 0.40)
YTICKS = [0.0, 0.1, 0.2, 0.3, 0.4]

LEGEND_LOC = "upper right"
LEGEND_BBOX_TO_ANCHOR = (0.985, 0.985)
LEGEND_NCOL = 2
LEGEND_FONT_SIZE = 6.7
LEGEND_HANDLE_LENGTH = 1.50
LEGEND_HANDLE_TEXT_PAD = 0.28
LEGEND_COLUMN_SPACING = 0.62

TRUTHRIDE_LABEL = "TruthRide"


STYLE_OVERRIDE = {
    "Random": {
        "color": "#8e7aa8",
        "marker": "o",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.72,
        "zorder": 3,
    },
    "Explore-First": {
        "color": "#ff7f0e",
        "marker": "D",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.72,
        "zorder": 3,
    },
    "UCB-Greedy": {
        "color": "#2ca02c",
        "marker": "^",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    "Trust-Aware": {
        "color": "#1f77b4",
        "marker": "s",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    "Membership-Aware": {
        "color": "#8c564b",
        "marker": "v",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    "TruthRide": {
        "color": "#b22222",
        "marker": "o",
        "linewidth": 2.35,
        "markersize": 5.0,
        "alpha": 1.0,
        "zorder": 7,
    },
}


def styled_config(config):
    merged = dict(config)
    override = STYLE_OVERRIDE.get(config["label"], {})
    merged.update(override)
    return merged


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


def sample_points(x, y, step):
    """
    Sample points every 'step' rounds, and always keep the last point.
    """
    if step <= 1:
        return x, y

    x_sampled = x[::step]
    y_sampled = y[::step]

    if x_sampled[-1] != x[-1]:
        x_sampled.append(x[-1])
        y_sampled.append(y[-1])

    return x_sampled, y_sampled


def save_rmse_plot_data_csv(output_csv, rounds, series_rows, series_config):
    headers = ["round"]

    for config in series_config:
        headers.append(config.get("csv_key", config["label"]))

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, round_id in enumerate(rounds):
            row = [round_id]

            for values in series_rows:
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


def plot_figure():
    set_figure_style()

    workers = load_workers()
    tasks_by_slot, task_detail_map, trusted_reference_map = build_worker_indexes(workers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    csv_rounds = None
    csv_series_rows = []
    legend_handles = []
    max_rmse = 0.0

    for raw_config in SERIES_CONFIG_RMSE:
        config = styled_config(raw_config)

        round_results = load_round_results(config["path"])

        rounds, per_round_rmse_raw, _ = compute_rmse_series(
            round_results,
            tasks_by_slot,
            task_detail_map,
            trusted_reference_map,
        )

        per_round_rmse = moving_average(per_round_rmse_raw, SMOOTH_WINDOW)

        if csv_rounds is None:
            csv_rounds = rounds

        csv_series_rows.append(per_round_rmse)

        if per_round_rmse:
            max_rmse = max(max_rmse, max(per_round_rmse))

        x_plot, y_plot = sample_points(rounds, per_round_rmse, POINT_EVERY)

        is_truthride = config["label"] == TRUTHRIDE_LABEL

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
            alpha=config.get("alpha", 1.0),
            zorder=config["zorder"],
            solid_capstyle="butt",
            solid_joinstyle="miter",
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
                alpha=config.get("alpha", 1.0),
            )
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Per-Round RMSE")

    ax.set_xlim(0, 144)

    if YLIM_MODE == "manual":
        ax.set_ylim(*YLIM)
    else:
        ax.set_ylim(0.0, max(0.5, round(max_rmse * 1.08, 2)))

    ax.set_xticks(range(0, 141, 20))
    ax.set_yticks(YTICKS)

    # Fig.2-like clean background: light y-grid only.
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
    ax.grid(False, axis="x")

    # Fig.2-like light axes.
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

    save_rmse_plot_data_csv(
        output_csv=OUTPUT_CSV,
        rounds=csv_rounds,
        series_rows=csv_series_rows,
        series_config=SERIES_CONFIG_RMSE,
    )

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()