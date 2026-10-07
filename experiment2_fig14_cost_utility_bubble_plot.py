import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator
from experiment2_paper_plot_style import (
    FIG2_SAVE_PAD,
    FIG2_SINGLE_FIGSIZE,
    LEGEND_BG,
    LEGEND_EDGE,
    SPINE_COLOR,
    TEXT_COLOR,
    TICK_COLOR,
    apply_fig2_rcparams,
    style_legend,
)


# ============================================================
# Fig. 14: Cost-Utility Bubble Plot
#
# Main-figure style:
# - bubble plot is kept
# - all algorithm labels are annotated beside bubbles
# - legend is kept, compact and semi-transparent
# - axis ticks are formatted as 10k / 20k / 60k
# - background texture is enhanced with soft panel color,
#   subtle horizontal bands, and gentle grid lines
# - bubble size represents cumulative task scale,
#   preferably cumulative covered tasks when available
# ============================================================


SERIES_CONFIG = [
    {
        "label": "Random",
        "csv_key": "random",
        "path": "experiment2_random_longrun_round_results.json",
        "color": "#8d8a83",
        "marker": "o",
        "zorder": 4,
        "offset": (5, -7),
        "fontsize": 6.6,
    },
    {
        "label": "Explore-First",
        "csv_key": "epsilon_first",
        "path": "experiment2_epsilon_first_longrun_round_results.json",
        "color": "#b7793c",
        "marker": "D",
        "zorder": 4,
        "offset": (5, 4),
        "fontsize": 6.6,
    },
    {
        "label": "UCB-Greedy",
        "csv_key": "cmab",
        "path": "experiment2_cmab_longrun_round_results.json",
        "color": "#5f6b3c",
        "marker": "^",
        "zorder": 5,
        "offset": (5, 4),
        "fontsize": 6.6,
    },
    {
        "label": "Trust-Aware",
        "csv_key": "cmab_trust",
        "path": "experiment2_cmab_trust_round_results.json",
        "color": "#2f5c85",
        "marker": "s",
        "zorder": 5,
        "offset": (5, 7),
        "fontsize": 6.6,
    },
    {
        "label": "Membership-Aware",
        "csv_key": "cmab_trust_pgrd",
        "path": "experiment2_cmab_trust_pgrd_round_results.json",
        "color": "#6a4c93",
        "marker": "v",
        "zorder": 5,
        "offset": (5, -10),
        "fontsize": 6.2,
    },
    {
        "label": "TruthRide",
        "csv_key": "truthride",
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results.json",
        "color": "#8f2d2d",
        "marker": "*",
        "zorder": 8,
        "offset": (6, 5),
        "fontsize": 7.0,
    },
]


OUTPUT_PNG = "experiment2_fig14_cost_utility_bubble_plot.png"
OUTPUT_PDF = "experiment2_fig14_cost_utility_bubble_plot.pdf"
OUTPUT_CSV = "experiment2_fig14_cost_utility_bubble_plot.csv"


# ============================================================
# Adjustable parameters
# ============================================================

FIGSIZE = FIG2_SINGLE_FIGSIZE

MIN_BUBBLE_SIZE = 105
MAX_BUBBLE_SIZE = 520

NORMAL_ALPHA = 0.88
TRUTH_ALPHA = 0.97

NORMAL_EDGE_WIDTH = 0.72
TRUTH_EDGE_WIDTH = 0.98

# Axis range.
# Use None for automatic range.
X_LIM = None
Y_LIM = None

# Example:
# X_LIM = (9000, 26000)
# Y_LIM = (-5000, 70000)

SHOW_ZERO_LINE = True

SHOW_ALL_LABELS = True
SHOW_LEGEND = False

LEGEND_LOC = "lower right"
LEGEND_NCOL = 2

# Background texture control
AXIS_FACE_COLOR = "#ffffff"
SHOW_BACKGROUND_BANDS = True
BACKGROUND_BAND_COLOR = "#f6f7fb"
BACKGROUND_BAND_ALPHA = 0.55
BACKGROUND_BAND_COUNT = 6


def load_round_results(path):
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    return [
        item for item in records
        if int(item.get("num_tasks", 0)) > 0
    ]


def compute_bubble_data(round_results):
    """
    X-axis:
        total platform cost =
        cumulative platform payment
        + cumulative bonus payment
        + cumulative validation cost

    Y-axis:
        cumulative platform utility

    Bubble size:
        cumulative covered tasks if available;
        otherwise cumulative completed tasks;
        otherwise cumulative number of tasks.
    """
    last = round_results[-1]

    cumulative_utility = float(last.get("cumulative_platform_utility", 0.0))

    cumulative_payment = float(last.get("cumulative_platform_payment", 0.0))
    cumulative_bonus = float(last.get("cumulative_bonus_payment", 0.0))
    cumulative_validation_cost = float(last.get("cumulative_validation_cost", 0.0))

    total_cost = cumulative_payment + cumulative_bonus + cumulative_validation_cost

    task_scale = float(
        last.get(
            "cumulative_covered_tasks",
            last.get(
                "cumulative_completed_tasks",
                last.get("cumulative_num_tasks", last.get("round_id", 1)),
            ),
        )
    )

    return (
        total_cost,
        cumulative_utility,
        task_scale,
        cumulative_payment,
        cumulative_bonus,
        cumulative_validation_cost,
    )


def scale_bubble_sizes(values, min_size=MIN_BUBBLE_SIZE, max_size=MAX_BUBBLE_SIZE):
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)

    if v_max == v_min:
        return [(min_size + max_size) / 2 for _ in values]

    return [
        min_size + (value - v_min) / (v_max - v_min) * (max_size - min_size)
        for value in values
    ]


def save_plot_data_csv(rows):
    headers = [
        "algorithm",
        "csv_key",
        "total_cost",
        "cumulative_platform_payment",
        "cumulative_bonus_payment",
        "cumulative_validation_cost",
        "cumulative_utility",
        "task_scale",
        "bubble_size",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def set_figure_style():
    apply_fig2_rcparams(
        font_size=8.0,
        axes_labelsize=8.6,
        xtick_labelsize=7.6,
        ytick_labelsize=7.6,
        legend_fontsize=5.9,
        axes_linewidth=0.86,
    )


def format_k(value, pos):
    """
    Format axis tick labels using k-units.
    Examples:
        10000 -> 10k
        12500 -> 12.5k
        60000 -> 60k
    """
    if abs(value) >= 1000:
        value_k = value / 1000

        if abs(value_k - round(value_k)) < 1e-6:
            return f"{value_k:.0f}k"

        return f"{value_k:.1f}k"

    return f"{value:.0f}"


def apply_axis_limits(ax, costs, utilities):
    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    elif costs:
        x_min, x_max = min(costs), max(costs)
        x_margin = max((x_max - x_min) * 0.16, 1.0)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)
    elif utilities:
        y_min, y_max = min(utilities), max(utilities)
        y_margin = max((y_max - y_min) * 0.16, 1.0)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)


def add_background_bands(ax):
    """
    Add very subtle horizontal bands to make the background less flat.
    This should stay light enough not to compete with bubbles.
    """
    if not SHOW_BACKGROUND_BANDS:
        return

    y_min, y_max = ax.get_ylim()
    band_height = (y_max - y_min) / BACKGROUND_BAND_COUNT

    for idx in range(BACKGROUND_BAND_COUNT):
        if idx % 2 == 0:
            y0 = y_min + idx * band_height
            y1 = y0 + band_height
            ax.axhspan(
                y0,
                y1,
                facecolor=BACKGROUND_BAND_COLOR,
                alpha=BACKGROUND_BAND_ALPHA,
                linewidth=0,
                zorder=0,
            )


def add_legend(ax):
    if not SHOW_LEGEND:
        return

    legend_handles = [
        Line2D(
            [0],
            [0],
            label=config["label"],
            marker=config["marker"],
            linestyle="None",
            markersize=4.6 if config["label"] != "TruthRide" else 5.3,
            markerfacecolor=config["color"],
            markeredgecolor="black",
            markeredgewidth=0.65,
            alpha=0.90,
        )
        for config in SERIES_CONFIG
    ]

    leg = ax.legend(
        handles=legend_handles,
        loc=LEGEND_LOC,
        ncol=LEGEND_NCOL,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        facecolor="#ffffff",
        edgecolor="#d3d7df",
        handletextpad=0.28,
        columnspacing=0.58,
        labelspacing=0.18,
        borderpad=0.24,
        borderaxespad=0.35,
    )

    style_legend(leg)
    leg.get_frame().set_linewidth(0.62)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#d3d7df")


def add_reference_guides(ax, x, y, color):
    ax.plot(
        [x, x],
        [ax.get_ylim()[0], y],
        linestyle=(0, (2.2, 2.4)),
        linewidth=0.72,
        color=color,
        alpha=0.28,
        zorder=2,
    )
    ax.plot(
        [ax.get_xlim()[0], x],
        [y, y],
        linestyle=(0, (2.2, 2.4)),
        linewidth=0.72,
        color=color,
        alpha=0.28,
        zorder=2,
    )


def plot_figure():
    set_figure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.set_axisbelow(True)

    costs = []
    utilities = []
    task_scales = []

    labels = []
    colors = []
    markers = []
    zorders = []
    offsets = []
    font_sizes = []

    payments = []
    bonuses = []
    validation_costs = []

    csv_rows = []

    for config in SERIES_CONFIG:
        round_results = load_round_results(config["path"])

        (
            total_cost,
            cumulative_utility,
            task_scale,
            cumulative_payment,
            cumulative_bonus,
            cumulative_validation_cost,
        ) = compute_bubble_data(round_results)

        costs.append(total_cost)
        utilities.append(cumulative_utility)
        task_scales.append(task_scale)

        labels.append(config["label"])
        colors.append(config["color"])
        markers.append(config["marker"])
        zorders.append(config["zorder"])
        offsets.append(config["offset"])
        font_sizes.append(config["fontsize"])

        payments.append(cumulative_payment)
        bonuses.append(cumulative_bonus)
        validation_costs.append(cumulative_validation_cost)

    bubble_sizes = scale_bubble_sizes(task_scales)

    for i, label in enumerate(labels):
        is_truthride = label == "TruthRide"

        scatter = ax.scatter(
            costs[i],
            utilities[i],
            s=bubble_sizes[i],
            color=colors[i],
            marker=markers[i],
            alpha=TRUTH_ALPHA if is_truthride else NORMAL_ALPHA,
            edgecolors="black",
            linewidth=TRUTH_EDGE_WIDTH if is_truthride else NORMAL_EDGE_WIDTH,
            zorder=zorders[i],
        )
        scatter.set_path_effects(
            [
                path_effects.SimplePatchShadow(
                    offset=(1.0, -1.0),
                    shadow_rgbFace=(0.0, 0.0, 0.0, 0.12),
                    alpha=0.22,
                ),
                path_effects.Normal(),
            ]
        )

        if SHOW_ALL_LABELS:
            annotation = ax.annotate(
                label,
                (costs[i], utilities[i]),
                xytext=offsets[i],
                textcoords="offset points",
                fontsize=font_sizes[i],
                fontweight="bold" if is_truthride else "normal",
                ha="left",
                va="bottom",
                color=TEXT_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": (1.0, 1.0, 1.0, 0.90),
                    "edgecolor": (0.80, 0.84, 0.89, 0.82),
                    "linewidth": 0.40,
                },
                zorder=zorders[i] + 1,
            )
            annotation.set_path_effects(
                [
                    path_effects.withStroke(linewidth=1.1, foreground="#fffdfa", alpha=0.92),
                    path_effects.Normal(),
                ]
            )

        csv_rows.append(
            [
                label,
                SERIES_CONFIG[i]["csv_key"],
                round(costs[i], 6),
                round(payments[i], 6),
                round(bonuses[i], 6),
                round(validation_costs[i], 6),
                round(utilities[i], 6),
                round(task_scales[i], 6),
                round(bubble_sizes[i], 6),
            ]
        )

    ax.set_xlabel("Total Platform Cost")
    ax.set_ylabel("Cumulative Utility")

    apply_axis_limits(ax, costs, utilities)

    # Background bands should be added after axis limits are fixed.
    add_background_bands(ax)
    add_reference_guides(ax, costs[-1], utilities[-1], colors[-1])

    if SHOW_ZERO_LINE:
        ax.axhline(
            0.0,
            color="#7d8694",
            linestyle=":",
            linewidth=0.82,
            alpha=0.92,
            zorder=2,
        )

    # k-style axis tick labels, e.g., 10k, 20k, 60k.
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax.yaxis.set_major_formatter(FuncFormatter(format_k))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Enhanced but soft background grid.
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.56,
        color="#d9dee7",
        alpha=0.82,
        zorder=1,
    )
    ax.grid(
        True,
        which="minor",
        axis="both",
        linestyle="-",
        linewidth=0.33,
        color="#eef1f6",
        alpha=0.95,
        zorder=1,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.92)
        spine.set_color(SPINE_COLOR)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.4,
        width=0.78,
        pad=2.0,
        color=TICK_COLOR,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=1.9,
        width=0.58,
        color=TICK_COLOR,
    )

    add_legend(ax)

    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.175, top=0.965)

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    plt.close(fig)

    save_plot_data_csv(csv_rows)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()
