import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

import os

# 图中文字语言：en（默认，英文期刊稿）/ zh（原中文版）
PLOT_LANG = os.environ.get("TRUTHRIDE_PLOT_LANG", "en")
EN_LABELS = {
    "随机调度方法": "Random",
    "探索优先调度方法": "Explore-First",
    "探索优先\n调度方法": "Explore-\nFirst",
    "质量估计调度方法": "UCB-Greedy",
    "质量估计\n调度方法": "UCB-\nGreedy",
    "本发明方法": "TruthRide",
    "信任感知": "Trust-Aware",
    "会员感知": "Membership-Aware",
    "可信": "Trusted",
    "未知": "Unknown",
    "低可信": "Malicious",
    "可信工人参与轮次": "trusted_participation_rounds",
    "未知工人参与轮次": "unknown_participation_rounds",
    "低可信工人参与轮次": "malicious_participation_rounds",
    "轮次": "Round",
    "每轮平均数据质量": "Per-Round Average Data Quality",
    "算法": "Algorithm",
    "平均参与轮次": "Average Participation Rounds",
    "任务覆盖率": "Task Coverage Rate",
    "任务完成率": "Task Completion Rate",
    "累计任务覆盖率": "Cumulative Task Coverage Rate",
    "累计任务完成率": "Cumulative Task Completion Rate",
}


def T(text):
    return EN_LABELS.get(text, text) if PLOT_LANG == "en" else text



# ============================================================
# Fig. 33-34: 任务覆盖率与完成率对比
#
# Style aligned with revised Fig.2 / Fig.11:
# - Per-round indicators:
#   trailing moving average + sampled points + straight lines
# - Cumulative indicators:
#   no extra smoothing by default + sampled points + straight lines
# - Clean white background
# - light y-grid only
# - remove top/right spines
# - in-panel legend
# - TruthRide highlighted
# ============================================================


SERIES_CONFIG = [
    {
        "algorithm": "random",
        "label": T("随机调度方法"),
        "path": "experiment2_random_longrun_round_results.json",
        "color": "#8e7aa8",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.72,
        "zorder": 3,
    },
    {
        "algorithm": "epsilon_first",
        "label": T("探索优先调度方法"),
        "path": "experiment2_epsilon_first_longrun_round_results.json",
        "color": "#ff7f0e",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 1.35,
        "markersize": 4.0,
        "alpha": 0.72,
        "zorder": 3,
    },
    {
        "algorithm": "cmab",
        "label": T("质量估计调度方法"),
        "path": "experiment2_cmab_longrun_round_results.json",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    {
        "algorithm": "trust",
        "label": "Trust-Aware",
        "path": "experiment2_cmab_trust_round_results.json",
        "color": "#1f77b4",
        "marker": "s",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    {
        "algorithm": "pgrd",
        "label": "Membership-Aware",
        "path": "experiment2_cmab_trust_pgrd_round_results.json",
        "color": "#8c564b",
        "marker": "v",
        "linewidth": 1.40,
        "markersize": 4.0,
        "alpha": 0.74,
        "zorder": 4,
    },
    {
        "algorithm": "lgsc",
        "label": T("本发明方法"),
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results.json",
        "color": "#b22222",
        "marker": "s",
        "linestyle": ":",
        "linewidth": 2.35,
        "markersize": 5.0,
        "alpha": 1.0,
        "zorder": 7,
    },
]

if PLOT_LANG == "en":
    for _config in SERIES_CONFIG:
        _config["linestyle"] = "-"


# ============================================================
# Adjustable figure settings
# ============================================================

FIGURE_SPECS = {
    "fig33": {
        "output_png": "experiment2_Fig. 33：任务覆盖率对比.png",
        "output_pdf": "experiment2_Fig. 33：任务覆盖率对比.pdf",
        "output_csv": "experiment2_Fig. 33：任务覆盖率对比.csv",
        "ylabel": T("任务覆盖率"),
        "metric_key": "coverage_rate",

        # Per-round indicator: follow revised Fig.2 style
        "is_cumulative": False,
        "smooth_window": 5,

        # Adjustable y-axis
        "ylim": (0.0, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "minor_yticks": None,

        # Legend
        "legend_loc": "lower left",
        "legend_bbox_to_anchor": (0.02, 0.03),
        "legend_ncol": 2,
    },

    "fig34": {
        "output_png": "experiment2_Fig. 34：任务完成率对比.png",
        "output_pdf": "experiment2_Fig. 34：任务完成率对比.pdf",
        "output_csv": "experiment2_Fig. 34：任务完成率对比.csv",
        "ylabel": T("任务完成率"),
        "metric_key": "completion_rate",

        # Per-round indicator: follow revised Fig.2 style
        "is_cumulative": False,
        "smooth_window": 5,

        # Adjustable y-axis
        "ylim": (0.0, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "minor_yticks": None,

        # Legend
        "legend_loc": "lower left",
        "legend_bbox_to_anchor": (0.02, 0.03),
        "legend_ncol": 2,
    },

    "fig33_supp": {
        "output_png": "experiment2_Fig. 33补充：累计任务覆盖率对比.png",
        "output_pdf": "experiment2_Fig. 33补充：累计任务覆盖率对比.pdf",
        "output_csv": "experiment2_Fig. 33补充：累计任务覆盖率对比.csv",
        "ylabel": T("累计任务覆盖率"),
        "metric_key": "cumulative_coverage_rate",

        # Cumulative indicator: no extra smoothing by default
        "is_cumulative": True,
        "smooth_window": 1,

        # Adjustable y-axis
        "ylim": (0.40, 1.02),
        "yticks": [0.4, 0.6, 0.8, 1.0],
        "minor_yticks": None,

        # Legend
        "legend_loc": "lower left",
        "legend_bbox_to_anchor": (0.02, 0.03),
        "legend_ncol": 2,
    },

    "fig34_supp": {
        "output_png": "experiment2_Fig. 34补充：累计任务完成率对比.png",
        "output_pdf": "experiment2_Fig. 34补充：累计任务完成率对比.pdf",
        "output_csv": "experiment2_Fig. 34补充：累计任务完成率对比.csv",
        "ylabel": T("累计任务完成率"),
        "metric_key": "cumulative_completion_rate",

        # Cumulative indicator: no extra smoothing by default
        "is_cumulative": True,
        "smooth_window": 1,

        # Adjustable y-axis
        "ylim": (0.25, 0.82),
        "yticks": [0.3, 0.5, 0.7, 0.8],
        "minor_yticks": None,

        # Legend
        "legend_loc": "lower left",
        "legend_bbox_to_anchor": (0.02, 0.03),
        "legend_ncol": 2,
    },
}


SUMMARY_CSV = "experiment2_Fig. 33-34：任务覆盖率与完成率最终统计.csv"

# Plot one point every POINT_EVERY rounds.
# If you want it closer to Fig.2, use 10.
# If these two curves are too sparse, use 6 or 8.
POINT_EVERY = 10

DEFAULT_PER_ROUND_SMOOTH_WINDOW = 5
DEFAULT_CUMULATIVE_SMOOTH_WINDOW = 1

TRUTHRIDE_LABEL = T("本发明方法")

FIGSIZE = (4.18, 2.70)
SAVE_PAD = 0.014


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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
    Sample points for plotting.

    The CSV saves the full series.
    The figure only draws sampled points connected by straight segments.
    """
    if step <= 1:
        return x, y

    x_sampled = x[::step]
    y_sampled = y[::step]

    if x_sampled[-1] != x[-1]:
        x_sampled.append(x[-1])
        y_sampled.append(y[-1])

    return x_sampled, y_sampled


def set_figure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Microsoft YaHei", "SimSun", "Times New Roman", "DejaVu Serif"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"],
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


def save_plot_data_csv(rounds, series_rows, labels, output_csv):
    headers = ["round"] + labels

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, round_id in enumerate(rounds):
            row = [round_id]
            row.extend(values[idx] for values in series_rows)
            writer.writerow(row)


def recompute_cumulative_task_metrics(round_results):
    """
    Recompute task coverage and completion indicators from primitive counters.

    Per-round:
        coverage_rate = num_covered / num_tasks
        completion_rate = num_completed / num_tasks

    Cumulative:
        cumulative_coverage_rate = sum(num_covered) / sum(num_tasks)
        cumulative_completion_rate = sum(num_completed) / sum(num_tasks)
    """
    cumulative_tasks = 0.0
    cumulative_covered = 0.0
    cumulative_completed = 0.0

    recomputed = []

    for entry in round_results:
        if int(entry.get("num_tasks", 0)) <= 0:
            continue

        item = dict(entry)

        num_tasks = float(item.get("num_tasks", 0))
        num_covered = float(item.get("num_covered", 0))
        num_completed = float(item.get("num_completed", 0))

        item["coverage_rate"] = round(
            num_covered / num_tasks if num_tasks > 0 else 0.0,
            4,
        )
        item["completion_rate"] = round(
            num_completed / num_tasks if num_tasks > 0 else 0.0,
            4,
        )

        cumulative_tasks += num_tasks
        cumulative_covered += num_covered
        cumulative_completed += num_completed

        item["cumulative_coverage_rate"] = round(
            cumulative_covered / cumulative_tasks if cumulative_tasks > 0 else 0.0,
            4,
        )
        item["cumulative_completion_rate"] = round(
            cumulative_completed / cumulative_tasks if cumulative_tasks > 0 else 0.0,
            4,
        )

        recomputed.append(item)

    return recomputed


def load_round_results(path):
    return recompute_cumulative_task_metrics(load_json(path))


def build_metric_curve(path, metric_key, smooth_window):
    round_results = load_round_results(path)

    rounds = [int(entry["round_id"]) for entry in round_results]
    values_raw = [float(entry.get(metric_key, 0.0)) for entry in round_results]

    values = moving_average(values_raw, smooth_window)

    return rounds, values


def apply_y_axis(ax, spec, y_values_all):
    """
    Apply adjustable y-axis settings.
    """
    if spec.get("ylim") is not None:
        ax.set_ylim(*spec["ylim"])

        if spec.get("yticks") is not None:
            ax.set_yticks(spec["yticks"])

        if spec.get("minor_yticks") is not None:
            ax.set_yticks(spec["minor_yticks"], minor=True)

    elif y_values_all:
        y_min = min(y_values_all)
        y_max = max(y_values_all)
        margin = max((y_max - y_min) * 0.08, 0.03)

        ax.set_ylim(
            max(0.0, y_min - margin),
            min(1.05, y_max + margin),
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def plot_figure(figure_key):
    spec = FIGURE_SPECS[figure_key]
    set_figure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    csv_rounds = None
    csv_series_rows = []
    labels = []
    legend_handles = []
    y_values_all = []

    for config in SERIES_CONFIG:
        labels.append(config["label"])

        rounds, values = build_metric_curve(
            config["path"],
            spec["metric_key"],
            spec["smooth_window"],
        )

        if csv_rounds is None:
            csv_rounds = rounds

        # Save full series to CSV.
        csv_series_rows.append(values)
        y_values_all.extend(values)

        # Plot sampled points only.
        x_plot, y_plot = sample_points(rounds, values, POINT_EVERY)

        is_truthride = config["label"] == TRUTHRIDE_LABEL

        ax.plot(
            x_plot,
            y_plot,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linestyle=config.get("linestyle", "-"),
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
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=config["label"],
                color=config["color"],
                linestyle=config.get("linestyle", "-"),
                linewidth=config["linewidth"],
                marker=config["marker"],
                markersize=config["markersize"],
                markerfacecolor=config["color"],
                markeredgecolor="#111111" if is_truthride else config["color"],
                markeredgewidth=0.82 if is_truthride else 0.60,
                alpha=config["alpha"],
                solid_capstyle="butt",
                solid_joinstyle="miter",
            )
        )

    ax.set_xlabel(T("轮次"))
    ax.set_ylabel(spec["ylabel"])

    ax.set_xlim(0, 144)
    ax.set_xticks(range(0, 141, 20))

    apply_y_axis(ax, spec, y_values_all)

    # Clean Fig.2/Fig.11-like background: light y-grid only.
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

    leg = ax.legend(
        handles=legend_handles,
        loc=spec.get("legend_loc", "lower left"),
        bbox_to_anchor=spec.get("legend_bbox_to_anchor", (0.02, 0.03)),
        ncol=spec.get("legend_ncol", 2),
        frameon=True,
        fancybox=False,
        framealpha=0.76,
        facecolor="white",
        edgecolor="#666666",
        fontsize=6.7,
        handlelength=1.50,
        handletextpad=0.28,
        columnspacing=0.62,
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

    fig.savefig(spec["output_png"], bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(spec["output_pdf"], bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)

    save_plot_data_csv(
        csv_rounds,
        csv_series_rows,
        labels,
        spec["output_csv"],
    )

    print(f"Saved {spec['output_png']}")
    print(f"Saved {spec['output_pdf']}")
    print(f"Saved {spec['output_csv']}")


def save_final_summary_csv():
    """
    Save final task coverage/completion statistics for paper tables.

    This file summarizes:
    - average per-round coverage rate
    - average per-round completion rate
    - final cumulative coverage rate
    - final cumulative completion rate
    """
    headers = [
        "algorithm",
        "avg_per_round_coverage_rate",
        "avg_per_round_completion_rate",
        "final_cumulative_coverage_rate",
        "final_cumulative_completion_rate",
    ]

    rows = []

    for config in SERIES_CONFIG:
        round_results = load_round_results(config["path"])

        if not round_results:
            rows.append([config["label"], 0.0, 0.0, 0.0, 0.0])
            continue

        avg_coverage = (
            sum(float(item.get("coverage_rate", 0.0)) for item in round_results)
            / len(round_results)
        )
        avg_completion = (
            sum(float(item.get("completion_rate", 0.0)) for item in round_results)
            / len(round_results)
        )

        final_entry = round_results[-1]
        final_cum_coverage = float(final_entry.get("cumulative_coverage_rate", 0.0))
        final_cum_completion = float(final_entry.get("cumulative_completion_rate", 0.0))

        rows.append(
            [
                config["label"],
                round(avg_coverage, 4),
                round(avg_completion, 4),
                round(final_cum_coverage, 4),
                round(final_cum_completion, 4),
            ]
        )

    with open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved {SUMMARY_CSV}")


def apply_window_overrides(window, cumulative_window):
    """
    Allow command-line overrides.

    Default:
    - per-round figures use trailing smoothing window = 5
    - cumulative figures use window = 1, namely no extra smoothing
    """
    if window is not None:
        FIGURE_SPECS["fig33"]["smooth_window"] = max(1, int(window))
        FIGURE_SPECS["fig34"]["smooth_window"] = max(1, int(window))

    if cumulative_window is not None:
        FIGURE_SPECS["fig33_supp"]["smooth_window"] = max(1, int(cumulative_window))
        FIGURE_SPECS["fig34_supp"]["smooth_window"] = max(1, int(cumulative_window))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_PER_ROUND_SMOOTH_WINDOW)
    parser.add_argument("--cumulative-window", type=int, default=DEFAULT_CUMULATIVE_SMOOTH_WINDOW)
    args = parser.parse_args()

    apply_window_overrides(args.window, args.cumulative_window)

    for figure_key in ("fig33", "fig34", "fig33_supp", "fig34_supp"):
        plot_figure(figure_key)

    save_final_summary_csv()


if __name__ == "__main__":
    main()
