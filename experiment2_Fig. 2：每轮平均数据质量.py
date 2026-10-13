import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
# Fig. 2: 每轮平均数据质量
# Style aligned with Fig. 11:
# - keep per-round metric with trailing moving average
# - sampled points connected by straight line segments
# - white background
# - light y-grid only
# - in-panel legend
# - TruthRide highlighted
# ============================================================

SERIES_CONFIG = [
    {
        "label": T("随机调度方法"),
        "csv_key": "random",
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
        "label": T("探索优先调度方法"),
        "csv_key": "epsilon_first",
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
        "label": T("质量估计调度方法"),
        "csv_key": "cmab",
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
        "label": T("本发明方法"),
        "csv_key": "truthride",
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


OUTPUT_PNG = "experiment2_Fig. 2：每轮平均数据质量.png"
OUTPUT_PDF = "experiment2_Fig. 2：每轮平均数据质量.pdf"
OUTPUT_CSV = "experiment2_Fig. 2：每轮平均数据质量.csv"

METRIC_KEY = "avg_quality"

# Keep Fig.2 original logic: trailing moving average + sampled line.
SMOOTH_WINDOW = 10
POINT_EVERY = 10

FIGSIZE = (4.18, 2.70)
SAVE_PAD = 0.014

YLIM = (0.2, 0.86)

LEGEND_LOC = "lower left"
LEGEND_BBOX_TO_ANCHOR = (0.02, 0.03)
LEGEND_NCOL = 2
LEGEND_FONT_SIZE = 6.7
LEGEND_HANDLE_LENGTH = 1.50
LEGEND_HANDLE_TEXT_PAD = 0.28
LEGEND_COLUMN_SPACING = 0.62

TRUTHRIDE_LABEL = T("本发明方法")


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

    This keeps the first point unchanged and avoids using future data.
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


def plot_figure():
    set_figure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    csv_rounds = set()
    csv_series_rows = []
    legend_handles = []

    for config in SERIES_CONFIG:
        records = load_round_results(config["path"])

        rounds = [int(item["round_id"]) for item in records]
        values_raw = [float(item[METRIC_KEY]) for item in records]
        values_smooth = moving_average(values_raw, SMOOTH_WINDOW)

        csv_rounds.update(rounds)
        csv_series_rows.append(dict(zip(rounds, values_smooth)))

        x_plot, y_plot = sample_points(rounds, values_smooth, POINT_EVERY)

        is_truthride = config["label"] == TRUTHRIDE_LABEL

        ax.plot(
            x_plot,
            y_plot,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linestyle=config.get("linestyle", "-"),
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
            )
        )

    ax.set_xlabel(T("轮次"))
    ax.set_ylabel(T("每轮平均数据质量"))

    ax.set_xlim(0, 144)
    ax.set_ylim(*YLIM)

    ax.set_xticks(range(0, 141, 20))
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # Fig.11-like clean background: light y-grid only.
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

    # Fig.11-like light axes: remove top/right spines.
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

    # In-panel legend, same feeling as Fig.11.
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

    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.17, top=0.965)

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)

    save_plot_data_csv(sorted(csv_rounds), csv_series_rows)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()
