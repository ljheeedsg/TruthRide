import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    "信任感知": "Trust-\nAware",
    "会员感知": "Member-\nAware",
    "可信": "Trusted",
    "未知": "Unknown",
    "低可信": "Malicious",
    "可信工人参与轮次": "trusted_participation_rounds",
    "未知工人参与轮次": "unknown_participation_rounds",
    "低可信工人参与轮次": "malicious_participation_rounds",
    "轮次": "Round",
    "每轮平均数据质量": "Per-Round Average Data Quality",
    "算法": "Algorithm",
    "平均参与轮次": "Avg. Participation Rounds",
    "任务覆盖率": "Task Coverage Rate",
    "任务完成率": "Task Completion Rate",
    "累计任务覆盖率": "Cumulative Task Coverage Rate",
    "累计任务完成率": "Cumulative Task Completion Rate",
}


def T(text):
    return EN_LABELS.get(text, text) if PLOT_LANG == "en" else text



# ============================================================
# Fig. 18: 工人参与轮次长度
#
# Unified style with revised Fig.2 / Fig.11:
# - white background
# - clean axes
# - light y-grid only
# - vivid color palette inspired by reference grouped-bar figures
# - compact in-panel legend
# - uncertainty bars
# - computation logic unchanged
# ============================================================

WORKER_OPTIONS_FILE = "experiment2_worker_options.json"

OUTPUT_PNG = "experiment2_Fig. 18：Worker Participation Length.png"
OUTPUT_PDF = "experiment2_Fig. 18：Worker Participation Length.pdf"
OUTPUT_CSV = "experiment2_Fig. 18：Worker Participation Length.csv"


# -----------------------------
# Legend settings
# -----------------------------
LEGEND_MODE = "manual"
LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.015, 0.985)
LEGEND_NCOL = 1


# -----------------------------
# Axis / figure settings
# -----------------------------
FIGSIZE = (4.18, 2.70)   # exactly align with Fig.11 for one-row layout
SAVE_PAD = 0.014

YLIM_MODE = "manual"
YLIM = (0, 9.8)
YTICKS = [0, 2, 4, 6, 8]
MINOR_YTICKS = [1, 3, 5, 7, 9]

BAR_WIDTH = 0.18

# Options: "std", "sem", "ci95"
# sem is cleaner visually; ci95 is more conservative but may look heavier.
ERROR_MODE = "sem"


XTICK_LABEL_MAP = {
    T("随机调度方法"): T("随机调度方法"),
    T("探索优先调度方法"): T("探索优先\n调度方法"),
    T("质量估计调度方法"): T("质量估计\n调度方法"),
    T("本发明方法"): T("本发明方法"),
    "Trust-Aware": T("信任感知"),
    "Membership-Aware": T("会员感知"),
    "TruthRide": T("本发明方法"),
}


ALGORITHM_CONFIG = [
    {
        "label": T("随机调度方法"),
        "csv_key": T("随机调度方法"),
        "path": "experiment2_random_longrun_round_results_all_runs.json",
    },
    {
        "label": T("探索优先调度方法"),
        "csv_key": T("探索优先调度方法"),
        "path": "experiment2_epsilon_first_longrun_round_results_all_runs.json",
    },
    {
        "label": T("质量估计调度方法"),
        "csv_key": T("质量估计调度方法"),
        "path": "experiment2_cmab_longrun_round_results_all_runs.json",
    },
    {
        "label": "Trust-Aware",
        "csv_key": "Trust-Aware",
        "path": "experiment2_cmab_trust_round_results_all_runs.json",
    },
    {
        "label": "Membership-Aware",
        "csv_key": "Membership-Aware",
        "path": "experiment2_cmab_trust_pgrd_round_results_all_runs.json",
    },
    {
        "label": T("本发明方法"),
        "csv_key": T("本发明方法"),
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results_all_runs.json",
    },
]


# Bright but still paper-friendly colors.
# Inspired by the reference style:
# cyan / green / red are more vivid and readable.
CATEGORY_CONFIG = [
    {
        "key": "trusted",
        "label": T("可信"),
        "csv_key": T("可信工人参与轮次"),
        "color": "#00b050",        # vivid green
        "hatch": "///",
        "error_color": "#007a38",
    },
    {
        "key": "unknown",
        "label": T("未知"),
        "csv_key": T("未知工人参与轮次"),
        "color": "#00bcd4",        # vivid cyan, close to reference cyan
        "hatch": "++",
        "error_color": "#008c9e",
    },
    {
        "key": "malicious",
        "label": T("低可信"),
        "csv_key": T("低可信工人参与轮次"),
        "color": "#e41a1c",        # vivid red
        "hatch": "--",
        "error_color": "#a50f15",
    },
]


def load_worker_categories(path):
    with Path(path).open("r", encoding="utf-8") as f:
        worker_options = json.load(f)

    worker_category_map = {}

    for worker in worker_options.values():
        worker_category_map[int(worker["worker_id"])] = worker["init_category"]

    return worker_category_map


def load_all_runs(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_run_participation_means(run_payload, worker_category_map):
    """
    For one run:
    1. Count how many rounds each worker is selected.
    2. Group workers by initial category.
    3. Compute average participation length for each category.
    """
    selected_counter = Counter()

    for round_result in run_payload["round_results"]:
        for worker_id in round_result.get("selected_workers", []):
            selected_counter[int(worker_id)] += 1

    grouped_values = defaultdict(list)

    for worker_id, category in worker_category_map.items():
        grouped_values[category].append(selected_counter.get(worker_id, 0))

    return {
        category: round(float(np.mean(grouped_values[category])), 4)
        if grouped_values[category] else 0.0
        for category in grouped_values
    }


def compute_algorithm_participation(path, worker_category_map):
    """
    For one algorithm:
    1. Compute category-level participation mean for each run.
    2. Aggregate across runs.
    3. Return mean, std, sem, ci95.

    Note:
    This figure consistently uses workers' initial categories for all algorithms.
    This is a posterior analysis and does not imply that all algorithms use
    category information during recruitment.
    """
    all_runs_payload = load_all_runs(path)

    per_run_means = []

    for run_payload in all_runs_payload:
        per_run_means.append(
            compute_run_participation_means(run_payload, worker_category_map)
        )

    aggregated = {}

    for category in ["trusted", "unknown", "malicious"]:
        values = np.array(
            [run_mean.get(category, 0.0) for run_mean in per_run_means],
            dtype=float,
        )

        mean = float(np.mean(values)) if len(values) else 0.0
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        ci95 = 1.96 * sem

        aggregated[category] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "sem": round(sem, 4),
            "ci95": round(ci95, 4),
            "n_runs": int(len(values)),
        }

    return aggregated


def save_plot_data_csv(rows):
    headers = ["algorithm"]

    for config in CATEGORY_CONFIG:
        csv_key = config["csv_key"]
        headers.extend(
            [
                csv_key,
                f"{csv_key}_std",
                f"{csv_key}_sem",
                f"{csv_key}_ci95",
                f"{csv_key}_n_runs",
            ]
        )

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            out_row = [row["algorithm"]]

            for config in CATEGORY_CONFIG:
                stats = row[config["key"]]
                out_row.extend(
                    [
                        stats["mean"],
                        stats["std"],
                        stats["sem"],
                        stats["ci95"],
                        stats["n_runs"],
                    ]
                )

            writer.writerow(out_row)


def set_figure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Microsoft YaHei", "SimSun", "Times New Roman", "DejaVu Serif"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"],
            "mathtext.fontset": "stix",

            # Match previous figures more closely.
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.7,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,

            # Slightly clearer hatch, similar to the reference style.
            "hatch.linewidth": 0.48,
            "axes.unicode_minus": False,
        }
    )


def get_error_value(stats):
    if ERROR_MODE == "std":
        return stats["std"]
    if ERROR_MODE == "sem":
        return stats["sem"]
    return stats["ci95"]


def plot_figure(rows):
    set_figure_style()

    algorithms = [row["algorithm"] for row in rows]
    x = np.arange(len(algorithms))
    xtick_labels = [XTICK_LABEL_MAP.get(name, name) for name in algorithms]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    all_upper_values = []

    for idx, category_config in enumerate(CATEGORY_CONFIG):
        offsets = x + (idx - 1) * BAR_WIDTH

        values = [row[category_config["key"]]["mean"] for row in rows]
        errors = [get_error_value(row[category_config["key"]]) for row in rows]

        all_upper_values.extend(
            [value + error for value, error in zip(values, errors)]
        )

        bars = ax.bar(
            offsets,
            values,
            width=BAR_WIDTH,
            label=category_config["label"],
            color=category_config["color"],
            edgecolor="#222222",
            linewidth=0.48,
            hatch=category_config["hatch"],
            alpha=0.98,
            yerr=errors,
            capsize=2.4,
            error_kw={
                "elinewidth": 0.70,
                "capthick": 0.70,
                "ecolor": category_config["error_color"],
                "alpha": 0.90,
                "zorder": 4,
            },
            zorder=3,
        )

        for bar, value, error in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + error + 0.12,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#222222",
                zorder=5,
            )

    ax.set_xlabel(T("算法"))
    ax.set_ylabel(T("平均参与轮次"))

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0)

    if YLIM_MODE == "manual":
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
        ax.set_yticks(MINOR_YTICKS, minor=True)
    elif all_upper_values:
        y_min = 0.0
        y_max = max(all_upper_values)
        margin = max(y_max * 0.15, 0.8)
        ax.set_ylim(y_min, y_max + margin)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Match previous figures: clean white background + light y-grid.
    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.50,
        color="#b5b5b5",
        alpha=0.28,
    )
    ax.grid(False, which="minor")
    ax.grid(False, axis="x")

    # To align with the previous line figures, use only left/bottom axes.
    # If you prefer the framed bar-chart style, see the alternative block below.
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=1.8,
        width=0.55,
        color="#444444",
    )

    # If you want full-frame style instead, replace the spine/tick block above with:
    #
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    #     spine.set_linewidth(0.80)
    #     spine.set_color("#444444")
    #
    # ax.tick_params(
    #     axis="both",
    #     which="major",
    #     direction="in",
    #     top=True,
    #     right=True,
    #     length=3.0,
    #     width=0.75,
    #     pad=2.0,
    #     color="#444444",
    # )
    # ax.tick_params(
    #     axis="both",
    #     which="minor",
    #     direction="in",
    #     top=True,
    #     right=True,
    #     length=1.8,
    #     width=0.55,
    #     color="#444444",
    # )

    legend_kwargs = {
        "loc": LEGEND_LOC if LEGEND_MODE == "manual" else "best",
        "ncol": LEGEND_NCOL,
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.78,
        "facecolor": "white",
        "edgecolor": "#666666",
        "handlelength": 1.45,
        "handletextpad": 0.32,
        "columnspacing": 0.60,
        "labelspacing": 0.20,
        "borderpad": 0.24,
        "borderaxespad": 0.30,
    }

    if LEGEND_MODE == "manual" and LEGEND_BBOX_TO_ANCHOR is not None:
        legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR

    leg = ax.legend(**legend_kwargs)
    leg.get_frame().set_linewidth(0.45)

    # Match Fig.11 axis box placement for one-row composition.
    fig.subplots_adjust(
        left=0.155,
        right=0.985,
        bottom=0.17,
        top=0.965,
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)


def main():
    dataset = os.environ.get("TRUTHRIDE_DATASET", "beijing")
    worker_options_file = Path(__file__).resolve().parent / "data" / dataset / WORKER_OPTIONS_FILE
    worker_category_map = load_worker_categories(worker_options_file)

    rows = []

    for algorithm_config in ALGORITHM_CONFIG:
        averaged = compute_algorithm_participation(
            algorithm_config["path"],
            worker_category_map,
        )

        rows.append(
            {
                "algorithm": algorithm_config["csv_key"],
                "trusted": averaged["trusted"],
                "unknown": averaged["unknown"],
                "malicious": averaged["malicious"],
            }
        )

    save_plot_data_csv(rows)
    plot_figure(rows)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
