import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experiment2_paper_plot_style import (
    AX_BG,
    FIG2_SAVE_PAD,
    PAPER_BG,
    apply_axis_frame,
    get_line_path_effects,
    get_palette,
    get_series_emphasis,
    set_paper_line_rcparams,
    style_legend,
)
from experiment2_parameter_visual_refine import (
    add_compact_legend,
    compute_interval_means,
    plot_bar_line_per_round,
    plot_grouped_bar_line_intervals,
    plot_retention_curve,
    save_interval_plot_data_csv,
)
from experiment2_重构_main import run_experiment


# ============================================================
# Malicious-ratio robustness experiment
#
# Plotting style:
# - per-round figures follow Fig.2 style:
#   trailing moving average + sampled points + straight line segments
# - cumulative figures follow Fig.3 style:
#   no smoothing + sampled points + straight line segments
# - retention figure follows Fig.15 style:
#   no smoothing + no sampling + complete line + marker every few rounds
# - no curved line joins
# - hollow markers
# - unified SCI-style frame/grid/legend
# ============================================================


MALICIOUS_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
TRUSTED_RATIO = 0.20

STEP1_SEED = 10
STEP2_SEED = 100
STEP3_SEED = 100

RUNS_PER_RATIO = 10
RESULTS_BUNDLE_FILE = "experiment2_恶意比例鲁棒性_results.json"


# Ratio curves: low malicious ratio -> cool color; high malicious ratio -> red.
COLOR_LIST = get_palette(len(MALICIOUS_RATIOS))
MARKER_LIST = ["o", "^", "D", "s", "P"]


FIGURE_SPECS = {
    # ========================================================
    # Per-round figures: Fig.2 style
    # smoothing + sampling + straight line connection
    # ========================================================
    "fig20": {
        "output_png": "experiment2_Fig. 20：不同恶意比例下每轮平均数据质量.png",
        "output_pdf": "experiment2_Fig. 20：不同恶意比例下每轮平均数据质量.pdf",
        "output_csv": "experiment2_Fig. 20：不同恶意比例下每轮平均数据质量.csv",
        "ylabel": "Per-Round Avg. Quality",
        "metric_key": "avg_quality",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "per_round",
    },
    "fig21": {
        "output_png": "experiment2_Fig. 21：不同恶意比例下每轮平台效用.png",
        "output_pdf": "experiment2_Fig. 21：不同恶意比例下每轮平台效用.pdf",
        "output_csv": "experiment2_Fig. 21：不同恶意比例下每轮平台效用.csv",
        "ylabel": "Per-Round Utility",
        "metric_key": "platform_utility",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": None,
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "per_round",
    },
    "fig23": {
        "output_png": "experiment2_Fig. 23：不同恶意比例下任务完成率.png",
        "output_pdf": "experiment2_Fig. 23：不同恶意比例下任务完成率.pdf",
        "output_csv": "experiment2_Fig. 23：不同恶意比例下任务完成率.csv",
        "ylabel": "Task Completion Rate",
        "metric_key": "completion_rate",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "per_round",
    },

    # ========================================================
    # Retention figure: keep same style as other per_round figures for alignment
    # ========================================================
    "fig22": {
        "output_png": "experiment2_Fig. 22：不同恶意比例下种子可信工人留存率.png",
        "output_pdf": "experiment2_Fig. 22：不同恶意比例下种子可信工人留存率.pdf",
        "output_csv": "experiment2_Fig. 22：不同恶意比例下种子可信工人留存率.csv",
        "ylabel": "Seed Trusted Retention Rate",
        "metric_key": "retention_rate",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "upper right",
        "legend_ncol": 2,
        "style_type": "per_round",
    },

    # ========================================================
    # Cumulative figures: Fig.3 style
    # no smoothing + sampling + straight line connection
    # ========================================================
    "fig20_supp": {
        "output_png": "experiment2_Fig. 20补充：不同恶意比例下累计平均数据质量.png",
        "output_pdf": "experiment2_Fig. 20补充：不同恶意比例下累计平均数据质量.pdf",
        "output_csv": "experiment2_Fig. 20补充：不同恶意比例下累计平均数据质量.csv",
        "ylabel": "Cumulative Avg. Quality",
        "metric_key": "cumulative_avg_quality",
        "smooth_window": 1,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "lower right",
        "legend_ncol": 2,
        "style_type": "cumulative",
    },
    "fig21_supp": {
        "output_png": "experiment2_Fig. 21补充：不同恶意比例下累计平台效用.png",
        "output_pdf": "experiment2_Fig. 21补充：不同恶意比例下累计平台效用.pdf",
        "output_csv": "experiment2_Fig. 21补充：不同恶意比例下累计平台效用.csv",
        "ylabel": "Cumulative Utility",
        "metric_key": "cumulative_platform_utility",
        "smooth_window": 1,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": None,
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "cumulative",
    },
    "fig23_supp": {
        "output_png": "experiment2_Fig. 23补充：不同恶意比例下累计任务完成率.png",
        "output_pdf": "experiment2_Fig. 23补充：不同恶意比例下累计任务完成率.pdf",
        "output_csv": "experiment2_Fig. 23补充：不同恶意比例下累计任务完成率.csv",
        "ylabel": "Cumulative Completion Rate",
        "metric_key": "cumulative_completion_rate",
        "smooth_window": 1,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "lower right",
        "legend_ncol": 2,
        "style_type": "cumulative",
    },
}


def save_json(payload, path):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def run_command(args):
    print("\n>>>", " ".join(str(arg) for arg in args))
    subprocess.run([str(arg) for arg in args], check=True)


def moving_average(values, window):
    """
    Trailing moving average.

    For round t, it only uses current and previous values:
    y_smooth(t) = average[y(t-window+1), ..., y(t)].

    This follows the Fig.2 style and avoids using future information.
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
    Sample points only for plotting.

    The CSV still stores the full data series.
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
    set_paper_line_rcparams()


def compute_adaptive_ylim(values, value_bounds=None):
    if not values:
        return None

    y_min = min(values)
    y_max = max(values)

    if value_bounds is not None:
        lower_bound, upper_bound = value_bounds

        span = max(y_max - y_min, 0.08)
        margin = max(span * 0.08, 0.015)

        y_min = max(lower_bound, y_min - margin)
        y_max = min(upper_bound, y_max + margin)

        if y_max - y_min < 0.05:
            center = (y_min + y_max) / 2.0
            half_span = 0.025
            y_min = max(lower_bound, center - half_span)
            y_max = min(upper_bound, center + half_span)

        return round(y_min, 4), round(y_max, 4)

    span = y_max - y_min
    margin = max(span * 0.08, 20.0)

    return round(y_min - margin, 4), round(y_max + margin, 4)


def ratio_label(malicious_ratio):
    return f"{int(round(malicious_ratio * 100))}%"


def find_preferred_default_label(labels, fallback_label=None):
    normalized_map = {}

    for label in labels:
        normalized = (
            str(label)
            .replace(" ", "")
            .replace("$", "")
            .replace("{", "")
            .replace("}", "")
            .replace("\\", "")
            .lower()
        )
        normalized_map[normalized] = label

    preferred_keys = (
        "m_v=3",
        "mv=3",
        "m=3",
        "3",
    )

    for key in preferred_keys:
        if key in normalized_map:
            return normalized_map[key]

    if fallback_label in labels:
        return fallback_label

    return labels[-1] if labels else fallback_label


def save_plot_data_csv(rounds, series_rows, labels, output_csv):
    headers = ["round"] + labels

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, round_id in enumerate(rounds):
            row = [round_id]
            row.extend(values[idx] for values in series_rows)
            writer.writerow(row)


def save_interval_plot_data_csv(interval_labels, series_map, output_csv):
    headers = ["interval"] + list(series_map.keys())

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, interval_label in enumerate(interval_labels):
            row = [interval_label]
            for label in series_map.keys():
                row.append(series_map[label][idx])
            writer.writerow(row)


def compute_interval_means(rounds, values, interval_size=20):
    if not rounds or not values:
        return [], []

    max_round = max(int(round_id) for round_id in rounds)
    interval_labels = []
    interval_means = []

    start_round = 1
    while start_round <= max_round:
        end_round = min(start_round + interval_size - 1, max_round)
        bucket = [
            float(value)
            for round_id, value in zip(rounds, values)
            if start_round <= int(round_id) <= end_round
        ]

        if bucket:
            interval_labels.append(f"{start_round}-{end_round}")
            interval_means.append(round(sum(bucket) / len(bucket), 4))

        start_round = end_round + 1

    return interval_labels, interval_means


def plot_grouped_bar_line_utility(ax, interval_labels, interval_values_dict, default_label="40%"):
    color_map = {
        "10%": "#1f77b4",
        "20%": "#ff7f0e",
        "30%": "#2ca02c",
        "40%": "#d62728",
        "50%": "#9467bd",
    }
    hatch_map = {
        "10%": "/",
        "20%": "\\\\",
        "30%": "x",
        "40%": ".",
        "50%": "///",
    }
    marker_map = {
        "10%": "o",
        "20%": "s",
        "30%": "^",
        "40%": "D",
        "50%": "v",
    }

    labels = list(interval_values_dict.keys())
    x_positions = list(range(len(interval_labels)))
    total_width = 0.82
    bar_width = total_width / max(1, len(labels))

    all_values = [value for values in interval_values_dict.values() for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    y_span = max(y_max - y_min, 1.0)

    legend_handles = []

    for idx, label in enumerate(labels):
        values = interval_values_dict[label]
        color = color_map.get(label, COLOR_LIST[idx % len(COLOR_LIST)])
        hatch = hatch_map.get(label, "/")
        marker = marker_map.get(label, "o")
        is_default = label == default_label
        offset = (idx - (len(labels) - 1) / 2.0) * bar_width

        ax.bar(
            [x_pos + offset for x_pos in x_positions],
            values,
            width=bar_width * 0.92,
            color=color,
            alpha=0.80 if is_default else 0.68,
            hatch=hatch,
            edgecolor="#444444",
            linewidth=0.45,
            zorder=2,
        )

        ax.plot(
            [x_pos + offset for x_pos in x_positions],
            values,
            color=color,
            linewidth=2.4 if is_default else 1.95,
            marker=marker,
            markersize=4.6 if is_default else 3.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            zorder=6 if is_default else 4,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=color,
                linewidth=2.4 if is_default else 1.95,
                marker=marker,
                markersize=4.6 if is_default else 3.8,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.85,
            )
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(interval_labels)
    ax.set_xlabel("Round interval")
    ax.set_ylabel("Average Per-Round Utility")
    ax.set_title("Malicious ratio: utility", fontsize=10, pad=5.0)

    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="#b5b5b5",
        alpha=0.30,
    )
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(axis="both", which="major", direction="out", length=3.0, width=0.75, pad=2.0)

    if y_min < 0 < y_max:
        ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.70, zorder=1)

    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.10 * y_span)

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=8,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def compact_round_entry(entry):
    return {
        "round_id": int(entry["round_id"]),
        "num_tasks": int(entry.get("num_tasks", 0)),
        "num_completed": int(entry.get("num_completed", 0)),
        "avg_quality": round(float(entry.get("avg_quality", 0.0)), 4),
        "platform_utility": round(float(entry.get("platform_utility", 0.0)), 4),
        "completion_rate": round(float(entry.get("completion_rate", 0.0)), 4),
        "cumulative_avg_quality": round(float(entry.get("cumulative_avg_quality", 0.0)), 4),
        "cumulative_platform_utility": round(
            float(entry.get("cumulative_platform_utility", 0.0)), 4
        ),
        "cumulative_completion_rate": round(
            float(entry.get("cumulative_completion_rate", 0.0)), 4
        ),
    }


def compact_retention_entry(entry, seed_trusted_ids, left_seed_ids, seed_count):
    left_ids = {int(worker_id) for worker_id in entry.get("left_worker_ids_this_round", [])}
    left_seed_ids.update(left_ids & seed_trusted_ids)

    retained = seed_count - len(left_seed_ids)

    return {
        "round_id": int(entry["round_id"]),
        "num_tasks": int(entry.get("num_tasks", 0)),
        "retention_rate": round(retained / seed_count, 4),
    }


def build_compact_round_metrics(round_results):
    return [compact_round_entry(entry) for entry in round_results]


def build_compact_retention_curve(all_runs_payload, seed_trusted_worker_ids):
    seed_trusted_ids = {int(worker_id) for worker_id in seed_trusted_worker_ids}
    seed_count = max(1, len(seed_trusted_ids))

    per_run_curves = []
    round_axis = None

    for run_payload in all_runs_payload:
        round_results = [
            entry for entry in run_payload["round_results"]
            if int(entry.get("num_tasks", 0)) > 0
        ]

        compact_curve = []
        left_seed_ids = set()

        for entry in round_results:
            compact_curve.append(
                compact_retention_entry(entry, seed_trusted_ids, left_seed_ids, seed_count)
            )

        rounds = [entry["round_id"] for entry in compact_curve]

        if round_axis is None:
            round_axis = rounds

        per_run_curves.append(compact_curve)

    averaged_curve = []

    for idx, round_id in enumerate(round_axis or []):
        averaged_curve.append(
            {
                "round_id": round_id,
                "num_tasks": per_run_curves[0][idx]["num_tasks"],
                "retention_rate": round(
                    sum(curve[idx]["retention_rate"] for curve in per_run_curves)
                    / len(per_run_curves),
                    4,
                ),
            }
        )

    return averaged_curve


def compact_item_payload(item):
    if "round_metrics" in item and "retention_curve" in item:
        return {
            "malicious_ratio": float(item["malicious_ratio"]),
            "ratio_tag": int(item["ratio_tag"]),
            "round_metrics": item["round_metrics"],
            "retention_curve": item["retention_curve"],
        }

    round_results = item["round_results"]
    all_runs_payload = item["all_runs_payload"]
    seed_trusted_worker_ids = item["seed_trusted_worker_ids"]

    return {
        "malicious_ratio": float(item["malicious_ratio"]),
        "ratio_tag": int(item["ratio_tag"]),
        "round_metrics": build_compact_round_metrics(round_results),
        "retention_curve": build_compact_retention_curve(
            all_runs_payload,
            seed_trusted_worker_ids,
        ),
    }


def compact_bundle(bundle):
    return {
        "experiment_name": bundle["experiment_name"],
        "bundle_format": "compact_round_metrics_v1",
        "trusted_ratio": float(bundle["trusted_ratio"]),
        "malicious_ratios": [float(ratio) for ratio in bundle["malicious_ratios"]],
        "runs_per_ratio": int(bundle["runs_per_ratio"]),
        "items": [compact_item_payload(item) for item in bundle["items"]],
    }


def load_round_results_from_item(item):
    round_entries = item.get("round_metrics", item.get("round_results", []))

    return [entry for entry in round_entries if int(entry.get("num_tasks", 0)) > 0]


def enrich_round_metrics(round_results):
    cumulative_num_tasks = 0
    cumulative_num_completed = 0
    cumulative_platform_utility = 0.0
    cumulative_quality_weighted_sum = 0.0

    enriched = []

    for entry in round_results:
        enriched_entry = dict(entry)

        num_tasks = int(enriched_entry.get("num_tasks", 0))
        completion_rate = float(enriched_entry.get("completion_rate", 0.0))

        num_completed = enriched_entry.get("num_completed")

        if num_completed is None:
            num_completed = int(round(num_tasks * completion_rate))
        else:
            num_completed = int(num_completed)

        avg_quality = float(enriched_entry.get("avg_quality", 0.0))
        platform_utility = float(enriched_entry.get("platform_utility", 0.0))

        cumulative_num_tasks += num_tasks
        cumulative_num_completed += num_completed
        cumulative_platform_utility += platform_utility
        cumulative_quality_weighted_sum += avg_quality * num_completed

        enriched_entry["num_completed"] = num_completed
        enriched_entry["cumulative_completion_rate"] = round(
            cumulative_num_completed / cumulative_num_tasks
            if cumulative_num_tasks > 0 else 0.0,
            4,
        )
        enriched_entry["cumulative_platform_utility"] = round(cumulative_platform_utility, 4)
        enriched_entry["cumulative_avg_quality"] = round(
            cumulative_quality_weighted_sum / cumulative_num_completed
            if cumulative_num_completed > 0 else 0.0,
            4,
        )

        enriched.append(enriched_entry)

    return enriched


def compute_seed_retention_curve(all_runs_payload, seed_trusted_ids):
    seed_trusted_ids = {int(worker_id) for worker_id in seed_trusted_ids}
    seed_count = max(1, len(seed_trusted_ids))

    per_run_curves = []
    round_axis = None

    for run_payload in all_runs_payload:
        round_results = [
            entry for entry in run_payload["round_results"]
            if int(entry.get("num_tasks", 0)) > 0
        ]

        rounds = [int(entry["round_id"]) for entry in round_results]

        if round_axis is None:
            round_axis = rounds

        left_seed_ids = set()
        retention_values = []

        for entry in round_results:
            left_ids = {
                int(worker_id)
                for worker_id in entry.get("left_worker_ids_this_round", [])
            }
            left_seed_ids.update(left_ids & seed_trusted_ids)

            retained = seed_count - len(left_seed_ids)
            retention_values.append(round(retained / seed_count, 4))

        per_run_curves.append(retention_values)

    averaged_curve = []

    for idx in range(len(per_run_curves[0])):
        averaged_curve.append(
            round(sum(curve[idx] for curve in per_run_curves) / len(per_run_curves), 4)
        )

    return round_axis, averaged_curve


def build_metric_curve(item, metric_key, smooth_window):
    if metric_key == "retention_rate":
        retention_entries = [
            entry for entry in item.get("retention_curve", [])
            if int(entry.get("num_tasks", 0)) > 0
        ]
        rounds = [int(entry["round_id"]) for entry in retention_entries]
        values = [float(entry["retention_rate"]) for entry in retention_entries]
        return rounds, moving_average(values, smooth_window)

    round_results = enrich_round_metrics(load_round_results_from_item(item))
    rounds = [int(entry["round_id"]) for entry in round_results]
    values = [float(entry[metric_key]) for entry in round_results]

    return rounds, moving_average(values, smooth_window)


def build_retention_curve(item, smooth_window):
    if "retention_curve" in item:
        retention_entries = [
            entry for entry in item["retention_curve"]
            if int(entry.get("num_tasks", 0)) > 0
        ]

        rounds = [int(entry["round_id"]) for entry in retention_entries]
        values = [float(entry["retention_rate"]) for entry in retention_entries]

        return rounds, moving_average(values, smooth_window)

    rounds, values = compute_seed_retention_curve(
        item["all_runs_payload"],
        item["seed_trusted_worker_ids"],
    )

    return rounds, moving_average(values, smooth_window)


def apply_common_axis_style(ax):
    apply_axis_frame(ax)


def apply_y_axis(ax, spec, y_values_all):
    if spec.get("ylim") is not None:
        ax.set_ylim(*spec["ylim"])
    else:
        ylim = compute_adaptive_ylim(
            y_values_all,
            spec.get("value_bounds"),
        )
        if ylim is not None:
            ax.set_ylim(*ylim)

    if spec.get("yticks") is not None:
        ax.set_yticks(spec["yticks"])

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Utility figures can have large values, so keep scientific notation.
    # Ratio/quality/completion/retention figures remain normal decimal format.
    if spec.get("value_bounds") is None:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)


def add_legend(ax, legend_handles, spec):
    legend_kwargs = {
        "handles": legend_handles,
        "loc": spec.get("legend_loc", "best"),
        "ncol": spec.get("legend_ncol", 2),
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.72,
        "facecolor": "white",
        "edgecolor": "#666666",
        "handlelength": 1.45,
        "handletextpad": 0.28,
        "columnspacing": 0.58,
        "labelspacing": 0.18,
        "borderpad": 0.24,
        "borderaxespad": 0.35,
    }

    leg = ax.legend(**legend_kwargs)
    style_legend(leg)


def add_final_retention_inset(ax, series_dict, colors, default_label):
    default_values = series_dict.get(default_label, ([], []))[1]
    default_final = default_values[-1] if default_values else 0.0
    default_end_y = default_values[-1] if default_values else 0.0

    inset_anchor = "upper right"
    if default_end_y >= 0.72:
        inset_anchor = "lower left"

    if inset_anchor == "upper right":
        inset = inset_axes(ax, width="33%", height="42%", loc="upper right", borderpad=1.1)
    else:
        inset = inset_axes(ax, width="33%", height="42%", loc="lower left", borderpad=1.0)

    labels = list(series_dict.keys())
    final_values = []
    bar_colors = []
    edge_colors = []
    bar_linewidths = []

    for label in labels:
        _, values = series_dict[label]
        final_value = float(values[-1]) if values else 0.0
        final_values.append(final_value)
        bar_colors.append(colors.get(label, "#888888"))
        is_default = label == default_label
        edge_colors.append("#333333" if is_default else "#666666")
        bar_linewidths.append(1.0 if is_default else 0.65)

    x_pos = list(range(len(labels)))
    bars = inset.bar(
        x_pos,
        final_values,
        color=bar_colors,
        alpha=0.92,
        edgecolor=edge_colors,
        linewidth=bar_linewidths,
        zorder=3,
    )

    for idx, bar in enumerate(bars):
        if labels[idx] == default_label:
            bar.set_alpha(1.0)

    inset.set_title("Final retention", fontsize=6.9, pad=1.6)
    inset.set_ylim(0.0, max(1.02, default_final + 0.08, max(final_values, default=0.0) + 0.08))
    inset.set_xticks(x_pos)
    inset.set_xticklabels(labels, fontsize=6.1)
    inset.set_yticks([0.0, 0.5, 1.0])
    inset.tick_params(axis="y", labelsize=6.1, length=2.0, width=0.6, pad=1.1)
    inset.tick_params(axis="x", labelsize=6.1, length=1.8, width=0.55, pad=0.8)

    inset.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.45,
        color="#b8b8b8",
        alpha=0.28,
        zorder=0,
    )
    inset.grid(False, which="minor")
    inset.set_facecolor("white")

    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.7)
    inset.spines["bottom"].set_linewidth(0.7)
    inset.spines["left"].set_color("#555555")
    inset.spines["bottom"].set_color("#555555")


def plot_retention_curve_with_inset(
    ax,
    rounds,
    series_dict,
    default_label,
    title,
    ylabel="Seed Trusted Retention Rate",
):
    discrete_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#7f7f7f", "#d62728"]
    marker_cycle = ["o", "s", "^", "D", "v", "P"]

    labels = list(series_dict.keys())
    non_default_labels = [label for label in labels if label != default_label]

    color_map = {}
    marker_map = {}

    for idx, label in enumerate(non_default_labels):
        color_map[label] = discrete_colors[idx % (len(discrete_colors) - 1)]
        marker_map[label] = marker_cycle[idx % len(marker_cycle)]

    color_map[default_label] = "#8b1e1e"
    marker_map[default_label] = "P"

    legend_handles = []
    all_rounds = list(rounds or [])

    for label in labels:
        current_rounds, values = series_dict[label]
        is_default = label == default_label
        color = color_map.get(label, "#4d4d4d")
        marker = marker_map.get(label, "o")

        line = ax.plot(
            current_rounds,
            values,
            color=color,
            linestyle="-",
            linewidth=2.4 if is_default else 1.55,
            marker=marker,
            markersize=5.2 if is_default else 4.0,
            markevery=10 if is_default else 12,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.95 if is_default else 0.8,
            alpha=1.0 if is_default else 0.78,
            zorder=6 if is_default else 3,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )[0]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=color,
                linestyle="-",
                linewidth=2.4 if is_default else 1.55,
                marker=marker,
                markersize=5.2 if is_default else 4.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.95 if is_default else 0.8,
                alpha=1.0 if is_default else 0.78,
            )
        )
        if current_rounds:
            all_rounds.extend(int(round_id) for round_id in current_rounds)

    max_round = max(all_rounds) if all_rounds else 144

    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=8.8, pad=2.5)
    ax.set_xlim(0, max(144, max_round))
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(range(0, max(145, max_round + 1), 20))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.5,
        color="#b5b5b5",
        alpha=0.30,
    )
    ax.grid(False, which="minor")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(axis="both", which="major", direction="out", length=3.0, width=0.75, pad=2.0, labelsize=8.5)

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=min(5, max(1, len(labels))),
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    add_final_retention_inset(ax, series_dict, color_map, default_label)
    return legend_handles


def plot_figure(bundle, figure_key):
    spec = FIGURE_SPECS[figure_key]
    items = sorted(bundle["items"], key=lambda item: item["malicious_ratio"])

    set_figure_style()

    fig, ax = plt.subplots(figsize=(4.18, 2.70))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)

    if figure_key == "fig22":
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

    csv_rounds = None
    csv_series_rows = []
    labels = []
    y_values_all = []
    series_map = {}

    for idx, item in enumerate(items):
        label = ratio_label(item["malicious_ratio"])
        labels.append(label)

        rounds, values = build_metric_curve(
            item,
            spec["metric_key"],
            spec["smooth_window"],
        )

        if csv_rounds is None:
            csv_rounds = rounds

        csv_series_rows.append(values)
        series_map[label] = (rounds, values)

    ax.set_xlabel("Round")
    ax.set_ylabel(spec["ylabel"])

    if spec.get("style_type") != "per_round":
        ax.set_xlim(0, 144)
        ax.set_xticks(range(0, 141, 20))
        ax.set_xticks(range(0, 145, 10), minor=True)

    if spec.get("style_type") == "per_round":
        interval_labels = None
        interval_values_dict = {}
        for label in labels:
            rounds, values = series_map[label]
            current_interval_labels, interval_means = compute_interval_means(rounds, values, 20)
            if interval_labels is None:
                interval_labels = current_interval_labels
            interval_values_dict[label] = interval_means
            y_values_all.extend(interval_means)
        save_interval_plot_data_csv(interval_labels or [], interval_values_dict, spec["output_csv"])
        plot_grouped_bar_line_intervals(
            ax,
            interval_labels or [],
            interval_values_dict,
            default_label=ratio_label(0.40),
            ylabel=spec["ylabel"],
            title="",
            legend_inside=True,
            legend_loc=spec.get("legend_loc", "upper left"),
            legend_ncol=spec.get("legend_ncol", 2),
        )
        ax.set_title("")
    elif spec.get("style_type") == "retention":
        legend_handles = plot_retention_curve(
            ax,
            series_map,
            ratio_label(0.40),
            marker_cycle=MARKER_LIST,
            y_values_all=y_values_all,
        )
        apply_y_axis(ax, spec, y_values_all)
        add_compact_legend(ax, legend_handles, spec)
    else:
        legend_handles = []
        for idx, label in enumerate(labels):
            rounds, values = series_map[label]
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            marker = MARKER_LIST[idx % len(MARKER_LIST)]
            emphasis = get_series_emphasis(idx, len(items))
            x_plot, y_plot = sample_points(rounds, values, spec.get("point_every", 1))
            line = ax.plot(
                x_plot, y_plot, color=color, marker=marker, linestyle="-",
                linewidth=emphasis["linewidth"], markersize=emphasis["markersize"],
                markevery=spec.get("mark_every", 1), markerfacecolor=AX_BG,
                markeredgecolor=color, markeredgewidth=emphasis["markeredgewidth"],
                alpha=emphasis["alpha"], zorder=emphasis["zorder"],
                solid_capstyle="round", solid_joinstyle="round",
            )[0]
            line.set_path_effects(get_line_path_effects(emphasis["linewidth"]))
            y_values_all.extend(values)
            legend_handles.append(
                Line2D([0], [0], label=label, color=color, linestyle="-",
                    linewidth=emphasis["linewidth"], marker=marker,
                    markersize=emphasis["markersize"], markerfacecolor=AX_BG,
                    markeredgecolor=color, markeredgewidth=emphasis["markeredgewidth"],
                    alpha=emphasis["alpha"])
            )
        apply_y_axis(ax, spec, y_values_all)
        apply_common_axis_style(ax)
        add_legend(ax, legend_handles, spec)

    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.18, top=0.955)

    fig.savefig(spec["output_png"], bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    fig.savefig(spec["output_pdf"], bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    if figure_key == "fig22":
        refined_png = str(Path(spec["output_png"]).with_name(Path(spec["output_png"]).stem + "_refined.png"))
        refined_pdf = str(Path(spec["output_pdf"]).with_name(Path(spec["output_pdf"]).stem + "_refined.pdf"))
        fig.savefig(refined_png, dpi=600, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
        fig.savefig(refined_pdf, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    plt.close(fig)

    if spec.get("style_type") != "per_round":
        save_plot_data_csv(csv_rounds, csv_series_rows, labels, spec["output_csv"])

    print(f"Saved {spec['output_png']}")
    print(f"Saved {spec['output_pdf']}")
    print(f"Saved {spec['output_csv']}")
    if figure_key == "fig22":
        print(f"Saved {refined_png}")
        print(f"Saved {refined_pdf}")


def plot_all_figures(bundle):
    for figure_key in (
        "fig20",
        "fig21",
        "fig22",
        "fig23",
        "fig20_supp",
        "fig21_supp",
        "fig23_supp",
    ):
        plot_figure(bundle, figure_key)


def run_single_ratio_experiment(malicious_ratio, runs_per_ratio, workspace_root):
    ratio_tag = int(round(malicious_ratio * 100))

    print(f"\n========== Malicious Ratio {ratio_tag}% ==========")

    with tempfile.TemporaryDirectory(
        prefix=f"experiment2_malicious_{ratio_tag}_",
        dir=str(workspace_root),
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        vehicle_file = temp_dir / "vehicle.csv"
        vehicle_plot = temp_dir / "vehicle.png"
        vehicle_summary = temp_dir / "vehicle_summary.json"
        vehicle_summary_all = temp_dir / "vehicle_summary_all_runs.json"

        task_csv = temp_dir / "tasks.csv"
        task_json = temp_dir / "task_segments.json"
        task_plot = temp_dir / "tasks_distribution.png"
        task_summary = temp_dir / "tasks_summary.json"
        task_summary_all = temp_dir / "tasks_summary_all_runs.json"

        worker_json = temp_dir / "worker_options.json"
        worker_summary = temp_dir / "worker_summary.json"
        worker_summary_all = temp_dir / "worker_summary_all_runs.json"

        prefix = temp_dir / f"truthride_malicious_{ratio_tag}"

        run_command(
            [
                sys.executable,
                "experiment2_第1步产生车数据.py",
                "--output-seg", vehicle_file,
                "--output-plot", vehicle_plot,
                "--summary-file", vehicle_summary,
                "--all-runs-summary-file", vehicle_summary_all,
                "--trusted-ratio", TRUSTED_RATIO,
                "--malicious-ratio", malicious_ratio,
                "--seed", STEP1_SEED,
                "--runs", 1,
            ]
        )

        run_command(
            [
                sys.executable,
                "experiment2_第2步产生任务数据.py",
                "--vehicle-file", vehicle_file,
                "--task-csv", task_csv,
                "--task-json", task_json,
                "--plot-file", task_plot,
                "--summary-file", task_summary,
                "--all-runs-summary-file", task_summary_all,
                "--seed", STEP2_SEED,
                "--runs", 1,
            ]
        )

        run_command(
            [
                sys.executable,
                "experiment2_第3步产生工人可选项.py",
                "--vehicle-file", vehicle_file,
                "--task-file", task_csv,
                "--output-json", worker_json,
                "--summary-file", worker_summary,
                "--all-runs-summary-file", worker_summary_all,
                "--seed", STEP3_SEED,
                "--runs", 1,
            ]
        )

        run_experiment(
            "lgsc",
            overrides={
                "WORKER_OPTIONS_FILE": str(worker_json),
                "NUM_EXPERIMENT_RUNS": runs_per_ratio,
                "prefix": str(prefix),
            },
        )

        worker_options = load_json(worker_json)

        seed_trusted_worker_ids = sorted(
            int(worker["worker_id"])
            for worker in worker_options.values()
            if worker.get("init_category") == "trusted"
        )

        return compact_item_payload(
            {
                "malicious_ratio": malicious_ratio,
                "ratio_tag": ratio_tag,
                "seed_trusted_worker_ids": seed_trusted_worker_ids,
                "round_results": load_json(f"{prefix}_round_results.json"),
                "all_runs_payload": load_json(f"{prefix}_round_results_all_runs.json"),
            }
        )


def run_full_experiment(ratios=None, runs_per_ratio=RUNS_PER_RATIO, keep_temp=False):
    workspace_root = Path.cwd()
    ratios = ratios or MALICIOUS_RATIOS

    items = []
    temp_root = workspace_root / "experiment2_恶意比例鲁棒性_tmp"

    try:
        if keep_temp:
            temp_root.mkdir(exist_ok=True)
            work_root = temp_root
        else:
            work_root = workspace_root

        for ratio in ratios:
            items.append(run_single_ratio_experiment(ratio, runs_per_ratio, work_root))

    finally:
        if temp_root.exists() and not keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    bundle = compact_bundle(
        {
            "experiment_name": "experiment2_恶意比例鲁棒性",
            "trusted_ratio": TRUSTED_RATIO,
            "malicious_ratios": ratios,
            "runs_per_ratio": runs_per_ratio,
            "items": items,
        }
    )

    save_json(bundle, RESULTS_BUNDLE_FILE)

    return bundle


def load_results_bundle():
    return compact_bundle(load_json(RESULTS_BUNDLE_FILE))
