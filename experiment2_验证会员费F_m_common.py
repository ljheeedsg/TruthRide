import csv
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

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
# Membership-fee sensitivity experiment
# Style version: Fig.2/Fig.3/retention unified SCI style
#
# Important:
# This version explicitly separates per-round and cumulative indicators.
#
# Per-round indicators:
# - per_round_avg_quality
# - per_round_completion_rate
# - per_round_platform_utility
#
# Cumulative indicators:
# - cumulative_avg_quality
# - cumulative_completion_rate
# - cumulative_platform_utility
#
# Cumulative values are recomputed from per-round values.
# ============================================================

MEMBERSHIP_FEE_VALUES = [1, 2, 5, 10,20]
RUNS_PER_VALUE = 10
RESULTS_BUNDLE_FILE = "experiment2_验证会员费F_m_results.json"
WORKER_OPTIONS_FILE = "experiment2_worker_options.json"

SUMMARY_PER_ROUND_CSV = "experiment2_验证会员费F_m_每轮指标统计.csv"
SUMMARY_CUMULATIVE_CSV = "experiment2_验证会员费F_m_累计指标统计.csv"


COLOR_LIST = get_palette(len(MEMBERSHIP_FEE_VALUES))
MARKER_LIST = ["o", "P", "^", "s", "v", "D"]
MARK_EVERY = 4


FIGURE_SPECS = {
    # --------------------------------------------------------
    # Per-round figures
    # --------------------------------------------------------
    "fig30": {
        "output_png": "experiment2_Fig. 30：不同F_m下每轮平均数据质量.png",
        "output_pdf": "experiment2_Fig. 30：不同F_m下每轮平均数据质量.pdf",
        "output_csv": "experiment2_Fig. 30：不同F_m下每轮平均数据质量.csv",
        "ylabel": "Per-Round Avg. Quality",
        "metric_key": "per_round_avg_quality",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "per_round",
    },
    "fig31": {
        "output_png": "experiment2_Fig. 31：不同F_m下每轮平台效用.png",
        "output_pdf": "experiment2_Fig. 31：不同F_m下每轮平台效用.pdf",
        "output_csv": "experiment2_Fig. 31：不同F_m下每轮平台效用.csv",
        "ylabel": "Per-Round Utility",
        "metric_key": "per_round_platform_utility",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": None,
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "per_round",
    },
    "fig32": {
        "output_png": "experiment2_Fig. 32：不同F_m下种子可信工人留存率.png",
        "output_pdf": "experiment2_Fig. 32：不同F_m下种子可信工人留存率.pdf",
        "output_csv": "experiment2_Fig. 32：不同F_m下种子可信工人留存率.csv",
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
    "fig33": {
        "output_png": "experiment2_Fig. 33：不同F_m下每轮任务完成率.png",
        "output_pdf": "experiment2_Fig. 33：不同F_m下每轮任务完成率.pdf",
        "output_csv": "experiment2_Fig. 33：不同F_m下每轮任务完成率.csv",
        "ylabel": "Task Completion Rate",
        "metric_key": "per_round_completion_rate",
        "smooth_window": 5,
        "point_every": 4,
        "mark_every": 1,
        "ylim": None,
        "value_bounds": (0.0, 1.0),
        "legend_loc": "upper left",
        "legend_ncol": 2,
        "style_type": "per_round",
    },

    # --------------------------------------------------------
    # Cumulative figures
    # --------------------------------------------------------
    "fig30_cum_quality": {
        "output_png": "experiment2_Fig. 30补充：不同F_m下累计平均数据质量.png",
        "output_pdf": "experiment2_Fig. 30补充：不同F_m下累计平均数据质量.pdf",
        "output_csv": "experiment2_Fig. 30补充：不同F_m下累计平均数据质量.csv",
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
    "fig31_cum_utility": {
        "output_png": "experiment2_Fig. 31补充：不同F_m下累计平台效用.png",
        "output_pdf": "experiment2_Fig. 31补充：不同F_m下累计平台效用.pdf",
        "output_csv": "experiment2_Fig. 31补充：不同F_m下累计平台效用.csv",
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
    "fig33_cum_completion": {
        "output_png": "experiment2_Fig. 33补充：不同F_m下累计任务完成率.png",
        "output_pdf": "experiment2_Fig. 33补充：不同F_m下累计任务完成率.pdf",
        "output_csv": "experiment2_Fig. 33补充：不同F_m下累计任务完成率.csv",
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


def moving_average(values, window):
    """
    Trailing moving average.

    For round t, it only uses current and previous values:
    y_smooth(t) = average[y(t-window+1), ..., y(t)].

    This follows the Fig.2 per-round style and avoids using future information.
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


def fm_label(membership_fee):
    return f"$F_m$={int(membership_fee)}"


def save_plot_data_csv(rounds, series_rows, labels, output_csv):
    headers = ["round"] + labels

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, round_id in enumerate(rounds):
            row = [round_id]
            row.extend(values[idx] for values in series_rows)
            writer.writerow(row)


def load_seed_trusted_worker_ids():
    worker_options = load_json(WORKER_OPTIONS_FILE)

    return sorted(
        int(worker["worker_id"])
        for worker in worker_options.values()
        if worker.get("init_category") == "trusted"
    )


def compact_round_entry(entry):
    """
    Keep original primitive fields as much as possible.
    Later, enrich_round_metrics() will recompute per-round and cumulative metrics.
    """
    return {
        "round_id": int(entry["round_id"]),
        "num_tasks": int(entry.get("num_tasks", 0)),
        "num_completed": int(entry.get("num_completed", 0)),
        "avg_quality": round(float(entry.get("avg_quality", 0.0)), 4),
        "platform_utility": round(float(entry.get("platform_utility", 0.0)), 4),
        "completion_rate": round(float(entry.get("completion_rate", 0.0)), 4),
        "trusted_count": int(entry.get("trusted_count", 0)),
        "unknown_count": int(entry.get("unknown_count", 0)),
        "malicious_count": int(entry.get("malicious_count", 0)),
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
            "membership_fee": int(item["membership_fee"]),
            "round_metrics": item["round_metrics"],
            "retention_curve": item["retention_curve"],
        }

    return {
        "membership_fee": int(item["membership_fee"]),
        "round_metrics": build_compact_round_metrics(item["round_results"]),
        "retention_curve": build_compact_retention_curve(
            item["all_runs_payload"],
            item["seed_trusted_worker_ids"],
        ),
    }


def compact_bundle(bundle):
    return {
        "experiment_name": bundle["experiment_name"],
        "bundle_format": "membership_fee_compact_v4_separate_round_and_cumulative",
        "membership_fee_values": [
            int(value) for value in bundle["membership_fee_values"]
        ],
        "runs_per_value": int(bundle["runs_per_value"]),
        "items": [compact_item_payload(item) for item in bundle["items"]],
    }


def load_round_results_from_item(item):
    round_entries = item.get("round_metrics", item.get("round_results", []))

    return [entry for entry in round_entries if int(entry.get("num_tasks", 0)) > 0]


def enrich_round_metrics(round_results):
    """
    Build a consistent set of per-round and cumulative indicators.

    Per-round indicators:
    - per_round_tasks
    - per_round_completed
    - per_round_completion_rate
    - per_round_avg_quality
    - per_round_platform_utility

    Cumulative indicators:
    - cumulative_completion_rate
    - cumulative_avg_quality
    - cumulative_platform_utility

    Important:
    If num_completed is detected as cumulative, it is converted into
    per-round completed tasks by differencing.
    """

    valid_entries = [
        dict(entry) for entry in round_results
        if int(entry.get("num_tasks", 0)) > 0
    ]

    raw_completed_values = [
        float(entry.get("num_completed", 0.0))
        for entry in valid_entries
    ]

    raw_task_values = [
        float(entry.get("num_tasks", 0.0))
        for entry in valid_entries
    ]

    # Detect whether num_completed is cumulative.
    # If any raw num_completed is larger than current round num_tasks,
    # it is very likely to be a cumulative count.
    completed_is_cumulative = any(
        completed > tasks
        for completed, tasks in zip(raw_completed_values, raw_task_values)
    )

    cumulative_tasks = 0.0
    cumulative_completed = 0.0
    cumulative_quality_weighted_sum = 0.0
    cumulative_platform_utility = 0.0

    previous_completed_value = 0.0

    trusted_count = None
    unknown_count = None
    malicious_count = None

    enriched = []

    for entry in valid_entries:
        enriched_entry = dict(entry)

        per_round_tasks = float(enriched_entry.get("num_tasks", 0.0))
        raw_completed = float(enriched_entry.get("num_completed", 0.0))

        if completed_is_cumulative:
            per_round_completed = raw_completed - previous_completed_value
            previous_completed_value = raw_completed
        else:
            per_round_completed = raw_completed

        # Defensive handling for abnormal reset or missing fields.
        per_round_completed = max(0.0, per_round_completed)
        per_round_completed = min(per_round_completed, per_round_tasks)

        per_round_avg_quality = float(enriched_entry.get("avg_quality", 0.0))
        per_round_platform_utility = float(enriched_entry.get("platform_utility", 0.0))

        per_round_completion_rate = (
            per_round_completed / per_round_tasks
            if per_round_tasks > 0 else 0.0
        )

        cumulative_tasks += per_round_tasks
        cumulative_completed += per_round_completed
        cumulative_quality_weighted_sum += per_round_avg_quality * per_round_completed
        cumulative_platform_utility += per_round_platform_utility

        cumulative_completion_rate = (
            cumulative_completed / cumulative_tasks
            if cumulative_tasks > 0 else 0.0
        )

        cumulative_avg_quality = (
            cumulative_quality_weighted_sum / cumulative_completed
            if cumulative_completed > 0 else 0.0
        )

        enriched_entry["per_round_tasks"] = round(per_round_tasks, 4)
        enriched_entry["per_round_completed"] = round(per_round_completed, 4)
        enriched_entry["per_round_completion_rate"] = round(per_round_completion_rate, 4)
        enriched_entry["per_round_avg_quality"] = round(per_round_avg_quality, 4)
        enriched_entry["per_round_platform_utility"] = round(per_round_platform_utility, 4)

        enriched_entry["cumulative_tasks"] = round(cumulative_tasks, 4)
        enriched_entry["cumulative_completed"] = round(cumulative_completed, 4)
        enriched_entry["cumulative_completion_rate"] = round(cumulative_completion_rate, 4)
        enriched_entry["cumulative_avg_quality"] = round(cumulative_avg_quality, 4)
        enriched_entry["cumulative_platform_utility"] = round(cumulative_platform_utility, 4)

        # Compatibility aliases.
        # These aliases now explicitly point to per-round values.
        enriched_entry["completion_rate"] = enriched_entry["per_round_completion_rate"]
        enriched_entry["avg_quality"] = enriched_entry["per_round_avg_quality"]
        enriched_entry["platform_utility"] = enriched_entry["per_round_platform_utility"]

        if "trusted_count" not in enriched_entry:
            trusted_count = 0 if trusted_count is None else trusted_count
            enriched_entry["trusted_count"] = trusted_count
        else:
            trusted_count = int(enriched_entry["trusted_count"])

        if "unknown_count" not in enriched_entry:
            unknown_count = 0 if unknown_count is None else unknown_count
            enriched_entry["unknown_count"] = unknown_count
        else:
            unknown_count = int(enriched_entry["unknown_count"])

        if "malicious_count" not in enriched_entry:
            malicious_count = 0 if malicious_count is None else malicious_count
            enriched_entry["malicious_count"] = malicious_count
        else:
            malicious_count = int(enriched_entry["malicious_count"])

        enriched.append(enriched_entry)

    return enriched


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
    retention_entries = [
        entry for entry in item["retention_curve"]
        if int(entry.get("num_tasks", 0)) > 0
    ]

    rounds = [int(entry["round_id"]) for entry in retention_entries]
    values = [float(entry["retention_rate"]) for entry in retention_entries]

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


def plot_figure(bundle, figure_key):
    spec = FIGURE_SPECS[figure_key]

    items_sorted = sorted(bundle["items"], key=lambda item: int(item["membership_fee"]))

    default_fee = 2

    normal_items = [
        item for item in items_sorted
        if int(item["membership_fee"]) != default_fee
    ]

    highlight_items = [
        item for item in items_sorted
        if int(item["membership_fee"]) == default_fee
    ]

    # Draw F_m=2 last so the black highlighted curve is not covered.
    items = normal_items + highlight_items

    set_figure_style()

    fig, ax = plt.subplots(figsize=(4.18, 2.70))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)

    csv_rounds = None
    csv_series_rows = []
    labels = []
    y_values_all = []
    series_map = {}

    for idx, item in enumerate(items):
        membership_fee = int(item["membership_fee"])
        label = fm_label(membership_fee)
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
            default_label=fm_label(default_fee),
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
            fm_label(default_fee),
            marker_cycle=MARKER_LIST,
            y_values_all=y_values_all,
        )
        apply_y_axis(ax, spec, y_values_all)
        add_compact_legend(ax, legend_handles, spec)
    else:
        legend_handles = []
        for idx, label in enumerate(labels):
            rounds, values = series_map[label]
            membership_fee = int(items[idx]["membership_fee"])
            color = COLOR_LIST[MEMBERSHIP_FEE_VALUES.index(membership_fee) % len(COLOR_LIST)]
            marker = MARKER_LIST[MEMBERSHIP_FEE_VALUES.index(membership_fee) % len(MARKER_LIST)]
            emphasis = get_series_emphasis(idx, len(items), highlight_index=len(items) - 1)
            if membership_fee == default_fee:
                color = COLOR_LIST[-1]
                marker = "P"
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
    plt.close(fig)

    if spec.get("style_type") != "per_round":
        save_plot_data_csv(csv_rounds, csv_series_rows, labels, spec["output_csv"])

    print(f"Saved {spec['output_png']}")
    print(f"Saved {spec['output_pdf']}")
    print(f"Saved {spec['output_csv']}")


def save_per_round_summary_csv(bundle):
    """
    Save average per-round statistics for each membership fee F_m.
    """
    items = sorted(bundle["items"], key=lambda item: int(item["membership_fee"]))

    headers = [
        "membership_fee_F_m",
        "final_seed_trusted_retention_rate",
        "avg_per_round_completion_rate",
        "avg_per_round_quality",
        "avg_per_round_platform_utility",
    ]

    rows = []

    for item in items:
        membership_fee = int(item["membership_fee"])

        round_results = enrich_round_metrics(load_round_results_from_item(item))

        _, retention_values = build_retention_curve(item, smooth_window=1)
        final_retention = retention_values[-1] if retention_values else 0.0

        if not round_results:
            rows.append([membership_fee, final_retention, 0.0, 0.0, 0.0])
            continue

        avg_completion = sum(
            float(entry.get("per_round_completion_rate", 0.0)) for entry in round_results
        ) / len(round_results)

        avg_quality = sum(
            float(entry.get("per_round_avg_quality", 0.0)) for entry in round_results
        ) / len(round_results)

        avg_utility = sum(
            float(entry.get("per_round_platform_utility", 0.0)) for entry in round_results
        ) / len(round_results)

        rows.append(
            [
                membership_fee,
                round(float(final_retention), 4),
                round(float(avg_completion), 4),
                round(float(avg_quality), 4),
                round(float(avg_utility), 4),
            ]
        )

    with open(SUMMARY_PER_ROUND_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved {SUMMARY_PER_ROUND_CSV}")


def save_cumulative_summary_csv(bundle):
    """
    Save final cumulative statistics for each membership fee F_m.
    """
    items = sorted(bundle["items"], key=lambda item: int(item["membership_fee"]))

    headers = [
        "membership_fee_F_m",
        "final_seed_trusted_retention_rate",
        "final_cumulative_completion_rate",
        "final_cumulative_avg_quality",
        "final_cumulative_platform_utility",
    ]

    rows = []

    for item in items:
        membership_fee = int(item["membership_fee"])

        round_results = enrich_round_metrics(load_round_results_from_item(item))

        _, retention_values = build_retention_curve(item, smooth_window=1)
        final_retention = retention_values[-1] if retention_values else 0.0

        if not round_results:
            rows.append([membership_fee, final_retention, 0.0, 0.0, 0.0])
            continue

        final_entry = round_results[-1]

        rows.append(
            [
                membership_fee,
                round(float(final_retention), 4),
                round(float(final_entry.get("cumulative_completion_rate", 0.0)), 4),
                round(float(final_entry.get("cumulative_avg_quality", 0.0)), 4),
                round(float(final_entry.get("cumulative_platform_utility", 0.0)), 4),
            ]
        )

    with open(SUMMARY_CUMULATIVE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved {SUMMARY_CUMULATIVE_CSV}")


def plot_all_figures(bundle):
    for figure_key in (
        "fig30",
        "fig31",
        "fig32",
        "fig33",
        "fig30_cum_quality",
        "fig31_cum_utility",
        "fig33_cum_completion",
    ):
        plot_figure(bundle, figure_key)

    save_per_round_summary_csv(bundle)
    save_cumulative_summary_csv(bundle)


def run_single_value_experiment(
    membership_fee,
    runs_per_value,
    workspace_root,
    seed_trusted_worker_ids,
):
    print(f"\n========== Membership Fee F_m = {membership_fee} ==========")

    with tempfile.TemporaryDirectory(
        prefix=f"experiment2_membership_fee_{membership_fee}_",
        dir=str(workspace_root),
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prefix = temp_dir / f"truthride_membership_fee_{membership_fee}"

        round_results, _ = run_experiment(
            "lgsc",
            overrides={
                "MEMBERSHIP_FEE": int(membership_fee),
                "NUM_EXPERIMENT_RUNS": runs_per_value,
                "prefix": str(prefix),
            },
        )

        return compact_item_payload(
            {
                "membership_fee": int(membership_fee),
                "round_results": round_results,
                "all_runs_payload": load_json(f"{prefix}_round_results_all_runs.json"),
                "seed_trusted_worker_ids": seed_trusted_worker_ids,
            }
        )


def run_full_experiment(
    membership_fee_values=None,
    runs_per_value=RUNS_PER_VALUE,
    keep_temp=False,
):
    workspace_root = Path.cwd()
    membership_fee_values = membership_fee_values or MEMBERSHIP_FEE_VALUES

    items = []
    seed_trusted_worker_ids = load_seed_trusted_worker_ids()
    temp_root = workspace_root / "experiment2_验证会员费F_m_tmp"

    try:
        if keep_temp:
            temp_root.mkdir(exist_ok=True)
            work_root = temp_root
        else:
            work_root = workspace_root

        for value in membership_fee_values:
            items.append(
                run_single_value_experiment(
                    value,
                    runs_per_value,
                    work_root,
                    seed_trusted_worker_ids,
                )
            )

    finally:
        if temp_root.exists() and not keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    bundle = compact_bundle(
        {
            "experiment_name": "experiment2_验证会员费F_m",
            "membership_fee_values": membership_fee_values,
            "runs_per_value": runs_per_value,
            "items": items,
        }
    )

    save_json(bundle, RESULTS_BUNDLE_FILE)

    return bundle


def load_results_bundle():
    return compact_bundle(load_json(RESULTS_BUNDLE_FILE))
