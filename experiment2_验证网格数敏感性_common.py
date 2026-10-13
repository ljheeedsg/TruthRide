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
from experiment2_rmse_plot_utils import (
    build_worker_indexes,
    compute_rmse_series,
    load_workers,
)


# ============================================================
# Validation-grid sensitivity experiment
#
# Plotting style:
# - per-round figures follow Fig.2 style:
#   trailing moving average + sampled points + straight line segments
# - cumulative figures follow Fig.3 style:
#   no smoothing + sampled points + straight line segments
# - retention figure follows Fig.15 style:
#   no smoothing + no sampling + complete line + marker every few rounds
# - category-count trends follow state-trend style:
#   no smoothing + no sampling + marker every few rounds
# - no curved line joins
# - hollow markers
# - unified SCI-style frame/grid/legend
# ============================================================


VALIDATION_TOP_M_VALUES = [0, 1, 2, 3]
RUNS_PER_VALUE = 10
RESULTS_BUNDLE_FILE = "experiment2_验证网格数敏感性_results.json"
WORKER_OPTIONS_FILE = "experiment2_worker_options.json"


COLOR_LIST = get_palette(len(VALIDATION_TOP_M_VALUES))
MARKER_LIST = ["o", "D", "s", "^", "v", "P"]


FIGURE_SPECS = {
    # ========================================================
    # Per-round figures: Fig.2 style
    # smoothing + sampling + straight line connection
    # ========================================================
    "fig24": {
        "output_png": "experiment2_Fig. 24：不同M_v下每轮平均数据质量.png",
        "output_pdf": "experiment2_Fig. 24：不同M_v下每轮平均数据质量.pdf",
        "output_csv": "experiment2_Fig. 24：不同M_v下每轮平均数据质量.csv",
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
    "fig25": {
        "output_png": "experiment2_Fig. 25：不同M_v下每轮平台效用.png",
        "output_pdf": "experiment2_Fig. 25：不同M_v下每轮平台效用.pdf",
        "output_csv": "experiment2_Fig. 25：不同M_v下每轮平台效用.csv",
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
    "fig24_completion": {
        "output_png": "experiment2_Fig. 24补充3：不同M_v下任务完成率.png",
        "output_pdf": "experiment2_Fig. 24补充3：不同M_v下任务完成率.pdf",
        "output_csv": "experiment2_Fig. 24补充3：不同M_v下任务完成率.csv",
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
    "fig24_rmse": {
        "output_png": "experiment2_Fig. 24补充2：不同M_v下活跃工人Trust RMSE.png",
        "output_pdf": "experiment2_Fig. 24补充2：不同M_v下活跃工人Trust RMSE.pdf",
        "output_csv": "experiment2_Fig. 24补充2：不同M_v下活跃工人Trust RMSE.csv",
        "ylabel": "Per-Round RMSE",
        "metric_key": "per_round_rmse",
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
    "fig26": {
        "output_png": "experiment2_Fig. 26：不同M_v下种子可信工人留存率.png",
        "output_pdf": "experiment2_Fig. 26：不同M_v下种子可信工人留存率.pdf",
        "output_csv": "experiment2_Fig. 26：不同M_v下种子可信工人留存率.csv",
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
    "fig24_supp": {
        "output_png": "experiment2_Fig. 24补充：不同M_v下累计平均数据质量.png",
        "output_pdf": "experiment2_Fig. 24补充：不同M_v下累计平均数据质量.pdf",
        "output_csv": "experiment2_Fig. 24补充：不同M_v下累计平均数据质量.csv",
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
    "fig25_supp": {
        "output_png": "experiment2_Fig. 25补充：不同M_v下累计平台效用.png",
        "output_pdf": "experiment2_Fig. 25补充：不同M_v下累计平台效用.pdf",
        "output_csv": "experiment2_Fig. 25补充：不同M_v下累计平台效用.csv",
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
    "fig24_supp_completion": {
        "output_png": "experiment2_Fig. 24补充4：不同M_v下累计任务完成率.png",
        "output_pdf": "experiment2_Fig. 24补充4：不同M_v下累计任务完成率.pdf",
        "output_csv": "experiment2_Fig. 24补充4：不同M_v下累计任务完成率.csv",
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


def mv_label(top_m):
    return f"$M_v$={int(top_m)}"


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
        "trust_rmse": round(float(entry.get("trust_rmse", 0.0)), 4),
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


def build_compact_rmse_curve(all_runs_payload):
    workers = load_workers()
    tasks_by_slot, task_detail_map, trusted_reference_map = build_worker_indexes(workers)

    per_run_rmse_curves = []
    round_axis = None

    for run_payload in all_runs_payload:
        round_results = [
            entry for entry in run_payload["round_results"]
            if int(entry.get("num_tasks", 0)) > 0
        ]

        rounds, per_round_rmse, cumulative_rmse = compute_rmse_series(
            round_results,
            tasks_by_slot,
            task_detail_map,
            trusted_reference_map,
        )

        if round_axis is None:
            round_axis = rounds

        per_run_rmse_curves.append((per_round_rmse, cumulative_rmse))

    averaged_curve = []

    for idx, round_id in enumerate(round_axis or []):
        averaged_curve.append(
            {
                "round_id": int(round_id),
                "per_round_rmse": round(
                    sum(curves[0][idx] for curves in per_run_rmse_curves)
                    / len(per_run_rmse_curves),
                    4,
                ),
                "cumulative_rmse": round(
                    sum(curves[1][idx] for curves in per_run_rmse_curves)
                    / len(per_run_rmse_curves),
                    4,
                ),
            }
        )

    return averaged_curve


def compact_item_payload(item):
    if "round_metrics" in item and "retention_curve" in item:
        return {
            "validation_top_m": int(item["validation_top_m"]),
            "round_metrics": item["round_metrics"],
            "retention_curve": item["retention_curve"],
            "rmse_curve": item.get("rmse_curve", []),
        }

    return {
        "validation_top_m": int(item["validation_top_m"]),
        "round_metrics": build_compact_round_metrics(item["round_results"]),
        "retention_curve": build_compact_retention_curve(
            item["all_runs_payload"],
            item["seed_trusted_worker_ids"],
        ),
        "rmse_curve": build_compact_rmse_curve(item["all_runs_payload"]),
    }


def compact_bundle(bundle):
    return {
        "experiment_name": bundle["experiment_name"],
        "bundle_format": "validation_top_m_compact_v1",
        "validation_top_m_values": [
            int(value) for value in bundle["validation_top_m_values"]
        ],
        "runs_per_value": int(bundle["runs_per_value"]),
        "items": [compact_item_payload(item) for item in bundle["items"]],
    }


def load_round_results_from_item(item):
    round_entries = item.get("round_metrics", item.get("round_results", []))

    return [entry for entry in round_entries if int(entry.get("num_tasks", 0)) > 0]


def enrich_round_metrics(round_results):
    trusted_count = None
    unknown_count = None
    malicious_count = None
    enriched = []

    for entry in round_results:
        enriched_entry = dict(entry)

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
    if metric_key in {"per_round_rmse", "cumulative_rmse"}:
        rmse_entries = item.get("rmse_curve", [])
        rounds = [int(entry["round_id"]) for entry in rmse_entries]
        values = [float(entry[metric_key]) for entry in rmse_entries]

        return rounds, moving_average(values, smooth_window)

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
    # Ratio/quality/completion/retention/RMSE figures remain normal decimal format.
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
    items = sorted(bundle["items"], key=lambda item: item["validation_top_m"])

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
        label = mv_label(item["validation_top_m"])
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
            default_label=mv_label(3),
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
            mv_label(3),
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
                x_plot,
                y_plot,
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=emphasis["linewidth"],
                markersize=emphasis["markersize"],
                markevery=spec.get("mark_every", 1),
                markerfacecolor=AX_BG,
                markeredgecolor=color,
                markeredgewidth=emphasis["markeredgewidth"],
                alpha=emphasis["alpha"],
                zorder=emphasis["zorder"],
                solid_capstyle="round",
                solid_joinstyle="round",
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


def plot_all_figures(bundle):
    for figure_key in (
        "fig24",
        "fig25",
        "fig26",
        "fig24_completion",
        "fig24_supp",
        "fig25_supp",
        "fig24_supp_completion",
        "fig24_rmse",
    ):
        plot_figure(bundle, figure_key)

    plot_category_trends(bundle)


def save_category_plot_csv(rounds, series_map, output_csv):
    headers = ["round"]

    for category_key in ("trusted_count", "unknown_count", "malicious_count"):
        for top_m in sorted(series_map.keys()):
            headers.append(f"{category_key}_M_v_{top_m}")

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, round_id in enumerate(rounds):
            row = [round_id]

            for category_key in ("trusted_count", "unknown_count", "malicious_count"):
                for top_m in sorted(series_map.keys()):
                    row.append(series_map[top_m][category_key][idx])

            writer.writerow(row)


def plot_category_trends(bundle):
    output_png = "experiment2_Fig. 26补充2：不同M_v下三类工人数量变化趋势.png"
    output_pdf = "experiment2_Fig. 26补充2：不同M_v下三类工人数量变化趋势.pdf"
    output_csv = "experiment2_Fig. 26补充2：不同M_v下三类工人数量变化趋势.csv"

    items = sorted(bundle["items"], key=lambda item: item["validation_top_m"])

    set_figure_style()

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.34), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER_BG)

    category_specs = [
        ("trusted_count", "Trusted"),
        ("unknown_count", "Unknown"),
        ("malicious_count", "Malicious"),
    ]

    csv_rounds = None
    series_map = {}
    y_values_all = []

    for idx, item in enumerate(items):
        round_results = enrich_round_metrics(load_round_results_from_item(item))
        rounds = [int(entry["round_id"]) for entry in round_results]

        if csv_rounds is None:
            csv_rounds = rounds

        top_m = int(item["validation_top_m"])
        series_map[top_m] = {}
        label = mv_label(top_m)

        color = COLOR_LIST[idx % len(COLOR_LIST)]
        marker = MARKER_LIST[idx % len(MARKER_LIST)]
        emphasis = get_series_emphasis(idx, len(items))
        linewidth = emphasis["linewidth"] - 0.18
        markersize = emphasis["markersize"] - 0.25
        zorder = emphasis["zorder"]

        for ax, (category_key, title) in zip(axes, category_specs):
            values = [float(entry.get(category_key, 0.0)) for entry in round_results]

            series_map[top_m][category_key] = values
            y_values_all.extend(values)

            ax.set_facecolor(AX_BG)

            line = ax.plot(
                rounds,
                values,
                label=label,
                color=color,
                marker=marker,
                linestyle="-",
                drawstyle="default",
                linewidth=linewidth,
                markersize=markersize,
                markevery=4,
                markerfacecolor=AX_BG,
                markeredgecolor=color,
                markeredgewidth=emphasis["markeredgewidth"],
                alpha=emphasis["alpha"],
                zorder=zorder,
                solid_capstyle="round",
                solid_joinstyle="round",
            )[0]
            line.set_path_effects(get_line_path_effects(linewidth))

            ax.set_title(title, fontsize=8.7, pad=3.0)
            ax.set_xlim(0, 144)
            ax.set_xticks(range(0, 141, 40))
            ax.set_xticks(range(0, 145, 20), minor=True)

            apply_axis_frame(ax)

    for ax in axes:
        ax.set_xlabel("Round")

    axes[0].set_ylabel("Number of Workers")

    if y_values_all:
        y_min = 0.0
        y_max = max(y_values_all)
        margin = max(y_max * 0.08, 8.0)
        axes[0].set_ylim(y_min, y_max + margin)

    legend_handles = []

    for idx, item in enumerate(items):
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        marker = MARKER_LIST[idx % len(MARKER_LIST)]
        emphasis = get_series_emphasis(idx, len(items))

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=mv_label(int(item["validation_top_m"])),
                color=color,
                linestyle="-",
                linewidth=emphasis["linewidth"] - 0.18,
                marker=marker,
                markersize=emphasis["markersize"] - 0.25,
                markerfacecolor=AX_BG,
                markeredgecolor=color,
                markeredgewidth=emphasis["markeredgewidth"],
                alpha=emphasis["alpha"],
                solid_capstyle="round",
                solid_joinstyle="round",
            )
        )

    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=0.72,
        facecolor="white",
        edgecolor="#666666",
        bbox_to_anchor=(0.5, 1.03),
        handlelength=1.45,
        handletextpad=0.30,
        columnspacing=0.80,
        labelspacing=0.22,
        borderpad=0.25,
    )
    style_legend(leg)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.79, wspace=0.12)

    fig.savefig(output_png, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    plt.close(fig)

    save_category_plot_csv(csv_rounds, series_map, output_csv)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")
    print(f"Saved {output_csv}")


def run_single_value_experiment(
    validation_top_m,
    runs_per_value,
    workspace_root,
    seed_trusted_worker_ids,
):
    print(f"\n========== Validation Top M = {validation_top_m} ==========")

    with tempfile.TemporaryDirectory(
        prefix=f"experiment2_validation_top_m_{validation_top_m}_",
        dir=str(workspace_root),
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prefix = temp_dir / f"truthride_validation_top_m_{validation_top_m}"

        round_results, _ = run_experiment(
            "lgsc",
            overrides={
                "VALIDATION_TOP_M": int(validation_top_m),
                "NUM_EXPERIMENT_RUNS": runs_per_value,
                "prefix": str(prefix),
            },
        )

        return compact_item_payload(
            {
                "validation_top_m": int(validation_top_m),
                "round_results": round_results,
                "all_runs_payload": load_json(f"{prefix}_round_results_all_runs.json"),
                "seed_trusted_worker_ids": seed_trusted_worker_ids,
            }
        )


def run_full_experiment(
    validation_top_m_values=None,
    runs_per_value=RUNS_PER_VALUE,
    keep_temp=False,
):
    workspace_root = Path.cwd()
    validation_top_m_values = validation_top_m_values or VALIDATION_TOP_M_VALUES

    items = []
    seed_trusted_worker_ids = load_seed_trusted_worker_ids()
    temp_root = workspace_root / "experiment2_验证网格数敏感性_tmp"

    try:
        if keep_temp:
            temp_root.mkdir(exist_ok=True)
            work_root = temp_root
        else:
            work_root = workspace_root

        for value in validation_top_m_values:
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
            "experiment_name": "experiment2_验证网格数敏感性",
            "validation_top_m_values": validation_top_m_values,
            "runs_per_value": runs_per_value,
            "items": items,
        }
    )

    save_json(bundle, RESULTS_BUNDLE_FILE)

    return bundle


def load_results_bundle():
    return compact_bundle(load_json(RESULTS_BUNDLE_FILE))
