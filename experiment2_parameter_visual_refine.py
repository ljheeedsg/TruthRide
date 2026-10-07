from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


DEFAULT_HIGHLIGHT_COLOR = "#8f2d2d"
NON_HIGHLIGHT_COLORS = [
    "#8d99a6",
    "#6f8599",
    "#9aa6b2",
    "#778899",
    "#8796a5",
    "#a4afba",
]


def hex_to_rgb(hex_color: str):
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(round(rgb[0] * 255)))),
        max(0, min(255, int(round(rgb[1] * 255)))),
        max(0, min(255, int(round(rgb[2] * 255)))),
    )


def lighten_color(hex_color: str, amount: float = 0.55) -> str:
    r, g, b = hex_to_rgb(hex_color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return rgb_to_hex((r, g, b))


def compute_interval_bars(rounds, values, interval_size=20):
    if not rounds or not values:
        return [], [], []

    bars_x = []
    bars_h = []
    bars_w = []

    start_idx = 0
    n = len(rounds)

    while start_idx < n:
        start_round = rounds[start_idx]
        end_round = min(start_round + interval_size - 1, rounds[-1])

        interval_indices = []
        idx = start_idx
        while idx < n and rounds[idx] <= end_round:
            interval_indices.append(idx)
            idx += 1

        if not interval_indices:
            break

        interval_rounds = [rounds[i] for i in interval_indices]
        interval_values = [values[i] for i in interval_indices]

        bars_x.append((interval_rounds[0] + interval_rounds[-1]) / 2.0)
        bars_h.append(sum(interval_values) / len(interval_values))
        bars_w.append(interval_rounds[-1] - interval_rounds[0] + 1)

        start_idx = interval_indices[-1] + 1

    return bars_x, bars_h, bars_w


def get_series_style(index, total, highlight_index, marker):
    is_highlight = index == highlight_index

    color = DEFAULT_HIGHLIGHT_COLOR if is_highlight else NON_HIGHLIGHT_COLORS[index % len(NON_HIGHLIGHT_COLORS)]

    return {
        "color": color,
        "marker": marker if is_highlight else None,
        "linewidth": 2.25 if is_highlight else 1.35,
        "markersize": 3.9 if is_highlight else 3.1,
        "markevery": 10 if is_highlight else 12,
        "alpha": 0.98 if is_highlight else 0.72,
        "zorder": 8 if is_highlight else 3 + index,
        "markeredgewidth": 0.8 if is_highlight else 0.7,
    }


def apply_refined_axes(ax, *, ratio_axis=False):
    ax.set_facecolor("white")
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.58,
        color="#b8b8b8",
        alpha=0.30,
    )
    ax.grid(False, which="minor", axis="both")

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
    )

    if ratio_axis:
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def add_compact_legend(ax, legend_handles, spec):
    leg = ax.legend(
        handles=legend_handles,
        loc=spec.get("legend_loc", "best"),
        ncol=spec.get("legend_ncol", 2),
        frameon=False,
        handlelength=1.3,
        handletextpad=0.25,
        columnspacing=0.55,
        labelspacing=0.18,
        borderaxespad=0.25,
    )
    return leg


def save_interval_plot_data_csv(interval_labels, series_map, output_csv):
    headers = ["interval"] + list(series_map.keys())

    with Path(output_csv).open("w", encoding="utf-8-sig", newline="") as f:
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


def plot_grouped_bar_line_intervals(
    ax,
    interval_labels,
    interval_values_dict,
    *,
    default_label,
    ylabel,
    title,
    color_cycle=None,
    hatch_cycle=None,
    marker_cycle=None,
    legend_inside=False,
    legend_loc="upper left",
    legend_ncol=2,
):
    labels = list(interval_values_dict.keys())
    x_positions = list(range(len(interval_labels)))
    total_width = 0.82
    bar_width = total_width / max(1, len(labels))

    color_cycle = color_cycle or ["#00A6FB", "#FF9800", "#2FBF71", "#FF3B30", "#9C27B0", "#00B8A9"]
    hatch_cycle = hatch_cycle or ["//", "\\\\", "xx", "--", "///", "++"]
    marker_cycle = marker_cycle or ["o", "s", "^", "D", "v", "P"]

    all_values = [value for values in interval_values_dict.values() for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    y_span = max(y_max - y_min, 1.0)

    legend_handles = []

    for idx, label in enumerate(labels):
        values = interval_values_dict[label]
        color = color_cycle[idx % len(color_cycle)]
        hatch = hatch_cycle[idx % len(hatch_cycle)]
        marker = marker_cycle[idx % len(marker_cycle)]
        is_default = label == default_label
        offset = (idx - (len(labels) - 1) / 2.0) * bar_width

        ax.bar(
            [x_pos + offset for x_pos in x_positions],
            values,
            width=bar_width * 0.92,
            color=color,
            alpha=1.00 if is_default else 0.95,
            hatch=hatch,
            edgecolor="#111111",
            linewidth=0.95 if is_default else 0.82,
            zorder=2 if not is_default else 3,
        )

        ax.plot(
            [x_pos + offset for x_pos in x_positions],
            values,
            color=color,
            linewidth=2.70 if is_default else 2.15,
            marker=marker,
            markersize=5.1 if is_default else 4.1,
            markerfacecolor=color,
            markeredgecolor="#111111",
            markeredgewidth=0.86,
            zorder=6 if is_default else 4,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=color,
                linewidth=2.70 if is_default else 2.15,
                marker=marker,
                markersize=5.1 if is_default else 4.1,
                markerfacecolor=color,
                markeredgecolor="#111111",
                markeredgewidth=0.86,
            )
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(interval_labels)
    ax.set_xlabel("Round interval")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10, pad=5.0)

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

    ratio_axis = "rate" in ylabel.lower() or "quality" in ylabel.lower()
    if ratio_axis:
        ax.set_ylim(0.0, max(1.02, y_max + 0.05))
    else:
        ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.10 * y_span)

    legend_kwargs = {
        "handles": legend_handles,
        "loc": legend_loc,
        "ncol": min(legend_ncol, len(labels)),
        "fontsize": 8,
        "handlelength": 1.5,
        "handletextpad": 0.35,
        "columnspacing": 0.8,
    }

    if legend_inside:
        legend_kwargs.update(
            {
                "frameon": True,
                "fancybox": False,
                "framealpha": 0.92,
                "facecolor": "white",
                "edgecolor": "#777777",
                "borderpad": 0.25,
                "borderaxespad": 0.35,
            }
        )
    else:
        legend_kwargs.update(
            {
                "frameon": False,
                "bbox_to_anchor": (0.5, 1.18),
                "loc": "upper center",
            }
        )

    ax.legend(**legend_kwargs)


def plot_bar_line_per_round(
    ax,
    series_map,
    default_label,
    *,
    marker_cycle,
    spec,
    y_values_all,
    interval_size=20,
):
    labels = list(series_map.keys())
    highlight_index = labels.index(default_label) if default_label in labels else len(labels) - 1

    default_rounds, default_values = series_map[labels[highlight_index]]
    bar_x, bar_h, bar_w = compute_interval_bars(default_rounds, default_values, interval_size=interval_size)
    bar_color = lighten_color(DEFAULT_HIGHLIGHT_COLOR, 0.72)

    if bar_x:
        ax.bar(
            bar_x,
            bar_h,
            width=bar_w,
            color=bar_color,
            edgecolor="none",
            alpha=0.28,
            align="center",
            zorder=1,
        )

    legend_handles = []

    draw_order = [idx for idx in range(len(labels)) if idx != highlight_index] + [highlight_index]

    for idx in draw_order:
        label = labels[idx]
        rounds, values = series_map[label]
        marker = marker_cycle[idx % len(marker_cycle)]
        style = get_series_style(idx, len(labels), highlight_index, marker)

        line = ax.plot(
            rounds,
            values,
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=style["markevery"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=style["markeredgewidth"],
            zorder=style["zorder"],
            solid_capstyle="round",
            solid_joinstyle="round",
        )[0]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=style["color"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
            )
        )

        y_values_all.extend(values)

    apply_refined_axes(ax, ratio_axis=spec.get("value_bounds") == (0.0, 1.0))
    return legend_handles


def plot_retention_curve(
    ax,
    series_map,
    default_label,
    *,
    marker_cycle,
    y_values_all,
):
    labels = list(series_map.keys())
    highlight_index = labels.index(default_label) if default_label in labels else len(labels) - 1
    legend_handles = []

    draw_order = [idx for idx in range(len(labels)) if idx != highlight_index] + [highlight_index]

    for idx in draw_order:
        label = labels[idx]
        rounds, values = series_map[label]
        marker = marker_cycle[idx % len(marker_cycle)]
        style = get_series_style(idx, len(labels), highlight_index, marker)

        line = ax.plot(
            rounds,
            values,
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=0.94 if idx == highlight_index else 0.62,
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=12,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=style["markeredgewidth"],
            zorder=style["zorder"],
            solid_capstyle="round",
            solid_joinstyle="round",
        )[0]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=style["color"],
                linewidth=style["linewidth"],
                alpha=0.94 if idx == highlight_index else 0.62,
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
            )
        )

        y_values_all.extend(values)

    apply_refined_axes(ax, ratio_axis=True)
    return legend_handles
