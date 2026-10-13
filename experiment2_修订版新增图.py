"""
修订版新增图（读取 run_revision_experiments.py 生成的 results/summary_*.csv）。

Fig. R1  Observable vs. Oracle quality feedback                (审稿人 2)
Fig. R2  Comparison with external baselines                     (审稿人 1, 意见 3)
Fig. R3  Ablation of the threshold / value function in selection (审稿人 1, 意见 1)
Fig. R4  Fairness of worker selection and income                (审稿人 1, 意见 2)
Fig. R5  Opportunistic (pedestrian-like) mobility               (审稿人 1, 意见 4)
Fig. R6  Sensitivity to the fraction of seed trusted workers     (审稿人 2 衍生)

风格与 Fig. 11 保持一致：白底、去掉上/右边框、浅色 y 向网格、TruthRide 红色突出。
输出到 results/figures/。
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

DATASETS = ["beijing", "rome", "chicago"]
DATASET_LABELS = {"beijing": "Beijing", "rome": "Rome", "chicago": "Chicago"}

MODE_STYLE = {
    "random": ("Random", "#8e7aa8"),
    "epsilon_first": ("Explore-First", "#ff7f0e"),
    "cmab": ("UCB-Greedy", "#2ca02c"),
    "ucb_classic": ("Classic CMAB-UCB", "#17becf"),
    "trust": ("Trust-Aware", "#1f77b4"),
    "beta_rep": ("Beta Reputation", "#bcbd22"),
    "crh_td": ("CRH Truth Discovery", "#7f7f7f"),
    "pgrd": ("Membership-Aware", "#8c564b"),
    "lgsc": ("TruthRide", "#b22222"),
}

FIGSIZE_ROW = (7.2, 2.35)


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
    })


def style_axis(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="#b5b5b5", alpha=0.28)
    ax.grid(False, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#444444")
    ax.tick_params(axis="both", direction="out", length=3.0, width=0.75, pad=2.0)


def legend(ax, **kwargs):
    leg = ax.legend(frameon=True, fancybox=False, framealpha=0.76, facecolor="white",
                    edgecolor="#666666", handlelength=1.4, handletextpad=0.3,
                    columnspacing=0.6, labelspacing=0.18, borderpad=0.24, **kwargs)
    leg.get_frame().set_linewidth(0.45)
    return leg


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.014)
    plt.close(fig)
    print(f"Saved {OUT / name}.png/pdf")


def read_csv(name):
    path = RESULTS / name
    if not path.exists():
        print(f"Skip: {path} not found")
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else np.nan


def grouped_bars(ax, groups, series, values, colors, labels, width=0.8):
    n = len(series)
    bar_w = width / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        offs = x - width / 2 + bar_w * (i + 0.5)
        is_tr = s == "lgsc" or labels[i] == "TruthRide"
        ax.bar(offs, [values[(g, s)] for g in groups], bar_w * 0.92, color=colors[i],
               label=labels[i], edgecolor="#111111" if is_tr else "none",
               linewidth=0.6 if is_tr else 0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(g, g) for g in groups])


# ------------------------------------------------------------------ R1
def fig_r1():
    rows = read_csv("summary_E2_observable_vs_oracle.csv")
    if not rows:
        return
    metrics = [
        ("final_cumulative_platform_utility", "Cumulative Platform Utility"),
        ("avg_avg_quality_all_non_empty", "Average Data Quality"),
        ("avg_completion_rate_all_non_empty", "Task Completion Rate"),
    ]
    modes = ["cmab", "trust", "lgsc"]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ROW)
    for ax, (key, ylabel) in zip(axes, metrics):
        style_axis(ax)
        x = np.arange(len(modes))
        for j, ds in enumerate(DATASETS):
            for k, fb in enumerate(["observable", "oracle"]):
                vals = []
                for m in modes:
                    match = [r for r in rows if r["dataset"] == ds and r["mode"] == m
                             and r["setting"] == f"feedback={fb}"]
                    vals.append(num(match[0], key) if match else np.nan)
                pos = x + (j - 1) * 0.27 + (k - 0.5) * 0.12
                color = ["#4c72b0", "#dd8452", "#55a868"][j]
                ax.bar(pos, vals, 0.115, color=color, alpha=1.0 if fb == "observable" else 0.38,
                       hatch=None if fb == "observable" else "////", edgecolor=color,
                       linewidth=0.4, zorder=3,
                       label=f"{DATASET_LABELS[ds]} ({'Observable' if fb == 'observable' else 'Oracle'})")
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_STYLE[m][0] for m in modes], fontsize=7.2)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig_R1_observable_vs_oracle")


# ------------------------------------------------------------------ R2
def fig_r2():
    rows = read_csv("summary_E1_main.csv")
    if not rows:
        return
    modes = ["ucb_classic", "cmab", "beta_rep", "crh_td", "trust", "lgsc"]
    metrics = [
        ("final_cumulative_platform_utility", "Cumulative Platform Utility"),
        ("avg_avg_quality_all_non_empty", "Average Data Quality"),
        ("malicious_f1", "Malicious Detection F1"),
        ("avg_trust_rmse_all_non_empty", "Average Trust RMSE"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.35))
    for ax, (key, ylabel) in zip(axes, metrics):
        style_axis(ax)
        use_modes = modes if key not in ("malicious_f1", "avg_trust_rmse_all_non_empty") else \
            ["beta_rep", "crh_td", "trust", "lgsc"]
        values = {}
        for r in rows:
            values[(r["dataset"], r["mode"])] = num(r, key)
        grouped_bars(ax, DATASETS, use_modes, values,
                     [MODE_STYLE[m][1] for m in use_modes], [MODE_STYLE[m][0] for m in use_modes])
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.0)
    save(fig, "Fig_R2_external_baselines")


# ------------------------------------------------------------------ R3
def fig_r3():
    rows = read_csv("summary_E3_threshold_ablation.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ROW)
    vfs = [("linear", "Linear (no threshold)", "#4c72b0"),
           ("hard", "Hard threshold", "#dd8452"),
           ("sigmoid", "Sigmoid threshold", "#b22222")]
    for ax, (key, ylabel) in zip(axes[:2], [
        ("final_cumulative_platform_utility", "Cumulative Platform Utility"),
        ("avg_avg_quality_all_non_empty", "Average Data Quality"),
    ]):
        style_axis(ax)
        values = {}
        for r in rows:
            if r["setting"].startswith("selection_value="):
                values[(r["dataset"], r["setting"].split("=")[1])] = num(r, key)
        grouped_bars(ax, DATASETS, [v[0] for v in vfs], values,
                     [v[2] for v in vfs], [v[1] for v in vfs])
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.36, 1.07))

    ax = axes[2]
    style_axis(ax)
    markers = {"beijing": "o", "rome": "s", "chicago": "^"}
    for ds in DATASETS:
        pts = sorted(
            (float(r["setting"].split("=")[1]), num(r, "avg_completion_rate_all_non_empty"))
            for r in rows if r["dataset"] == ds and r["setting"].startswith("delta=")
        )
        if pts:
            ax.plot(*zip(*pts), marker=markers[ds], linewidth=1.4, markersize=4,
                    label=DATASET_LABELS[ds])
    ax.set_xlabel(r"Completion threshold $\delta$")
    ax.set_ylabel("Task Completion Rate")
    legend(ax, loc="lower left", fontsize=6.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig_R3_threshold_ablation")


# ------------------------------------------------------------------ R4
def fig_r4():
    rows = read_csv("summary_E1_main.csv")
    if not rows:
        return
    modes = ["random", "epsilon_first", "cmab", "trust", "pgrd", "lgsc"]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ROW)
    for ax, (key, ylabel) in zip(axes, [
        ("fairness_jain_selection", "Jain's Index (Selections)"),
        ("fairness_jain_reward", "Jain's Index (Income)"),
        ("fairness_participation_ratio", "Participation Ratio"),
    ]):
        style_axis(ax)
        values = {(r["dataset"], r["mode"]): num(r, key) for r in rows}
        grouped_bars(ax, DATASETS, modes, values,
                     [MODE_STYLE[m][1] for m in modes], [MODE_STYLE[m][0] for m in modes])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig_R4_fairness")


# ------------------------------------------------------------------ R5 / R6
def line_vs_param(csv_name, prefix, xlabel, modes, metrics, name, reverse_x=False):
    rows = read_csv(csv_name)
    if not rows:
        return
    fig, axes = plt.subplots(len(metrics), 3, figsize=(7.2, 2.15 * len(metrics)), squeeze=False)
    for i, (key, ylabel) in enumerate(metrics):
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            style_axis(ax)
            for m in modes:
                pts = sorted(
                    (float(r["setting"].split("=")[1]), num(r, key))
                    for r in rows if r["dataset"] == ds and r["mode"] == m and r["setting"].startswith(prefix)
                )
                if not pts:
                    continue
                label, color = MODE_STYLE[m]
                is_tr = m == "lgsc"
                ax.plot(*zip(*pts), color=color, marker="o", markersize=4.6 if is_tr else 3.6,
                        linewidth=2.2 if is_tr else 1.3, label=label, zorder=7 if is_tr else 4,
                        markeredgecolor="#111111" if is_tr else color, markeredgewidth=0.6)
            if reverse_x:
                ax.invert_xaxis()
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=7.0, bbox_to_anchor=(0.5, 1.0))
    save(fig, name)


def fig_r5():
    line_vs_param(
        "summary_E6_opportunistic_mobility.csv", "availability_keep=",
        "Availability retention probability", ["cmab", "trust", "lgsc"],
        [("avg_completion_rate_all_non_empty", "Task Completion Rate"),
         ("final_cumulative_platform_utility", "Cum. Platform Utility")],
        "Fig_R5_opportunistic_mobility", reverse_x=True,
    )


def fig_r6():
    line_vs_param(
        "summary_E7_seed_trusted_fraction.csv", "seed_trusted_fraction=",
        "Fraction of seed trusted workers", ["beta_rep", "crh_td", "trust", "lgsc"],
        [("malicious_f1", "Malicious Detection F1"),
         ("final_cumulative_platform_utility", "Cum. Platform Utility")],
        "Fig_R6_seed_trusted_fraction",
    )


def main():
    set_style()
    for fn in (fig_r1, fig_r2, fig_r3, fig_r4, fig_r5, fig_r6):
        fn()


if __name__ == "__main__":
    main()
