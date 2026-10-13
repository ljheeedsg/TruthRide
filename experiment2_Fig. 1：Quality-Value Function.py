import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from experiment2_paper_plot_style import FIG2_SAVE_PAD, FIG2_SINGLE_FIGSIZE, apply_fig2_rcparams

from experiment2_重构_配置 import BASE_CONFIG
from experiment2_重构_评价 import quality_value


OUTPUT_PNG = "experiment2_Fig. 1：Quality-Value Function.png"
OUTPUT_PDF = "experiment2_Fig. 1：Quality-Value Function.pdf"
OUTPUT_CSV = "experiment2_Fig. 1：Quality-Value Function.csv"
LEGEND_MODE = "auto"
LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.015, 0.995)
LEGEND_NCOL = 1
YLIM_MODE = "manual"
YLIM = (-32.0, 32.0)
NUM_POINTS = 201

R_MAX = float(BASE_CONFIG["QUALITY_VALUE_R_MAX"])
K = float(BASE_CONFIG["QUALITY_VALUE_K"])
Q0 = float(BASE_CONFIG["QUALITY_VALUE_Q0"])
RHO = float(BASE_CONFIG["RHO"])


def save_plot_data_csv(q_values, nonlinear_values, linear_values):
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quality_q", "nonlinear_value", "linear_baseline"])
        for q, nonlinear, linear in zip(q_values, nonlinear_values, linear_values):
            writer.writerow([q, nonlinear, linear])


def set_figure_style():
    apply_fig2_rcparams(
        font_size=8.0,
        axes_labelsize=8.6,
        xtick_labelsize=7.6,
        ytick_labelsize=7.6,
        legend_fontsize=6.0,
        axes_linewidth=0.8,
    )

def build_series():
    q_values = [round(i / (NUM_POINTS - 1), 4) for i in range(NUM_POINTS)]
    nonlinear_values = [
        round(quality_value(q, r_max=R_MAX, k=K, q0=Q0), 4) for q in q_values
    ]
    linear_values = [round(RHO * q, 4) for q in q_values]
    return q_values, nonlinear_values, linear_values


def plot_figure():
    set_figure_style()
    q_values, nonlinear_values, linear_values = build_series()
    zero_crossing_q = min(q_values, key=lambda q: abs(quality_value(q, r_max=R_MAX, k=K, q0=Q0)))

    fig, ax = plt.subplots(figsize=FIG2_SINGLE_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    ax.fill_between(
        q_values,
        nonlinear_values,
        0.0,
        where=[value >= 0.0 for value in nonlinear_values],
        color="#9ecae1",
        alpha=0.22,
        interpolate=True,
        zorder=1,
    )
    ax.fill_between(
        q_values,
        nonlinear_values,
        0.0,
        where=[value < 0.0 for value in nonlinear_values],
        color="#fdd0a2",
        alpha=0.24,
        interpolate=True,
        zorder=1,
    )

    ax.plot(
        q_values,
        nonlinear_values,
        color="#1a1a1a",
        linestyle="-",
        linewidth=2.45,
        zorder=5,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )
    ax.plot(
        q_values,
        linear_values,
        color="#3a86c8",
        linestyle="--",
        linewidth=1.85,
        zorder=4,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )

    ax.axhline(0.0, color="#6b6b6b", linestyle=":", linewidth=0.9, alpha=0.85, zorder=2)
    ax.axvline(Q0, color="#4c9f70", linestyle=(0, (1.5, 2.2)), linewidth=1.05, alpha=0.95, zorder=2)
    ax.scatter(
        [Q0, zero_crossing_q],
        [quality_value(Q0, r_max=R_MAX, k=K, q0=Q0), 0.0],
        s=[18, 18],
        color=["#4c9f70", "#1a1a1a"],
        zorder=6,
    )

    ax.text(
        Q0 + 0.016,
        YLIM[0] + (YLIM[1] - YLIM[0]) * 0.07,
        rf"$q_0={Q0:.1f}$",
        color="#3e7f59",
        fontsize=8.8,
    )
    ax.annotate(
        "gain region",
        xy=(0.72, 24.0),
        xytext=(0.60, 27.6),
        fontsize=8.4,
        color="#355c7d",
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#355c7d"},
    )
    ax.annotate(
        "penalty region",
        xy=(0.20, -20.0),
        xytext=(0.08, -26.2),
        fontsize=8.4,
        color="#9a5b2a",
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#9a5b2a"},
    )
    ax.annotate(
        "utility turns positive",
        xy=(zero_crossing_q, 0.0),
        xytext=(0.47, -6.0),
        fontsize=8.2,
        color="#444444",
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "#666666"},
    )

    ax.set_xlabel("Data Quality $q$")
    ax.set_ylabel("Task Value $R(q)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([i / 10 for i in range(0, 11, 2)])
    if YLIM_MODE == "manual":
        ax.set_ylim(*YLIM)
        ax.set_yticks(range(-30, 31, 10))

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.55,
        color="#a6a6a6",
        alpha=0.55,
    )
    ax.grid(
        True,
        axis="x",
        which="major",
        linestyle=":",
        linewidth=0.45,
        color="#bfbfbf",
        alpha=0.32,
    )

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=3.2,
        width=0.75,
        color="#444444",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            label=rf"Nonlinear $R(q)$",
            color="#1a1a1a",
            linestyle="-",
            linewidth=2.45,
        ),
        Line2D(
            [0],
            [0],
            label=rf"Linear baseline $(\rho={RHO:.0f})$",
            color="#3a86c8",
            linestyle="--",
            linewidth=1.85,
        ),
        Line2D(
            [0],
            [0],
            label="Zero-value boundary",
            color="#6b6b6b",
            linestyle=":",
            linewidth=0.9,
        ),
        Patch(
            facecolor="#9ecae1",
            edgecolor="none",
            alpha=0.22,
            label="Positive utility",
        ),
        Patch(
            facecolor="#fdd0a2",
            edgecolor="none",
            alpha=0.24,
            label="Negative utility",
        ),
    ]

    legend_kwargs = {
        "handles": legend_handles,
        "loc": LEGEND_LOC if LEGEND_MODE == "manual" else "best",
        "ncol": LEGEND_NCOL,
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#7a7a7a",
        "handlelength": 2.2,
        "handletextpad": 0.5,
        "columnspacing": 1.0,
        "labelspacing": 0.38,
        "borderpad": 0.45,
    }
    if LEGEND_MODE == "manual":
        legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
    legend = ax.legend(**legend_kwargs)

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.965)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.012)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    save_plot_data_csv(q_values, nonlinear_values, linear_values)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()

