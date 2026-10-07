import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# Fig. 19: Final Retention and Trusted Pool Size
# Style version: compact SCI / engineering paper style
# Only style is changed. Metric computation logic is kept.
# ============================================================

WORKER_OPTIONS_FILE = "experiment2_worker_options.json"

OUTPUT_PNG = "experiment2_fig19_supplement_final_retention_and_trusted_pool.png"
OUTPUT_PDF = "experiment2_fig19_supplement_final_retention_and_trusted_pool.pdf"
OUTPUT_CSV = "experiment2_fig19_supplement_final_retention_and_trusted_pool.csv"

LEGEND_MODE = "manual"
LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = None
LEGEND_NCOL = 1

YLIM_MODE = "manual"
YLIM = (0.0, 1.1)
YTICK_STEP = 0.2

BAR_WIDTH = 0.34


VARIANT_CONFIG = [
    {
        "label": "UCB-Greedy",
        "path": "experiment2_cmab_longrun_round_results_all_runs.json",
        "has_trust_metric": False,
    },
    {
        "label": "Trust-Aware",
        "path": "experiment2_cmab_trust_round_results_all_runs.json",
        "has_trust_metric": True,
    },
    {
        "label": "Membership-Aware",
        "path": "experiment2_cmab_trust_pgrd_round_results_all_runs.json",
        "has_trust_metric": True,
    },
    {
        "label": "TruthRide",
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results_all_runs.json",
        "has_trust_metric": True,
    },
]


METRIC_CONFIG = [
    {
        "key": "final_seed_retention",
        "label": "Final Seed Retention",
        "color": "#1f77b4",
        "hatch": "///",
    },
    {
        "key": "final_trusted_pool_normalized",
        "label": "Final Trusted Pool",
        "color": "#d62728",
        "hatch": "\\\\\\",
    },
]


def load_worker_stats(path):
    with Path(path).open("r", encoding="utf-8") as f:
        worker_options = json.load(f)

    seed_trusted_ids = {
        int(worker["worker_id"])
        for worker in worker_options.values()
        if worker.get("init_category") == "trusted"
    }

    return seed_trusted_ids, len(worker_options)


def load_all_runs(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_final_seed_retention(all_runs_payload, seed_trusted_ids):
    seed_count = max(1, len(seed_trusted_ids))
    final_values = []

    for run_payload in all_runs_payload:
        left_seed_ids = set()

        round_results = [
            item for item in run_payload["round_results"]
            if int(item.get("num_tasks", 0)) > 0
        ]

        for item in round_results:
            left_ids = {
                int(worker_id)
                for worker_id in item.get("left_worker_ids_this_round", [])
            }

            left_seed_ids.update(left_ids & seed_trusted_ids)

        retained = seed_count - len(left_seed_ids)
        final_values.append(retained / seed_count)

    return round(float(np.mean(final_values)), 4) if final_values else 0.0


def compute_final_trusted_pool_normalized(
    all_runs_payload,
    total_workers,
    has_trust_metric,
    seed_trusted_ids,
):
    if not has_trust_metric:
        final_values = []

        for run_payload in all_runs_payload:
            left_seed_ids = set()

            round_results = [
                item for item in run_payload["round_results"]
                if int(item.get("num_tasks", 0)) > 0
            ]

            for item in round_results:
                left_ids = {
                    int(worker_id)
                    for worker_id in item.get("left_worker_ids_this_round", [])
                }

                left_seed_ids.update(left_ids & seed_trusted_ids)

            retained_seed_count = len(seed_trusted_ids) - len(left_seed_ids)
            final_values.append(retained_seed_count / max(1, total_workers))

        return round(float(np.mean(final_values)), 4) if final_values else 0.0

    final_values = []

    for run_payload in all_runs_payload:
        round_results = [
            item for item in run_payload["round_results"]
            if int(item.get("num_tasks", 0)) > 0
        ]

        if not round_results:
            continue

        final_trusted_count = float(round_results[-1].get("trusted_count", 0.0))
        final_values.append(final_trusted_count / max(1, total_workers))

    return round(float(np.mean(final_values)), 4) if final_values else 0.0


def save_plot_data_csv(rows):
    headers = ["algorithm", "final_seed_retention", "final_trusted_pool_normalized"]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            writer.writerow(
                [
                    row["variant"],
                    row["final_seed_retention"],
                    row["final_trusted_pool_normalized"],
                ]
            )


def set_figure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",

            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 5.9,

            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,

            "hatch.linewidth": 0.45,
            "axes.unicode_minus": False,
        }
    )


def plot_figure(rows):
    set_figure_style()

    variants = [row["variant"] for row in rows]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(3.48, 2.26))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    all_values = []

    for idx, metric_config in enumerate(METRIC_CONFIG):
        offsets = x + (idx - 0.5) * BAR_WIDTH
        values = [
            row[metric_config["key"]]
            if row[metric_config["key"]] is not None else np.nan
            for row in rows
        ]

        all_values.extend([value for value in values if not np.isnan(value)])

        bars = ax.bar(
            offsets,
            values,
            width=BAR_WIDTH,
            label=metric_config["label"],
            color=metric_config["color"],
            edgecolor="black",
            linewidth=0.65,
            hatch=metric_config["hatch"],
            alpha=0.92,
            zorder=3,
        )

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color="black",
                zorder=5,
            )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Normalized Value")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=12, ha="right")

    if YLIM_MODE == "manual":
        ax.set_ylim(*YLIM)
        ax.set_yticks(np.arange(YLIM[0], YLIM[1] + 1e-8, YTICK_STEP))
    elif all_values:
        y_min = 0.0
        y_max = max(all_values)
        margin = max(y_max * 0.12, 0.05)
        ax.set_ylim(y_min, y_max + margin)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.48,
        color="#9a9a9a",
        alpha=0.68,
    )
    ax.grid(
        True,
        axis="y",
        which="minor",
        linestyle=":",
        linewidth=0.32,
        color="#b8b8b8",
        alpha=0.45,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_color("#000000")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.0,
        width=0.75,
        pad=2.0,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=1.8,
        width=0.55,
    )

    legend_kwargs = {
        "loc": LEGEND_LOC if LEGEND_MODE == "manual" else "best",
        "ncol": LEGEND_NCOL,
        "frameon": True,
        "fancybox": False,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "black",
        "handlelength": 1.45,
        "handletextpad": 0.30,
        "columnspacing": 0.60,
        "labelspacing": 0.22,
        "borderpad": 0.25,
        "borderaxespad": 0.35,
    }

    if LEGEND_MODE == "manual":
        legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR

    leg = ax.legend(**legend_kwargs)
    leg.get_frame().set_linewidth(0.55)

    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.255, top=0.965)

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.012)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


def main():
    seed_trusted_ids, total_workers = load_worker_stats(WORKER_OPTIONS_FILE)

    rows = []

    for variant in VARIANT_CONFIG:
        all_runs_payload = load_all_runs(variant["path"])

        rows.append(
            {
                "variant": variant["label"],
                "final_seed_retention": compute_final_seed_retention(
                    all_runs_payload,
                    seed_trusted_ids,
                ),
                "final_trusted_pool_normalized": compute_final_trusted_pool_normalized(
                    all_runs_payload,
                    total_workers,
                    variant["has_trust_metric"],
                    seed_trusted_ids,
                ),
            }
        )

    save_plot_data_csv(rows)
    plot_figure(rows)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()