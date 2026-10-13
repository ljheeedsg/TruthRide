import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment2_paper_plot_style import FIG2_SAVE_PAD, FIG2_SINGLE_FIGSIZE, apply_fig2_rcparams


INPUT_CSV = "experiment2_vehicle.csv"
OUTPUT_PNG = "experiment2_Fig. 17补充：原始每轮可用工人数.png"
OUTPUT_PDF = "experiment2_Fig. 17补充：原始每轮可用工人数.pdf"
OUTPUT_CSV = "experiment2_Fig. 17补充：原始每轮可用工人数.csv"
SLOT_SEC = 600
TOTAL_SLOTS = 144
SMOOTH_WINDOW = 1
YLIM = (0,300)
YTICK_STEP = 20


def load_raw_available_counts():
    slot_vehicle_map = defaultdict(set)

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vehicle_id = int(row["vehicle_id"])
                start_time = int(row["start_time"])
            except (TypeError, ValueError, KeyError):
                continue

            slot_id = start_time // SLOT_SEC
            if 0 <= slot_id < TOTAL_SLOTS:
                slot_vehicle_map[slot_id].add(vehicle_id)

    rounds = list(range(1, TOTAL_SLOTS + 1))
    values = [len(slot_vehicle_map[slot_id]) for slot_id in range(TOTAL_SLOTS)]
    return rounds, values


def moving_average(values, window):
    if window <= 1 or len(values) <= 2:
        return [round(float(value), 4) for value in values]

    half_window = window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        smoothed.append(round(sum(values[start:end]) / (end - start), 4))
    return smoothed


def save_plot_data_csv(rounds, values):
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "raw_available_workers"])
        for round_id, value in zip(rounds, values):
            writer.writerow([round_id, value])


def set_figure_style():
    apply_fig2_rcparams()

def plot_figure():
    set_figure_style()

    rounds, raw_values = load_raw_available_counts()
    values = moving_average(raw_values, SMOOTH_WINDOW)

    fig, ax = plt.subplots(figsize=FIG2_SINGLE_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    ax.plot(
        rounds,
        values,
        color="#4C78A8",
        marker="o",
        linestyle="-",
        linewidth=2.4,
        markersize=4.2,
        markevery=1,
        markerfacecolor="white",
        markeredgewidth=0.55,
        alpha=0.98,
        zorder=4,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Raw Available Workers")
    ax.set_xlim(0, 145)
    ax.set_xticks(range(0, 141, 20))
    ax.set_xticks(range(0, 145, 10), minor=True)
    ax.set_ylim(*YLIM)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yticks(range(YLIM[0], YLIM[1] + 1, YTICK_STEP))

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.65,
        color="#b8b8b8",
        alpha=0.55,
    )
    ax.grid(
        True,
        axis="x",
        which="major",
        linestyle=":",
        linewidth=0.5,
        color="#c7c7c7",
        alpha=0.35,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=3.0, width=0.75, pad=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=1.8, width=0.55)

    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.96)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.012)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    save_plot_data_csv(rounds, values)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    plot_figure()

