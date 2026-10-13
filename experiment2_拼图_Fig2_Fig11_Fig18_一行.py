from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from experiment2_paper_plot_style import FIG2_SAVE_PAD, PAPER_BG


INPUT_FILES = [
    "experiment2_Fig. 2：每轮平均数据质量.png",
    "experiment2_Fig. 11：Per-Round Platform Utility.png",
    "experiment2_Fig. 18：Worker Participation Length.png",
]

OUTPUT_PNG = "experiment2_拼图_Fig2_Fig11_Fig18_一行.png"
OUTPUT_PDF = "experiment2_拼图_Fig2_Fig11_Fig18_一行.pdf"

FIGURE_HEIGHT = 3.0
HSPACE = 0.03


def load_images():
    image_paths = [Path(path) for path in INPUT_FILES]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input images:\n" + "\n".join(missing))
    return [mpimg.imread(path) for path in image_paths]


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    images = load_images()
    width_ratios = [image.shape[1] / image.shape[0] for image in images]
    total_width = FIGURE_HEIGHT * sum(width_ratios) + FIGURE_HEIGHT * HSPACE * 2

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(total_width, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.patch.set_facecolor(PAPER_BG)

    for ax, image in zip(axes, images):
        ax.imshow(image, aspect="auto")
        ax.axis("off")

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=HSPACE,
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    plt.close(fig)


if __name__ == "__main__":
    main()
