from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from experiment2_paper_plot_style import FIG2_SAVE_PAD, PAPER_BG


INPUT_FILES = [
    "experiment2_Fig. 20：不同恶意比例下每轮平均数据质量.png",
    "experiment2_Fig. 21：不同恶意比例下每轮平台效用.png",
    "experiment2_Fig. 23：不同恶意比例下任务完成率.png",
    "experiment2_Fig. 22：不同恶意比例下种子可信工人留存率.png",
]

OUTPUT_PNG = "experiment2_拼图_恶意比例参数实验_一行四图.png"
OUTPUT_PDF = "experiment2_拼图_恶意比例参数实验_一行四图.pdf"

def main():
    image_paths = [Path(path) for path in INPUT_FILES]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input images:\n" + "\n".join(missing))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 3.6))
    fig.patch.set_facecolor(PAPER_BG)

    for ax, image_path in zip(axes, image_paths):
        image = mpimg.imread(image_path)
        ax.imshow(image, aspect="auto")
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.04)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=FIG2_SAVE_PAD)
    plt.close(fig)


if __name__ == "__main__":
    main()
