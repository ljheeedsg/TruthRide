from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from experiment2_paper_plot_style import FIG2_SAVE_PAD, PAPER_BG


CITIES = ["北京", "罗马", "芝加哥"]

FIGURE_GROUPS = [
    {
        "name": "每轮平均数据质量",
        "inputs": [
            "北京每轮平均数据质量.png",
            "罗马每轮平均数据质量.png",
            "芝加哥每轮平均数据质量.png",
        ],
        "output_png": "三城每轮平均数据质量_横向拼图.png",
        "output_pdf": "三城每轮平均数据质量_横向拼图.pdf",
    },
    {
        "name": "任务完成率对比",
        "inputs": [
            "北京任务完成率对比.png",
            "罗马任务完成率对比.png",
            "芝加哥任务完成率对比.png",
        ],
        "output_png": "三城任务完成率对比_横向拼图.png",
        "output_pdf": "三城任务完成率对比_横向拼图.pdf",
    },
    {
        "name": "工人参与时长",
        "inputs": [
            "北京工人参与时长.png",
            "罗马工人参与时长.png",
            "芝加哥工人参与时长.png",
        ],
        "output_png": "三城工人参与时长_横向拼图.png",
        "output_pdf": "三城工人参与时长_横向拼图.pdf",
    },
]


FIGURE_HEIGHT = 3.55
HSPACE = 0.032
SAVE_DPI = 1000


def load_images(image_names):
    image_paths = [Path(name) for name in image_names]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入图片：\n" + "\n".join(missing))
    return [mpimg.imread(path) for path in image_paths]


def stitch_triptych(group):
    images = load_images(group["inputs"])
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
        left=0.008,
        right=0.992,
        top=0.99,
        bottom=0.02,
        wspace=HSPACE,
    )

    fig.savefig(
        group["output_png"],
        bbox_inches="tight",
        pad_inches=FIG2_SAVE_PAD,
        dpi=SAVE_DPI,
    )
    fig.savefig(
        group["output_pdf"],
        bbox_inches="tight",
        pad_inches=FIG2_SAVE_PAD,
        dpi=SAVE_DPI,
    )
    plt.close(fig)

    print(f"Saved {group['output_png']}")
    print(f"Saved {group['output_pdf']}")


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Microsoft YaHei", "SimSun", "Times New Roman", "DejaVu Serif"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 200,
            "savefig.dpi": SAVE_DPI,
            "axes.unicode_minus": False,
        }
    )

    for group in FIGURE_GROUPS:
        stitch_triptych(group)


if __name__ == "__main__":
    main()
