import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


FIG2_SINGLE_FIGSIZE = (3.48, 2.26)
FIG2_DOUBLE_FIGSIZE = FIG2_SINGLE_FIGSIZE
FIG2_TRIPLE_FIGSIZE = FIG2_SINGLE_FIGSIZE
FIG2_SAVE_PAD = 0.012

PAPER_BG = "#f7f3eb"
AX_BG = "#fbf9f4"
SPINE_COLOR = "#3f3a34"
TICK_COLOR = "#2f2a25"
GRID_MAJOR = "#d8d0c3"
GRID_MINOR = "#ece6db"
LEGEND_EDGE = "#b9ad9a"
LEGEND_BG = "#f5efe4"
TEXT_COLOR = "#2f2a25"

PALETTES = {
    4: ["#8d8a83", "#b7793c", "#2f5c85", "#8f2d2d"],
    5: ["#8d8a83", "#b7793c", "#2f5c85", "#5f6b3c", "#8f2d2d"],
    6: ["#8d8a83", "#b7793c", "#2f5c85", "#5f6b3c", "#6a4c93", "#8f2d2d"],
}


def get_palette(n_series):
    if n_series in PALETTES:
        return PALETTES[n_series]
    if n_series < 4:
        return PALETTES[4][:n_series]

    base = PALETTES[6][:]
    while len(base) < n_series:
        base.extend(PALETTES[6])
    return base[:n_series]


def apply_fig2_rcparams(
    *,
    font_size=8.6,
    axes_labelsize=9.0,
    xtick_labelsize=8.0,
    ytick_labelsize=8.0,
    legend_fontsize=6.7,
    axes_linewidth=0.95,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.labelsize": axes_labelsize,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.labelsize": xtick_labelsize,
            "ytick.labelsize": ytick_labelsize,
            "xtick.color": TICK_COLOR,
            "ytick.color": TICK_COLOR,
            "text.color": TEXT_COLOR,
            "legend.fontsize": legend_fontsize,
            "axes.linewidth": axes_linewidth,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 800,
            "axes.unicode_minus": False,
        }
    )


def set_paper_line_rcparams():
    apply_fig2_rcparams()


def apply_axis_frame(ax):
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.62,
        color=GRID_MAJOR,
        alpha=0.85,
    )
    ax.grid(
        True,
        which="minor",
        axis="both",
        linestyle="-",
        linewidth=0.36,
        color=GRID_MINOR,
        alpha=0.78,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.02)
        spine.set_color(SPINE_COLOR)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.8,
        width=0.86,
        pad=2.2,
        color=SPINE_COLOR,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.1,
        width=0.62,
        color=SPINE_COLOR,
    )


def style_legend(legend):
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_BG)
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(0.72)
    frame.set_alpha(0.94)


def get_series_emphasis(index, total, highlight_index=None):
    if highlight_index is None:
        highlight_index = total - 1

    is_highlight = index == highlight_index
    return {
        "linewidth": 1.90 if is_highlight else 1.34,
        "markersize": 4.1 if is_highlight else 3.35,
        "markeredgewidth": 0.95 if is_highlight else 0.80,
        "alpha": 0.98 if is_highlight else 0.92,
        "zorder": 8 if is_highlight else 4 + index,
    }


def get_line_path_effects(linewidth):
    return [
        path_effects.Stroke(
            linewidth=linewidth + 1.05,
            foreground="#fffdfa",
            alpha=0.92,
        ),
        path_effects.Normal(),
    ]
