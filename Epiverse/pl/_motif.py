import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D

def plot_seq_logo(
    counts,
    title="Motif logo",
    use_information=True,
    show_frame=True,
    ax=None,
    figsize_scale=0.4,
    colors=None,
):
    """
    绘制 DNA/蛋白质序列 logo（WebLogo 样式）

    Parameters
    ----------
    counts : dict | np.ndarray
        - dict 形式：{'A': [...], 'C': [...], 'G': [...], 'T': [...]}  
        - ndarray  : shape=(L, 4)；按 A,C,G,T 列顺序
    title : str
        图标题
    use_information : bool
        True → 信息量 logo；False → 频率 logo（每列总高 1）
    show_frame : bool
        是否保留外框四条边
    ax : matplotlib.axes.Axes | None
        若为 None 则自动创建新 figure
    figsize_scale : float
        控制横向尺寸；单个位点宽度约 = figsize_scale
    colors : dict | None
        字母颜色映射；默认 A/C/G/T = 绿/蓝/橙/红
    """
    # --- 0 准备数据 ---
    letters = ["A", "C", "G", "T"]
    if colors is None:
        colors = {"A": "green", "C": "blue", "G": "orange", "T": "red"}

    # 转成 numpy 矩阵 shape=(L,4)
    if isinstance(counts, dict):
        count_mat = np.array([counts[l] for l in letters]).T
    else:  # 假设是 ndarray
        count_mat = np.asarray(counts)
        assert count_mat.shape[1] == 4, "矩阵列数必须为 4（A,C,G,T）"

    L = count_mat.shape[0]

    # 1 计数 → 频率
    freq_mat = count_mat / count_mat.sum(axis=1, keepdims=True)

    # 2 信息量（若需要）
    if use_information:
        entropy = -np.sum(freq_mat * np.log2(freq_mat + 1e-9), axis=1)
        IC = 2 - entropy                          # DNA 最高 2 bits
        height_mat = freq_mat * IC[:, None]       # 每个字母的高度
        ymax = 2
    else:
        height_mat = freq_mat                     # 每列总高 1
        ymax = 1

    # --- 3 开始画图 ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(L * figsize_scale, 2.5))
    fp = FontProperties(family="DejaVu Sans", weight="bold")

    for pos in range(L):
        y_offset = 0
        # 堆叠顺序：先矮后高
        for Ltr in sorted(letters, key=lambda x: height_mat[pos, letters.index(x)]):
            h = height_mat[pos, letters.index(Ltr)]
            if h == 0:
                continue
            tp = TextPath((0, 0), Ltr, size=1, prop=fp)
            trans = (
                Affine2D().scale(1, h)               # 垂直缩放
                + Affine2D().translate(pos + 0.05, y_offset)
            )
            patch = PathPatch(
                tp,
                transform=trans + ax.transData,
                fc=colors[Ltr],
                ec=colors[Ltr],
                lw=0,
            )
            ax.add_patch(patch)
            y_offset += h

    # --- 4 坐标轴/外框样式 ---
    ax.set_xlim(0, L)
    ax.set_ylim(0, ymax)
    ax.set_title(title, pad=6)

    # 关掉刻度线 & 标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)

    if show_frame:
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)
    else:
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(False)

    ax.grid(False)
    plt.tight_layout()
    return ax
