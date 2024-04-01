# -------------------------------------------------------------------------------
# Name:        ScatterD
# Purpose:     scatterplot with distributions
#
# Author:      EPervago
#
# Created:     28/03/2023
# Copyright:   (c) EPervago 2023
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import openturns as ot
import openturns.viewer as otv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_scatter_with_distrib(
    x, y, title="", namex="var1", namey="var2", size=(700, 700)
):
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    fig = plt.figure()
    dpi = fig.get_dpi()
    new_size = [p / dpi for p in size]
    fig.set_size_inches(new_size)
    fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0, wspace=0.0, w_pad=0.0)
    fig.suptitle(title)
    mosaic = [
        ["HistX", "Text", "Text"],
        ["BoxX", "Text", "Text"],
        ["Scatter", "BoxY", "HistY"],
    ]
    axd = fig.subplot_mosaic(
        mosaic,
        height_ratios=[3, 1, 10],
        width_ratios=[10, 1, 3],
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    #    identify_axes(axd)

    axd["HistX"].sharex(axd["Scatter"])
    axd["HistY"].sharey(axd["Scatter"])
    axd["BoxX"].sharex(axd["Scatter"])
    axd["BoxY"].sharey(axd["Scatter"])
    # axd['HistX'].xaxis.set_ticklabels([])
    # axd['HistY'].yaxis.set_ticklabels([])
    plt.setp(axd["HistX"].get_xticklabels(), visible=False)
    plt.setp(axd["HistY"].get_yticklabels(), visible=False)
    axd["Text"].get_xaxis().set_ticks([])
    axd["Text"].get_yaxis().set_ticks([])
    axd["BoxX"].axis("off")
    axd["BoxY"].axis("off")
    axd["Text"].axis("off")
    axd["Scatter"].plot(x, y, ".")
    axd["Scatter"].set_xlabel(namex)
    axd["Scatter"].set_ylabel(namey)
    axd["Scatter"].grid(True)
    boxprops = dict(facecolor="C0", edgecolor="black", linewidth=0.5)
    whiskerprops = dict(color="black", linewidth=1.0, linestyle="--")
    capprops = dict(color="black", linewidth=1.0)
    flierprops = dict(
        marker=".", markerfacecolor="C0", markersize=8, markeredgecolor="none"
    )
    medianprops = dict(color="black", linewidth=2.0)
    axd["BoxX"].boxplot(
        x,
        vert=False,
        patch_artist=True,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops,
    )
    boxprops["facecolor"] = "C1"
    flierprops["markerfacecolor"] = "C1"
    axd["BoxY"].boxplot(
        y,
        vert=True,
        patch_artist=True,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops,
    )
    axd["HistX"].hist(x, bins="auto", facecolor="C0", edgecolor="black", linewidth=0.5)
    axd["HistY"].hist(
        y,
        bins="auto",
        orientation="horizontal",
        facecolor="C1",
        edgecolor="black",
        linewidth=0.5,
    )
    axd["HistX"].set_xlim(axd["Scatter"].get_xlim())
    axd["HistY"].set_ylim(axd["Scatter"].get_ylim())
    #    axd['D'].set_aspect('equal')
    # identify_axes(axd)
    R_Spearman, _ = stats.spearmanr(x, y)
    R_Kendall, _ = stats.kendalltau(x, y)
    R_Pearson, _ = stats.pearsonr(x, y)
    txt = "Spearman = %g\nKendall = %g\nPearson = %g" % (
        R_Spearman,
        R_Kendall,
        R_Pearson,
    )
    # kw = dict(ha="center", va="center", fontsize=10, color="black")
    kw = dict(ha="center", va="center", fontsize="medium", color="black")
    axd["Text"].text(0.5, 0.5, txt, transform=axd["Text"].transAxes, **kw)
    plt.tight_layout()
    return fig  # , axd


def main():
    marginals = [ot.Normal(0.0, 0.5), ot.Normal(0.0, 0.9)]
    R = ot.CorrelationMatrix(2)
    R[0, 1] = -0.6
    copula = ot.NormalCopula(R)
    test_distrib = ot.ComposedDistribution(marginals, copula)
    N = 500
    sample = test_distrib.getSample(N)
    x = sample[:, 0]
    y = sample[:, 1]
    plot_scatter_with_distrib(
        x, y, title="AAAAAAA", namex="var x", namey="var y", size=(700, 700)
    )
    plt.show()


if __name__ == "__main__":
    main()
