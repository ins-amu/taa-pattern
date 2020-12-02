"""
Various utilities for plotting
"""


import itertools
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.stats as stats



class Background():
    def __init__(self, fig=None, visible=False, spacing=0.1, linecolor='0.5', linewidth=1):
        if fig is not None:
            plt.scf(fig)
        ax = plt.axes([0,0,1,1], facecolor=None, zorder=-1000)
        plt.xticks(np.arange(0, 1 + spacing/2., spacing))
        plt.yticks(np.arange(0, 1 + spacing/2., spacing))
        plt.grid()
        if not visible:
            plt.axis('off')
        self.axes = ax
        self.linecolor = linecolor
        self.linewidth = linewidth

    def vline(self, x, y0=0, y1=1, **args):
        defargs = dict(color=self.linecolor, linewidth=self.linewidth)
        defargs.update(args)
        self.axes.add_line(lines.Line2D([x, x], [y0, y1], **defargs))

    def hline(self, y, x0=0, x1=1, **args):
        defargs = dict(color=self.linecolor, linewidth=self.linewidth)
        defargs.update(args)
        self.axes.add_line(lines.Line2D([x0, x1], [y, y], **defargs))

    def labels(self, xs, ys, fontsize=30):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        assert len(xs) == len(ys)
        for x, y, letter in zip(xs, ys, letters):
            self.axes.text(x, y, letter, transform=self.axes.transAxes, size=fontsize, 
                           weight='bold', ha='left', va='bottom')


def axtext(ax, text, **args):
    defargs = {'fontsize': 14, 'ha': 'center', 'va': 'center'}
    defargs.update(args)
    plt.text(0.5, 0.5, text, **defargs)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.axis('off')



def scale_bars(ax, pos, sizex=None, labelx=None, sizey=None, labely=None, barwidth=4, fontsize=14):
    if sizex:
        sizex_ax = ax.transLimits.transform((sizex, 0))[0] - ax.transLimits.transform((0, 0))[0]
        ax.add_artist(mpatches.Rectangle(pos, sizex_ax, 0, lw=barwidth, ec='black', transform=ax.transAxes))
        if labelx:
            ax.annotate(labelx, xy=(pos[0] + 0.5*sizex_ax, pos[1]), xycoords='axes fraction',
                        xytext=(0, -0.5*fontsize), textcoords='offset points', ha='center', va='top',
                        fontsize=fontsize)

    if sizey:
        sizey_ax = ax.transLimits.transform((0, sizey))[1] - ax.transLimits.transform((0, 0))[1]
        ax.add_artist(mpatches.Rectangle(pos, 0, sizey_ax, lw=barwidth, ec='black', transform=ax.transAxes))
        if labely:
            ax.annotate(labely, xy=(pos[0], pos[1] + 0.5*sizey_ax), xycoords='axes fraction',
                        xytext=(-0.5*fontsize, 0), textcoords='offset points', ha='right', va='center',
                        fontsize=fontsize)


def add_panel_letters(fig, axes=None, fontsize=30, xpos=-0.04, ypos=1.05):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if axes is None:
        axes = fig.get_axes()

    if type(xpos) == float:
        xpos = itertools.repeat(xpos)
    if type(ypos) == float:
        ypos = itertools.repeat(ypos)

    for i, (ax, x, y) in enumerate(zip(axes, xpos, ypos)):
        ax.text(x, y, labels[i],
                transform=ax.transAxes, size=fontsize, weight='bold')


def add_line_coords(df, geom_direc):
    df['line_coord'] = [[] for _ in range(len(df))]
    df['slope'] = 0.
    df['intercept'] = 0.

    for i, row in df.iterrows():
        contact_file = os.path.join(geom_direc, f"{row.subject}/seeg.txt")
        cnames = list(np.genfromtxt(contact_file, usecols=(0,), dtype=str))
        cpos = np.genfromtxt(contact_file, usecols=(1,2,3), dtype=float)
        cpos_taa = np.array([cpos[cnames.index(name)] for name in row.contacts])
        line_coord = list(np.linalg.norm(cpos_taa - cpos_taa[0], axis=1))

        slope, intercept, rval, _, _ = stats.linregress(line_coord, row.tfr)

        df.at[i, 'line_coord'] = line_coord
        df.loc[i, 'slope'] = slope
        df.loc[i, 'intercept'] = intercept



def plot_groups(df_file, geom_direc, rec_direc, md_file, img_direc):
    df = pd.read_pickle(df_file)
    add_line_coords(df, geom_direc)

    os.makedirs(img_direc)

    report = """
---
title: "Supplementary information: Detected TAA groups"
geometry: "left=2.6cm,right=2.6cm,top=1.6cm,bottom=2cm"
output: pdf_document
---

Figures below show the detected occurences of TAA groups.
The layout of all figures is the same as of panels B and C in Figure 3 of the main text.
"""

    for irow, taa in df.iterrows():
        filename = os.path.join(img_direc, f"taa-{irow:05d}_{taa.subject}_{taa.rec:04d}.png")
        report += f"\n![TAA group {irow+1} (Subject {taa.subject})]({filename})\n"
        if irow % 2 == 1:
            report += f"\n\clearpage\n"

        rec = np.load(f"{rec_direc}/{taa.subject}/rec_{taa.rec:04d}.npz")
        inds = [list(rec['names']).index(c) for c in taa.contacts]
        nc = len(inds)

        tfr = np.array(taa.tfr)
        tto = np.array(taa.tto)
        tdur = tto - tfr

        t = rec['t']
        mask = (t > np.min(tfr) - 5) * (t < np.max(tto) + 5)
        t = t[mask]

        seeg = rec['seeg'][inds][:, mask]
        seeg *= 3 / np.max(np.abs(seeg))
        taamask = (t > np.min(taa.tfr)) * (t < max(np.max(taa.tto), np.min(taa.tfr) + 5.0))
        taaseeg = seeg[:, taamask]

        pca = PCA(n_components=nc)
        comps = pca.fit_transform(taaseeg.T)
        var_explained = pca.explained_variance_ratio_
        pca_ve_acc = np.cumsum(var_explained)

        # Plotting ---------------------------------------------------------------------------------------------
        plt.figure(figsize=(16, 8))

        # Panel A: Traces, linregress, duration
        ax0 = plt.subplot2grid((2, 5), (0, 0), rowspan=2, colspan=3)
        for i, ind in enumerate(inds):
            plt.plot(t, seeg[i] + taa.line_coord[i], color='k', lw=0.5)
            plt.scatter([tfr[i]], [taa.line_coord[i]], color='r', marker='x', s=140)
            lc = taa.line_coord[i]
            plt.fill_between([tfr[i], tto[i]], [lc-1.2, lc-1.2], [lc+1.2, lc+1.2], color='b', alpha=0.2, zorder=-1, lw=0.)

        x = np.linspace(-1.5, max(taa.line_coord) + 2, 100)
        plt.plot(taa.intercept + taa.slope * x, x, color='r', ls='--')
        plt.text(0.05, 0.1, f"Slope = {taa.tfr_slope:.2f} s/mm\n$R^2$ = {taa.tfr_r2:.2f}",
                 fontsize=12, color='r', ha='left', va='top', transform=plt.gca().transAxes)
        plt.text(0.53, 0.1, f"Duration = {np.mean(tdur):.2f} s",
                 fontsize=12, color='b', ha='left', va='top', transform=plt.gca().transAxes)
        plt.yticks(taa.line_coord, taa.contacts)
        plt.xticks([])

        plt.ylim(-3.5, max(taa.line_coord) + 3.)
        scale_bars(ax0, (0.1, 0.97), sizex=1, labelx="1 s")

        tt0 = np.min(taa.tfr)
        tt1 = max(np.max(taa.tto), np.min(taa.tfr) + 5.0)
        lc = np.max(taa.line_coord[-1])
        plt.plot([tt0, tt1, tt1, tt0, tt0], [-2, -2, lc+2, lc+2, -2], color='g', ls='--', lw=1)

        # Panel C: PCA components
        ax1 = plt.subplot2grid((2, 5), (0, 3), colspan=2)
        plt.title("Principal component analysis", fontsize=14)
        for i in range(nc):
            plt.plot(t[taamask], 0.4*comps[:, i] + i, color='k', lw=0.5)
        plt.xticks([])
        # tticks = np.array([10, 20])
        # plt.xticks(tticks + rec['onset'], [f"{tt} s" for tt in tticks])
        # scale_bars(ax3, (0.1, 0.9), sizex=1, labelx="1 s")
        plt.ylim(-1, nc+1)
        plt.yticks(np.r_[:nc], np.r_[:nc])
        plt.ylabel("Principal components")
        scale_bars(ax1, (0.1, 0.9), sizex=1, labelx="1 s")


        # Panel C2: Variance explained
        ax2 = plt.subplot2grid((2, 5), (1, 3))
        plt.plot(pca_ve_acc, 'kx-')
        plt.axhline(1.0, ls='--', color='0.5')
        plt.xticks(np.r_[:nc], np.r_[:nc] + 1)
        plt.ylabel("Variance explained")
        for i in [0, 1]:
            plt.annotate(f"PCA VE{i+1}", xy=(i, pca_ve_acc[i]),
                        # f"PCA VE{i+1} = {pca_ve_acc[i]:.3f}", xy=(i, pca_ve_acc[i]),
                        textcoords='axes fraction',
                        xytext=(0.4, 0.3 + 0.2*i),
                        fontsize=12, color='g',
                        arrowprops=dict(fc='g', ec='g', arrowstyle="->"), va='center')

        # Panel C3: Principal axes
        ax3 = plt.subplot2grid((2, 5), (1, 4))
        plt.title("Principal axes")
        vmax = np.max(np.abs(pca.components_))
        plt.imshow(pca.components_.T, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(orientation='vertical', ticks=[-0.5, 0, 0.5], shrink=0.6)
        plt.xticks(np.r_[:nc], np.r_[:nc] + 1)
        plt.yticks(np.r_[:nc], taa.contacts)
        plt.xlim(-0.5, nc-0.5)
        plt.ylim(-0.5, nc-0.5)

        plt.subplots_adjust(wspace=0.5, hspace=0.2, left=0.05, right=0.98, top=0.92, bottom=0.1)
        plt.savefig(filename)
        plt.close()
        # ---------------------------------------------------------------------------------------------------------

    with open(md_file, 'w') as fh:
        fh.write(report)

