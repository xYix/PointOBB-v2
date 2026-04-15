#!/usr/bin/env python3
"""Combine sigma visualization into a publication-quality figure.

Reads raw_data.npz (saved by vis_sigma.py) and renders a uniform grid
directly from numpy arrays — no cropping from saved PNGs, pixel-perfect.

Layout: 3 rows (Image, P_cls, sigma) x N columns.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--panels', nargs='+', required=True,
                   help='Panel directory names (stems)')
    p.add_argument('--base-dir', default='visualization/sigma_maps')
    p.add_argument('--out', default='visualization/sigma_combined.png')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--cell-size', type=float, default=1.4,
                   help='Cell size in inches')
    p.add_argument('--col-labels', nargs='*', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    N = len(args.panels)

    # Load raw data
    data_list = []
    for name in args.panels:
        npz_path = os.path.join(args.base_dir, name, 'raw_data.npz')
        d = np.load(npz_path)
        data_list.append({
            'img_small': d['img_small'],       # (H, W, 3) uint8
            'sigma': d['sigma'],               # (2, H, W)
            'center_unc': d['center_unc'],     # (H, W)
            'max_cls': d['max_cls'],           # (H, W)
            'gt': d['gt_centers_feat'],        # (N, 2)
        })
        print(f'  Loaded {name}: img {d["img_small"].shape}, sigma {d["sigma"].shape}')

    rows_cfg = [
        ('image',    'Image'),
        ('cls_prob', '$P_{cls}$'),
        ('sigma',    '$\\sigma$'),
    ]
    n_rows = len(rows_cfg)

    col_labels = args.col_labels or [p.split('__')[0] for p in args.panels]

    # Figure layout
    cs = args.cell_size
    fig_w = 0.35 + cs * N + 0.03 * (N - 1)
    fig_h = 0.30 + cs * n_rows + 0.03 * (n_rows - 1)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = GridSpec(n_rows, N, figure=fig,
                  left=0.35 / fig_w, right=0.995,
                  top=1.0 - 0.30 / fig_h, bottom=0.005,
                  wspace=0.03, hspace=0.03)

    for ri, (rtype, rlabel) in enumerate(rows_cfg):
        for ci, dd in enumerate(data_list):
            ax = fig.add_subplot(gs[ri, ci])

            if rtype == 'image':
                ax.imshow(dd['img_small'])
                dot_color, ms = 'r', 0.8
            elif rtype == 'cls_prob':
                ax.imshow(dd['img_small'], alpha=0.25)
                ax.imshow(dd['max_cls'], cmap='RdYlBu_r',
                          alpha=0.75, vmin=0, vmax=0.5)
                dot_color, ms = 'w', 0.8
            else:  # sigma
                v0 = np.percentile(dd['center_unc'], 2)
                v1 = np.percentile(dd['center_unc'], 98)
                ax.imshow(dd['img_small'], alpha=0.25)
                ax.imshow(dd['center_unc'], cmap='RdYlBu',
                          alpha=0.75, vmin=v0, vmax=v1)
                dot_color, ms = 'w', 0.8

            # GT center dots
            gt = dd['gt']
            if len(gt) > 0:
                ax.plot(gt[:, 0], gt[:, 1], '.', color=dot_color,
                        markersize=ms, markeredgewidth=0)

            ax.set_xlim(0, dd['img_small'].shape[1])
            ax.set_ylim(dd['img_small'].shape[0], 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color('#555555')

            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=8, pad=2)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=8, rotation=90, labelpad=4)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight',
                pad_inches=0.01, facecolor='white')
    plt.close(fig)

    from PIL import Image
    im = Image.open(args.out)
    w, h = im.size
    print(f'\nSaved: {args.out}  ({w}x{h}px, {w/args.dpi:.1f}x{h/args.dpi:.1f} in)')


if __name__ == '__main__':
    main()
