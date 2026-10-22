"""Top-down annotation tiles for the object benchmark: points between 5 cm and 2.3 m above the local floor coloured
by height (floor dimmed), with a 0.5 m grid labelled every metre. Also saves the area's points and their height
above the floor, used later to snap hand-placed boxes to the LiDAR.

python gt_tiles.py <laz glob> <x0 x1 y0 y1> <out_dir> [tile_m]
"""
import glob
import os
import sys

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

lazg = sys.argv[1]; win = list(map(float, sys.argv[2:6])); out = sys.argv[6]
tile = float(sys.argv[7]) if len(sys.argv) > 7 else 6.0
os.makedirs(out, exist_ok=True)
P, C = [], []
for f in sorted(glob.glob(lazg)):
    with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
        h = r.header
        if h.maxs[0] < win[0] or h.mins[0] > win[1] or h.maxs[1] < win[2] or h.mins[1] > win[3]:
            continue
        for c in r.chunk_iterator(10_000_000):
            x, y = np.asarray(c.x), np.asarray(c.y)
            k = (x >= win[0]) & (x <= win[1]) & (y >= win[2]) & (y <= win[3])
            if k.any():
                P.append(np.stack([x[k], y[k], np.asarray(c.z)[k]], 1))
                C.append(np.stack([np.asarray(c.red)[k], np.asarray(c.green)[k], np.asarray(c.blue)[k]], 1) / 65535.0)
P = np.concatenate(P); C = np.concatenate(C)
g = np.floor(P / 0.015).astype(np.int64); _, iu = np.unique(g, axis=0, return_index=True); P, C = P[iu], C[iu]
cell = np.floor(P[:, :2] / 0.5).astype(np.int64)
key = (cell[:, 0] - cell[:, 0].min()) * 100000 + (cell[:, 1] - cell[:, 1].min())
uk, inv = np.unique(key, return_inverse=True)
o = np.argsort(inv, kind="stable"); s = np.searchsorted(inv[o], np.arange(len(uk)))
fl = np.full(len(uk), np.nan)
for i in range(len(uk)):
    zz = P[o[s[i]:(s[i + 1] if i + 1 < len(uk) else len(o))], 2]
    if len(zz) >= 30:
        fl[i] = np.percentile(zz, 2)
gf = np.nanmedian(fl)
fl = np.where(np.isnan(fl) | (np.abs(fl - gf) > 0.3), gf, fl)
H = P[:, 2] - fl[inv]
np.savez_compressed(os.path.join(out, "area_points.npz"), P=P.astype(np.float64), C=C.astype(np.float16), H=H.astype(np.float32))
print("points", len(P), "floor", round(float(gf), 3), flush=True)
obj = (H > 0.05) & (H < 2.3)
xs = np.arange(win[0], win[1], tile); ys = np.arange(win[2], win[3], tile)
cm = plt.get_cmap("turbo")
for j, y0 in enumerate(ys[::-1]):
    for i, x0 in enumerate(xs):
        x1, y1 = min(x0 + tile, win[1]), min(y0 + tile, win[3])
        k = (P[:, 0] >= x0) & (P[:, 0] < x1) & (P[:, 1] >= y0) & (P[:, 1] < y1)
        fig, ax = plt.subplots(figsize=(10, 10 * (y1 - y0) / (x1 - x0)), dpi=120)
        kf = k & ~obj
        st = max(1, kf.sum() // 400000)
        ax.scatter(P[kf][::st, 0], P[kf][::st, 1], c=np.clip(C[kf][::st].astype(float) * 0.6, 0, 1), s=0.3, linewidths=0)
        ko = k & obj
        o2 = np.argsort(H[ko])
        ax.scatter(P[ko][o2, 0], P[ko][o2, 1], c=cm(np.clip(H[ko][o2] / 2.3, 0, 1)), s=0.6, linewidths=0)
        for gx in np.arange(np.ceil(x0 * 2) / 2, x1, 0.5):
            ax.axvline(gx, color="w", lw=0.6 if gx % 1 == 0 else 0.25, alpha=0.6)
        for gy in np.arange(np.ceil(y0 * 2) / 2, y1, 0.5):
            ax.axhline(gy, color="w", lw=0.6 if gy % 1 == 0 else 0.25, alpha=0.6)
        ax.set_xticks(np.arange(np.ceil(x0), x1 + 0.01, 1)); ax.set_yticks(np.arange(np.ceil(y0), y1 + 0.01, 1))
        ax.tick_params(labelsize=9); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_facecolor("#000")
        ax.set_title(f"x {x0:.0f} to {x1:.0f}, y {y0:.0f} to {y1:.0f}   (colour = height above floor, blue 0 m to red 2.3 m)", fontsize=9)
        fig.savefig(os.path.join(out, f"tile_{x0:.0f}_{y0:.0f}.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)
print("tiles", len(xs) * len(ys), flush=True)
