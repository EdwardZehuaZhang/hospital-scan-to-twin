"""Snap hand-placed annotation outlines to the LiDAR: the points inside each outline (dilated 5 cm) between 5 cm and
2.3 m above the local floor give the object's oriented box. Writes gt_boxes.json and an overlay image per area.

python gt_snap.py <gt_seeds.json> <bench_dir> <area> [<area> ...]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

seeds = json.load(open(sys.argv[1])); bench = sys.argv[2]


def obox(X):
    xy = X[:, :2] - X[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T)); th = float(np.arctan2(v[1, 1], v[0, 1]))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = xy @ R
    lo, hi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
    c = X[:, :2].mean(0) + R @ ((lo + hi) / 2)
    return c, hi - lo, th


def corners(b):
    R = np.array([[np.cos(b["yaw"]), -np.sin(b["yaw"])], [np.sin(b["yaw"]), np.cos(b["yaw"])]])
    return (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [b["sx"] / 2, b["sy"] / 2]) @ R.T + [b["cx"], b["cy"]]


for area in sys.argv[3:]:
    d = np.load(os.path.join(bench, area, "area_points.npz"))
    P, H = d["P"], d["H"]
    ok = (H > 0.05) & (H < 2.3)
    Q, QH = P[ok], H[ok]
    floor = np.median(P[~ok & (H <= 0.05) & (H > -0.05), 2]) if np.any((H <= 0.05) & (H > -0.05)) else float(np.median(P[:, 2]))
    boxes = []
    for s in seeds[area]:
        poly = np.array(s["poly"]); c0 = poly.mean(0)
        dil = c0 + (poly - c0) * (1 + 0.05 / max(1e-3, np.min(np.linalg.norm(poly - c0, axis=1))))
        m = Path(dil).contains_points(Q[:, :2])
        X = Q[m]
        if len(X) >= 80:
            c, sz, th = obox(X)
            z0 = float(np.percentile(X[:, 2] - QH[m], 50))  # local floor under the object
            top = float(np.percentile(QH[m], 99))
            b = {"cx": float(c[0]), "cy": float(c[1]), "sx": float(sz[0]), "sy": float(sz[1]), "yaw": th,
                 "z0": z0, "sz": top}
        else:  # too few points: keep the hand outline
            c, sz, th = obox(np.c_[poly, np.zeros(len(poly))]) if len(poly) > 2 else (c0, [0.3, 0.3], 0.0)
            b = {"cx": float(c0[0]), "cy": float(c0[1]), "sx": float(np.ptp(poly[:, 0])), "sy": float(np.ptp(poly[:, 1])),
                 "yaw": 0.0, "z0": floor, "sz": 1.0}
        b.update({"id": s["id"], "cls": s["cls"], "n_pts": int(len(X))})
        boxes.append(b)
    json.dump(boxes, open(os.path.join(bench, area, "gt_boxes.json"), "w"), indent=1)
    # overlay for checking
    st = max(1, len(Q) // 1500000)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
    ax.scatter(Q[::st, 0], Q[::st, 1], c=np.clip(QH[::st] / 2.3, 0, 1), cmap="turbo", s=0.15, linewidths=0)
    cols = {"seating": "#00e5ff", "table": "#ffea00", "kiosk": "#ff4081", "sign": "#76ff03", "plant": "#00c853", "other": "#ffffff"}
    for b in boxes:
        cr = corners(b); cr = np.vstack([cr, cr[:1]])
        ax.plot(cr[:, 0], cr[:, 1], color=cols[b["cls"]], lw=1.0)
        ax.text(b["cx"], b["cy"], b["id"], color="w", fontsize=6, ha="center")
    ax.set_aspect("equal"); ax.set_facecolor("#000")
    fig.savefig(os.path.join(bench, area, "gt_overlay.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(area, len(boxes), "boxes;", sum(b["n_pts"] < 80 for b in boxes), "kept as outline", flush=True)
