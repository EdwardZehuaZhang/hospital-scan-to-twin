"""Ground-truth candidates for the object benchmark, from the LiDAR alone (no image detector involved).

Points between 5 cm and 2.3 m above the local floor are kept, large vertical planes (walls, glass, railings) are
removed by iterative RANSAC, and the rest is clustered on a 4 cm grid. Each cluster becomes a candidate with an
oriented box; contact sheets (top and side views in scan colour) and an overview map are rendered for labelling.

python gt_candidates.py <laz glob> <x0 x1 y0 y1> <out_dir>
"""
import glob
import json
import os
import sys

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

lazg = sys.argv[1]; win = list(map(float, sys.argv[2:6])); out = sys.argv[6]
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
            k[1::2] = False
            if k.any():
                P.append(np.stack([x[k], y[k], np.asarray(c.z)[k]], 1))
                C.append(np.stack([np.asarray(c.red)[k], np.asarray(c.green)[k], np.asarray(c.blue)[k]], 1) / 65535.0)
P = np.concatenate(P); C = np.concatenate(C)
# 2 cm grid to keep it manageable
g = np.floor(P / 0.02).astype(np.int64); _, iu = np.unique(g, axis=0, return_index=True); P, C = P[iu], C[iu]
print("points", len(P), flush=True)

# local floor per 0.5 m cell (2nd percentile), smoothed by the median of neighbouring cells
cell = np.floor(P[:, :2] / 0.5).astype(np.int64)
key = (cell[:, 0] - cell[:, 0].min()) * 100000 + (cell[:, 1] - cell[:, 1].min())
uk, inv = np.unique(key, return_inverse=True)
fl = np.full(len(uk), np.nan)
o = np.argsort(inv); s = np.searchsorted(inv[o], np.arange(len(uk)))
for i in range(len(uk)):
    zz = P[o[s[i]:(s[i + 1] if i + 1 < len(uk) else len(o))], 2]
    if len(zz) >= 30:
        fl[i] = np.percentile(zz, 2)
glob_floor = np.nanmedian(fl)
fl = np.where(np.isnan(fl) | (np.abs(fl - glob_floor) > 0.3), glob_floor, fl)
hgt = P[:, 2] - fl[inv]
keep = (hgt > 0.05) & (hgt < 2.3)
Q, QC, QH = P[keep], C[keep], hgt[keep]
print("object-height points", len(Q), flush=True)

# remove large vertical planes (walls, glass, railings)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))
alive = np.ones(len(Q), bool)
for it in range(40):
    idx = np.where(alive)[0]
    if len(idx) < 5000:
        break
    sub = pc.select_by_index(idx)
    plane, inl = sub.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1500)
    n = np.array(plane[:3]); inl = idx[np.asarray(inl)]
    if abs(n[2]) > 0.2:  # not vertical: stop removing (tables and seats are horizontal)
        break
    q = Q[inl]
    # keep only the largest connected strip of the plane, and only if it is wall-like (long and tall)
    gq = np.floor(q / 0.05).astype(np.int64); uq, iq = np.unique(gq, axis=0, return_inverse=True)
    lab = DBSCAN(eps=0.12, min_samples=3).fit_predict((uq + 0.5) * 0.05, sample_weight=np.bincount(iq.ravel()))[iq.ravel()]
    removed = 0
    for l in set(lab) - {-1}:
        m = lab == l
        ext_h = np.ptp(q[m] @ np.array([-n[1], n[0], 0]))
        ext_v = np.ptp(QH[inl][m])
        if ext_h > 2.5 and ext_v > 1.4:
            alive[inl[m]] = False; removed += m.sum()
    print("plane", it, "inliers", len(inl), "removed", removed, flush=True)
    if removed == 0:
        break
Q, QC, QH = Q[alive], QC[alive], QH[alive]

# clusters on a 4 cm grid
gv = np.floor(Q / 0.04).astype(np.int64)
ug, inv2 = np.unique(gv, axis=0, return_inverse=True); cnt = np.bincount(inv2.ravel())
lab = DBSCAN(eps=0.09, min_samples=4).fit_predict((ug + 0.5) * 0.04, sample_weight=cnt)[inv2.ravel()]


def obox(X):
    xy = X[:, :2] - X[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T)); th = float(np.arctan2(v[1, 1], v[0, 1]))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = xy @ R
    lo, hi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
    c = X[:, :2].mean(0) + R @ ((lo + hi) / 2)
    z0, z1 = np.percentile(X[:, 2], 1), np.percentile(X[:, 2], 99)
    return {"cx": float(c[0]), "cy": float(c[1]), "cz": float((z0 + z1) / 2), "sx": float(hi[0] - lo[0]),
            "sy": float(hi[1] - lo[1]), "sz": float(z1 - z0), "yaw": th}


cands = []
for l in sorted(set(lab) - {-1}):
    m = lab == l
    X = Q[m]
    if len(X) < 150:
        continue
    b = obox(X)
    mx = max(b["sx"], b["sy"], b["sz"])
    if mx < 0.2 or mx > 6.0 or b["sz"] < 0.15:
        continue
    cands.append({"cid": len(cands), "n": int(m.sum()), "box": b, "top": float(QH[m].max()), "label": l})
np.savez_compressed(os.path.join(out, "cand_points.npz"), **{f"c{c['cid']}": Q[lab == c["label"]].astype(np.float32) for c in cands},
                    **{f"rgb{c['cid']}": QC[lab == c["label"]].astype(np.float16) for c in cands})
for c in cands:
    c.pop("label")
json.dump(cands, open(os.path.join(out, "candidates.json"), "w"), indent=1)
print("candidates", len(cands), flush=True)

# overview map
fig, ax = plt.subplots(figsize=(14, 14 * (win[3] - win[2]) / (win[1] - win[0])), dpi=110)
st = max(1, len(P) // 1500000)
ax.scatter(P[::st, 0], P[::st, 1], c=np.clip(C[::st] * 1.4, 0, 1), s=0.2, linewidths=0)
for c in cands:
    b = c["box"]; R = np.array([[np.cos(b["yaw"]), -np.sin(b["yaw"])], [np.sin(b["yaw"]), np.cos(b["yaw"])]])
    cr = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * [b["sx"] / 2, b["sy"] / 2]
    pts = cr @ R.T + [b["cx"], b["cy"]]
    ax.plot(pts[:, 0], pts[:, 1], color="#ff3", lw=0.8); ax.text(b["cx"], b["cy"], str(c["cid"]), color="#f0f", fontsize=7, ha="center")
ax.set_aspect("equal"); ax.set_facecolor("#111"); ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
fig.savefig(os.path.join(out, "overview.png"), facecolor="#111", bbox_inches="tight"); plt.close(fig)

# contact sheets: 4 x 5 candidates, each with a top view and a side view
D = np.load(os.path.join(out, "cand_points.npz"))
for s0 in range(0, len(cands), 20):
    fig, axs = plt.subplots(5, 8, figsize=(16, 10.5), dpi=90)
    for k in range(20):
        a1, a2 = axs[k // 4, (k % 4) * 2], axs[k // 4, (k % 4) * 2 + 1]
        a1.axis("off"); a2.axis("off")
        if s0 + k >= len(cands):
            continue
        c = cands[s0 + k]; X = D[f"c{c['cid']}"]; RGB = np.clip(D[f"rgb{c['cid']}"].astype(float) * 1.4, 0, 1)
        st = max(1, len(X) // 6000)
        a1.scatter(X[::st, 0], X[::st, 1], c=RGB[::st], s=1, linewidths=0); a1.set_aspect("equal"); a1.set_facecolor("#222")
        b = c["box"]; R = np.array([[np.cos(b["yaw"]), -np.sin(b["yaw"])], [np.sin(b["yaw"]), np.cos(b["yaw"])]])
        Xr = (X[:, :2] - [b["cx"], b["cy"]]) @ R
        a2.scatter(Xr[::st, 0], X[::st, 2], c=RGB[::st], s=1, linewidths=0); a2.set_aspect("equal"); a2.set_facecolor("#222")
        a1.set_title(f"#{c['cid']}  {b['sx']:.2f}x{b['sy']:.2f}x{b['sz']:.2f} m", fontsize=7, color="k")
        a2.set_title(f"side, top {c['top']:.2f} m", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"sheet_{s0 // 20:02d}.png"), facecolor="white"); plt.close(fig)
print("done", flush=True)
