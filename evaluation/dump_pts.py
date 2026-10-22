"""Baselines for the generated objects: an oriented box and the convex hull, both fitted to the same LiDAR points
the generated mesh was fitted to, scored with the same metrics.

python baseline.py <Result> <laz glob> <x0 x1 y0 y1> <work_dir>
Reads work_dir/gen3/metrics.csv, writes work_dir/gen3/baselines.csv
"""
import csv
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinlib import Mission

res, lazg = sys.argv[1], sys.argv[2]
win = list(map(float, sys.argv[3:7]))
work = sys.argv[7]
out = os.path.join(work, "gen3")
M = Mission(res, lazg)
objs = json.load(open(os.path.join(work, "objects.json")))
arr = np.load(os.path.join(work, "objects_points.npz"))
gen = {r["id"]: r for r in csv.DictReader(open(os.path.join(out, "metrics.csv")))}


# same object preparation as gen_place_v2.py step 1 (the wall filter and area clip do not change surviving points)
def split(P3, C3):
    z0 = np.percentile(P3[:, 2], 2)
    keep = P3[:, 2] > z0 + 0.05
    if keep.sum() < 300:
        return [(P3, C3)]
    Qk = P3[keep]
    g = np.floor(Qk[:, :2] / 0.04).astype(np.int64)
    ug, inv = np.unique(g, axis=0, return_inverse=True)
    cnt = np.bincount(inv.ravel())
    vl = DBSCAN(eps=0.12, min_samples=3).fit_predict((ug + 0.5) * 0.04, sample_weight=cnt)
    lab = vl[inv.ravel()]
    parts = []
    for l in set(lab) - {-1}:
        idx = np.where(keep)[0][lab == l]
        Q = P3[idx]
        ext = np.percentile(Q, 98, 0) - np.percentile(Q, 2, 0)
        if len(idx) >= 300 and np.max(ext[:2]) > 0.2 and ext[2] > 0.25:
            lo, hi = Q[:, :2].min(0), Q[:, :2].max(0)
            base = (~keep) & np.all((P3[:, :2] >= lo) & (P3[:, :2] <= hi), 1)
            sel = np.concatenate([idx, np.where(base)[0]])
            parts.append((P3[sel], C3[sel]))
    return parts or [(P3, C3)]


items = {}
for o in objs:
    if o["label"] == "person" or o["n_views"] < 2 or o["n_points"] < 300:
        continue
    P3, C3 = arr[f"o{o['id']}_xyz"].astype(np.float64), arr[f"o{o['id']}_rgb"]
    low = o["label"] in ("chair", "bench", "table", "chair bench")
    if low:
        zf = np.percentile(P3[:, 2], 2)
        k = P3[:, 2] < zf + 1.3
        P3, C3 = P3[k], C3[k]
    ext = np.percentile(P3, 98, 0) - np.percentile(P3, 2, 0)
    parts = split(P3, C3) if np.max(ext[:2]) > 1.5 else [(P3, C3)]
    for j, (Q, C) in enumerate(parts):
        e = np.percentile(Q, 98, 0) - np.percentile(Q, 2, 0)
        if 0.25 < np.max(e) < (4.5 if low else 3.0):
            items[f"{o['id']}" + (f".{j}" if len(parts) > 1 else "")] = Q



n = 0
for k, g in gen.items():
    if k in items and len(items[k]) == int(g["lidar_pts"]):
        np.save(os.path.join(out, f"{k}_pts.npy"), items[k]); n += 1
    else:
        print(k, "points do not match the generation run; not saved", flush=True)
print("saved", n, "of", len(gen), flush=True)
