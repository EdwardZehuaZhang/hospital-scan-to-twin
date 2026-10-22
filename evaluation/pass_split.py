"""Split an area of one mission by pass (the route crosses it more than once, minutes apart) and use the passes as
independent observations: pass-to-pass distance (repeatability inside the hospital), floor height difference
(vertical drift), and a held-out test of the object fits (fit on pass A, score on pass B), with an optimised box
baseline under the same protocol.

python pass_split.py <laz glob> <x0 x1 y0 y1> <work_dir>
Writes work_dir/gen3/passes.json and work_dir/gen3/heldout.csv
"""
import csv
import glob
import json
import os
import sys

import laspy
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN

lazg = sys.argv[1]
win = list(map(float, sys.argv[2:6]))
work = sys.argv[6]
out = os.path.join(work, "gen3")
rng = np.random.default_rng(0)

# ---------- 1. every point in the area, with time ----------
P, T = [], []
for f in sorted(glob.glob(lazg)):
    with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
        h = r.header
        if h.maxs[0] < win[0] or h.mins[0] > win[1] or h.maxs[1] < win[2] or h.mins[1] > win[3]:
            continue
        for c in r.chunk_iterator(10_000_000):
            x, y = np.asarray(c.x), np.asarray(c.y)
            k = (x >= win[0]) & (x <= win[1]) & (y >= win[2]) & (y <= win[3])
            if k.any():
                P.append(np.stack([x[k], y[k], np.asarray(c.z)[k]], 1)); T.append(np.asarray(c.gps_time)[k])
P = np.concatenate(P); T = np.concatenate(T)
print("area points", len(P), flush=True)

# ---------- 2. passes = time clusters separated by > 60 s ----------
order = np.argsort(T); ts = T[order]
gaps = np.where(np.diff(ts) > 60)[0]
bounds = [ts[0]] + [ts[g + 1] for g in gaps]
ends = [ts[g] for g in gaps] + [ts[-1]]
passes = [(a, b, int(((T >= a) & (T <= b)).sum())) for a, b in zip(bounds, ends)]
passes = [p for p in passes if p[2] > 0.05 * len(P)]
info = {"area_points": int(len(P)), "passes": [{"start_s": float(a - ts[0]), "end_s": float(b - ts[0]), "points": n}
                                             for a, b, n in passes]}
print(json.dumps(info), flush=True)
if len(passes) < 2:
    json.dump(info, open(os.path.join(out, "passes.json"), "w"), indent=1); sys.exit(0)
A = (T >= passes[0][0]) & (T <= passes[0][1])
B = (T >= passes[1][0]) & (T <= passes[-1][1]) & ~A
info["gap_minutes"] = float((passes[1][0] - passes[0][1]) / 60)


# ---------- 3. pass-to-pass distance ----------
def vox(Q, v):
    g = np.floor(Q / v).astype(np.int64)
    _, i = np.unique(g, axis=0, return_index=True)
    return Q[i]


PA, PB = vox(P[A], 0.02), vox(P[B], 0.02)
d, _ = cKDTree(PA).query(PB, distance_upper_bound=0.3)
ov = np.isfinite(d); dv = d[ov]
info["c2c"] = {"overlap_frac_of_B": round(float(ov.mean()), 3), "median_cm": round(100 * float(np.median(dv)), 2),
               "p90_cm": round(100 * float(np.percentile(dv, 90)), 2), "within_2cm_pct": round(100 * float(np.mean(dv < 0.02)), 1),
               "within_5cm_pct": round(100 * float(np.mean(dv < 0.05)), 1)}
np.save(os.path.join(out, "pass_dist.npy"), dv.astype(np.float32))

# ---------- 4. floor height per 0.5 m cell, pass A vs pass B ----------
def floor_cells(Q):
    g = np.floor(Q[:, :2] / 0.5).astype(np.int64)
    key = g[:, 0] * 100000 + g[:, 1]
    o = np.argsort(key); key, Qz = key[o], Q[o, 2]
    u, s = np.unique(key, return_index=True)
    res = {}
    for kk, a, b in zip(u, s, list(s[1:]) + [len(key)]):
        z = Qz[a:b]
        if len(z) < 50:
            continue
        z0 = np.percentile(z, 2)
        fl = z[z < z0 + 0.05]
        if len(fl) >= 20:
            res[kk] = np.median(fl)
    return res


fa, fb = floor_cells(P[A]), floor_cells(P[B])
common = [k for k in fa if k in fb]
dz = np.array([fb[k] - fa[k] for k in common])
info["floor_dz"] = {"cells": len(common), "median_cm": round(100 * float(np.median(dz)), 2),
                    "mad_cm": round(100 * float(np.median(np.abs(dz - np.median(dz)))), 2),
                    "p90_abs_cm": round(100 * float(np.percentile(np.abs(dz), 90)), 2)}
print(json.dumps(info), flush=True)
json.dump(info, open(os.path.join(out, "passes.json"), "w"), indent=1)

# ---------- 5. held-out object fits ----------
objs = json.load(open(os.path.join(work, "objects.json")))
arr = np.load(os.path.join(work, "objects_points.npz"))
gen = {r["id"]: r for r in csv.DictReader(open(os.path.join(out, "metrics.csv")))}
tree = cKDTree(P)


def split(P3):  # as gen_place_v2.py
    z0 = np.percentile(P3[:, 2], 2); keep = P3[:, 2] > z0 + 0.05
    if keep.sum() < 300:
        return [P3]
    Qk = P3[keep]; g = np.floor(Qk[:, :2] / 0.04).astype(np.int64)
    ug, inv = np.unique(g, axis=0, return_inverse=True); cnt = np.bincount(inv.ravel())
    lab = DBSCAN(eps=0.12, min_samples=3).fit_predict((ug + 0.5) * 0.04, sample_weight=cnt)[inv.ravel()]
    parts = []
    for l in set(lab) - {-1}:
        idx = np.where(keep)[0][lab == l]; Q = P3[idx]
        ext = np.percentile(Q, 98, 0) - np.percentile(Q, 2, 0)
        if len(idx) >= 300 and np.max(ext[:2]) > 0.2 and ext[2] > 0.25:
            lo, hi = Q[:, :2].min(0), Q[:, :2].max(0)
            base = (~keep) & np.all((P3[:, :2] >= lo) & (P3[:, :2] <= hi), 1)
            parts.append(P3[np.concatenate([idx, np.where(base)[0]])])
    return parts or [P3]


items = {}
for o in objs:
    if o["label"] == "person" or o["n_views"] < 2 or o["n_points"] < 300:
        continue
    P3 = arr[f"o{o['id']}_xyz"].astype(np.float64)
    low = o["label"] in ("chair", "bench", "table", "chair bench")
    if low:
        P3 = P3[P3[:, 2] < np.percentile(P3[:, 2], 2) + 1.3]
    ext = np.percentile(P3, 98, 0) - np.percentile(P3, 2, 0)
    parts = split(P3) if np.max(ext[:2]) > 1.5 else [P3]
    for j, Q in enumerate(parts):
        e = np.percentile(Q, 98, 0) - np.percentile(Q, 2, 0)
        if 0.25 < np.max(e) < (4.5 if low else 3.0):
            items[f"{o['id']}" + (f".{j}" if len(parts) > 1 else "")] = Q


def fit_generic(V, F, P3):
    """Same fit as gen_place_v2.fit (per-axis scale in the principal frame, 4 headings + Nelder-Mead on heading and
    offset, base on the floor), for a shape already z-up."""
    surf, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), 20000)
    lo = np.percentile(P3, 2, 0)
    xy = P3[:, :2] - P3[:, :2].mean(0)
    w, vecs = np.linalg.eigh(np.cov(xy.T)); th0 = np.arctan2(vecs[1, 1], vecs[0, 1])
    Rl = Rot.from_euler("z", th0).as_matrix()
    Pl = (P3 - P3.mean(0)) @ Rl
    elo, ehi = np.percentile(Pl, 2, 0), np.percentile(Pl, 98, 0); ext = ehi - elo
    sub = P3[rng.choice(len(P3), min(6000, len(P3)), replace=False)]

    def build(k, dth, dx, dy, X):
        Rk = Rot.from_euler("z", k * np.pi / 2).as_matrix()
        Xr = X @ Rk.T; sl, sh = (surf @ Rk.T).min(0), (surf @ Rk.T).max(0)
        Xr = (Xr - (sl + sh) / 2) * (ext / np.maximum(sh - sl, 1e-6))
        Xr = Xr @ Rot.from_euler("z", th0 + dth).as_matrix().T
        cxy = P3[:, :2].mean(0) + Rl[:2, :2] @ ((elo + ehi) / 2)[:2]
        return Xr + np.array([cxy[0] + dx, cxy[1] + dy, lo[2] + ext[2] / 2])

    best = None
    for k in range(4):
        sol = minimize(lambda x: np.mean(np.minimum(cKDTree(build(k, *x, surf)).query(sub)[0], 0.3)), [0, 0, 0],
                       method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 250})
        if best is None or sol.fun < best[0]:
            best = (sol.fun, k, sol.x)
    _, k, x = best
    return trimesh.Trimesh(build(k, *x, V), F, process=False)


def scores(mesh, Pfit, Peval):
    s, _ = trimesh.sample.sample_surface(mesh, 30000)
    de, _ = cKDTree(s).query(Peval)
    dfit, _ = cKDTree(Pfit).query(s)
    deval, _ = cKDTree(Peval).query(s)
    uns = dfit > 0.10  # surface the fitting pass does not support
    return {"mean_cm": round(100 * float(np.mean(de)), 1), "within5_pct": round(100 * float(np.mean(de < 0.05)), 1),
            "unsupported_by_fit_pct": round(100 * float(np.mean(uns)), 1),
            "of_which_seen_by_other_pass_pct": round(100 * float(np.mean(deval[uns] < 0.05)), 1) if uns.any() else 0.0}


box = trimesh.creation.box(extents=(1, 1, 1))
BV, BF = np.asarray(box.vertices, float), np.asarray(box.faces)
rows, boxrows = [], []
for k, g in gen.items():
    if k not in items:
        continue
    P3 = items[k]
    a = scores(fit_generic(BV, BF, P3), P3, P3)  # optimised box on all points, for the main baseline table
    boxrows.append({"id": k, "label": g["label"], "boxopt_mean_cm": a["mean_cm"], "boxopt_within5_pct": a["within5_pct"],
                    "boxopt_unsupported_pct": a["unsupported_by_fit_pct"]})
    _, ii = tree.query(P3)
    tA, tB = A[ii], B[ii]
    if tA.sum() < 200 or tB.sum() < 200:
        print(k, "not seen in both passes", int(tA.sum()), int(tB.sum()), flush=True); continue
    sc = trimesh.load(os.path.join(out, f"{k}.glb"), force="mesh")
    V = np.asarray(sc.vertices); F = np.asarray(sc.faces)
    r = {"id": k, "label": g["label"], "pts_A": int(tA.sum()), "pts_B": int(tB.sum())}
    for name, (VV, FF) in (("gen", (V, F)), ("box", (BV, BF))):
        m = fit_generic(VV, FF, P3[tA])
        ins = scores(m, P3[tA], P3[tA]); ho = scores(m, P3[tA], P3[tB])
        r.update({f"{name}_insample_mean_cm": ins["mean_cm"], f"{name}_heldout_mean_cm": ho["mean_cm"],
                  f"{name}_heldout_within5_pct": ho["within5_pct"], f"{name}_unsupported_by_A_pct": ho["unsupported_by_fit_pct"],
                  f"{name}_unsupported_seen_by_B_pct": ho["of_which_seen_by_other_pass_pct"]})
    rows.append(r); print(json.dumps(r), flush=True)

with open(os.path.join(out, "boxopt.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(boxrows[0])); w.writeheader(); w.writerows(boxrows)
if rows:
    with open(os.path.join(out, "heldout.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("done", len(rows), flush=True)
