"""Fair comparison of the generated objects against controls, all fitted by the same procedure to the same points
and scored with exact point-to-triangle distances.

Models per object:
  gen       the generated mesh as placed by gen_place_v2.py
  gen_unif  the generated mesh refitted with one uniform scale (no per-axis stretch)
  shuffled  another object's generated mesh, refitted to this object's points (does the image matter?)
  box       a unit box, refitted (four headings + Nelder-Mead, per-axis scale), same as the mesh
  hull      convex hull of the points (no free parameters)
  scanmesh  screened Poisson mesh of the points themselves (upper bound on fit, nothing completed)
Metrics: scan-to-model mean distance and share within 5 cm (exact, Open3D raycasting); precision, recall and
F-score at 5 cm and at 5 % of the object's bounding-box diagonal; invented surface (model samples > 10 cm from the
scan), split into 'visible' (in line of sight of at least one panorama position within 10 m, so a scan point
should exist there) and 'unseen' (never in line of sight).

python robust.py <work_dir>        reads work_dir/gen3/{metrics.csv,<id>.glb,<id>_pts.npy}, work_dir/poses.json
writes work_dir/gen3/robust.csv
"""
import csv
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(0)


def fit_generic(V, F, P3, uniform=False):
    """As gen_place_v2.fit for a z-up shape: scale to the 2-98 % extents of the points in their horizontal principal
    frame (per axis, or one uniform factor), base on the floor, heading among 4 quarter turns + Nelder-Mead on heading
    and horizontal offset minimising the truncated (30 cm) mean distance from points to the shape."""
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
        sc = ext / np.maximum(sh - sl, 1e-6)
        if uniform:
            sc = np.full(3, np.exp(np.mean(np.log(sc))))
        Xr = (Xr - (sl + sh) / 2) * sc
        Xr = Xr @ Rot.from_euler("z", th0 + dth).as_matrix().T
        cxy = P3[:, :2].mean(0) + Rl[:2, :2] @ ((elo + ehi) / 2)[:2]
        return Xr + np.array([cxy[0] + dx, cxy[1] + dy, lo[2] + (sh - sl)[2] * sc[2] / 2])

    best = None
    for k in range(4):
        sol = minimize(lambda x: np.mean(np.minimum(cKDTree(build(k, *x, surf)).query(sub)[0], 0.3)), [0, 0, 0],
                       method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 250})
        if best is None or sol.fun < best[0]:
            best = (sol.fun, k, sol.x)
    _, k, x = best
    return trimesh.Trimesh(build(k, *x, V), F, process=False)


def scene_of(mesh):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))))
    return sc


def score(mesh, P3, cen, cams=None):
    sc = scene_of(mesh)
    d_obs = sc.compute_distance(o3d.core.Tensor(P3.astype(np.float32))).numpy()
    S, fi = trimesh.sample.sample_surface(mesh, 20000)
    d_gen, _ = cKDTree(P3).query(S)
    diag = float(np.linalg.norm(np.percentile(P3, 98, 0) - np.percentile(P3, 2, 0)))
    r = {"mean_cm": 100 * float(np.mean(d_obs)), "within5_pct": 100 * float(np.mean(d_obs < 0.05))}
    for tag, tau in (("5cm", 0.05), ("5pctdiag", 0.05 * diag)):
        prec, rec = float(np.mean(d_gen < tau)), float(np.mean(d_obs < tau))
        r[f"F_{tag}"] = 100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    inv = d_gen > 0.10
    r["invented_pct"] = 100 * float(np.mean(inv))
    # visibility of each invented sample from the panorama positions within 10 m (self-occlusion only)
    near = np.zeros((0, 3)) if cams is None else cams[np.linalg.norm(cams[:, :2] - cen[:2], axis=1) < 10]
    if len(near) > 40:
        near = near[rng.choice(len(near), 40, replace=False)]
    vis = np.zeros(len(S), bool)
    idx = np.where(inv)[0]
    if len(idx) and len(near):
        Si = S[idx]
        for c in near:
            dirs = Si - c; dist = np.linalg.norm(dirs, axis=1); dirs /= dist[:, None]
            rays = np.concatenate([np.repeat(c[None], len(Si), 0), dirs], 1).astype(np.float32)
            t = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            vis[idx] |= t >= dist - 0.02
    r["invented_visible_pct"] = 100 * float(np.mean(inv & vis))
    r["invented_unseen_pct"] = 100 * float(np.mean(inv & ~vis))
    return r


def scanmesh(P3):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P3))
    p = p.voxel_down_sample(0.01)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    p.orient_normals_towards_camera_location(P3.mean(0) + np.array([0, 0, 3.0]))
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p, depth=8)
    dens = np.asarray(dens); m.remove_vertices_by_mask(dens < np.quantile(dens, 0.1))
    m = m.crop(o3d.geometry.AxisAlignedBoundingBox(P3.min(0) - 0.02, P3.max(0) + 0.02))
    return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)


if __name__ == "__main__":
    work = sys.argv[1]
    out = os.path.join(work, "gen3")
    gen = list(csv.DictReader(open(os.path.join(out, "metrics.csv"))))
    cams = np.array([p["c"] for p in json.load(open(os.path.join(work, "poses.json"))).values()])
    objs = []
    for g in gen:
        k = g["id"]
        pp = os.path.join(out, f"{k}_pts.npy")
        if not os.path.exists(pp):
            print(k, "no saved points", flush=True); continue
        P3 = np.load(pp)
        m = trimesh.load(os.path.join(out, f"{k}.glb"), force="mesh")
        objs.append((k, g["label"], P3, np.asarray(m.vertices, float), np.asarray(m.faces)))

    box = trimesh.creation.box(extents=(1, 1, 1))
    rows = []
    for i, (k, label, P3, V, F) in enumerate(objs):
        cen = P3.mean(0)
        # control: the mesh of the next object (in list order) that carries a different label
        j = next((jj % len(objs) for jj in range(i + 1, i + len(objs)) if objs[jj % len(objs)][1] != label), (i + 1) % len(objs))
        _, _, _, Vs, Fs = objs[j]
        models = {"gen": trimesh.Trimesh(V, F, process=False),
                  "gen_unif": fit_generic(V, F, P3, uniform=True),
                  "shuffled": fit_generic(Vs, Fs, P3),
                  "box": fit_generic(np.asarray(box.vertices, float), np.asarray(box.faces), P3),
                  "hull": trimesh.convex.convex_hull(P3[rng.choice(len(P3), min(20000, len(P3)), replace=False)]),
                  "scanmesh": scanmesh(P3)}
        r = {"id": k, "label": label, "pts": len(P3),
             "size_m": "x".join(f"{v:.2f}" for v in np.percentile(P3, 98, 0) - np.percentile(P3, 2, 0))}
        for name, mesh in models.items():
            for kk, vv in score(mesh, P3, cen, cams).items():
                r[f"{name}_{kk}"] = round(vv, 2)
        rows.append(r); print(json.dumps(r), flush=True)

    with open(os.path.join(out, "robust.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("done", len(rows), flush=True)
