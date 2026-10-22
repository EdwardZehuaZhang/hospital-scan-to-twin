"""Visibility split of each generated object's unsupported surface, with the whole scene as occluder.

A sample on the generated mesh that is farther than 10 cm from the object's scan points is 'unsupported'. It is
'in line of sight' if, from at least one panorama position within 10 m, the first surface hit along the ray is the
sample itself (within 3 cm); occluders are the room shell and every object mesh in scene_twin.glb. People are not
modelled, so 'in line of sight' still overstates what the scanner could actually see.

python vis2.py <work_dir>      writes <work_dir>/gen3/vis2.csv
"""
import csv
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

work = sys.argv[1]
out = os.path.join(work, "gen3")
rng = np.random.default_rng(0)
cams = np.array([p["c"] for p in json.load(open(os.path.join(work, "poses.json"))).values()])
scene = trimesh.load(os.path.join(out, "scene_twin.glb"))
sc = o3d.t.geometry.RaycastingScene()
for g in scene.geometry.values():  # untransformed: world frame (the scene graph adds a local offset)
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(g.vertices, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(g.faces, np.int32)))))
rows = []
for g in csv.DictReader(open(os.path.join(out, "metrics.csv"))):
    k = g["id"]; pp = os.path.join(out, f"{k}_pts.npy")
    if not os.path.exists(pp):
        continue
    P3 = np.load(pp)
    m = trimesh.load(os.path.join(out, f"{k}.glb"), force="mesh")
    S, _ = trimesh.sample.sample_surface(m, 20000)
    d, _ = cKDTree(P3).query(S)
    inv = d > 0.10
    near = cams[np.linalg.norm(cams[:, :2] - P3.mean(0)[:2], axis=1) < 10]
    if len(near) > 60:
        near = near[rng.choice(len(near), 60, replace=False)]
    vis = np.zeros(len(S), bool); idx = np.where(inv)[0]
    for c in near:
        if not len(idx):
            break
        Si = S[idx]; dirs = Si - c; dist = np.linalg.norm(dirs, axis=1); dirs /= dist[:, None]
        rays = np.concatenate([np.repeat(c[None], len(Si), 0), dirs], 1).astype(np.float32)
        t = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis[idx] |= np.isfinite(t) & (t >= dist - 0.03)
    r = {"id": k, "label": g["label"], "cams": len(near), "unsupported_pct": round(100 * float(inv.mean()), 1),
         "unsupported_in_sight_pct": round(100 * float((inv & vis).mean()), 1),
         "unsupported_unseen_pct": round(100 * float((inv & ~vis).mean()), 1)}
    rows.append(r); print(json.dumps(r), flush=True)
with open(os.path.join(out, "vis2.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("done", len(rows))
