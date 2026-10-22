"""Step 4: room shell (floor, walls, fixed fittings) as a coloured mesh from the LiDAR, objects removed, then one
scene GLB: shell + every generated object as a separate node.

python shell.py <Result> <laz glob> <x0 x1 y0 y1> <work_dir>
"""
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinlib import Mission

res, lazg = sys.argv[1], sys.argv[2]
GD = os.environ.get("GEN_DIR", "gen2")
win = list(map(float, sys.argv[3:7])); work = sys.argv[7]
M = Mission(res, lazg)
P, C, _ = M.load_points(win, pad=0, every=2)
arr = np.load(os.path.join(work, "objects_points.npz"))
objs = json.load(open(os.path.join(work, "objects.json")))
# remove points belonging to detected objects (people included), by 5 cm voxels
def vkey(Q):
    g = np.floor(Q / 0.05).astype(np.int64) + 100000
    return (g[:, 0] * 200003 + g[:, 1]) * 200003 + g[:, 2]
obj_keys = np.unique(np.concatenate([vkey(arr[f"o{o['id']}_xyz"]) for o in objs])) if objs else np.array([], np.int64)
keep = ~np.isin(vkey(P), obj_keys)
P, C = P[keep], C[keep]
# cut the ceiling off above 3.5 m over the floor so the scene can be viewed from above
z0 = np.percentile(P[:, 2], 1)
k = P[:, 2] < z0 + 3.5
P, C = P[k], C[k]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.colors = o3d.utility.Vector3dVector(C)
pc = pc.voxel_down_sample(0.03)
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
pc.orient_normals_consistent_tangent_plane(15)
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=10)
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.08))
bb = o3d.geometry.AxisAlignedBoundingBox(np.array([win[0], win[2], z0 - 0.2]), np.array([win[1], win[3], z0 + 3.5]))
mesh = mesh.crop(bb)
mesh = mesh.simplify_quadric_decimation(400000)
V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
_, ii = cKDTree(np.asarray(pc.points)).query(V)
col = (np.asarray(pc.colors)[ii] * 255).astype(np.uint8)
sh = trimesh.Trimesh(V, F, vertex_colors=np.concatenate([col, np.full((len(col), 1), 255, np.uint8)], 1), process=False)
os.makedirs(os.path.join(work, GD), exist_ok=True)
sh.export(os.path.join(work, GD, "shell.glb"))
scene = trimesh.Scene(); scene.add_geometry(sh, node_name="shell")
meta = json.load(open(os.path.join(work, GD, "objects_v2.json"))) if os.path.exists(os.path.join(work, GD, "objects_v2.json")) else []
for m in meta:
    f = os.path.join(work, GD, f"{m['id']}.glb")
    if os.path.exists(f):
        g = trimesh.load(f, force="mesh")
        scene.add_geometry(g, node_name=f"{m['id']}_{m['label'].replace(' ', '_')}")
# shift to a local origin so viewers do not lose float precision
off = np.array([(win[0] + win[1]) / 2, (win[2] + win[3]) / 2, z0])
scene.apply_translation(-off)
scene.export(os.path.join(work, GD, "scene_twin.glb"))
json.dump({"origin_offset": off.tolist(), "shell_faces": int(len(F)), "objects": len(meta)},
          open(os.path.join(work, GD, "scene_twin.json"), "w"), indent=1)
print("shell faces", len(F), "objects", len(meta))
