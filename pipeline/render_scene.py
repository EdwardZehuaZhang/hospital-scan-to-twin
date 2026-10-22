"""Oblique preview of scene_twin.glb: shell in scan colour (dimmed), objects in scan colour with outline tint.
python render_scene.py <scene_twin.glb> <out.png>"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sc = trimesh.load(sys.argv[1])
pts, cols, isobj = [], [], []
shell_name = max(sc.geometry, key=lambda k: len(sc.geometry[k].faces))
for name, g in sc.geometry.items():
    node = [n for n in sc.graph.nodes_geometry if sc.graph[n][1] == name][0]
    T = sc.graph[node][0]
    m = g.copy(); m.apply_transform(T)
    is_shell = name == shell_name
    n = 500000 if is_shell else 25000
    p, fi = trimesh.sample.sample_surface(m, n)
    vc = m.visual.vertex_colors[m.faces[fi][:, 0], :3] / 255.0
    pts.append(p); cols.append(vc); isobj.append(np.full(len(p), not is_shell))
P = np.concatenate(pts); C = np.concatenate(cols); O = np.concatenate(isobj)
keep = P[:, 2] < 2.6  # drop ceiling for the view
P, C, O = P[keep], C[keep], O[keep]
C = np.where(O[:, None], np.clip(C * 0.5 + np.array([0.45, 0.3, 0.0]), 0, 1), C * 0.8)
fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=100)
for ax, (elev, az) in zip(axes, [(90, -90), (35, -60)]):
    ax.remove()
for i, (elev, az) in enumerate([(90, -90), (35, -60)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    o = np.argsort(O)
    ax.scatter(P[o, 0], P[o, 1], P[o, 2], c=C[o], s=0.3, linewidths=0)
    ax.view_init(elev=elev, azim=az); ax.set_axis_off(); ax.set_facecolor('#111')
    r = (P[:, :2].max(0) - P[:, :2].min(0)).max() / 2; c = (P[:, :2].max(0) + P[:, :2].min(0)) / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(-0.5, 2 * r - 0.5)
fig.patch.set_facecolor("#111")
plt.tight_layout(); plt.savefig(sys.argv[2], facecolor="#111")
print("objects pts", int(O.sum()), "shell pts", int((~O).sum()))
