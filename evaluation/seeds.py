"""Generation variance: re-generate every object from its saved crop with seeds 0, 1 and 2, fit each mesh with the same
procedure, and score it with exact distances.

python seeds.py <work_dir> [<work_dir> ...]     writes <work_dir>/gen3/seeds.csv
"""
import csv
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

works = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robust  # noqa: E402  (fit_generic, score)
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini",
                                                        use_safetensors=True, device="cuda")
for work in works:
    out = os.path.join(work, "gen3")
    rows = []
    for g in csv.DictReader(open(os.path.join(out, "metrics.csv"))):
        k = g["id"]
        cp, pp = os.path.join(out, f"{k}_crop.png"), os.path.join(out, f"{k}_pts.npy")
        if not (os.path.exists(cp) and os.path.exists(pp)):
            continue
        P3 = np.load(pp); rgba = Image.open(cp)
        r = {"id": k, "label": g["label"]}
        for seed in (0, 1, 2):
            mesh = pipe(image=rgba, num_inference_steps=30, octree_resolution=256, num_chunks=8000,
                        generator=torch.manual_seed(seed), output_type="trimesh")[0]
            if len(mesh.faces) > 20000:
                mesh = mesh.simplify_quadric_decimation(face_count=20000)
            V = mesh.vertices @ Rot.from_euler("x", 90, degrees=True).as_matrix().T  # y-up -> z-up
            m = robust.fit_generic(V, np.asarray(mesh.faces), P3)
            s = robust.score(m, P3, P3.mean(0))  # no visibility split here
            r.update({f"s{seed}_mean_cm": round(s["mean_cm"], 2), f"s{seed}_within5_pct": round(s["within5_pct"], 1),
                      f"s{seed}_invented_pct": round(s["invented_pct"], 1)})
            torch.cuda.empty_cache()
        rows.append(r); print(r, flush=True)
    if rows:
        with open(os.path.join(out, "seeds.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("done", flush=True)
