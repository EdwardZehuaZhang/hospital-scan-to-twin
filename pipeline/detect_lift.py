"""Step 2: open-vocabulary detection in panorama views, SAM masks, lift to LiDAR points, fuse across frames.

python detect_lift.py <Result> <laz glob> <x0 x1 y0 y1> <work_dir> [frame_step]
Writes work_dir/objects.json, objects_points.npz, det_samples/*.jpg
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinlib import Mission, cube_faces, visible

res, lazg = sys.argv[1], sys.argv[2]
win = list(map(float, sys.argv[3:7]))
work = sys.argv[7]
step = int(sys.argv[8]) if len(sys.argv) > 8 else 3
os.makedirs(os.path.join(work, "det_samples"), exist_ok=True)
GROUPS = [["potted plant", "trash bin", "wet floor sign"],
          ["kiosk machine", "vending machine", "standing sign board"],
          ["chair", "bench", "table"],
          ["luggage trolley", "wheelchair", "stretcher"],
          ["queue stanchion", "display stand", "hospital bed"]]
dev = "cuda"
gdp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gdm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
samp = SamProcessor.from_pretrained("facebook/sam-vit-base")
samm = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()

M = Mission(res, lazg)
poses = {int(k): v for k, v in json.load(open(os.path.join(work, "poses.json"))).items()}
pts, rgb, _ = M.load_points(win, pad=8, every=2)
print("points", len(pts), flush=True)
frames = sorted(poses)[::step]
dets = []
nsamp = 0
for fi in frames:
    p = poses[fi]
    if p["cc"] < 0.2:
        continue
    c, R = np.array(p["c"]), np.array(p["R"])
    d = pts - c
    dist = np.linalg.norm(d, axis=1)
    near = np.where((dist > 0.5) & (dist < 10))[0]
    vis = near[visible(d[near])]
    img = M.image(fi, (3104, 1552))
    for face, Rf, K, (fy, fp) in cube_faces(img, R, size=768):
        if fp < 0:
            continue  # downward face is mostly the operator
        boxes, scores, labs = [], [], []
        for grp in GROUPS:
            with torch.no_grad():
                inp = gdp(images=face, text=". ".join(grp) + ".", return_tensors="pt").to(dev)
                outp = gdm(**inp)
                r = gdp.post_process_grounded_object_detection(outp, inp.input_ids, threshold=0.3, text_threshold=0.25,
                                                               target_sizes=[face.size[::-1]])[0]
            ll = r.get("text_labels", r.get("labels"))
            for b, s, l in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), ll):
                if l and l in " ".join(grp):
                    boxes.append(b); scores.append(s); labs.append(l)
        if not boxes:
            continue
        boxes, scores = np.array(boxes), np.array(scores)
        # class-agnostic NMS
        order = np.argsort(-scores); kept = []
        for i in order:
            b = boxes[i]
            def iou(a, c):
                ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1]))
                inter = ix * iy; return inter / ((a[2]-a[0])*(a[3]-a[1]) + (c[2]-c[0])*(c[3]-c[1]) - inter + 1e-9)
            if all(iou(b, boxes[j]) < 0.6 for j in kept):
                kept.append(i)
        boxes, scores, labs = boxes[kept], scores[kept], [labs[i] for i in kept]
        keep = [i for i, b in enumerate(boxes) if (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * 768 * 768]
        if not keep:
            continue
        boxes, scores, labs = boxes[keep], scores[keep], [labs[i] for i in keep]
        with torch.no_grad():
            si = samp(face, input_boxes=[boxes.tolist()], return_tensors="pt").to(dev)
            so = samm(**si, multimask_output=False)
            masks = samp.image_processor.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu(),
                                                            si["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
        # project visible points into this face
        q = (pts[vis] - c) @ Rf  # face camera coords (x right, y down, z fwd)
        front = q[:, 2] > 0.1
        uv = (q[front] @ K.T)
        u = (uv[:, 0] / uv[:, 2]).astype(int); v = (uv[:, 1] / uv[:, 2]).astype(int)
        ok = (u >= 0) & (u < 768) & (v >= 0) & (v < 768)
        idx_face = vis[front][ok]; u, v = u[ok], v[ok]
        if nsamp < 16:
            dr = face.copy(); g = ImageDraw.Draw(dr)
            for b, l, s in zip(boxes, labs, scores):
                g.rectangle(b.tolist(), outline=(255, 0, 0), width=3); g.text((b[0] + 3, b[1] + 3), f"{l} {s:.2f}", fill=(255, 255, 0))
            dr.save(os.path.join(work, "det_samples", f"f{fi}_y{fy}_p{fp}.jpg")); nsamp += 1
        for b, l, s, mk in zip(boxes, labs, scores, masks):
            inside = mk[v, u]
            ids = idx_face[inside]
            if len(ids) < 50:
                continue
            # keep the depth layer of the object: points within 0.6 m of the 30th-percentile range
            rng = np.linalg.norm(pts[ids] - c, axis=1)
            ids = ids[rng < np.percentile(rng, 30) + 0.6]
            if len(ids) < 50:
                continue
            if len(ids) > 100000:  # very dense masks (open campus scenes): cluster 3 cm voxels weighted by count to fit in memory
                gv = np.floor(pts[ids] / 0.03).astype(np.int64)
                ug, inv = np.unique(gv, axis=0, return_inverse=True)
                cnt = np.bincount(inv.ravel())
                lab = DBSCAN(eps=0.12, min_samples=8).fit_predict((ug + 0.5) * 0.03, sample_weight=cnt)[inv.ravel()]
            else:
                lab = DBSCAN(eps=0.12, min_samples=8).fit_predict(pts[ids])
            if (lab >= 0).sum() == 0:
                continue
            main = np.bincount(lab[lab >= 0]).argmax()
            ids = ids[lab == main]
            ext = np.percentile(pts[ids], 98, 0) - np.percentile(pts[ids], 2, 0)
            if ext[2] < 0.2 or np.max(ext) > 3.5:
                continue  # floor/wall patches or building parts
            dets.append({"frame": int(fi), "label": l, "score": float(s), "ids": ids})
    print("frame", fi, "detections so far", len(dets), flush=True)

# fuse detections into objects: union-find on overlapping point sets (Jaccard on voxels) with label vote
vox = lambda ids: set(map(tuple, np.floor(pts[ids] / 0.1).astype(int)))
V = [vox(dd["ids"]) for dd in dets]
cent = np.array([pts[dd["ids"]].mean(0) for dd in dets]) if dets else np.zeros((0, 3))
parent = list(range(len(dets)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
tree = cKDTree(cent)
for i, j in tree.query_pairs(1.5):
    inter = len(V[i] & V[j])
    if inter / max(1, min(len(V[i]), len(V[j]))) > 0.3:
        parent[find(i)] = find(j)
groups = {}
for i in range(len(dets)):
    groups.setdefault(find(i), []).append(i)
objs, arrays = [], {}
for k, members in enumerate(sorted(groups.values(), key=len, reverse=True)):
    votes = {}
    for i in members:
        votes[dets[i]["label"]] = votes.get(dets[i]["label"], 0) + dets[i]["score"]
    label = max(votes, key=votes.get)
    ids = np.unique(np.concatenate([dets[i]["ids"] for i in members]))
    P3 = pts[ids]
    lo, hi = np.percentile(P3, 2, 0), np.percentile(P3, 98, 0)
    o = {"id": k, "label": label, "n_views": len(members), "n_points": int(len(ids)),
         "score": float(np.mean([dets[i]["score"] for i in members])),
         "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "size": (hi - lo).tolist(),
         "frames": sorted({dets[i]["frame"] for i in members})}
    objs.append(o)
    arrays[f"o{k}_xyz"] = P3.astype(np.float32); arrays[f"o{k}_rgb"] = rgb[ids]
json.dump(objs, open(os.path.join(work, "objects.json"), "w"), indent=1)
np.savez_compressed(os.path.join(work, "objects_points.npz"), **arrays)
print("detections", len(dets), "objects", len(objs))
for o in objs[:40]:
    print(o["id"], o["label"], "views", o["n_views"], "pts", o["n_points"], "size", np.round(o["size"], 2).tolist())
