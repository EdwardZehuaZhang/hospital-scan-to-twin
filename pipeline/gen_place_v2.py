"""Step 3 (v2): split merged objects, choose a clean view per object, SAM mask in the crop, label from the crop,
generate a mesh (Hunyuan3D-2mini), fit it with per-axis scale to the LiDAR, colour from the scan, score, export.

python gen_place_v2.py <Result> <laz glob> <x0 x1 y0 y1> <work_dir> [max_objects]
Writes work_dir/gen2/<id>_crop.png, <id>.glb, metrics.csv, objects_v2.json, compare.jpg
"""
import csv
import json
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinlib import Mission, lonlat, to_pix

res, lazg = sys.argv[1], sys.argv[2]
win = list(map(float, sys.argv[3:7]))
work = sys.argv[7]
maxo = int(sys.argv[8]) if len(sys.argv) > 8 else 16
out = os.path.join(work, os.environ.get("GEN_DIR", "gen2")); os.makedirs(out, exist_ok=True)
dev = "cuda"
M = Mission(res, lazg)
poses = {int(k): v for k, v in json.load(open(os.path.join(work, "poses.json"))).items()}
objs = json.load(open(os.path.join(work, "objects.json")))
arr = np.load(os.path.join(work, "objects_points.npz"))
scene_pts, _, _ = M.load_points(win, pad=8, every=4)
GROUPS = [["potted plant", "trash bin", "wet floor sign"], ["kiosk machine", "vending machine", "standing sign board"],
          ["chair", "bench", "table"], ["luggage trolley", "wheelchair", "stretcher"],
          ["queue stanchion", "display stand", "hospital bed"]]
OPERATOR_LON = (-170.0, -35.0)  # camera-frame longitude sector where the operator's body fills the panorama


# ---------- 1. split merged objects ----------
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
            # give the part back its floor-level points within its footprint
            lo, hi = Q[:, :2].min(0), Q[:, :2].max(0)
            base = (~keep) & np.all((P3[:, :2] >= lo) & (P3[:, :2] <= hi), 1)
            sel = np.concatenate([idx, np.where(base)[0]])
            parts.append((P3[sel], C3[sel]))
    return parts or [(P3, C3)]


items, size_drops = [], []
for o in objs:
    if o["label"] == "person" or o["n_views"] < 2 or o["n_points"] < 300:
        continue
    P3, C3 = arr[f"o{o['id']}_xyz"].astype(np.float64), arr[f"o{o['id']}_rgb"]
    low = o["label"] in ("chair", "bench", "table", "chair bench")
    if low:  # seating and tables are under ~1.2 m: cut away wall points the mask bled into
        zf = np.percentile(P3[:, 2], 2)
        k = P3[:, 2] < zf + 1.3
        P3, C3 = P3[k], C3[k]
    ext = np.percentile(P3, 98, 0) - np.percentile(P3, 2, 0)
    parts = split(P3, C3) if np.max(ext[:2]) > 1.5 else [(P3, C3)]
    for j, (Q, C) in enumerate(parts):
        e = np.percentile(Q, 98, 0) - np.percentile(Q, 2, 0)
        if 0.25 < np.max(e) < (4.5 if low else 3.0):
            items.append({"id": f"{o['id']}" + (f".{j}" if len(parts) > 1 else ""), "label0": o["label"],
                          "n_views": o["n_views"], "P": Q, "C": C})
        else:
            size_drops.append({"id": f"{o['id']}.{j}", "label": o["label"], "reason": f"size gate (max extent {np.max(e):.2f} m)"})


def wall_piece(Q):
    """True when the object is a near-planar patch of a larger surface (wall, door, floor): the scan continues in
    the same plane just outside its footprint. Free-standing flat things (sign boards) have empty space around them."""
    c = Q.mean(0)
    w, V = np.linalg.eigh(np.cov((Q - c).T))
    thick = 2 * np.sqrt(max(w[0], 0))
    if thick > 0.08:
        return False, thick, 0.0
    nrm, a1, a2 = V[:, 0], V[:, 2], V[:, 1]
    q = (Q - c) @ np.stack([a1, a2], 1)
    lo, hi = q.min(0), q.max(0)
    S = scene_pts - c
    near = np.abs(S @ nrm) < 0.04
    qs = S[near] @ np.stack([a1, a2], 1)
    m = 0.5
    ring = np.all((qs > lo - m) & (qs < hi + m), 1) & ~np.all((qs > lo) & (qs < hi), 1)
    inside = np.all((qs > lo) & (qs < hi), 1).sum()
    ring_area = (hi[0] - lo[0] + 2 * m) * (hi[1] - lo[1] + 2 * m) - (hi[0] - lo[0]) * (hi[1] - lo[1])
    in_area = max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-3)
    ratio = (ring.sum() / ring_area) / max(inside / in_area, 1e-6)
    return ratio > 0.35, thick, float(ratio)


dropped = []
kept = []
for it in items:
    cen = it["P"].mean(0)
    if not (win[0] <= cen[0] <= win[1] and win[2] <= cen[1] <= win[3]):
        dropped.append({"id": it["id"], "label": it["label0"], "reason": "outside area"}); continue
    wp, thick, ratio = wall_piece(it["P"])
    if wp:
        dropped.append({"id": it["id"], "label": it["label0"], "reason": f"wall/door piece (thickness {thick:.3f} m, coplanar ratio {ratio:.2f})"}); continue
    kept.append(it)
capped = sorted(kept, key=lambda it: (-it["n_views"], -len(it["P"])))[maxo:]
dropped += size_drops + [{"id": it["id"], "label": it["label0"], "reason": f"object cap ({maxo})"} for it in capped]
json.dump(dropped, open(os.path.join(out, "dropped.json"), "w"), indent=1)
print("dropped", len(dropped), dropped, flush=True)
items = sorted(kept, key=lambda it: (-it["n_views"], -len(it["P"])))[:maxo]
print("objects after split", len(items), flush=True)

# ---------- 2. clean view, SAM mask, crop label ----------
gdp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gdm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
samp = SamProcessor.from_pretrained("facebook/sam-vit-base")
samm = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()


def visible_frac(c, Pobj):
    d_obj = Pobj - c
    d_sc = scene_pts - c
    n_sc = np.linalg.norm(d_sc, axis=1)
    near = n_sc < np.linalg.norm(d_obj, axis=1).max() + 1
    lon, lat, n = lonlat(d_sc[near])
    u, v = to_pix(lon, lat, 1024, 512)
    zb = np.full(1024 * 512, np.inf); np.minimum.at(zb, v * 1024 + u, n)
    lo, la, no = lonlat(d_obj)
    uo, vo = to_pix(lo, la, 1024, 512)
    return float(np.mean(no <= zb[vo * 1024 + uo] + 0.08))


def best_view(Pobj):
    cen = Pobj.mean(0)
    best = None
    for fi, p in poses.items():
        if p["cc"] < 0.2:
            continue
        c, R = np.array(p["c"]), np.array(p["R"])
        d = cen - c; dist = np.linalg.norm(d)
        if not 1.0 < dist < 7:
            continue
        lon, lat, _ = lonlat((d @ R)[None])
        lon_d, lat_d = np.degrees(lon[0]), np.degrees(lat[0])
        if OPERATOR_LON[0] < lon_d < OPERATOR_LON[1] or lat_d < -45:
            continue
        cand = (dist, fi, c, R, lat_d)
        if best is None:
            best = []
        best.append(cand)
    if not best:
        return None
    best = sorted(best)[:12]  # nearest dozen usable frames, then score visibility
    scored = []
    for dist, fi, c, R, lat_d in best:
        vf = visible_frac(c, Pobj[::max(1, len(Pobj) // 3000)])
        scored.append((vf / (1 + 0.15 * dist) - 0.004 * abs(lat_d), fi, c, R, vf))
    return max(scored, key=lambda s: s[0])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def crop(it):
    bv = best_view(it["P"])
    if bv is None:
        return None
    _, fi, c, R, vf = bv
    P3 = it["P"]; cen = P3.mean(0)
    fwd = (cen - c) / np.linalg.norm(cen - c)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    Rf = np.stack([right, -up, fwd], 1)
    q = (P3 - c) @ Rf
    ang = np.max(np.abs(np.arctan2(q[:, :2], q[:, 2:3])))
    fov = min(100, max(25, np.degrees(2 * ang) * 1.4))
    S = 512; f = S / 2 / np.tan(np.radians(fov) / 2); K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]])
    jj, ii = np.meshgrid(np.arange(S), np.arange(S))
    rays = np.stack([(jj - S / 2) / f, (ii - S / 2) / f, np.ones_like(jj, float)], -1) @ Rf.T
    lon, lat, _ = lonlat(rays.reshape(-1, 3) @ R)
    img = np.asarray(M.image(fi, (3104, 1552)))
    u, v = to_pix(lon, lat, 3104, 1552)
    rgb = img[v, u].reshape(S, S, 3)
    uv = q @ K.T
    pu = (uv[:, 0] / uv[:, 2]).astype(int); pv = (uv[:, 1] / uv[:, 2]).astype(int)
    k = (pu >= 0) & (pu < S) & (pv >= 0) & (pv < S)
    ptmask = np.zeros((S, S), bool); ptmask[pv[k], pu[k]] = True
    box = [np.percentile(pu[k], 2), np.percentile(pv[k], 2), np.percentile(pu[k], 98), np.percentile(pv[k], 98)]
    face = Image.fromarray(rgb)
    with torch.no_grad():
        si = samp(face, input_boxes=[[box]], input_points=[[[[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]]],
                  return_tensors="pt").to(dev)
        so = samm(**si, multimask_output=False)
        mk = samp.image_processor.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu(),
                                                     si["reshaped_input_sizes"].cpu())[0][0, 0].numpy()
    mask = mk & binary_dilation(ptmask, iterations=14)
    # label from the crop
    label, lscore = it["label0"], 0.0
    for grp in GROUPS:
        with torch.no_grad():
            inp = gdp(images=face, text=". ".join(grp) + ".", return_tensors="pt").to(dev)
            r = gdp.post_process_grounded_object_detection(gdm(**inp), inp.input_ids, threshold=0.3, text_threshold=0.25,
                                                           target_sizes=[(S, S)])[0]
        for b, s, l in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r.get("text_labels", r.get("labels"))):
            if l and iou(b, box) > 0.3 and s > lscore:
                label, lscore = l, float(s)
    rgba = np.concatenate([rgb, (mask * 255).astype(np.uint8)[..., None]], -1)
    return Image.fromarray(rgba, "RGBA"), fi, vf, label, lscore, float(mask.mean())


# ---------- 3. generate and fit ----------
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini",
                                                          use_safetensors=True, device="cuda")


def fit(mesh, P3):
    """Per-axis scale in the object's principal frame, heading among 4 + refine, base on the floor."""
    V = mesh.vertices @ Rot.from_euler("x", 90, degrees=True).as_matrix().T  # y-up -> z-up
    surf, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V, mesh.faces, process=False), 20000)
    lo, hi = np.percentile(P3, 2, 0), np.percentile(P3, 98, 0)
    xy = P3[:, :2] - P3[:, :2].mean(0)
    w, vecs = np.linalg.eigh(np.cov(xy.T)); th0 = np.arctan2(vecs[1, 1], vecs[0, 1])
    Rl = Rot.from_euler("z", th0).as_matrix()
    Pl = (P3 - P3.mean(0)) @ Rl  # LiDAR in its principal frame
    elo, ehi = np.percentile(Pl, 2, 0), np.percentile(Pl, 98, 0)
    ext = ehi - elo
    sub = P3[np.random.default_rng(0).choice(len(P3), min(6000, len(P3)), replace=False)]

    def build(k, dth, dx, dy, V_or_S):
        Rk = Rot.from_euler("z", k * np.pi / 2).as_matrix()
        A = V_or_S @ Rk.T
        sl, sh = (surf @ Rk.T).min(0), (surf @ Rk.T).max(0)
        sc = ext / np.maximum(sh - sl, 1e-6)
        A = (A - (sl + sh) / 2) * sc  # centred, scaled per axis in the principal frame
        Rw = Rot.from_euler("z", th0 + dth).as_matrix()
        A = A @ Rw.T
        cxy = (P3[:, :2].mean(0) + Rl[:2, :2] @ ((elo + ehi) / 2)[:2])
        return A + np.array([cxy[0] + dx, cxy[1] + dy, lo[2] + ext[2] / 2])

    def cost(x, k):
        Q = build(k, x[0], x[1], x[2], surf)
        d, _ = cKDTree(Q).query(sub)
        return np.mean(np.minimum(d, 0.3))

    best = None
    for k in range(4):
        sol = minimize(lambda x: cost(x, k), [0, 0, 0], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 250})
        if best is None or sol.fun < best[0]:
            best = (sol.fun, k, sol.x)
    _, k, x = best
    return trimesh.Trimesh(build(k, x[0], x[1], x[2], V), mesh.faces, process=False)


rows, panels, meta = [], [], []
for it in items:
    k = it["id"]; P3, C3 = it["P"], it["C"]
    cr = crop(it)
    if cr is None:
        print(k, "no clean view", flush=True); continue
    rgba, fi, vf, label, lscore, mfrac = cr
    if mfrac < 0.01:
        print(k, "empty mask", flush=True); continue
    rgba.save(os.path.join(out, f"{k}_crop.png"))
    mesh = pipe(image=rgba, num_inference_steps=30, octree_resolution=256, num_chunks=8000,
                generator=torch.manual_seed(0), output_type="trimesh")[0]
    if len(mesh.faces) > 20000:
        mesh = mesh.simplify_quadric_decimation(face_count=20000)
    placed = fit(mesh, P3)
    samp_m, _ = trimesh.sample.sample_surface(placed, 30000)
    d_obs, _ = cKDTree(samp_m).query(P3)
    d_gen, _ = cKDTree(P3).query(samp_m)
    lo, hi = np.percentile(P3, 2, 0), np.percentile(P3, 98, 0)
    glo, ghi = placed.bounds
    dd, ii = cKDTree(P3).query(placed.vertices)
    vc = np.where(dd[:, None] < 0.15, C3[ii], 0.6)
    placed.visual.vertex_colors = np.concatenate([(vc * 255).astype(np.uint8), np.full((len(vc), 1), 255, np.uint8)], 1)
    placed.export(os.path.join(out, f"{k}.glb"))
    np.save(os.path.join(out, f"{k}_pts.npy"), P3.astype(np.float64))  # the exact points this object was fitted to
    r = {"id": k, "label": label, "label_score": round(lscore, 2), "label_fused": it["label0"], "views": it["n_views"],
         "lidar_pts": len(P3), "frame": fi, "visible_frac": round(vf, 2),
         "scan_to_mesh_mean_cm": round(100 * float(np.mean(d_obs)), 1),
         "scan_to_mesh_p90_cm": round(100 * float(np.percentile(d_obs, 90)), 1),
         "scan_within_5cm_pct": round(100 * float(np.mean(d_obs < 0.05)), 1),
         "mesh_unsupported_pct": round(100 * float(np.mean(d_gen > 0.10)), 1),
         "size_lidar_m": np.round(hi - lo, 2).tolist(), "size_mesh_m": np.round(ghi - glo, 2).tolist()}
    rows.append(r); meta.append({**r, "bbox_min": lo.tolist(), "bbox_max": hi.tolist()}); print(json.dumps(r), flush=True)

    def side(Pts, col):
        im = Image.new("RGB", (256, 256), (20, 20, 20)); dr = ImageDraw.Draw(im)
        c0 = (lo + hi) / 2; sc = 220 / max(1e-3, *(hi - lo))
        az = np.radians(30); x = (Pts[:, 0] - c0[0]) * np.cos(az) + (Pts[:, 1] - c0[1]) * np.sin(az); y = Pts[:, 2] - c0[2]
        st = max(1, len(x) // 6000)
        for xi, yi, cc in zip(x[::st], y[::st], col[::st]):
            dr.point((128 + xi * sc, 128 - yi * sc), fill=tuple(int(255 * t) for t in cc))
        return im
    sv, _ = trimesh.sample.sample_surface(placed, 8000); _, si_ = cKDTree(P3).query(sv)
    panel = Image.new("RGB", (768, 280)); bg = Image.new("RGB", (512, 512), (40, 40, 40)); bg.paste(rgba, (0, 0), rgba)
    panel.paste(bg.resize((256, 256)), (0, 24)); panel.paste(side(P3, C3), (256, 24)); panel.paste(side(sv, C3[si_]), (512, 24))
    ImageDraw.Draw(panel).text((4, 4), f"#{k} {label}  scan->mesh {r['scan_to_mesh_mean_cm']} cm  within5cm {r['scan_within_5cm_pct']}%  vis {vf:.2f}", fill=(255, 255, 0))
    panels.append(panel); torch.cuda.empty_cache()

if rows:
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(meta, open(os.path.join(out, "objects_v2.json"), "w"), indent=1)
    sheet = Image.new("RGB", (768, 280 * len(panels)))
    for i, p in enumerate(panels):
        sheet.paste(p, (0, 280 * i))
    sheet.save(os.path.join(out, "compare.jpg"), quality=85)
print("done", len(rows))
