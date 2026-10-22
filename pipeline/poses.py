"""Step 1: refined camera poses for all panoramas in the pilot window, plus a blend check image.
python poses.py <Result> <laz glob> <x0 x1 y0 y1> <out_dir>"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinlib import Mission, lonlat, to_pix, refine_pose, visible
from scipy.spatial.transform import Rotation as Rot
res, lazg = sys.argv[1], sys.argv[2]; win = list(map(float, sys.argv[3:7])); out = sys.argv[7]
os.makedirs(out, exist_ok=True)
M = Mission(res, lazg)
pts, rgb, _ = M.load_points(win, every=4)
gray = rgb.mean(1)
frames = M.frames_in(win)
print("points", len(pts), "frames", len(frames), flush=True)
poses = {}
for fi in frames:
    c, R, cc1, cc0 = refine_pose(M, fi, pts, gray)
    tilt = np.degrees(Rot.from_matrix(Rot.from_euler('Z', -M.P[fi, 6], degrees=True).as_matrix().T @ R).magnitude())
    poses[int(fi)] = {"c": c.tolist(), "R": R.tolist(), "cc": cc1, "cc0": cc0, "corr_deg": tilt}
    print(fi, f"cc {cc0:.3f} -> {cc1:.3f}  correction {tilt:.1f} deg", flush=True)
json.dump(poses, open(os.path.join(out, "poses.json"), "w"))
# blend check on two frames
W, H = 1240, 620
panels = []
for fi in [frames[len(frames) // 3], frames[2 * len(frames) // 3]]:
    p = poses[int(fi)]; c = np.array(p["c"]); R = np.array(p["R"])
    d = pts - c; vis = visible(d); d, col = d[vis], rgb[vis]
    lon, lat, n = lonlat(d @ R); u, v = to_pix(lon, lat, W, H)
    o = np.argsort(-n); ren = np.zeros((H, W, 3), np.float32); ren[v[o], u[o]] = col[o] * 255
    pano = np.asarray(M.image(fi, (W, H)), np.float32)
    filled = ren.sum(2) > 0; b = pano.copy(); b[filled] = 0.5 * pano[filled] + 0.5 * ren[filled]
    panels.append(b.astype(np.uint8))
Image.fromarray(np.concatenate(panels, 0)).save(os.path.join(out, "pose_check.jpg"), quality=80)
ccs = [v["cc"] for v in poses.values()]
print("median cc", np.median(ccs), "min", np.min(ccs))
