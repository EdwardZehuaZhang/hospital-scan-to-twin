"""Shared helpers for the UREx twin pilot: LiGrip panorama poses, LiDAR loading, projection.

Pose convention (found 20 Oct 2026 by overlaying LiDAR on panoramas): the imglist position is the camera
centre in the LAZ frame; the logged yaw is a clockwise heading, so R0 = Rz(-yaw) maps camera axes
(x forward, y left, z up) to world. The residual tilt (handheld, about 5 to 10 deg) is refined per frame
by maximising the correlation between panorama brightness and the vendor point colours.
Equirect mapping: u = (0.5 - lon / 2pi) * W, v = (0.5 - lat / pi) * H.
"""
import glob
import json
import os

import laspy
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot


class Mission:
    def __init__(self, result_dir, laz_glob):
        self.res = result_dir
        self.m = os.path.basename(os.path.dirname(result_dir.rstrip("\\/")))
        rows = [l.strip().split(",") for l in open(os.path.join(result_dir, self.m + ".imglist"))
                if l.strip() and not l.startswith("#")]
        self.paths = [r[1] for r in rows]
        self.P = np.array([[float(v) for v in (r[0], *r[3:15])] for r in rows])  # t X Y Z r p y ...
        self.laz_glob = laz_glob

    def frames_in(self, win):
        x0, x1, y0, y1 = win
        P = self.P
        return np.where((P[:, 1] > x0) & (P[:, 1] < x1) & (P[:, 2] > y0) & (P[:, 2] < y1))[0]

    def image(self, fi, size=None):
        im = Image.open(os.path.join(self.res, self.paths[fi])).convert("RGB")
        return im.resize(size) if size else im

    def load_points(self, win, pad=12.0, every=1):
        x0, x1, y0, y1 = win
        pts, rgb, gt = [], [], []
        for f in sorted(glob.glob(self.laz_glob)):
            with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
                h = r.header
                if h.maxs[0] < x0 - pad or h.mins[0] > x1 + pad or h.maxs[1] < y0 - pad or h.mins[1] > y1 + pad:
                    continue
                for c in r.chunk_iterator(5_000_000):
                    x, y = np.asarray(c.x), np.asarray(c.y)
                    k = (x > x0 - pad) & (x < x1 + pad) & (y > y0 - pad) & (y < y1 + pad)
                    if every > 1:
                        k[np.arange(len(x)) % every != 0] = False
                    pts.append(np.stack([x[k], y[k], np.asarray(c.z)[k]], 1))
                    rgb.append(np.stack([np.asarray(c.red)[k], np.asarray(c.green)[k], np.asarray(c.blue)[k]], 1) / 65535.0)
                    gt.append(np.asarray(c.gps_time)[k])
        return np.concatenate(pts), np.concatenate(rgb).astype(np.float32), np.concatenate(gt)


def lonlat(v):
    n = np.linalg.norm(v, axis=1) + 1e-9
    return np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2] / n, -1, 1)), n


def to_pix(lon, lat, W, H):
    u = ((0.5 - lon / (2 * np.pi)) * W).astype(int) % W
    v = np.clip(((0.5 - lat / np.pi) * H).astype(int), 0, H - 1)
    return u, v


def visible(d, res_u=2048, res_v=1024, tol=0.04):
    """Spherical z-buffer: True for points not hidden behind a nearer point (world-axis directions)."""
    lon, lat, n = lonlat(d)
    u, v = to_pix(lon, lat, res_u, res_v)
    key = v * res_u + u
    near = np.full(res_u * res_v, np.inf)
    np.minimum.at(near, key, n)
    return n <= near[key] * (1 + tol) + 0.05


def refine_pose(mission, fi, pts, gray, W=1552, H=776):
    """Return (centre, R camera->world, cc) for frame fi, refining tilt and heading near Rz(-yaw)."""
    c = mission.P[fi, 1:4]
    yaw = mission.P[fi, 6]
    d = pts - c
    dist = np.linalg.norm(d, axis=1)
    k = (dist > 0.7) & (dist < 12)
    d, g = d[k], gray[k]
    vis = visible(d, 1024, 512)
    d, g = d[vis], g[vis]
    if len(d) > 60000:
        s = np.random.default_rng(0).choice(len(d), 60000, replace=False)
        d, g = d[s], g[s]
    img = gaussian_filter(np.asarray(mission.image(fi, (W, H)).convert("L"), np.float32) / 255.0, 1.5)
    R0 = Rot.from_euler("Z", -yaw, degrees=True)

    def score(rv):
        R = (R0 * Rot.from_rotvec(rv)).as_matrix()
        lon, lat, _ = lonlat(d @ R)
        keep = lat > np.radians(-35)
        u, v = to_pix(lon[keep], lat[keep], W, H)
        return -np.corrcoef(img[v, u], g[keep])[0, 1]

    best = (score(np.zeros(3)), np.zeros(3))
    for a in [-8, -4, 4, 8]:
        for ax in range(3):
            rv = np.zeros(3); rv[ax] = np.radians(a)
            s = score(rv)
            if s < best[0]:
                best = (s, rv)
    sol = minimize(score, best[1], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 600})
    R = (R0 * Rot.from_rotvec(sol.x)).as_matrix()
    return c, R, -sol.fun, -score(np.zeros(3))


def cube_faces(img, R, size=768, fov=90):
    """Six perspective views from an equirect panorama. Returns list of (face_img, Rface camera->world, K)."""
    W, H = img.size
    a = np.asarray(img)
    f = size / 2 / np.tan(np.radians(fov) / 2)
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    out = []
    # face camera frame: x right, y down, z forward (OpenCV); express its axes in the panorama camera frame
    for yaw_deg, pitch_deg in [(0, 0), (90, 0), (180, 0), (270, 0), (0, 60), (0, -60)]:
        Rp = Rot.from_euler("ZY", [yaw_deg, -pitch_deg], degrees=True).as_matrix()  # pano-cam: fwd x, left y, up z
        fwd, left, up = Rp[:, 0], Rp[:, 1], Rp[:, 2]
        Rf = np.stack([-left, -up, fwd], 1)  # columns: face x, y, z in pano-cam coords
        jj, ii = np.meshgrid(np.arange(size), np.arange(size))
        rays = np.stack([(jj - size / 2) / f, (ii - size / 2) / f, np.ones_like(jj, float)], -1) @ Rf.T
        lon, lat, _ = lonlat(rays.reshape(-1, 3))
        u, v = to_pix(lon, lat, W, H)
        face = a[v, u].reshape(size, size, 3)
        out.append((Image.fromarray(face), R @ Rf, K, (yaw_deg, pitch_deg)))
    return out
