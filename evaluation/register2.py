"""Cross-scan repeatability between two separately walked missions that overlap.

Registration: FPFH + RANSAC on 40 cm voxels (2 seeds), then point-to-plane ICP (50, 15, 5 cm) on 5 cm voxels, using
ONLY the western half of the overlap (x below its median). Evaluation: on the eastern half, which ICP never saw,
with every 16th raw point (no voxelisation) of scan B measured against scan A as point-to-plane distance along A's
local normal (an M3C2-like signed distance with a 10 cm normal radius). No distance cut is applied to the statistics;
the share of points beyond 30 cm is reported separately. A sampling floor is computed by comparing scan A against
a disjoint subsample of itself (offset 8 of every 16th point) with the same distance.

python register2.py <laz_dir> <out_json> <mA> <mB> [<mA> <mB> ...]
"""
import glob
import json
import os
import sys
import time

import laspy
import numpy as np
import open3d as o3d

laz_dir, out = sys.argv[1], sys.argv[2]
pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))
reg = o3d.pipelines.registration


def load(m, every=16, offset=0):
    pts = []
    for f in sorted(glob.glob(os.path.join(laz_dir, f"{m}_*.laz"))):
        with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
            for c in r.chunk_iterator(5_000_000):
                s = slice(offset, None, every)
                pts.append(np.stack([np.asarray(c.x)[s], np.asarray(c.y)[s], np.asarray(c.z)[s]], 1))
    return np.concatenate(pts)


def pc(X):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))


def feats(p, v):
    d = p.voxel_down_sample(v)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * v, max_nn=30))
    return d, reg.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * v, max_nn=100))


def p2plane(Xb, A_pc, A_tree, A_n):
    """Distance from each point of Xb to A along A's normal at the nearest A point (abs), plus raw nearest distance."""
    d_nn = np.asarray(pc(Xb).compute_point_cloud_distance(A_pc))
    idx = np.empty(len(Xb), np.int64)
    for i, x in enumerate(Xb):
        idx[i] = A_tree.search_knn_vector_3d(x, 1)[1][0]
    Ap = np.asarray(A_pc.points)[idx]
    return np.abs(np.einsum("ij,ij->i", Xb - Ap, A_n[idx])), d_nn


def stats(d, d_nn, prefix):
    near = d_nn < 0.30
    dd = d[near]
    return {f"{prefix}n": int(near.sum()), f"{prefix}median_cm": round(100 * float(np.median(dd)), 2),
            f"{prefix}p90_cm": round(100 * float(np.percentile(dd, 90)), 2),
            f"{prefix}p95_cm": round(100 * float(np.percentile(dd, 95)), 2),
            f"{prefix}within_2cm_pct": round(100 * float(np.mean(dd < 0.02)), 1),
            f"{prefix}within_5cm_pct": round(100 * float(np.mean(dd < 0.05)), 1)}


results = []
for a, b in pairs:
    t0 = time.time()
    XA, XB = load(a), load(b)
    A, B = pc(XA), pc(XB)
    best = None
    Ad, Af = feats(A, 0.4); Bd, Bf = feats(B, 0.4)
    for seed in range(2):
        o3d.utility.random.seed(seed)
        g = reg.registration_ransac_based_on_feature_matching(
            Bd, Ad, Bf, Af, True, 0.6, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(0.6)],
            reg.RANSACConvergenceCriteria(200000, 0.999))
        if best is None or g.fitness > best.fitness:
            best = g
    T = best.transformation
    # overlap after the coarse alignment, split into an ICP half and an evaluation half
    Bc = pc(XB); Bc.transform(T)
    dB = np.asarray(Bc.compute_point_cloud_distance(A))
    XBt = np.asarray(Bc.points)
    ov = dB < 1.0
    if ov.sum() < 1000:
        results.append({"a": a, "b": b, "overlap_pts": int(ov.sum()), "note": "no overlap after coarse alignment"})
        json.dump(results, open(out, "w"), indent=1); continue
    xmed = np.median(XBt[ov, 0])
    icp_half = XB[ov & (XBt[:, 0] < xmed)]
    A5 = A.voxel_down_sample(0.05); B5 = pc(icp_half).voxel_down_sample(0.05)
    for p in (A5, B5):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    for th in (0.5, 0.15, 0.05):
        icp = reg.registration_icp(B5, A5, th, T, reg.TransformationEstimationPointToPlane(),
                                   reg.ICPConvergenceCriteria(max_iteration=60))
        T = icp.transformation
    # evaluation half: raw every-8th points of B, transformed with the final T
    Bf_ = pc(XB); Bf_.transform(T); XBf = np.asarray(Bf_.points)
    dBf = np.asarray(Bf_.compute_point_cloud_distance(A))
    ev = (dBf < 1.0) & (XBf[:, 0] >= xmed)
    rng = np.random.default_rng(0)
    evi = np.where(ev)[0]; evi = rng.choice(evi, min(400000, len(evi)), replace=False)
    lo, hi = XBf[evi].min(0) - 1.0, XBf[evi].max(0) + 1.0
    Ae = A.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    Ae.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30))
    A_n = np.asarray(Ae.normals); A_tree = o3d.geometry.KDTreeFlann(Ae)
    d, dnn = p2plane(XBf[evi], Ae, A_tree, A_n)
    # sampling floor: A against a disjoint subsample of itself, same region
    XA2 = load(a, 16, 8)
    reg_box = np.all((XA2 >= lo + 1.0) & (XA2 <= hi - 1.0), 1) & (XA2[:, 0] >= xmed)
    near_a = np.asarray(pc(XA2[reg_box]).compute_point_cloud_distance(Ae)) < 1.0
    XA2e = XA2[reg_box][near_a]
    XA2e = XA2e[rng.choice(len(XA2e), min(200000, len(XA2e)), replace=False)]
    df, dfnn = p2plane(XA2e, Ae, A_tree, A_n)
    r = {"a": a, "b": b, "ransac_fitness": round(best.fitness, 3), "icp_fitness_5cm": round(icp.fitness, 3),
         "icp_rmse_cm": round(100 * icp.inlier_rmse, 2), "overlap_frac_of_b": round(float(ov.mean()), 3),
         "overlap_area_m2": int(len(np.unique(np.floor(XBt[ov, :2]).astype(np.int64), axis=0))),
         "eval_beyond_30cm_pct": round(100 * float(np.mean(dnn >= 0.30)), 1),
         **stats(d, dnn, ""), **stats(df, dfnn, "floor_"),
         "T": np.round(T, 5).tolist(), "seconds": round(time.time() - t0)}
    np.save(os.path.join(os.path.dirname(out), f"reg2_{a}_{b}_dist.npy"), d[dnn < 0.30].astype(np.float32))
    results.append(r); print(json.dumps({k: v for k, v in r.items() if k != "T"}), flush=True)
    json.dump(results, open(out, "w"), indent=1)
print("done", flush=True)
