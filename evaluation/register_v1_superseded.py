"""Independent repeatability check: register two separately walked missions that overlap (global FPFH+RANSAC,
then point-to-plane ICP) and measure cloud-to-cloud disagreement in the overlap.

python register.py <laz_dir> <out_json> <mA> <mB> [<mA> <mB> ...]
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


def load(m, every=8):
    pts = []
    for f in sorted(glob.glob(os.path.join(laz_dir, f"{m}_*.laz"))):
        with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
            for c in r.chunk_iterator(5_000_000):
                x, y, z = np.asarray(c.x), np.asarray(c.y), np.asarray(c.z)
                s = slice(None, None, every)
                pts.append(np.stack([x[s], y[s], z[s]], 1))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    return pc


def feats(pc, v):
    d = pc.voxel_down_sample(v)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * v, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * v, max_nn=100))
    return d, f


results = []
for a, b in pairs:
    t0 = time.time()
    A, B = load(a), load(b)
    best = None
    for v in (0.25, 0.4):
        Ad, Af = feats(A, v); Bd, Bf = feats(B, v)
        for seed in range(3):
            o3d.utility.random.seed(seed)
            g = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                Bd, Ad, Bf, Af, True, 1.5 * v,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(1.5 * v)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))
            if best is None or g.fitness > best.fitness:
                best = g
    # ICP refine at 10 cm then 3 cm
    A10 = A.voxel_down_sample(0.05); B10 = B.voxel_down_sample(0.05)
    for pc in (A10, B10):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    T = best.transformation
    for th in (0.5, 0.15, 0.05):
        icp = o3d.pipelines.registration.registration_icp(B10, A10, th, T,
                                                           o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                           o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = icp.transformation
    Bt = o3d.geometry.PointCloud(B10); Bt.transform(T)
    d = np.asarray(Bt.compute_point_cloud_distance(A10))
    ov = d < 0.30  # overlap region
    dov = d[ov]
    r = {"a": a, "b": b, "ransac_fitness": round(best.fitness, 3), "icp_fitness_5cm": round(icp.fitness, 3),
         "icp_rmse_cm": round(100 * icp.inlier_rmse, 2), "overlap_pts": int(ov.sum()),
         "overlap_frac_of_b": round(float(ov.mean()), 3),
         "c2c_median_cm": round(100 * float(np.median(dov)), 2) if ov.any() else None,
         "c2c_p90_cm": round(100 * float(np.percentile(dov, 90)), 2) if ov.any() else None,
         "c2c_p95_cm": round(100 * float(np.percentile(dov, 95)), 2) if ov.any() else None,
         "within_2cm_pct": round(100 * float(np.mean(dov < 0.02)), 1) if ov.any() else None,
         "within_5cm_pct": round(100 * float(np.mean(dov < 0.05)), 1) if ov.any() else None,
         "translation_m": np.round(T[:3, 3], 2).tolist(), "seconds": round(time.time() - t0)}
    np.save(os.path.join(os.path.dirname(out), f"reg_{a}_{b}_dist.npy"), dov.astype(np.float32))
    results.append(r)
    print(json.dumps(r), flush=True)
    json.dump(results, open(out, "w"), indent=1)
