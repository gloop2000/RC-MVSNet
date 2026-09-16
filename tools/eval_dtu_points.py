#!/usr/bin/env python3
"""DTU point-cloud evaluation in Python: accuracy, completeness, overall, and
precision / recall / F1 at several distance thresholds.

Accuracy/completeness follow the official MATLAB code in matlab_eval/
(BaseEvalMain_web_pt.m + ComputeStat_web_pt.m + compute_mean.m):

  1. reduce the fused cloud to 0.2 mm minimum point spacing
  2. data -> stl distance for every point, capped at 60 mm
     stl  -> data distance for every GT point, capped at 60 mm
  3. keep data points inside ObsMask<scan>_10.mat and GT points above Plane<scan>.mat
  4. drop distances >= 20 mm (outliers), then take the mean per scan
  5. accuracy = mean over scans of data->stl, completeness = mean of stl->data,
     overall = (accuracy + completeness) / 2

F1 is NOT part of the official DTU protocol. It is computed here on the same
masked distances (step 3, no outlier removal) as:
  precision(t) = % of data points with data->stl distance < t
  recall(t)    = % of GT points  with stl->data distance < t
  F1(t)        = 2PR / (P + R)
State that definition in the report.

GT layout expected under --gt_dir (DTU "SampleSet/MVS Data" + Points.zip):
  Points/stl/stl001_total.ply
  ObsMask/ObsMask1_10.mat
  ObsMask/Plane1.mat

Runs in the rcmvsnet environment (numpy, scipy, plyfile).
"""

import argparse
import csv
import os
import time

import numpy as np
from plyfile import PlyData
from scipy.io import loadmat
from scipy.spatial import cKDTree

MAX_DIST = 60.0   # MaxDistCP cap (PointCompareMain.m)
OUTLIER = 20.0    # ComputeStat_web_pt.m
MARGIN = 10


def read_ply_xyz(path):
    v = PlyData.read(path)['vertex']
    return np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)


def reduce_points(pts, dst, seed=0):
    """Greedy 0.2 mm thinning, equivalent in effect to reducePts_haa.m."""
    rng = np.random.RandomState(seed)
    pts = pts[rng.permutation(len(pts))]
    tree = cKDTree(pts)
    neighbours = tree.query_ball_point(pts, r=dst, workers=-1)
    keep = np.ones(len(pts), dtype=bool)
    for i, nb in enumerate(neighbours):
        if keep[i]:
            keep[nb] = False
            keep[i] = True
    return pts[keep]


def nn_dist(src, dst_pts):
    d, _ = cKDTree(dst_pts).query(src, k=1, distance_upper_bound=MAX_DIST, workers=-1)
    return np.minimum(d, MAX_DIST)


def eval_scan(ply_path, scan_id, gt_dir, dst, thresholds):
    data = reduce_points(read_ply_xyz(ply_path), dst)
    stl = read_ply_xyz(os.path.join(gt_dir, 'Points', 'stl', 'stl{:03d}_total.ply'.format(scan_id)))

    m = loadmat(os.path.join(gt_dir, 'ObsMask', 'ObsMask{}_{}.mat'.format(scan_id, MARGIN)))
    obs, bb, res = m['ObsMask'], m['BB'], float(np.asarray(m['Res']).ravel()[0])
    plane = loadmat(os.path.join(gt_dir, 'ObsMask', 'Plane{}.mat'.format(scan_id)))['P'].ravel()

    d_data = nn_dist(data, stl)
    d_stl = nn_dist(stl, data)

    # data inside observation mask (MATLAB: round((Q-BB1)/Res + 1), 1-based)
    idx = np.round((data - bb[0]) / res).astype(np.int64)
    inb = np.all((idx >= 0) & (idx < np.array(obs.shape)), axis=1)
    in_mask = np.zeros(len(data), dtype=bool)
    in_mask[inb] = obs[idx[inb, 0], idx[inb, 1], idx[inb, 2]].astype(bool)
    # GT above ground plane
    above = (np.hstack([stl, np.ones((len(stl), 1))]) @ plane) > 0

    dd, ds = d_data[in_mask], d_stl[above]
    row = {
        'scan': scan_id,
        'n_data': int(len(dd)),
        'acc': float(np.mean(dd[dd < OUTLIER])),
        'comp': float(np.mean(ds[ds < OUTLIER])),
    }
    for t in thresholds:
        p = 100.0 * np.mean(dd < t)
        r = 100.0 * np.mean(ds < t)
        row['P@{}'.format(t)] = p
        row['R@{}'.format(t)] = r
        row['F1@{}'.format(t)] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ply_dir', required=True, help='--outdir of eval_rcmvsnet_dtu.py (holds mvsnetXXX_l3.ply)')
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--testlist', default='lists/dtu/test.txt')
    ap.add_argument('--thresholds', default='1,2,5', help='mm, comma separated')
    ap.add_argument('--dst', type=float, default=0.2)
    ap.add_argument('--out_csv', default=None, help='default: <ply_dir>/dtu_metrics.csv')
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(',')]
    with open(args.testlist) as f:
        scans = [int(l.strip()[4:]) for l in f if l.strip()]

    rows = []
    for s in scans:
        ply = os.path.join(args.ply_dir, 'mvsnet{:03d}_l3.ply'.format(s))
        if not os.path.exists(ply):
            print('skip scan{} (missing {})'.format(s, ply))
            continue
        t0 = time.time()
        row = eval_scan(ply, s, args.gt_dir, args.dst, thresholds)
        rows.append(row)
        print('scan{:<4} acc {:.4f}  comp {:.4f}  '.format(s, row['acc'], row['comp'])
              + '  '.join('F1@{:g} {:.2f}'.format(t, row['F1@{}'.format(t)]) for t in thresholds)
              + '  ({:.0f}s)'.format(time.time() - t0))

    if not rows:
        raise SystemExit('no point clouds evaluated')
    keys = [k for k in rows[0] if k != 'scan']
    mean = {'scan': 'mean'}
    mean.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    mean['overall'] = (mean['acc'] + mean['comp']) / 2
    rows.append(mean)

    out_csv = args.out_csv or os.path.join(args.ply_dir, 'dtu_metrics.csv')
    fields = ['scan'] + keys + ['overall']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print('\nMEAN over {} scans: acc {:.4f} mm  comp {:.4f} mm  overall {:.4f} mm'.format(
        len(rows) - 1, mean['acc'], mean['comp'], mean['overall']))
    for t in thresholds:
        print('  t={:g} mm  P {:.2f}  R {:.2f}  F1 {:.2f}'.format(
            t, mean['P@{}'.format(t)], mean['R@{}'.format(t)], mean['F1@{}'.format(t)]))
    print('written', out_csv)


if __name__ == '__main__':
    main()
