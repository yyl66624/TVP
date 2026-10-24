#!/usr/bin/env python3
"""Build a TF-VPR-format benchmark from NYC-Indoor-VPR (indoor_2023.zip).

    make_nyc_indoor.py --raw <.../indoor_2023> --out <datasets dir> --name nyc_indoor

Every site folder holds one pair of sessions recorded weeks apart plus
coordinate1.txt / coordinate2.txt (x, y in metres, one row per image; a leading
[0,0] row appears when the file has one row more than the session has images).
Sites are laid out in separate 1 km blocks so that indoor trajectories that
both start at the local origin never become mutual positives.  Ground truth is
computed at run time from the metric coordinates.
"""
import argparse
import glob
import os

import numpy as np

SITES = [
    # (pair dir, session 1 dir, session 2 dir)
    ("b4_1/0330_0411_b4", "b4_03_30_23", "b4_04_11_23"),
    ("s2_1/0330_0411_s2", "s2_03_30_23", "s2_04_11_23"),
    ("s4_1/0330_0411_s4", "s4_03_30_23", "s4_04_11_23"),
    ("s5_1/0330_0411_s5", "s5_03_30_23", "s5_04_11_23"),
    ("s6_1/0330_0411_s6", "s6_03_30_23", "s6_04_11_23"),
]


def load_coords(path, n_images):
    a = np.loadtxt(path)
    if len(a) == n_images + 1:
        a = a[1:]
    if len(a) != n_images:
        raise RuntimeError(f"{path}: {len(a)} rows for {n_images} images")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="nyc_indoor")
    args = ap.parse_args()

    test = os.path.join(args.out, args.name, "images", "test")
    db_dir, q_dir = os.path.join(test, "database"), os.path.join(test, "queries")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(q_dir, exist_ok=True)

    n_db = n_q = 0
    for rank, (pair, sess_db, sess_q) in enumerate(SITES):
        base = os.path.join(args.raw, pair)
        imgs_db = sorted(glob.glob(os.path.join(base, sess_db, "*.jpg")))
        imgs_q = sorted(glob.glob(os.path.join(base, sess_q, "*.jpg")))
        c_db = load_coords(os.path.join(base, "coordinate1.txt"), len(imgs_db))
        c_q = load_coords(os.path.join(base, "coordinate2.txt"), len(imgs_q))
        off = rank * 1000.0
        for i, (p, (x, y)) in enumerate(zip(imgs_db, c_db)):
            dst = os.path.join(db_dir, f"{pair.split('/')[0]}_{i:04d}@{x + off:.3f}@{y + off:.3f}@.jpg")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(p), dst)
        for i, (p, (x, y)) in enumerate(zip(imgs_q, c_q)):
            dst = os.path.join(q_dir, f"{pair.split('/')[0]}_{i:04d}@{x + off:.3f}@{y + off:.3f}@.jpg")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(p), dst)
        n_db += len(imgs_db)
        n_q += len(imgs_q)
        print(f"  {pair}: {len(imgs_db)} db, {len(imgs_q)} queries")
    print(f"{args.name}: {n_db} db, {n_q} queries")


if __name__ == "__main__":
    main()
