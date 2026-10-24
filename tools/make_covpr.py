#!/usr/bin/env python3
"""Build a TF-VPR-format benchmark from CoVPR-Indoor.

    make_covpr.py --raw <.../Indoor/Test> --out <datasets dir> --name covpr_indoor

Every file is named <prefix>_<x>_<y>_<z>.png, so the metric coordinates come
from the filename.  The database is Test/DataBase, the queries are the two
images per location under Test/Query/<id>/.  Ground truth is a radius search,
so evaluate with a small threshold (0.5 m) -- the database is dense.
"""
import argparse
import glob
import os

NAMED = ""


def link(src, dst_dir, tag):
    parts = os.path.basename(src).rsplit(".", 1)[0].split("_")
    x, y = float(parts[-3]), float(parts[-2])
    dst = os.path.join(dst_dir, f"{tag}_{os.path.basename(src).split('_')[0]}@{x:.3f}@{y:.3f}@.jpg")
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="covpr_indoor")
    args = ap.parse_args()

    test = os.path.join(args.out, args.name, "images", "test")
    db_dir = os.path.join(test, "database")
    q_dir = os.path.join(test, "queries")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(q_dir, exist_ok=True)

    db = sorted(glob.glob(os.path.join(args.raw, "DataBase", "*.png")))
    qs = sorted(glob.glob(os.path.join(args.raw, "Query", "*", "*.png")))
    seen = set()
    for p in db:
        stem = os.path.basename(p).rsplit(".", 1)[0]
        if stem in seen:
            continue
        seen.add(stem)
        link(p, db_dir, "db")
    for i, p in enumerate(qs):
        link(p, q_dir, f"q{i:04d}")
    print(f"{args.name}: {len(os.listdir(db_dir))} db, {len(os.listdir(q_dir))} queries")


if __name__ == "__main__":
    main()
