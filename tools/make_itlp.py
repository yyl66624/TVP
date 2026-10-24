#!/usr/bin/env python3
"""Build TF-VPR-format benchmarks from ITLP-Campus-Outdoor / -Indoor.

    make_itlp.py --raw <itlp dir> --out <datasets dir> --name itlp_day_night_0506 \
                 --db 05_2023-08-15-day --query 06_2023-08-18-night [--indoor]

Poses come from track.csv (outdoor) or dataset.csv (indoor, columns
track,floor,timestamp,...,tx,ty,tz and front_cam_ts).  Each image is named by
its capture timestamp in nanoseconds, identical to front_cam_ts.
Every pose keeps the one image captured closest to it, and the image becomes a
symlink named <ts>@<x>@<y>@.jpg, which is the convention BaseDataset parses.
For the indoor tracks the floor is folded into y so that stacked floors do not
become mutual positives.
"""
import argparse
import csv
import os
from glob import glob

TOL_NS = 500_000_000  # 0.5 s between image and pose (both stamps are ns)
FLOOR_GAP = 100.0  # metres added per floor


def read_poses(path, track=None, indoor=False):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if track is not None:
        rows = [r for r in rows if r["track"] == track]
    out = []
    for r in rows:
        x, y = float(r["tx"]), float(r["ty"])
        floor = int(r["floor"]) if indoor else None
        if indoor:
            y += floor * FLOOR_GAP
        out.append((int(r["front_cam_ts"]), x, y, floor))
    return out


def pick_images(img_dir, poses):
    imgs = sorted(glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg")))
    stamps = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            stamps.append((int(stem.split("_")[-1]), p))
        except ValueError:
            continue
    stamps.sort()
    best = {}  # pose index -> (dt, image)
    keys = [s for s, _, _, _ in poses]
    j = 0
    for ts, p in stamps:
        while j + 1 < len(keys) and abs(keys[j + 1] - ts) <= abs(keys[j] - ts):
            j += 1
        dt = abs(keys[j] - ts)
        if dt <= TOL_NS and (j not in best or dt < best[j][0]):
            best[j] = (dt, p)
    return [(poses[j][1], poses[j][2], best[j][1]) for j in sorted(best)]


def build(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for x, y, src in images:
        name = f"{os.path.basename(src).split('.')[0]}@{x:.3f}@{y:.3f}@.jpg"
        dst = os.path.join(out_dir, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)


def indoor_images(raw, track):
    rows = read_poses(os.path.join(raw, "dataset.csv"), track, indoor=True)
    by_floor = {}
    for ts, x, y, floor in rows:
        by_floor.setdefault(floor, []).append((ts, x, y, floor))
    out = []
    for floor, poses in sorted(by_floor.items()):
        out.extend(pick_images(os.path.join(raw, track, f"floor_{floor}", "front_cam"), poses))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--db", required=True, help="track name used as database")
    ap.add_argument("--query", required=True, help="track name used as queries")
    ap.add_argument("--indoor", action="store_true")
    args = ap.parse_args()

    if args.indoor:
        db_imgs = indoor_images(args.raw, args.db)
        q_imgs = indoor_images(args.raw, args.query)
    else:
        db_poses = read_poses(os.path.join(args.raw, args.db, "track.csv"))
        q_poses = read_poses(os.path.join(args.raw, args.query, "track.csv"))
        db_imgs = pick_images(os.path.join(args.raw, args.db, "front_cam"), db_poses)
        q_imgs = pick_images(os.path.join(args.raw, args.query, "front_cam"), q_poses)

    test = os.path.join(args.out, args.name, "images", "test")
    build(db_imgs, os.path.join(test, "database"))
    build(q_imgs, os.path.join(test, "queries"))
    print(f"{args.name}: {len(db_imgs)} db ({args.db}), {len(q_imgs)} queries ({args.query})")


if __name__ == "__main__":
    main()
