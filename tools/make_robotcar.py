#!/usr/bin/env python3
"""Build TF-VPR-format benchmarks from Oxford RobotCar (OPR release).

    make_robotcar.py --raw <raw dir> --out <datasets dir> --name robotcar_sun_night \
                     --db 2014-07-14-14-49-50 --query 2014-12-10-18-10-50

Each date holds images_small/stereo_centre/<ts_us>.png and a
pointcloud_locations_20m_10overlap.csv with columns timestamp,northing,easting.
Images are mapped to the nearest ground-truth pose and symlinked as
<ts>@<easting>@<northing>@.jpg.
"""
import argparse
import csv
import os
from glob import glob

TOL_US = 200_000  # 0.2 s


def read_poses(path):
    out = []
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == "timestamp":
                continue
            ts, northing, easting = int(r[0]), float(r[1]), float(r[2])
            out.append((ts, easting, northing))
    return sorted(out)


def pick_images(img_dir, poses):
    imgs = sorted(glob(os.path.join(img_dir, "*.png")))
    stamps = []
    for p in imgs:
        try:
            stamps.append((int(os.path.splitext(os.path.basename(p))[0]), p))
        except ValueError:
            continue
    stamps.sort()
    best = {}
    keys = [p[0] for p in poses]
    j = 0
    for ts, p in stamps:
        while j + 1 < len(keys) and abs(keys[j + 1] - ts) <= abs(keys[j] - ts):
            j += 1
        dt = abs(keys[j] - ts)
        if dt <= TOL_US and (j not in best or dt < best[j][0]):
            best[j] = (dt, p)
    return [(poses[j][1], poses[j][2], best[j][1]) for j in sorted(best)]


def collect(raw, date):
    poses = read_poses(os.path.join(raw, date, "pointcloud_locations_20m_10overlap.csv"))
    imgs = pick_images(os.path.join(raw, date, "images_small", "stereo_centre"), poses)
    print(f"  {date}: {len(poses)} poses, {len(imgs)} images")
    return imgs


def build(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for x, y, src in images:
        name = f"{os.path.splitext(os.path.basename(src))[0]}@{x:.3f}@{y:.3f}@.jpg"
        dst = os.path.join(out_dir, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    db_imgs = collect(args.raw, args.db)
    q_imgs = collect(args.raw, args.query)
    test = os.path.join(args.out, args.name, "images", "test")
    build(db_imgs, os.path.join(test, "database"))
    build(q_imgs, os.path.join(test, "queries"))
    print(f"{args.name}: {len(db_imgs)} db ({args.db}), {len(q_imgs)} queries ({args.query})")


if __name__ == "__main__":
    main()
