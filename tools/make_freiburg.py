#!/usr/bin/env python3
"""Build a TF-VPR-format benchmark from the Freiburg Across Seasons parquets.

    make_freiburg.py --raw <raw dir> --out <datasets dir> --pairs summer2012_winter

The parquet shards hold 500 JPEGs each (filename + bytes); the matchings file
gives global row indices into those shards for a pair of sessions.  We extract
only the images that take part in at least one match, then write symlinks and
an explicit positives.json (no UTM coordinates exist for this benchmark).
"""
import argparse
import json
import os
from glob import glob

import pyarrow.parquet as pq


def load_pairs(raw, pair_name):
    t = pq.read_table(os.path.join(raw, "matchings", f"{pair_name}_matchings.parquet"))
    q = t.column_names[0]
    return t.column(q).to_pylist(), t.column("winter_idx").to_pylist()


def extract_split(raw, split, wanted, out_img_dir):
    """Extract rows of `wanted` (global ids) from the shard sequence."""
    os.makedirs(out_img_dir, exist_ok=True)
    shards = sorted(glob(os.path.join(raw, split, "*.parquet")))
    offset = 0
    for shard in shards:
        t = pq.read_table(shard, columns=["filename", "image"])
        n = t.num_rows
        names = t.column("filename").to_pylist()
        imgs = t.column("image").to_pylist()
        for r in range(n):
            gid = offset + r
            if gid not in wanted:
                continue
            dest = os.path.join(out_img_dir, f"{gid:06d}.jpg")
            if os.path.exists(dest):
                continue
            with open(dest, "wb") as f:
                f.write(imgs[r]["bytes"])
        offset += n
    missing = [g for g in wanted if not os.path.exists(os.path.join(out_img_dir, f"{g:06d}.jpg"))]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing from {split}, e.g. {missing[:5]}")
    return offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", required=True, help="e.g. summer2012_winter")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    q_split = args.pairs.split("_winter")[0]
    name = args.name or f"freiburg_{q_split.replace('summer', '')}w"

    q_ids, w_ids = load_pairs(args.raw, args.pairs)
    # query -> list of positive winter ids, and the reverse for sanity
    q2w = {}
    for q, w in zip(q_ids, w_ids):
        q2w.setdefault(q, set()).add(w)
    wanted_q = set(q2w)
    wanted_w = {w for ws in q2w.values() for w in ws}
    print(f"{args.pairs}: {len(q_ids)} matchings, {len(wanted_q)} queries, {len(wanted_w)} db images")

    ex = os.path.abspath(os.path.join(args.raw, "extracted"))
    n_q = extract_split(args.raw, q_split, wanted_q, os.path.join(ex, q_split))
    n_w = extract_split(args.raw, "winter", wanted_w, os.path.join(ex, "winter"))
    print(f"shards cover {n_q} {q_split} rows and {n_w} winter rows")

    root = os.path.join(args.out, name)
    test = os.path.join(root, "images", "test")
    db_dir, q_dir = os.path.join(test, "database"), os.path.join(test, "queries")
    for d in (db_dir, q_dir):
        os.makedirs(d, exist_ok=True)

    db_ids = sorted(wanted_w)
    db_index = {g: i for i, g in enumerate(db_ids)}
    for g, i in db_index.items():
        dst = os.path.join(db_dir, f"{i:06d}.jpg")
        if not os.path.lexists(dst):
            os.symlink(os.path.join(ex, "winter", f"{g:06d}.jpg"), dst)
    queries = sorted(wanted_q)
    for i, g in enumerate(queries):
        dst = os.path.join(q_dir, f"{i:06d}.jpg")
        if not os.path.lexists(dst):
            os.symlink(os.path.join(ex, q_split, f"{g:06d}.jpg"), dst)

    gt = {"database": [f"database/{i:06d}.jpg" for i in range(len(db_ids))],
          "queries": [f"queries/{i:06d}.jpg" for i in range(len(queries))],
          "positives": [sorted(db_index[w] for w in q2w[g]) for g in queries]}
    with open(os.path.join(test, "positives.json"), "w") as f:
        json.dump(gt, f)
    print(f"wrote {root}: {len(db_ids)} db, {len(queries)} queries")


if __name__ == "__main__":
    main()
