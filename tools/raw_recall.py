"""Raw (uncalibrated) recall for a multi-view descriptor dump.

The dump is (N, sum(splits) * D); D is inferred as cols // sum(splits).  Each
view block is L2-normalised per descriptor, the blocks are concatenated and the
whole descriptor is L2-normalised -- the same order as the inference model
(`F.normalize(x, p=2, dim=2)` then `F.normalize(x, p=2, dim=1)`).  Retrieval is
FAISS IndexFlatL2 on the resulting unit vectors (equivalent to cosine).

This gives the `raw` column of the benchmark tables for the exact dump that was
calibrated, so the calibration delta is not confounded with the choice of views.

Run from TF-VPR-main:
  .venv/bin/python -I ../tools/raw_recall.py --features <dump.npy> \
      --eval_dataset_name st_lucia --splits 17,13,1 --positive_dist 25
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import faiss
import numpy as np

import datasets_ws
import parser
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2norm(x, axis=1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--splits", default="17,13,1")
    ap.add_argument("--views", default=None,
                    help="comma-separated view indices to keep, e.g. 0 for "
                         "the TF view of a 17,13,1 dump")
    ap.add_argument("--positive_dist", type=float, default=25.0)
    ap.add_argument("--no_intra_norm", action="store_true",
                    help="skip the per-view L2 normalisation")
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2",
                "--val_positive_dist_threshold", str(args_cli.positive_dist)]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder,
                                 args_cli.eval_dataset_name, "test")
    positives = ds.soft_positives_per_query
    db_n = ds.database_num

    X = np.load(args_cli.features)
    splits = [int(s) for s in args_cli.splits.split(",")]
    keep = (list(range(len(splits))) if args_cli.views is None
            else [int(s) for s in args_cli.views.split(",")])
    D = X.shape[1] // sum(splits)
    assert D * sum(splits) == X.shape[1], (X.shape, splits)

    views = []
    off = 0
    for i, n_tok in enumerate(splits):
        V = X[:, off:off + n_tok * D]
        off += n_tok * D
        if i not in keep:
            continue
        V = V.astype("float32")
        if not args_cli.no_intra_norm:
            V = l2norm(V)
        views.append(V)
    Z = l2norm(views[0] if len(views) == 1
               else np.concatenate(views, axis=1))
    del X, views

    db, q = Z[:db_n], Z[db_n:]
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    _, pred = index.search(q, 10)

    for k in (1, 5, 10):
        hits = [len(np.intersect1d(pred[i, :k], positives[i])) > 0
                for i in range(len(pred))]
        print(f"raw {args_cli.eval_dataset_name} dim={Z.shape[1]} "
              f"R@{k} {100.0 * float(np.mean(hits)):.1f}", flush=True)


if __name__ == "__main__":
    main()
