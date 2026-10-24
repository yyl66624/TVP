"""Check whether all-fit calibration is inflated by using the evaluated queries
in its own PCA fit.

Protocol: split the query set into even/odd folds.  Fit PCA-whitening on
database + the *opposite* fold, then evaluate on the held-out fold.  Compare
with db-only and with the all-fit (db + all queries) fit evaluated on the same
fold.  No labels are used for fitting; ground truth is used only for recall.

Run from TF-VPR-main (the working directory is the code root):
  .venv/bin/python -I ../tools/crossfit_check.py --features <dump.npy> \
      --eval_dataset_name nordland --dims 768 --positive_dist 25
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import faiss
import numpy as np
from sklearn.decomposition import PCA

import datasets_ws
import parser
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def recalls(q, db, positives, ks=(1, 5, 10)):
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    _, pred = index.search(q, max(ks))
    return {k: 100.0 * float(np.mean(
        [len(np.intersect1d(pred[i, :k], positives[i])) > 0 for i in range(len(pred))]))
        for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", default="nordland")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--n_tokens", type=int, default=1)
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--positive_dist", type=float, default=25.0)
    ap.add_argument("--fold_seed", type=int, default=0)
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2",
                "--val_positive_dist_threshold", str(args_cli.positive_dist)]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder,
                                 args.eval_dataset_name, "test")
    positives = ds.soft_positives_per_query
    db_n = ds.database_num

    X = np.load(args_cli.features, mmap_mode="r")
    D = X.shape[1] // args_cli.n_tokens
    assert D * args_cli.n_tokens == X.shape[1], (X.shape, args_cli.n_tokens)
    X = X[:, :min(args_cli.n_tokens * D, args_cli.dims * args_cli.n_tokens)]
    if X.shape[1] != args_cli.n_tokens * D:
        raise SystemExit("dims must equal the per-token width")
    X = np.asarray(X, dtype="float32")

    q_n = X.shape[0] - db_n
    rng = np.random.default_rng(args_cli.fold_seed)
    perm = rng.permutation(q_n)
    folds = [perm[0::2], perm[1::2]]

    k = min(args_cli.dims, X.shape[1], db_n - 1)
    sets = {
        "db-fit": np.arange(db_n),
        "cross-fit": [np.arange(db_n), None],
        "all-fit": np.arange(X.shape[0]),
    }

    for name in ("db-fit", "cross-fit", "all-fit"):
        for f, eval_fold in enumerate(folds):
            if name == "db-fit":
                rows = sets["db-fit"]
            elif name == "all-fit":
                rows = sets["all-fit"]
            else:
                other = folds[1 - f]
                rows = np.concatenate([np.arange(db_n), db_n + other])
            pca = PCA(n_components=k, whiten=True, svd_solver="randomized",
                      random_state=0)
            n_fit = len(rows)
            pca.fit(X[rows])
            del rows
            Zdb = l2norm(pca.transform(X[:db_n]).astype("float32"))
            qrows = db_n + eval_fold
            Zq = l2norm(pca.transform(X[qrows]).astype("float32"))
            out = recalls(Zq, Zdb, [positives[i] for i in eval_fold])
            print(f"{name} fold{f} n_fit={n_fit} dim={k} "
                  f"R@1 {out[1]:.1f}  R@5 {out[5]:.1f}  R@10 {out[10]:.1f}",
                  flush=True)
            del Zdb, Zq


if __name__ == "__main__":
    main()
