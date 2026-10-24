"""Evaluate combinations of independently calibrated descriptor views.

Each view is a separate dump (TF tokens, GeM+CLS, Mean, CLS+Mean); every view
is PCA-whitened on the database descriptors with its own budget, then the
calibrated views are concatenated.

  .venv/bin/python ../tools/multiview_eval.py --dataset msls
"""
import argparse
import glob
import itertools
import os
import sys

sys.path.insert(0, os.getcwd())

import faiss
import numpy as np
from sklearn.decomposition import PCA

import datasets_ws
import parser
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# dataset -> {view: (glob, whitening dim)}
LAYOUT = {
    "msls": {
        "tf": ("test/feat_msls_tf/*/features_hard_resize.npy", 2048),
        "gem": ("test/feat_msls_gemcls/*/features_hard_resize.npy", 1024),
        "mean": ("test/msls_mean/*/features_hard_resize.npy", 768),
        "clsmean": ("test/msls_cls_mean/*/features_hard_resize.npy", 1536),
    },
    "st_lucia": {
        "tf": ("test/feat_stlucia_tf/*/features_hard_resize.npy", 256),
        "gem": ("test/feat_stlucia_gemcls/*/features_hard_resize.npy", 256),
    },
}


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def recalls(q, db, positives, ks=(1, 5, 10)):
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    _, pred = index.search(q, max(ks))
    return [100.0 * np.mean([len(np.intersect1d(pred[i, :k], positives[i])) > 0
                             for i in range(len(pred))]) for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msls", choices=list(LAYOUT))
    ap.add_argument("--datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--fit", default="db", choices=["db", "all"])
    args_cli = ap.parse_args()

    views = {}
    for name, (pattern, dim) in LAYOUT[args_cli.dataset].items():
        files = sorted(glob.glob(pattern))
        if files:
            views[name] = (files[-1], dim)
        else:
            print(f"missing view {name} ({pattern})")
    print("views:", views)

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.dataset,
                "--eval_datasets_folder", args_cli.datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2"]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args_cli.datasets_folder, args_cli.dataset, "test")
    db_n = ds.database_num
    pos = ds.soft_positives_per_query

    Z = {}
    for name, (path, dim) in views.items():
        X = np.load(path).astype("float32")
        fit_n = db_n if args_cli.fit == "db" else len(X)
        k = min(dim, X.shape[1], fit_n - 1)
        p = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=0)
        p.fit(X[:fit_n])
        Z[name] = l2n(p.transform(X).astype("float32"))
        del X
        print(f"calibrated {name}: {Z[name].shape}")

    for r in range(1, len(Z) + 1):
        for combo in itertools.combinations(sorted(Z), r):
            Zm = l2n(np.concatenate([Z[c] for c in combo], axis=1))
            res = recalls(Zm[db_n:], Zm[:db_n], pos)
            print(f"{'+'.join(combo):20s} R@1 {res[0]:5.1f}  R@5 {res[1]:5.1f}  R@10 {res[2]:5.1f}")


if __name__ == "__main__":
    main()
