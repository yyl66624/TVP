"""Post-processing on top of block-calibrated multi-view descriptors.

Loads a raw multi-view dump (e.g. the 3-view MVC dump: semantic 17 tokens,
multi-scale 13 tokens, global 1 token = 31*768 dims), calibrates each view
separately (PCA-whitening fitted on the database), optionally weights the
views, and then evaluates training-free post-processing:

  none      : plain nearest neighbour
  aqe       : alpha query expansion (query average of top-k db neighbours)
  dba       : database-side augmentation (same operation on the database)
  qcenter   : subtract the mean calibrated query descriptor
  qwhiten   : PCA-whiten the calibrated query block by its own statistics
  snorm     : score normalisation (z-score of the top-100 distances per query)

All operations are label-free.  Ground truth is only used for the metrics.

Run from TF-VPR-main:
  .venv/bin/python ../tools/calib_postproc.py --features <raw.npy> \
      --eval_dataset_name msls --dims 2048,1024,768 --post aqe --aqe_k 3
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


def l2norm(x, axis=1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def recalls(q, db, positives, ks=(1, 5, 10), index=None):
    if index is None:
        index = faiss.IndexFlatL2(db.shape[1])
        index.add(db)
    _, pred = index.search(q, max(ks))
    out = {}
    for k in ks:
        hits = [len(np.intersect1d(pred[i, :k], positives[i])) > 0 for i in range(len(pred))]
        out[k] = 100.0 * float(np.mean(hits))
    return out, pred


def alpha_qe(X_feat, X_db, k=3, alpha=3.0, beta=0.1):
    """Classic alpha-QE on the query set: features are the query descriptors."""
    index = faiss.IndexFlatL2(X_db.shape[1])
    index.add(X_db)
    D, I = index.search(X_feat, k)
    w = D[:, :, None] ** alpha if alpha > 0 else np.ones_like(D)[:, :, None]
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    Xq = X_feat.copy()
    acc = (X_db[I] * w).sum(axis=1)
    return l2norm((1 - beta) * Xq + beta * acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--splits", default="17,13,1")
    ap.add_argument("--dims", default="2048,1024,768")
    ap.add_argument("--weights", default="1,1,1")
    ap.add_argument("--post", default="none")
    ap.add_argument("--aqe_k", type=int, default=3)
    ap.add_argument("--aqe_alpha", type=float, default=3.0)
    ap.add_argument("--aqe_beta", type=float, default=0.1)
    ap.add_argument("--qw_dim", type=int, default=0)
    ap.add_argument("--snorm_k", type=int, default=100)
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--fit", default="db", choices=["db", "all"])
    ap.add_argument("--positive_dist", type=float, default=25.0)
    ap.add_argument("--no_whiten", action="store_true",
                    help="plain PCA (no variance equalisation) for the ablation")
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2",
                "--val_positive_dist_threshold", str(args_cli.positive_dist)]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    positives = ds.soft_positives_per_query
    db_n = ds.database_num

    # --features / --splits / --dims accept ';'-separated per-file specs, so
    # views dumped by different runs (e.g. a second input resolution) can be
    # fused with the single-run views.
    files = args_cli.features.split(";")
    split_specs = args_cli.splits.split(";")
    dim_specs = args_cli.dims.split(";")
    assert len(files) == len(split_specs) == len(dim_specs), (files, split_specs, dim_specs)

    views = []
    for path, sspec, dspec in zip(files, split_specs, dim_specs):
        splits = [int(s) for s in sspec.split(",")]
        dims = [int(d) for d in dspec.split(",")]
        assert len(splits) == len(dims)
        X = np.load(path)
        D = X.shape[1] // sum(splits)
        assert D * sum(splits) == X.shape[1], (X.shape, splits)
        offset = 0
        for n_tok, k in zip(splits, dims):
            sl = slice(offset, offset + n_tok * D)
            offset += n_tok * D
            Xv = X[:, sl].astype("float32")
            fit_n = db_n if args_cli.fit == "db" else len(Xv)
            pca = PCA(n_components=min(k, n_tok * D, fit_n - 1), whiten=not args_cli.no_whiten,
                      svd_solver="randomized", random_state=0)
            pca.fit(Xv[:fit_n])
            Z = pca.transform(Xv).astype("float32")
            del Xv
            views.append(l2norm(Z))
            print(f"view {n_tok} tokens -> {Z.shape[1]} dims ({os.path.basename(os.path.dirname(os.path.dirname(path)))})", flush=True)
        del X

    if args_cli.max_queries:
        views = [np.concatenate([v[:db_n], v[db_n:db_n + args_cli.max_queries]]) for v in views]
        positives = positives[: args_cli.max_queries]

    for wcfg in args_cli.weights.split(";"):
        weights = [float(w) for w in wcfg.split(",")]
        assert len(weights) == len(views)
        Z = l2norm(np.concatenate([w * v for w, v in zip(weights, views)], axis=1))
        db0, q0 = Z[:db_n], Z[db_n:]
        del Z
        for post in args_cli.post.split(","):
            db, q = db0.copy(), q0.copy()
            index = faiss.IndexFlatL2(db.shape[1])
            index.add(db)

            if post == "none":
                pass
            elif post == "aqe":
                q = alpha_qe(q, db, args_cli.aqe_k, args_cli.aqe_alpha, args_cli.aqe_beta)
                index = faiss.IndexFlatL2(db.shape[1]); index.add(db)
            elif post == "dba":
                db = alpha_qe(db, db, args_cli.aqe_k, args_cli.aqe_alpha, args_cli.aqe_beta)
                index = faiss.IndexFlatL2(db.shape[1]); index.add(db)
            elif post == "qcenter":
                mu = q.mean(axis=0, keepdims=True)
                q = l2norm(q - mu)
                index = faiss.IndexFlatL2(db.shape[1]); index.add(db)
            elif post == "qwhiten":
                k = args_cli.qw_dim or db.shape[1]
                pca = PCA(n_components=min(k, db.shape[1], len(q) - 1), whiten=True,
                          svd_solver="randomized", random_state=0)
                pca.fit(q)
                Zq = pca.transform(q)
                Zd = pca.transform(db)
                q, db = l2norm(Zq), l2norm(Zd)
                index = faiss.IndexFlatL2(db.shape[1]); index.add(db)
            else:
                raise ValueError(post)

            out, pred = recalls(q, db, positives, index=index)
            print(f"{args_cli.eval_dataset_name} w={wcfg} post={post} dim={db.shape[1]} "
                  f"R@1 {out[1]:.1f}  R@5 {out[5]:.1f}  R@10 {out[10]:.1f}", flush=True)
            del db, q
        del db0, q0


if __name__ == "__main__":
    main()
