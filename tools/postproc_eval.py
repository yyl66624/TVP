"""Offline descriptor post-processing experiments.

Loads a descriptor dump produced with `--save_features` and re-evaluates it
under training-free, transductive post-processing:

  l2          : baseline (descriptors are already L2-normalised)
  center      : subtract the test-set mean descriptor (mean of db + queries)
  center_db   : subtract the database mean
  power       : signed power-law on the raw descriptor, then L2
  center_power: centre then power
  pca_w       : PCA-whiten to k dimensions, then L2
  aqe         : alpha query expansion from the top-k database neighbours

The dataset is only used for ground-truth positives and db_num, no images are
decoded, so this runs in seconds.

Run from TF-VPR-main:
  .venv/bin/python ../tools/postproc_eval.py --features <path.npy> \
    --eval_dataset_name msls --eval_datasets_folder $WORKSHOP_ROOT/datasets
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


def recalls(q, db, positives, ks=(1, 5, 10)):
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    _, pred = index.search(q, max(ks))
    out = {}
    for k in ks:
        hits = [len(np.intersect1d(pred[i, :k], positives[i])) > 0 for i in range(len(pred))]
        out[k] = 100.0 * np.mean(hits)
    return out


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def signed_power(x, alpha=0.5):
    return np.sign(x) * np.abs(x) ** alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--num_clusters", type=int, default=17)
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--aqe_k", type=int, default=5)
    ap.add_argument("--aqe_alpha", type=float, default=3.0)
    ap.add_argument("--max_queries", type=int, default=0, help="0 = all")
    ap.add_argument("--shrink", action="store_true", help="also evaluate shrinkage whitening")
    ap.add_argument("--block_dim", type=int, default=768,
                    help="token block size for token-wise whitening (0 disables)")
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", str(args_cli.num_clusters),
                "--backbone", "dinov2"]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    positives = ds.soft_positives_per_query[: args_cli.max_queries or None]

    X = np.load(args_cli.features).astype("float32")
    db_n = ds.database_num
    db, q = X[:db_n], X[db_n:]
    if args_cli.max_queries:
        q = q[: args_cli.max_queries]
    print(f"features {X.shape}  db {db.shape}  q {q.shape}")

    results = {}

    def report(name, db_, q_):
        r = recalls(q_, db_, positives)
        results[name] = r
        print(f"{name:14s} R@1 {r[1]:5.1f}  R@5 {r[5]:5.1f}  R@10 {r[10]:5.1f}")

    report("l2", l2norm(db), l2norm(q))

    mu_all = X.mean(0, keepdims=True)
    mu_db = db.mean(0, keepdims=True)
    report("center", l2norm(db - mu_all), l2norm(q - mu_all))
    report("center_db", l2norm(db - mu_db), l2norm(q - mu_db))

    report("power", l2norm(signed_power(l2norm(db))), l2norm(signed_power(l2norm(q))))
    dbc, qc = l2norm(db - mu_all), l2norm(q - mu_all)
    report("center_power", l2norm(signed_power(dbc)), l2norm(signed_power(qc)))

    # PCA whitening fitted on the test set (transductive, label-free)
    from sklearn.decomposition import PCA
    k = min(args_cli.pca_dim, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=0)
    pca.fit(X)
    Z = pca.transform(X).astype("float32")
    Z = l2norm(Z)
    report(f"pca_w{k}", Z[:db_n], Z[db_n:][: len(q)])

    # database-only fit: no query statistics, deployable transductive setup
    pca_db = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=0)
    pca_db.fit(X[:db_n])
    Zb = l2norm(pca_db.transform(X).astype("float32"))
    report(f"pca_w{k}_db", Zb[:db_n], Zb[db_n:][: len(q)])

    # shrinkage whitening: 1/sqrt(lambda + eps * mean(lambda)) instead of 1/sqrt(lambda).
    # PCA is fitted once per fit-set and reused for every eps.
    if args_cli.shrink:
        ap_shrink = [0.002, 0.01, 0.05, 0.1, 0.25]
        for fit_name, fit_X in (("all", X), ("db", X[:db_n])):
            p = PCA(n_components=k, whiten=False, svd_solver="randomized", random_state=0)
            p.fit(fit_X)
            Zs = p.transform(X).astype("float32")
            lam = p.explained_variance_.astype("float32")
            for eps in ap_shrink:
                Zt = l2norm(Zs / np.sqrt(lam + eps * lam.mean()))
                report(f"sw{k}_e{eps}_{fit_name}", Zt[:db_n], Zt[db_n:][: len(q)])

    # token-wise whitening: each 768-d token block gets its own PCA-whitening
    bd = args_cli.block_dim
    if bd and X.shape[1] % bd == 0:
        T = X.shape[1] // bd
        for bk in (32, 64):
            blocks_all, blocks_db = [], []
            ok = True
            for t in range(T):
                sl = slice(t * bd, (t + 1) * bd)
                pb = PCA(n_components=bk, whiten=True, svd_solver="randomized", random_state=0)
                try:
                    pb.fit(X[:db_n, sl])
                except Exception:
                    ok = False
                    break
                blocks_db.append(l2norm(pb.transform(X[:db_n, sl]).astype("float32")))
                blocks_all.append(l2norm(pb.transform(X[db_n:, sl]).astype("float32")))
            if ok:
                Zb = l2norm(np.concatenate(blocks_db, axis=1))
                Zq = l2norm(np.concatenate(blocks_all, axis=1))
                report(f"block_w{bk}", Zb, Zq[: len(q)])

    # alpha query expansion on the plain descriptor
    dbn, qn = l2norm(db), l2norm(q)
    index = faiss.IndexFlatIP(dbn.shape[1])
    index.add(dbn)
    kq = min(args_cli.aqe_k, dbn.shape[0])
    sim, nn = index.search(qn, kq)
    w = np.clip(sim, 0, None) ** args_cli.aqe_alpha
    qe = qn + np.einsum("qk,qkd->qd", w, dbn[nn])
    report("aqe", dbn, l2norm(qe))


if __name__ == "__main__":
    main()
