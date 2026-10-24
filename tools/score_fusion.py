"""Score-level fusion of independently calibrated views.

Each view is PCA-whitened on the database and L2-normalised, but the views are
*not* concatenated.  Instead every view produces its own similarity scores and
the per-view scores are standardised (mean 0, std 1 over the top-K neighbours)
before being summed.  This equalises the views without hand-tuned weights.

  .venv/bin/python ../tools/score_fusion.py --features <mvc3 dump.npy> \
      --eval_dataset_name msls --dims 2048,1024,768 --topk 200
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


def recalls_from_ranked(ranked, positives, ks=(1, 5, 10)):
    return {k: 100.0 * float(np.mean([len(np.intersect1d(r[:k], p)) > 0
                                      for r, p in zip(ranked, positives)]))
            for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--splits", default="17,13,1")
    ap.add_argument("--dims", default="2048,1024,768")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--norm", default="z", choices=["z", "none", "mean"])
    ap.add_argument("--max_queries", type=int, default=0)
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2"]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    positives = ds.soft_positives_per_query
    db_n = ds.database_num

    splits = [int(s) for s in args_cli.splits.split(",")]
    dims = [int(d) for d in args_cli.dims.split(",")]
    X = np.load(args_cli.features)
    D = X.shape[1] // sum(splits)
    assert D * sum(splits) == X.shape[1]

    total = db_n + (args_cli.max_queries or len(X) - db_n)
    scores = np.zeros((total - db_n, db_n), dtype="float32")
    offset = 0
    for n_tok, k in zip(splits, dims):
        sl = slice(offset, offset + n_tok * D)
        offset += n_tok * D
        Xv = X[:total, sl].astype("float32")
        pca = PCA(n_components=min(k, n_tok * D, db_n - 1), whiten=True,
                  svd_solver="randomized", random_state=0)
        pca.fit(Xv[:db_n])
        Z = l2norm(pca.transform(Xv).astype("float32"))
        del Xv
        index = faiss.IndexFlatL2(Z.shape[1])
        index.add(Z[:db_n])
        Dm, Im = index.search(Z[db_n:], args_cli.topk)
        del Z
        sim = -Dm
        if args_cli.norm == "z":
            sim = (sim - sim.mean()) / (sim.std() + 1e-8)
        elif args_cli.norm == "mean":
            sim = sim - sim.mean()
        for i in range(sim.shape[0]):
            scores[i, Im[i]] += sim[i]
        print(f"view {n_tok} -> {k} dims fused", flush=True)

    top = np.argsort(-scores, axis=1)[:, :10]
    pos = positives[: scores.shape[0]] if args_cli.max_queries else positives
    out = recalls_from_ranked(top, pos)
    print(f"{args_cli.eval_dataset_name} score-fusion ({args_cli.norm}) "
          f"R@1 {out[1]:.1f}  R@5 {out[5]:.1f}  R@10 {out[10]:.1f}", flush=True)


if __name__ == "__main__":
    main()
