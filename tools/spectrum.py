"""Spectral analysis of a multi-view descriptor dump.

Reports, for each view and for the joint concatenation:
  * participation ratio of the covariance (effective rank)
  * variance share of the first 5 / first 256 principal directions
  * how the energy of the joint top-256 directions splits across the views

  .venv/bin/python ../tools/spectrum.py --features <dump.npy> --split 17,13 --db 18871
"""
import argparse

import numpy as np
from sklearn.decomposition import PCA


def participation_ratio(Z):
    Z = Z - Z.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh((Z.T @ Z / len(Z)).astype(np.float64))[::-1]
    lam = np.clip(lam, 0, None)
    return (lam.sum() ** 2) / (lam ** 2).sum(), lam / lam.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="17,13")
    ap.add_argument("--db", type=int, required=True)
    ap.add_argument("--dims", type=int, default=256)
    args = ap.parse_args()

    X = np.load(args.features).astype("float32")[: args.db]
    tokens = [int(t) for t in args.split.split(",")]
    D = X.shape[1] // sum(tokens)
    views, off = [], 0
    for t in tokens:
        views.append(X[:, off:off + t * D])
        off += t * D
    for i, V in enumerate(views):
        pr, share = participation_ratio(V)
        print(f"view {i}: tokens {tokens[i]:2d}  dim {V.shape[1]:6d}  "
              f"PR {pr:6.1f}  top5 {np.round(share[:5], 3)}  "
              f"top256 {share[:256].sum():.3f}  var {V.var(0).sum():.4f}")
    p = PCA(n_components=args.dims, whiten=False, svd_solver="randomized",
            random_state=0).fit(X)
    Vc = p.components_
    e = [(Vc[:, off:off + t * D] ** 2).sum(1) for t, off in
         zip(tokens, np.cumsum([0] + [t * D for t in tokens[:-1]]))]
    tot = sum(x.sum() for x in e)
    print("joint top-%d energy share: %s" % (
        args.dims, "  ".join(f"view{i} {x.sum() / tot:.3f}" for i, x in enumerate(e))))
    print("joint explained variance: %.3f" % p.explained_variance_ratio_.sum())


if __name__ == "__main__":
    main()
