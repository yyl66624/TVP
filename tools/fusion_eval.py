"""Late fusion of two descriptor dumps (whitened separately, then concatenated).

Concatenating two L2-normalised whitened descriptors makes the retrieval score
the sum of the two cosine similarities, i.e. an equal-weight ensemble.

Run from TF-VPR-main:
  .venv/bin/python ../tools/fusion_eval.py --feat_a <tf.npy> --feat_b <gem.npy> \
    --pca_a 2048 --pca_b 1024 --eval_dataset_name msls \
    --eval_datasets_folder $WORKSHOP_ROOT/datasets
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


def whiten(X, fit_n, k, seed=1):
    p = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=seed)
    p.fit(X[:fit_n])
    Z = p.transform(X).astype("float32")
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def evaluate(q, db, positives, ks=(1, 5, 10)):
    idx = faiss.IndexFlatL2(db.shape[1])
    idx.add(db)
    _, pred = idx.search(q, max(ks))
    out = []
    for k in ks:
        hits = [len(np.intersect1d(pred[i, :k], positives[i])) > 0 for i in range(len(pred))]
        out.append(100.0 * np.mean(hits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_a", required=True)
    ap.add_argument("--feat_b", required=True)
    ap.add_argument("--pca_a", type=int, default=2048)
    ap.add_argument("--pca_b", type=int, default=1024)
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    a = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", a.eval_dataset_name,
                "--eval_datasets_folder", a.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2"]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, a.eval_datasets_folder, a.eval_dataset_name, "test")
    db_n = ds.database_num
    positives = ds.soft_positives_per_query

    A = np.load(a.feat_a).astype("float32")
    B = np.load(a.feat_b).astype("float32")
    print(f"A {A.shape}  B {B.shape}  db {db_n}")
    Za = whiten(A, db_n, a.pca_a)
    Zb = whiten(B, db_n, a.pca_b)
    for name, Z in (("A", Za), ("B", Zb), ("concat", np.concatenate([Za, Zb], 1))):
        r = evaluate(Z[db_n:], Z[:db_n], positives)
        print(f"{name:8s} R@1 {r[0]:5.1f}  R@5 {r[1]:5.1f}  R@10 {r[2]:5.1f}")


if __name__ == "__main__":
    main()
