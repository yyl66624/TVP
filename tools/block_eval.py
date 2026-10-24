"""Block-wise descriptor calibration.

Splits a token-major descriptor (T tokens x 768) into contiguous blocks,
PCA-whitens every block independently on the database descriptors, then
concatenates the calibrated blocks and L2-normalises.  This is the offline
version of `--whiten_blocks` in test.py.

  .venv/bin/python ../tools/block_eval.py --features <dump.npy> \
      --split 17,13 --dims 2048,1024 --eval_dataset_name msls
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


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def recalls(q, db, positives, ks=(1, 5, 10)):
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    _, pred = index.search(q, max(ks))
    return [100.0 * np.mean([len(np.intersect1d(pred[i, :k], positives[i])) > 0
                             for i in range(len(pred))]) for k in ks]


def whiten_blocks(X, fit_n, splits, dims, seed=0):
    from sklearn.decomposition import PCA
    blocks = []
    off = 0
    for s, d in zip(splits, dims):
        sl = slice(off, off + s * 768)
        off += s * 768
        p = PCA(n_components=min(d, s * 768, fit_n - 1), whiten=True,
                svd_solver="randomized", random_state=seed)
        p.fit(X[:fit_n, sl])
        blocks.append(l2n(p.transform(X[:, sl]).astype("float32")))
    assert off == X.shape[1], (off, X.shape)
    return l2n(np.concatenate(blocks, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="17,13", help="tokens per block, comma separated")
    ap.add_argument("--dims", default="2048,1024", help="whitening dim per block")
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    api = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", api.eval_dataset_name,
                "--eval_datasets_folder", api.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2"]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, api.eval_datasets_folder, api.eval_dataset_name, "test")
    db_n = ds.database_num
    X = np.load(api.features).astype("float32")
    splits = [int(s) for s in api.split.split(",")]
    dims = [int(d) for d in api.dims.split(",")]
    print(f"features {X.shape}  db {db_n}  split {splits} dims {dims}")
    Z = whiten_blocks(X, db_n, splits, dims)
    r = recalls(Z[db_n:], Z[:db_n], ds.soft_positives_per_query)
    print(f"block_calib R@1 {r[0]:5.1f}  R@5 {r[1]:5.1f}  R@10 {r[2]:5.1f}")


if __name__ == "__main__":
    main()
