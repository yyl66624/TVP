"""Database/query domain-gap diagnostics for descriptor calibration.

Given a single-view descriptor dump, fits the PCA-whitening on the database
(what most of our earlier runs did), maps both sets into that space and
reports how far the query distribution is from the database distribution:

  mean_shift : ||mu_q - mu_db|| / sqrt(k)   (per-coordinate, in db std units)
  var_ratio  : E||z_q||^2 / E||z_db||^2     (1 = same total variance)
  cos_sim    : mean cosine between the db and query means

These are the quantities that decide whether a database-fitted whitening
transfers to the queries.

  .venv/bin/python ../tools/domain_shift.py --features <dump.npy> \
      --eval_dataset_name msls --tokens 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
from sklearn.decomposition import PCA

import datasets_ws
import parser
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", required=True)
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--num_clusters", type=int, default=17)
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", str(args_cli.num_clusters),
                "--backbone", "dinov2"]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    db_n = ds.database_num

    X = np.load(args_cli.features).astype("float32")
    pca = PCA(n_components=min(args_cli.dims, X.shape[1], db_n - 1), whiten=True,
              svd_solver="randomized", random_state=0)
    pca.fit(X[:db_n])
    Z = pca.transform(X)
    zd, zq = Z[:db_n], Z[db_n:]

    md, mq = zd.mean(0), zq.mean(0)
    mean_shift = float(np.linalg.norm(mq - md) / np.sqrt(Z.shape[1]))
    var_ratio = float((zq ** 2).sum(1).mean() / (zd ** 2).sum(1).mean())
    cos_sim = float(l2norm(md[None])[0] @ l2norm(mq[None])[0])
    print(f"{args_cli.eval_dataset_name:>10s}  k={Z.shape[1]:5d}  "
          f"mean_shift {mean_shift:.3f}  var_ratio {var_ratio:.3f}  cos(means) {cos_sim:.3f}")


if __name__ == "__main__":
    main()
