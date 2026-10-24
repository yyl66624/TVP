"""Fit a spherical k-means vocabulary on DINOv2 patch tokens for --mode VLAD.

Sampling is label-free: every s-th database image, a random subset of patches
per image.  The vocabulary is then reused for every image (db + queries).

Run from TF-VPR-main:
  .venv/bin/python -I ../tools/fit_vlad.py --eval_dataset_name msls \
      --sample_images 1500 --patches_per_image 96 --k 64 \
      --out ../weights/vlad_k64_dinov2b_336.npy
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import datasets_ws
import parser
from model import network_infer
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--foundation_model_path",
                    default=os.path.join(ROOT, "weights", "dinov2_vitb14_pretrain.pth"))
    ap.add_argument("--sample_images", type=int, default=1500)
    ap.add_argument("--patches_per_image", type=int, default=96)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the patch subsample and the k-means init; a "
                         "second vocabulary with another seed gives an "
                         "independent codebook for cross-codebook fusion")
    ap.add_argument("--out", required=True)
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0],
                "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "CLS", "--backbone", "dinov2",
                "--foundation_model_path", args_cli.foundation_model_path,
                "--val_positive_dist_threshold", "25"]
    args = parser.parse_arguments()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = network_infer.Infer_Model(
        args, pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path).to(args.device).eval()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder,
                                 args.eval_dataset_name, "test")
    db_n = ds.database_num
    step = max(1, db_n // args_cli.sample_images)
    ids = list(range(0, db_n, step))[: args_cli.sample_images]
    loader = DataLoader(Subset(ds, ids), batch_size=args_cli.batch_size,
                        num_workers=8, pin_memory=(args.device == "cuda"))

    torch.manual_seed(args_cli.seed)
    chunks = []
    with torch.no_grad():
        for bi, (imgs, _) in enumerate(loader):
            out = model.backbone(imgs.to(args.device))
            tok = out["x_norm_patchtokens"]
            tok = F.normalize(tok, p=2, dim=-1)
            p = min(args_cli.patches_per_image, tok.shape[1])
            idx = torch.randperm(tok.shape[1], device=tok.device)[:p]
            chunks.append(tok[:, idx].reshape(-1, tok.shape[-1]).float().cpu())
            if bi % 10 == 0:
                print(f"batch {bi}/{len(loader)}", flush=True)
    X = torch.cat(chunks).numpy()
    print("tokens:", X.shape, flush=True)

    kmeans = faiss.Kmeans(d=X.shape[1], k=args_cli.k, niter=30,
                          verbose=True, spherical=True, seed=args_cli.seed, gpu=False)
    kmeans.train(X)
    C = kmeans.centroids.astype("float32")
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    np.save(args_cli.out, C)
    print("saved", args_cli.out, C.shape, flush=True)


if __name__ == "__main__":
    main()
