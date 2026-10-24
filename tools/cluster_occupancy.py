"""Diagnose how TF-CAM distributes patches over the k-means prototypes.

For a sample of test images this reports, per image:
  * effective number of used clusters (exp of the entropy of cluster mass),
  * fraction of patches that are >0.9 assigned to a single cluster,
  * largest / smallest cluster mass.

Run from TF-VPR-main:
    .venv/bin/python ../tools/cluster_occupancy.py --eval_dataset_name st_lucia \
      --eval_datasets_folder $WORKSHOP_ROOT/datasets --backbone dinov2 \
      --mode TF_VPR --num_clusters 17 \
      --foundation_model_path $WORKSHOP_ROOT/weights/dinov2_vitb14_pretrain.pth \
      --cluster_cache $WORKSHOP_ROOT/cache/stlucia_c17.pt \
      --infer_batch_size 64 --num_workers 8 --save_dir occ
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import parser
import datasets_ws
from model import network_infer
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    args = parser.parse_arguments()
    model = network_infer.Infer_Model(args, pretrained_foundation=True,
                                      foundation_model_path=args.foundation_model_path)
    model = model.to(args.device).eval()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    model.agg_cross.Init_Cluster(args, ds, model)
    C = model.agg_cross.clusters_num

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), min(args.infer_batch_size, len(ds)), replace=False)
    dl = DataLoader(Subset(ds, idx), batch_size=min(args.infer_batch_size, 32),
                    num_workers=args.num_workers)

    eff, peaked, frac_max, empty, cos_cls = [], [], [], [], []
    cluster_mass = np.zeros(C)
    with torch.no_grad():
        for inputs, _ in dl:
            x = model.backbone(inputs.to(args.device))
            B, P, D = x["x_prenorm"].shape
            W, H = model._patch_grid(P - 1)
            tokens = x["x_norm_patchtokens"]
            tokens = model.agg_Graph(tokens)
            x1 = tokens.reshape(B, W, H, D).permute(0, 3, 1, 2).contiguous()

            xx = F.normalize(x1, p=2, dim=1)
            logits = model.agg_cross.conv(xx).reshape(B, C, -1)
            att = F.softmax(logits / getattr(model.agg_cross, "temperature", 1.0), dim=1)
            mass = att.sum(dim=2)
            cluster_mass += mass.sum(0).cpu().numpy()
            p = mass / mass.sum(dim=1, keepdim=True)
            eff.append(torch.exp(-(p * (p + 1e-9).log()).sum(1)).cpu())
            frac_max.append((att.max(dim=1).values > 0.9).float().mean(1).cpu())
            empty.append((mass < 1e-3).sum(1).cpu())
            # cosine between cluster tokens and the CLS token
            pooled = torch.einsum("ncl,ndl->ncd", att, x1.reshape(B, D, -1))
            pooled = F.normalize(pooled, p=2, dim=2)
            cls = F.normalize(x["x_norm_clstoken"], p=2, dim=1)
            cos_cls.append((pooled * cls[:, None, :]).sum(-1).cpu())

    eff = torch.cat(eff)
    frac_max = torch.cat(frac_max)
    empty = torch.cat(empty).float()
    cos_cls = torch.cat(cos_cls)
    mass_share = cluster_mass / cluster_mass.sum()
    print(f"dataset {args.eval_dataset_name}  images {len(eff)}  clusters {C}")
    print(f"effective clusters used: mean {eff.mean():.2f}  median {eff.median():.2f}  "
          f"min {eff.min():.2f}  max {eff.max():.2f}")
    print(f"patches with w>0.9 on their best cluster: mean {frac_max.mean()*100:.1f}%")
    print(f"empty clusters per image: mean {empty.mean():.2f}")
    print("cluster mass share: " + " ".join(f"{v:.3f}" for v in sorted(mass_share, reverse=True)))
    print(f"cos(cluster token, CLS): mean {cos_cls.mean():.3f}  "
          f"max {cos_cls.max():.3f}  min {cos_cls.min():.3f}")


if __name__ == "__main__":
    main()
