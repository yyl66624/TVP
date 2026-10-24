"""Retrieval diagnostics: where do the retrieved images actually land?

Runs the standard evaluation pipeline, then reports the geodesic (UTM) distance
between each query and its top-1 / top-5 predictions.  This separates
"descriptor is weak" (predictions are cities away) from "positives/protocol is
off" (predictions are tens of metres from the query).

Usage (from TF-VPR-main, so that eval-time modules are importable):
    .venv/bin/python ../tools/retrieval_diag.py --eval_dataset_name msls \
        --eval_datasets_folder $WORKSHOP_ROOT/datasets \
        --backbone dinov2 --mode GeM_CLS --num_clusters 14 \
        --foundation_model_path $WORKSHOP_ROOT/weights/dinov2_vitb14_pretrain.pth \
        --infer_batch_size 32 --num_workers 8 --save_dir diag_msls_gemcls
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # run from TF-VPR-main so the eval modules import

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import parser
import datasets_ws
from model import network_infer
ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract(args, model, ds, offset=0):
    feats = np.empty((len(ds), args.features_dim), dtype="float32")
    dl = DataLoader(ds, num_workers=args.num_workers, batch_size=args.infer_batch_size,
                    shuffle=False, pin_memory=(args.device == "cuda"))
    with torch.no_grad():
        for inputs, indices in tqdm(dl, ncols=100):
            out = model(inputs.to(args.device)).cpu().numpy()
            feats[indices.numpy() - offset, :] = out
    return feats


def main():
    args = parser.parse_arguments()
    args.save_dir = os.path.join("test", args.save_dir, "diag")
    model = network_infer.Infer_Model(args, pretrained_foundation=True,
                                      foundation_model_path=args.foundation_model_path)
    model = model.to(args.device).eval()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    if args.mode == "TF_VPR":
        model.agg_cross.Init_Cluster(args, ds, model)
    args.features_dim *= getattr(model, "out_tokens", args.num_clusters)

    db_ds = Subset(ds, list(range(ds.database_num)))
    q_ds = Subset(ds, list(range(ds.database_num, len(ds))))
    db = extract(args, model, db_ds)
    q = extract(args, model, q_ds, offset=ds.database_num)

    index = faiss.IndexFlatL2(args.features_dim)
    index.add(db)
    D, P = index.search(q, 10)
    del db

    db_utms = ds.database_utms
    q_utms = ds.queries_utms
    top_d = np.linalg.norm(db_utms[P] - q_utms[:, None, :], axis=2)  # (Q, 10)

    positives = ds.soft_positives_per_query
    hit = np.array([len(p) > 0 for p in positives])
    in_top1 = np.array([len(np.intersect1d(P[i, :1], positives[i])) > 0 for i in range(len(P))])
    in_top5 = np.array([len(np.intersect1d(P[i, :5], positives[i])) > 0 for i in range(len(P))])
    best_rank = np.array([
        (np.where(np.isin(P[i], positives[i]))[0].min() + 1) if len(np.intersect1d(P[i], positives[i])) else 0
        for i in range(len(P))
    ])

    print(f"queries: {len(q)}  db: {len(db_utms)}  queries with positives: {hit.sum()}")
    print(f"R@1 {in_top1.mean()*100:.1f}  R@5 {in_top5.mean()*100:.1f}  "
          f"oracle-in-top10 {(best_rank>0).mean()*100:.1f}")
    for name, d in [("top1", top_d[:, 0]), ("top5-min", top_d[:, :5].min(1))]:
        print(f"{name} UTM distance (m): median {np.median(d):.1f}  mean {d.mean():.1f}  "
              f"<25 {(d<25).mean()*100:.1f}%  <50 {(d<50).mean()*100:.1f}%  "
              f"<100 {(d<100).mean()*100:.1f}%  <1km {(d<1000).mean()*100:.1f}%")
    # distance of the true nearest positive if it appears in top10
    found = best_rank > 0
    if found.any():
        rows = np.where(found)[0]
        dpos = top_d[rows, best_rank[rows] - 1]
        print(f"retrieved-positive UTM distance (m): median {np.median(dpos):.1f}  max {dpos.max():.1f}")


if __name__ == "__main__":
    main()
