#!/usr/bin/env python3
"""Write manifest.json: datasets, weights, dumps, caches, code and environment.

Run from the project root:
  python3 tools/make_manifest.py [--out manifest.json]

Only paths that exist are listed; hashes are SHA-256.  Dataset counts and
radii are the documented protocol (EXPERIMENTS.md), not re-derived here.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WORKSHOP_ROOT",
                           Path(__file__).resolve().parents[1])).resolve()

DATASETS = {
    # name: (db, queries, positive radius m, source, licence)
    "st_lucia": (1549, 1464, 25.0, "QUT St Lucia", "research use"),
    "msls": (18871, 11084, 25.0, "Mapillary MSLS val", "CC BY-NC-SA 4.0"),
    "eynsham": (23935, 23935, 25.0, "Eynsham official downloader layout",
                "research use"),
    "nordland": (27592, 27592, 25.0, "Nordland summer db / winter queries",
                 "research use"),
    "freiburg_12w": (2684, 3656, None,
                     "fansel/freiburg-across-seasons, summer2012->winter",
                     "CC BY-SA 4.0"),
    "freiburg_15w": (3337, 4477, None,
                     "fansel/freiburg-across-seasons, summer2015->winter",
                     "CC BY-SA 4.0"),
    "itlp_season_0005": (609, 720, 25.0,
                         "OPR-Project/ITLP-Campus-Outdoor 00->05", "CC BY 4.0"),
    "itlp_day_night_0506": (720, 720, 25.0,
                            "OPR-Project/ITLP-Campus-Outdoor 05->06",
                            "CC BY 4.0"),
    "itlp_day_night_0708": (464, 505, 25.0,
                            "OPR-Project/ITLP-Campus-Outdoor 07->08",
                            "CC BY 4.0"),
    "itlp_indoor": (1310, 1233, 5.0,
                    "OPR-Project/ITLP-Campus-Indoor 01->00, floors separated",
                    "CC BY 4.0"),
    "robotcar_sun_night": (581, 841, 25.0,
                           "Oxford RobotCar 2014-07-14 -> 2014-12-10",
                           "CC BY-NC-SA 4.0"),
    "robotcar_snow": (844, 799, 25.0,
                      "Oxford RobotCar 2014-11-18 -> 2015-02-03",
                      "CC BY-NC-SA 4.0"),
    "robotcar_day_night": (844, 841, 25.0,
                           "Oxford RobotCar 2014-11-18 -> 2014-12-10",
                           "CC BY-NC-SA 4.0"),
    "covpr_indoor": (10378, 558, 0.5, "ai4ce/CoVPR Indoor (test)", "Apache-2.0"),
    "nyc_indoor": (625, 625, 5.0, "deansmile123/NYC-Indoor-VPR", "unstated"),
}

WEIGHTS = [
    "weights/dinov2_vitb14_pretrain.pth",
    "weights/dinov2_vitl14_pretrain.pth",
    "weights/clip-vit-base-patch16/pytorch_model.bin",
]

CODE = [
    "TF-VPR-main/eval.py", "TF-VPR-main/test.py",
    "TF-VPR-main/datasets_ws.py", "TF-VPR-main/parser.py",
    "TF-VPR-main/util.py", "TF-VPR-main/commons.py",
    "TF-VPR-main/model/network_infer.py", "TF-VPR-main/model/ot_cam.py",
    "TF-VPR-main/model/free_module.py", "TF-VPR-main/model/lang_prompt.py",
    "TF-VPR-main/run_eval.sh",
    "tools/calib_postproc.py", "tools/gamma_mix.py", "tools/block_eval.py",
    "tools/multiview_eval.py", "tools/fusion_eval.py", "tools/score_fusion.py",
    "tools/fit_vlad.py", "tools/raw_recall.py", "tools/crossfit_check.py",
    "tools/domain_shift.py", "tools/postproc_eval.py", "tools/spectrum.py",
    "tools/retrieval_diag.py", "tools/cluster_occupancy.py",
    "tools/run_benchmark.sh", "tools/queue_bench.sh",
    "tools/make_figure.py", "tools/make_metrics.py", "tools/make_manifest.py",
]

PACKAGES = ["torch", "torchvision", "faiss-cpu", "numpy", "scikit-learn",
            "timm", "transformers", "Pillow", "matplotlib"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree(path):
    p = ROOT / path
    if not p.exists():
        return None
    files, total = 0, 0
    for dirpath, _, names in os.walk(p):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
                files += 1
            except OSError:
                pass
    return {"path": path, "files": files, "bytes": total}


def dump_inventory():
    base = ROOT / "TF-VPR-main" / "test"
    rows = []
    if not base.exists():
        return rows
    for run in sorted(base.iterdir()):
        if not run.is_dir():
            continue
        npy = list(run.rglob("features*.npy"))
        if not npy:
            continue
        f = npy[0]
        rows.append({"run": run.name, "file": str(f.relative_to(ROOT)),
                     "bytes": f.stat().st_size})
    return rows


def env_versions():
    out = {}
    try:
        import platform
        out["python"] = platform.python_version()
    except Exception:
        pass
    try:
        from importlib import metadata
        for p in PACKAGES:
            try:
                out[p] = metadata.version(p)
            except Exception:
                out[p] = None
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--skip-hashes", action="store_true")
    args = ap.parse_args()

    manifest = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "git": {"is_repo": False, "commit": None},
        "environment": env_versions(),
        "protocol": {
            "backbone": "dinov2_vitb14",
            "input_size": 336,
            "tokens": "17 TF + 13 GeM + 1 mean",
            "descriptor": "per-view L2, concatenate, global L2",
            "retrieval": "FAISS IndexFlatL2",
            "calibration": "PCA-whitening per view",
            "fit": "database + queries (all-fit), label-free",
            "dim_rule": "k = clip(round32(N/15), 64, 4096) per view; "
                        "VLAD block floor 256",
            "weights": "sqrt(k) of per-view dims, normalised; equal when equal",
        },
        "datasets": [
            {"name": n, "db": v[0], "queries": v[1],
             "positive_radius_m": v[2], "source": v[3], "licence": v[4],
             "path": f"datasets/{n}",
             "present": (ROOT / "datasets" / n).exists()}
            for n, v in DATASETS.items()
        ],
        "weights": [],
        "code": [],
        "inventory": {
            "note": "file counts and logical bytes from os.path.getsize; "
                    "symlinked image files are counted at target size, "
                    "symlinked directories are not traversed",
            "datasets": tree("datasets"),
            "datasets_raw": tree("datasets_raw"),
            "feature_dumps": tree("TF-VPR-main/test"),
            "cache": tree("cache"),
        },
        "feature_dumps": dump_inventory(),
        "missing_or_external": ["pitts30k (released on request by authors)"],
    }
    for rel in WEIGHTS:
        p = ROOT / rel
        manifest["weights"].append({
            "path": rel, "present": p.exists(),
            "bytes": p.stat().st_size if p.exists() else None,
            "sha256": None if (args.skip_hashes or not p.exists())
            else sha256(p),
        })
    for rel in CODE:
        p = ROOT / rel
        manifest["code"].append({
            "path": rel, "present": p.exists(),
            "sha256": None if (args.skip_hashes or not p.exists())
            else sha256(p),
        })
    out = ROOT / args.out
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({(out.stat().st_size / 1024):.1f} KiB)")


if __name__ == "__main__":
    main()
