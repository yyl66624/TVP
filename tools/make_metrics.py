#!/usr/bin/env python3
"""Generate metrics.csv: every reported metric in one flat schema.

Run from the project root:
    python3 tools/make_metrics.py [--out metrics.csv]

Values are copied from EXPERIMENTS.md / REPORT_FINAL.md, which are generated
from the run logs.  Timings, VRAM and git metadata were not recorded during the
sprint and are written as NA.  `config_hash` is a short hash of the row's
(dataset, method, dims, weights, fit, radius) tuple so two rows with different
settings cannot collide.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path

ROOT = Path(os.environ.get("WORKSHOP_ROOT",
                           Path(__file__).resolve().parents[1])).resolve()

FIELDS = [
    "timestamp", "method", "method_origin", "encoder", "layer", "facet",
    "input_size", "precision", "dataset", "split", "db_count", "query_count",
    "calibration_count", "calibration_access", "seed", "recall_at_1",
    "recall_at_5", "recall_at_10", "setup_s", "feature_extract_s",
    "aggregate_s", "retrieval_s", "peak_vram_gib", "cache_size_gib", "git_sha",
    "config_hash", "notes",
]

# name: (db, q, radius, rule d, rule d_v)
BENCH = {
    "st_lucia": (1549, 1464, 25.0, 192, 256),
    "msls": (18871, 11084, 25.0, 1984, 1984),
    "eynsham": (23935, 23935, 25.0, 3168, 3168),
    "nordland": (27592, 27592, 25.0, 3648, 3648),
    "freiburg_12w": (2684, 3656, None, 416, 416),
    "freiburg_15w": (3337, 4477, None, 512, 512),
    "itlp_season_0005": (609, 720, 25.0, 64, 256),
    "itlp_day_night_0506": (720, 720, 25.0, 96, 256),
    "itlp_day_night_0708": (464, 505, 25.0, 64, 256),
    "itlp_indoor": (1310, 1233, 5.0, 160, 256),
    "robotcar_sun_night": (581, 841, 25.0, 64, 256),
    "robotcar_snow": (844, 799, 25.0, 96, 256),
    "robotcar_day_night": (844, 841, 25.0, 96, 256),
    "covpr_indoor": (10378, 558, 0.5, 704, 704),
    "nyc_indoor": (625, 625, 5.0, 64, 256),
}

# uncalibrated three-view / db-fit / all-fit, R@1 / R@5 / R@10
MVC = {
    "st_lucia": ((89.6, 95.8, 97.3), (96.9, 99.4, 99.8), (98.0, 99.5, 99.9)),
    "msls": ((39.6, 50.3, 54.9), (66.3, 78.8, 83.9), (69.3, 80.6, 85.0)),
    "eynsham": ((79.2, 88.7, 91.6), (89.2, 94.3, 95.7), (90.2, 94.9, 96.0)),
    "nordland": ((31.0, 47.1, 54.8), (2.7, 22.6, 32.0), (67.7, 78.8, 82.7)),
    "freiburg_12w": ((18.7, 54.5, 68.7), (21.5, 52.3, 66.5), (35.2, 69.0, 74.0)),
    "freiburg_15w": ((29.4, 57.2, 65.5), (31.0, 57.9, 65.3), (47.8, 71.0, 75.6)),
    "itlp_season_0005": ((73.5, 80.4, 83.2), (71.2, 79.0, 81.0), (75.6, 80.7, 82.4)),
    "itlp_day_night_0506": ((90.1, 96.8, 98.6), (88.1, 97.2, 98.6), (91.9, 97.1, 97.9)),
    "itlp_day_night_0708": ((89.9, 96.6, 97.2), (88.3, 96.8, 97.0), (93.5, 97.0, 98.2)),
    "itlp_indoor": ((73.2, 90.2, 95.4), (80.0, 95.2, 98.1), (81.8, 95.9, 98.5)),
    "robotcar_sun_night": ((54.8, 64.0, 67.1), (51.1, 64.1, 68.3), (56.6, 66.8, 70.4)),
    "robotcar_snow": ((86.2, 96.5, 97.7), (86.5, 96.5, 98.5), (91.7, 97.7, 98.7)),
    "robotcar_day_night": ((71.9, 86.2, 88.9), (65.4, 84.1, 87.3), (84.9, 90.5, 91.4)),
    "covpr_indoor": ((81.7, 89.4, 92.7), (84.6, 93.9, 95.0), (84.8, 94.1, 95.3)),
    "nyc_indoor": ((83.4, 94.9, 97.8), (83.7, 95.5, 99.2), (82.1, 95.0, 98.6)),
}

# TF view only (baseline reproduction)
TF_RAW = {
    "st_lucia": (87.5, 94.4, 96.9),
    "msls": (38.7, 50.1, 55.1),
    "eynsham": (79.7, 89.1, 92.0),
    "nordland": (26.0, 41.5, 49.0),
}

# one-codebook VLAD fusion, rule dims, R@1 only
FUSE_K32 = {
    "st_lucia": 98.6, "msls": 72.0, "eynsham": 90.5, "nordland": 73.3,
    "freiburg_12w": 37.7, "freiburg_15w": 49.6, "itlp_season_0005": 75.8,
    "itlp_day_night_0506": 92.8, "itlp_day_night_0708": 93.3,
    "itlp_indoor": 83.5, "robotcar_sun_night": 60.6, "robotcar_snow": 93.6,
    "robotcar_day_night": 85.1, "covpr_indoor": 85.1, "nyc_indoor": 84.5,
}

# tuned per-view dims, all-fit
TUNED = {
    "st_lucia": ((98.4, 99.5, 100.0), (256, 256, 256)),
    "msls": ((70.6, 81.8, 85.7), (6144, 3072, 1536)),
    "eynsham": ((89.9, 94.8, 96.0), (4096, 2048, 1024)),
    "nordland": ((74.2, 84.7, 88.1), (4096, 2048, 1024)),
}

DUAL = [
    ("st_lucia", (99.1, 99.9, 99.9), "512,512,0,512,512",
     "single pass, K32+K64"),
    ("msls", (76.7, 86.7, 90.2), "NA",
     "single pass, tuned dims not recorded per view; two-dump 76.8"),
    ("msls", (73.6, 84.7, 88.5), "1984,1984,1984,1984",
     "single pass, rule dims"),
    ("covpr_indoor", (85.1, 93.9, 95.7), "704,704,704,704",
     "rule 704, 0.5 m"),
]

# calibrated training-free baselines (raw, calibrated, k)
BASELINES = [
    ("st_lucia", "CLS", (69.1, 85.0, 90.2), (78.6, 91.6, 94.3), 768),
    ("st_lucia", "Mean", (77.7, 90.0, 93.6), (85.5, 95.4, 97.1), 768),
    ("st_lucia", "Mean+CLS", (71.4, 86.8, 91.5), (67.7, 83.7, 89.5), 1536),
    ("st_lucia", "GeM+Mean", (90.1, 95.4, 97.3), (95.9, 98.2, 98.8), 1024),
    ("st_lucia", "GeM+CLS", (90.6, 95.4, 96.9), None, 1024),
    ("st_lucia", "TF-VPR", (87.5, 94.4, 96.9), (96.5, 99.4, 99.8), 256),
    ("msls", "CLS", (35.6, 48.2, 52.8), (53.8, 67.3, 73.1), 768),
    ("msls", "Mean", (34.3, 44.6, 49.2), (61.7, 74.4, 79.4), 768),
    ("msls", "Mean+CLS", (36.3, 48.4, 53.0), (64.7, 77.1, 81.9), 1536),
    ("msls", "GeM+Mean", (37.5, 47.4, 51.8), (60.1, 72.6, 77.7), 1024),
    ("msls", "GeM+CLS", (39.5, 48.9, 53.0), (61.7, 74.5, 79.9), 1024),
    ("msls", "TF-VPR", (38.7, 50.1, 55.1), (65.7, 78.6, 83.5), 2048),
    ("eynsham", "GeM+CLS", (75.4, 85.6, 88.9), (87.8, 93.9, 95.4), 1024),
    ("eynsham", "Mean", (62.1, 74.4, 79.1), (88.9, 94.3, 95.4), 768),
    ("eynsham", "TF view", None, (86.6, 93.1, 94.9), 2048),
    ("nordland", "CLS", (14.0, 24.0, 30.1), (33.1, 46.6, 52.8), 768),
    ("nordland", "Mean", (12.5, 22.2, 28.0), (32.4, 44.9, 50.8), 768),
    ("nordland", "GeM+CLS", (39.4, 55.3, 62.9), None, 3072),
    ("nordland", "GeM view", None, (72.7, 82.2, 85.4), 3072),
]

TIMESTAMP = "2026-10-24/25"


def cfg(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def row(method, origin, dataset, r1, r5, r10, fit, dims, weights,
        calibration_access, notes, radius=None):
    db, q, default_radius, _, _ = BENCH[dataset]
    radius = default_radius if radius is None else radius
    cal = 0 if fit in ("none", "raw") else (db if fit == "db" else db + q)
    return {
        "timestamp": TIMESTAMP,
        "method": method,
        "method_origin": origin,
        "encoder": "dinov2_vitb14",
        "layer": "last",
        "facet": "multi-view" if "MVC" in method else "pooled/global",
        "input_size": 336,
        "precision": "fp16 features / fp32 compute",
        "dataset": dataset,
        "split": "test",
        "db_count": db,
        "query_count": q,
        "calibration_count": cal,
        "calibration_access": calibration_access,
        "seed": 0,
        "recall_at_1": r1,
        "recall_at_5": r5 if r5 is not None else "NA",
        "recall_at_10": r10 if r10 is not None else "NA",
        "setup_s": "NA",
        "feature_extract_s": "NA",
        "aggregate_s": "NA",
        "retrieval_s": "NA",
        "peak_vram_gib": "NA",
        "cache_size_gib": "NA",
        "git_sha": "none",
        "config_hash": cfg(dataset, method, dims, weights, fit, radius),
        "notes": notes,
    }


def build():
    rows = []
    for ds, (r1, r5, r10) in TF_RAW.items():
        rows.append(row("TF-VPR (TF view)", "independent_reimplementation", ds,
                        r1, r5, r10, "raw", "17x768", "-", "none",
                        "baseline reproduction; uncalibrated TF view only"))
    for ds, (raw, dbfit, allfit) in MVC.items():
        _, _, radius, d, _ = BENCH[ds]
        rows.append(row("MVC-VPR (uncalibrated)", "new_method", ds, *raw,
                        "raw", f"{d},{d},{d}", "1,1,1", "none",
                        f"three-view TF|GeM|Mean, radius {radius} m"))
        rows.append(row("MVC-VPR (db-fit)", "new_method", ds, *dbfit, "db",
                        f"{d},{d},{d}", "1,1,1", "database-only",
                        "PCA-whitening fitted on database only"))
        rows.append(row("MVC-VPR (all-fit)", "new_method", ds, *allfit, "all",
                        f"{d},{d},{d}", "1,1,1", "transductive",
                        "PCA-whitening fitted on database + queries, no labels"))
    for ds, r1 in FUSE_K32.items():
        _, _, radius, d, dv = BENCH[ds]
        rows.append(row("MVC-VPR+VLAD-K32", "new_method", ds, r1, None, None,
                        "all", f"{d},{d},{d},{dv}", "1,1,1,1",
                        "transductive",
                        f"one VLAD codebook, rule dims d={d} dv={dv}, "
                        f"radius {radius} m; R@1 only in source table"))
    for ds, (vals, dims) in TUNED.items():
        rows.append(row("MVC-VPR (tuned)", "new_method", ds, *vals, "all",
                        ",".join(map(str, dims)), "-", "transductive",
                        "per-dataset tuned dims and sqrt(k) weights"))
    for ds, vals, dims, note in DUAL:
        rows.append(row("MVC-VPR+VLAD-K32+K64", "new_method", ds, *vals, "all",
                        dims, "-", "transductive", note))
    for ds, name, raw, cal, k in BASELINES:
        if raw is not None:
            rows.append(row(f"TF-VPR baseline: {name} (raw)",
                            "independent_reimplementation", ds, *raw, "raw",
                            f"{k}", "-", "none",
                            f"pooled/TF baseline, uncalibrated; k={k}"))
        if cal is not None:
            rows.append(row(f"TF-VPR baseline: {name} (calibrated)",
                            "independent_reimplementation", ds, *cal, "all",
                            f"{k}", "-", "transductive",
                            f"pooled/TF baseline, all-fit PCA-whitening k={k}"))
    # single-pass TF_VLAD on MSLS (tuned and rule)
    for method, vals, note in [
        ("MVC-VPR+VLAD-K32 (single pass)", (71.9, 83.0, 87.2),
         "rule dims 1984 each"),
        ("MVC-VPR+VLAD-K32 (single pass)", (75.3, 85.9, 89.3),
         "tuned 6144/3072/2048"),
    ]:
        rows.append(row(method, "new_method", "msls", *vals, "all", note, "-",
                        "transductive", note))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics.csv")
    args = ap.parse_args()
    rows = build()
    out = ROOT / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
