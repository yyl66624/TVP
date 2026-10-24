# TVP — training-free visual place recognition with multi-view block calibration

This repository contains the **code framework only** for a training-free VPR
pipeline built on a frozen DINOv2 ViT-B/14 backbone.  No datasets, model
weights, feature dumps, result logs or paper sources are included.

The method (referred to as MVC-VPR in the accompanying notes) describes each
image with three complementary views and calibrates each view independently
before retrieval:

| view | tokens | what it captures |
|---|---|---|
| semantic | 17 | CLS + 16 TF-CAM cluster tokens |
| multi-scale | 13 | GeM over a 4×4 grid (border cells merged) |
| global | 1 | mean of patch tokens |

Each view is PCA-whitened separately (fitting on database + queries is
label-free/transductive), scaled by `sqrt(k)` and concatenated.  An optional
spherical k-means VLAD block can be appended to fuse the token and residual
families in a single forward pass.

## Layout

```
TF-VPR-main/            evaluation and model code
  eval.py               feature extraction, calibration and recall evaluation
  datasets_ws.py        dataset indexing and positive-pair generation
  parser.py             command-line options
  test.py               retrieval / metric helpers
  util.py, commons.py   utilities
  run_eval.sh           thin wrapper around eval.py
  model/
    network_infer.py    DINOv2 backbone, TF-CAM aggregation, GeM/global views, VLAD
    free_module.py      frozen-backbone helper module
    ot_cam.py           optional OT-based token purification (negative result)
    lang_prompt.py      optional CLIP language-prompt tokens (negative result)
    vision_transformer.py, normalization.py
tools/                  offline evaluation and dataset-building scripts
  calib_postproc.py     block calibration / PCA-whitening of saved descriptors
  block_eval.py         per-block whitening
  multiview_eval.py     view-combination evaluation
  postproc_eval.py      whitening, AQE and related post-processing
  gamma_mix.py          spectral calibration kernel and score mixing
  fit_vlad.py           spherical k-means codebook fitting
  fusion_eval.py, score_fusion.py   VLAD / score fusion
  raw_recall.py         recall from a saved descriptor dump (optionally one view)
  domain_shift.py, spectrum.py, retrieval_diag.py, cluster_occupancy.py,
  crossfit_check.py     diagnostics
  run_benchmark.sh      extraction + db-fit/all-fit evaluation for one benchmark
  queue_bench.sh        sequential driver for several benchmarks
  make_*.py             dataset builders (Freiburg, ITLP, RobotCar, CoVPR, NYC)
  hf_list.py, hf_fetch.py           Hugging Face mirror helpers
  smoke_model.py        import/forward smoke test
```

## Environment

Tested with Python 3.12 and CUDA 12.8:

```bash
python3.12 -m venv TF-VPR-main/.venv
TF-VPR-main/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
TF-VPR-main/.venv/bin/pip install -r requirements.txt
```

`run_eval.sh` invokes `TF-VPR-main/.venv/bin/python`; create the venv there or
edit the last line of the script.

## Data and weights (not included)

`run_eval.sh` and the tools expect:

```
$WORKSHOP_ROOT/
  datasets/<benchmark>/images/{database,queries}/...
  weights/dinov2_vitb14_pretrain.pth
```

`WORKSHOP_ROOT` defaults to the repository root.  Dataset layout follows the
[VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)
convention.  The DINOv2 weights are the public `dinov2_vitb14` `_pretrain`
checkpoint.

## Usage

Run the pipeline from `TF-VPR-main/` (the tools import modules from the
working directory):

```bash
cd TF-VPR-main

# three-view MVC descriptor, st_lucia (tuned per-view dims)
DATASET=st_lucia ./run_eval.sh mvc \
  --tf_fusion_gem --tf_mean_view \
  --whiten_blocks 17,13,1 --whiten_block_dims 256,256,256 --save_features

# one benchmark end-to-end with the dimension rule k = clip(round(N/15), 64, 4096)
../tools/run_benchmark.sh st_lucia "" 25
```

Offline evaluation of a saved descriptor dump (run from `TF-VPR-main/`):

```bash
.venv/bin/python ../tools/calib_postproc.py \
  --features test/st_lucia_mvc/*/features_hard_resize.npy \
  --eval_dataset_name st_lucia --splits 17,13,1 \
  --dims 256,256,256 --fit all --weights 1,.707,.5
```

The optional cross-family fusion is enabled with
`--mode TF_VLAD --vlad_centers2 ...`; codebooks are fitted with
`tools/fit_vlad.py`.

## Protocol

- frozen DINOv2 ViT-B/14, input 336×336 hard resize, no training;
- positives at 25 m (indoor sets use 0.5 m / 5 m);
- per-view PCA-whitening fitted on database + queries (no labels);
- per-view dimension `k = clip(round_down_32(N/15), 64, 4096)`,
  `N = database + queries`, with a floor of 256 for the VLAD block;
- recall at 1/5/10 by exact L2 search on L2-normalised descriptors.

## Attribution and license

`TF-VPR-main/` is derived from the official TF-VPR code
(https://github.com/ddfs430/TF-VPR, MIT License, Copyright (c) 2024 Feng Lu)
and has been
modified for multi-view calibration and VLAD fusion.  The original MIT license
is retained in `LICENSE`; see `NOTICE` for the list of changes.  The scripts in
`tools/` are released under the same license.
