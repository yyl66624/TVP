"""Mixture of spectral calibration kernels (TFA-style) for multi-view descriptors.

For every view v we fit a PCA on the chosen rows (db or db+queries, label-free)
and keep the un-whitened projection P_v = U_v^T (x - mu_v) and the spectrum
lambda_v.  A single reversible kernel is parameterised by gamma:

    phi_gamma(x) = L2( diag(lambda_v^{-gamma/2}) U_v^T (x - mu_v) )
    gamma = 0   -> plain PCA projection (no variance equalisation)
    gamma = 1   -> full PCA-whitening
    gamma = .5  -> half whitening

The final score for a (query, db) pair is a convex mixture of the cosine
similarities of the concatenated multi-view descriptors over a set of gammas:

    S = sum_gamma alpha_gamma * S_gamma(q, d)

(score fusion, default) or a reciprocal-rank fusion of the per-gamma rankings.
This mirrors the spectral-kernel mixture of the reliability-guided training-free
aggregation baseline, but applied per view and over our multi-view descriptor.

Run from TF-VPR-main:
  .venv/bin/python -I ../tools/gamma_mix.py --features <dump.npy> \
      --eval_dataset_name msls --splits 1,1,1,64 --dims 768,768,768,2048 \
      --fit all --gammas 0,0.5,1 --topk 100
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


def l2norm(x, axis=1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def evaluate(ranked, positives, ks=(1, 5, 10), topk=100):
    """ranked: (n_q, topk) candidate db ids, best first."""
    out = {}
    for k in ks:
        kk = min(k, topk)
        hits = [len(np.intersect1d(ranked[i, :kk], positives[i])) > 0
                for i in range(len(ranked))]
        out[k] = 100.0 * float(np.mean(hits))
    return out


def search_scores(index, q, topk):
    sims, ids = index.search(q, topk)          # inner product on unit vectors
    return sims, ids


def fuse_late(res, n_db, adapt="consensus", tau=0.1, rk=10, score_norm="z"):
    """Late fusion of several retrievals with per-query reliability weights.

    res: list of (sims, ids, base_weight) where base_weight folds in the view
    weight, the gamma weight and any global calibration.  Each entry is a voter
    that proposes its top-k candidates.  A voter's reliability for a query is
    how much its top-rk neighbours are corroborated by the other voters
    (cross-view / cross-kernel agreement); the per-query mixture weights are a
    softmax over that agreement.  This is the query-adaptive counterpart of a
    fixed reliability gate.
    """
    n_q, T = res[0][1].shape
    rk = min(rk, T)
    tops = [ids[:, :rk] for _, ids, _ in res]

    if adapt == "consensus" and len(res) > 1:
        agree = np.zeros((len(res), n_q), dtype="float32")
        for a in range(len(res)):
            A = tops[a]
            acc = np.zeros((n_q, rk), dtype="float32")
            for b in range(len(res)):
                if a == b:
                    continue
                B = tops[b]
                hit = (A[:, :, None] == B[:, None, :]).any(axis=-1)  # (n_q, rk)
                acc += hit
            agree[a] = acc.mean(axis=1) / (len(res) - 1)
        base = np.array([w for _, _, w in res], dtype="float32")[:, None]
        z = (agree - agree.mean(axis=0, keepdims=True)) / (tau + 1e-8)
        w_qv = base * np.exp(z)
        w_qv /= w_qv.sum(axis=0, keepdims=True) + 1e-8
        print(f"  consensus reliability: mean {agree.mean():.3f} "
              f"min {agree.min():.3f} max {agree.max():.3f}", flush=True)
    else:
        base = np.array([w for _, _, w in res], dtype="float32")[:, None]
        w_qv = np.repeat(base / base.sum(), n_q, axis=1)

    acc = np.zeros((n_q, n_db + 1), dtype="float32")
    rows = np.arange(n_q, dtype="int64")[:, None]
    for j, (sims, ids, _) in enumerate(res):
        if score_norm == "z":
            mu = sims.mean(axis=1, keepdims=True)
            sd = sims.std(axis=1, keepdims=True) + 1e-8
            s = (sims - mu) / sd
        else:
            s = sims
        np.add.at(acc, (rows, ids), (w_qv[j][:, None] * s).astype("float32"))
    acc[:, n_db] = -1e9
    return np.argsort(-acc, axis=1)[:, :T], w_qv


def fuse(scores_ids, alphas, n_db, mode="score", rrf_k=60.0, score_norm="z"):
    """Union the per-gamma candidate lists and merge their scores/ranks.

    The similarity scales of different kernels are not comparable: without
    whitening (gamma=0) every cosine is close to 1, so an un-normalised sum is
    dominated by the least calibrated kernel.  Per-query standardisation of the
    candidate scores puts the kernels on a common scale before mixing.
    """
    sims_all, ids_all = scores_ids
    n_q, T = ids_all[0].shape
    acc = np.zeros((n_q, n_db + 1), dtype="float32")   # last column = sentinel
    for a, (sims, ids) in zip(alphas, zip(sims_all, ids_all)):
        if mode == "score":
            if score_norm == "z":
                mu = sims.mean(axis=1, keepdims=True)
                sd = sims.std(axis=1, keepdims=True) + 1e-8
                contrib = a * (sims - mu) / sd
            else:
                contrib = a * sims
        else:
            contrib = a / (rrf_k + np.arange(1, T + 1)[None, :].astype("float32"))
        rows = np.arange(n_q, dtype="int64")[:, None]
        np.add.at(acc, (rows, ids), contrib.astype("float32"))
    acc[:, n_db] = -1e9
    order = np.argsort(-acc, axis=1)[:, :T]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval_dataset_name", default="msls")
    ap.add_argument("--eval_datasets_folder", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--splits", default="1,1,1,64")
    ap.add_argument("--dims", default="768,768,768,2048")
    ap.add_argument("--weights", default="1,1,1,1")
    ap.add_argument("--gammas", default="0,0.5,1")
    ap.add_argument("--gamma_weights", default="")
    ap.add_argument("--fit", default="all", choices=["db", "all"])
    ap.add_argument("--fusion", default="score", choices=["score", "rrf"])
    ap.add_argument("--score_norm", default="z", choices=["z", "none"],
                    help="Per-query standardisation of the candidate scores before "
                         "mixing the kernels")
    ap.add_argument("--mix", default="concat",
                    choices=["concat", "perview", "late"],
                    help="concat: concatenate views then mix gammas (one index per "
                         "gamma); perview: mix within a view, then concatenate "
                         "the mixed views and search once; late: one retrieval per "
                         "(view, gamma) fused with per-query reliability weights")
    ap.add_argument("--adapt", default="consensus", choices=["consensus", "none"],
                    help="Late fusion: per-query weights from cross-view agreement")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="Softmax temperature of the reliability weights")
    ap.add_argument("--rk", type=int, default=10,
                    help="Top-rk list used to measure cross-view agreement")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--positive_dist", type=float, default=25.0)
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--query_subset", default="",
                    help="Text file with the queries to evaluate: one integer index "
                         "(0-based, sorted query order) or one query filename per line. "
                         "Calibration still uses every query unless --fit_subset is set.")
    ap.add_argument("--fit_subset", action="store_true",
                    help="Fit the PCA on db + the query subset instead of db + all queries")
    ap.add_argument("--cache_dir", default="",
                    help="If set, cache the per-view projections + spectra here so "
                         "gamma/weight sweeps skip the PCA refit")
    args_cli = ap.parse_args()

    sys.argv = [sys.argv[0], "--eval_dataset_name", args_cli.eval_dataset_name,
                "--eval_datasets_folder", args_cli.eval_datasets_folder,
                "--mode", "TF_VPR", "--num_clusters", "17", "--backbone", "dinov2",
                "--val_positive_dist_threshold", str(args_cli.positive_dist)]
    args = parser.parse_arguments()
    ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder,
                                 args.eval_dataset_name, "test")
    positives = ds.soft_positives_per_query
    db_n = ds.database_num

    # ---- query selection (e.g. one protocol's query list) ----
    q_sel = np.arange(ds.queries_num)
    if args_cli.query_subset:
        by_name = {os.path.basename(p): i for i, p in enumerate(ds.queries_paths)}
        idx = []
        with open(args_cli.query_subset) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.lstrip("-").isdigit():
                    idx.append(int(s))
                elif s in by_name:
                    idx.append(by_name[s])
                elif os.path.basename(s) in by_name:
                    idx.append(by_name[os.path.basename(s)])
                else:
                    raise SystemExit(f"query not found: {s}")
        q_sel = np.array(sorted(set(idx)), dtype="int64")
    if args_cli.max_queries:
        q_sel = q_sel[: args_cli.max_queries]
    use_subset = len(q_sel) != ds.queries_num
    if use_subset:
        print(f"query subset: {len(q_sel)} / {ds.queries_num}", flush=True)

    files = args_cli.features.split(";")
    split_specs = args_cli.splits.split(";")
    dim_specs = args_cli.dims.split(";")
    assert len(files) == len(split_specs) == len(dim_specs)

    gammas = [float(g) for g in args_cli.gammas.split(",")]
    if args_cli.gamma_weights:
        alphas = [float(a) for a in args_cli.gamma_weights.split(",")]
        assert len(alphas) == len(gammas)
    else:
        alphas = [1.0 / len(gammas)] * len(gammas)
    # one weight per declared view; a view with dim 0 is skipped (useful when
    # fusing dumps that share the same CLS/GeM/mean tokens)
    weight_specs = args_cli.weights.split(";")
    all_weights = []
    for cfg in weight_specs:
        all_weights += [float(w) for w in cfg.split(",")]
    declared_k = []
    for dspec in dim_specs:
        declared_k += [int(k) for k in dspec.split(",")]
    assert len(all_weights) == len(declared_k), (all_weights, declared_k)
    view_weights = [w for w, k in zip(all_weights, declared_k) if k > 0]
    keep_view = [k > 0 for k in declared_k]

    # ---- per view: project once (gamma-independent), keep the spectrum ----
    subset_sig = ""
    if args_cli.fit_subset and use_subset:
        import hashlib
        subset_sig = hashlib.md5(q_sel.astype("int64").tobytes()).hexdigest()[:8]
    stem = ""
    if args_cli.cache_dir:
        import hashlib
        key_parts = [",".join(os.path.abspath(f) for f in files),
                     args_cli.splits, args_cli.dims, args_cli.fit,
                     str(db_n), args_cli.eval_dataset_name,
                     f"mq{args_cli.max_queries}"]
        if args_cli.fit_subset:
            key_parts.append(f"fs{subset_sig}")
        key = "|".join(key_parts)
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        os.makedirs(args_cli.cache_dir, exist_ok=True)
        stem = os.path.join(args_cli.cache_dir, tag)

    def fit_views():
        out_p, out_s = [], []
        vi = 0
        for path, sspec, dspec in zip(files, split_specs, dim_specs):
            splits = [int(s) for s in sspec.split(",")]
            klist = [int(k) for k in dspec.split(",")]
            assert len(splits) == len(klist)
            X = np.load(path, mmap_mode="r")   # only the kept view is read
            D = X.shape[1] // sum(splits)
            assert D * sum(splits) == X.shape[1], (X.shape, splits)
            offset = 0
            for n_tok, k in zip(splits, klist):
                sl = slice(offset, offset + n_tok * D)
                offset += n_tok * D
                if not keep_view[vi]:
                    vi += 1
                    continue
                vi += 1
                Xv = X[:, sl].astype("float32")
                if args_cli.fit == "db":
                    fit_rows = np.arange(db_n)
                elif args_cli.fit_subset and use_subset:
                    fit_rows = np.concatenate([np.arange(db_n), db_n + q_sel])
                else:
                    fit_rows = np.arange(len(Xv))
                k = min(k, n_tok * D, len(fit_rows) - 1)
                pca = PCA(n_components=k, whiten=False,
                          svd_solver="randomized", random_state=0)
                pca.fit(Xv[fit_rows] if len(fit_rows) != len(Xv) else Xv)
                out_p.append(pca.transform(Xv).astype("float32"))
                out_s.append(np.maximum(pca.explained_variance_, 1e-12))
                del Xv, pca
                # Large views (a 64-word VLAD block on 50k images is ~10 GB)
                # leave freed heap in the glibc arena; give it back so the next
                # view does not push the machine into swap.
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                print(f"view {n_tok} tok -> {k} comps", flush=True)
            del X
        return out_p, out_s

    if stem and os.path.exists(f"{stem}_v0.npy"):
        proj, spectra = [], []
        nv = 0
        while os.path.exists(f"{stem}_v{nv}.npy"):
            proj.append(np.load(f"{stem}_v{nv}.npy"))
            spectra.append(np.load(f"{stem}_s{nv}.npy"))
            nv += 1
        print(f"cache hit: {nv} views", flush=True)
    else:
        proj, spectra = fit_views()
        if stem:
            for i, (P, lam) in enumerate(zip(proj, spectra)):
                np.save(f"{stem}_v{i}.npy", P)
                np.save(f"{stem}_s{i}.npy", lam)
            print(f"cached -> {stem}", flush=True)

    assert len(view_weights) == len(proj), (view_weights, len(proj))
    if use_subset:
        keep = np.concatenate([np.arange(db_n), db_n + q_sel])
        proj = [P[keep] for P in proj]
        positives = [positives[i] for i in q_sel]
    n = len(proj[0])

    def build(gamma):
        """Concatenated, weighted, unit-norm descriptors for one gamma."""
        blocks = []
        for P, lam, w in zip(proj, spectra, view_weights):
            blocks.append(w * l2norm(P * lam[None, :] ** (-gamma / 2.0)))
        Z = np.concatenate(blocks, axis=1)
        del blocks
        return l2norm(Z)

    if args_cli.mix == "perview":
        # mix the *views* per gamma at the score level is not possible in a
        # single index; instead take the (weighted) mean of the per-view
        # descriptors across gammas, then search once.
        blocks = []
        for vi in range(len(proj)):
            acc = None
            for a, g in zip(alphas, gammas):
                lam = spectra[vi]
                Zv = l2norm(proj[vi] * lam[None, :] ** (-g / 2.0))
                acc = a * Zv if acc is None else acc + a * Zv
            blocks.append(view_weights[vi] * l2norm(acc))
            del acc
        Z = l2norm(np.concatenate(blocks, axis=1))
        print("perview-mixed dim", Z.shape[1], flush=True)
        index = faiss.IndexFlatIP(Z.shape[1])
        index.add(Z[:db_n])
        sims, ids = index.search(Z[db_n:], args_cli.topk)
        out = evaluate(ids, positives, topk=args_cli.topk)
        print(f"{args_cli.eval_dataset_name} mix=perview g={args_cli.gammas} "
              f"w={args_cli.weights} dim={Z.shape[1]} "
              f"R@1 {out[1]:.1f}  R@5 {out[5]:.1f}  R@10 {out[10]:.1f}", flush=True)
        return

    if args_cli.mix == "late":
        res = []
        for gi, g in enumerate(gammas):
            for vi, (P, lam) in enumerate(zip(proj, spectra)):
                Zv = l2norm(P * lam[None, :] ** (-g / 2.0))
                index = faiss.IndexFlatIP(Zv.shape[1])
                index.add(Zv[:db_n])
                sims, ids = index.search(Zv[db_n:], args_cli.topk)
                res.append((sims, ids, view_weights[vi] * alphas[gi]))
                print(f"  view {vi} gamma {g} done", flush=True)
                del Zv, index
        ranked, w_qv = fuse_late(res, db_n, adapt=args_cli.adapt,
                                 tau=args_cli.tau, rk=args_cli.rk,
                                 score_norm=args_cli.score_norm)
        out = evaluate(ranked, positives, topk=args_cli.topk)
        print(f"{args_cli.eval_dataset_name} mix=late adapt={args_cli.adapt} "
              f"g={args_cli.gammas} w={args_cli.weights} "
              f"R@1 {out[1]:.1f}  R@5 {out[5]:.1f}  R@10 {out[10]:.1f}", flush=True)
        return

    per_gamma = []
    for g in gammas:
        Z = build(g)
        index = faiss.IndexFlatIP(Z.shape[1])
        index.add(Z[:db_n])
        sims, ids = index.search(Z[db_n:], args_cli.topk)
        per_gamma.append((sims, ids, Z.shape[1]))
        print(f"  gamma={g} done", flush=True)
        del Z

    ranked = fuse(([s for s, _, _ in per_gamma], [i for _, i, _ in per_gamma]),
                  alphas, db_n, mode=args_cli.fusion,
                  score_norm=args_cli.score_norm)
    out = evaluate(ranked, positives, topk=args_cli.topk)
    print(f"{args_cli.eval_dataset_name} mix={args_cli.mix} fusion={args_cli.fusion} "
          f"g={args_cli.gammas} a={alphas} w={args_cli.weights} "
          f"dim={per_gamma[0][2]} "
          f"R@1 {out[1]:.1f}  R@5 {out[5]:.1f}  R@10 {out[10]:.1f}", flush=True)


if __name__ == "__main__":
    main()
