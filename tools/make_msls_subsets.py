"""Write MSLS query-subset lists for protocol / robustness checks.

The evaluation pipeline sorts query filenames, so every list here contains the
basenames of a subset of that sorted list; gamma_mix.py --query_subset maps
them back by name.

    python3 -I tools/make_msls_subsets.py --queries datasets/msls/images/val/queries \
        --out subsets/msls

Produces:
  all_<N>.txt            every query (sanity check)
  rand<N>_s<seed>.txt    random size-N subsets (query-count sensitivity)
  note_<note>.txt        day_forward / day_sideways / day_backward / night_forward
  city_<city>.txt        cph / sf
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=7401)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    names = sorted(f for f in os.listdir(a.queries) if f.endswith(".jpg"))
    os.makedirs(a.out, exist_ok=True)

    def write(tag, sel):
        path = os.path.join(a.out, f"{tag}.txt")
        with open(path, "w") as f:
            for i in sel:
                f.write(names[i] + "\n")
        print(f"{tag}: {len(sel)}")

    write(f"all_{len(names)}", np.arange(len(names)))

    rng = np.random.default_rng(0)
    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(names), size=min(a.size, len(names)),
                                 replace=False))
        write(f"rand{a.size}_s{seed}", sel)

    notes = {}
    cities = {}
    for i, n in enumerate(names):
        note = n.split("@")[-2]
        notes.setdefault(note, []).append(i)
        city = n.rsplit("_", 1)[-1].removesuffix(".jpg").rstrip("@")
        cities.setdefault(city, []).append(i)
    for k, v in sorted(notes.items()):
        write(f"note_{k}", np.array(v))
    for k, v in sorted(cities.items()):
        write(f"city_{k}", np.array(v))


if __name__ == "__main__":
    main()
