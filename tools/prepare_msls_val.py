"""Prepare the MSLS validation split in the layout expected by TF-VPR.

Raw input: Mapillary SLS release (metadata CSVs + image volumes), as mirrored
by deansmile123/msls on HF. Output:
  datasets/msls/images/val/database/*.jpg
  datasets/msls/images/val/queries/*.jpg
  datasets/msls/images/test -> val
Filenames follow the VPR-datasets-downloader convention:
  @easting@northing@zone@letter@lat@lon@pano_id@...@note@.jpg
"""
import argparse, csv, os, shutil, sys
import utm

CITIES = {"cph": "cph", "sf": "sf"}

def fmt(num, left=2, right=5):
    return f"{float(num):0={left+right+1}.{right}f}"

def dst_name(lat, lon, pano_id, timestamp, note):
    e, n, zn, zl = utm.from_latlon(float(lat), float(lon))
    return (f"@{fmt(e,7,2)}@{fmt(n,7,2)}@{zn:02d}@{zl}@{fmt(lat,3,5)}@{fmt(lon,4,5)}"
            f"@{pano_id}@@@@@@@@{timestamp}@{note}@.jpg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help=".../msls_raw/meta/train_val")
    ap.add_argument("--images", required=True, help="staging root with train_val/<city>/<split>/images")
    ap.add_argument("--out", required=True, help="datasets/msls/images/val")
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy")
    a = ap.parse_args()

    total = 0
    for city in CITIES:
        for split, out_split in (("database", "database"), ("query", "queries")):
            src_dir = os.path.join(a.images, "train_val", city, split, "images")
            csv_dir = os.path.join(a.meta, city, split)
            out_dir = os.path.join(a.out, out_split)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(csv_dir, "raw.csv"), newline="") as f:
                raw = list(csv.reader(f))[1:]
            with open(os.path.join(csv_dir, "postprocessed.csv"), newline="") as f:
                post = list(csv.reader(f))[1:]
            assert len(raw) == len(post), (len(raw), len(post), city, split)
            n_done = 0
            for r, p in zip(raw, post):
                # raw: idx,key,lon,lat,ca,captured_at,pano
                pano_id, lon, lat = r[1], r[2], r[3]
                timestamp = r[5].replace("-", "")
                is_pano = r[6]
                if is_pano == "True":
                    continue
                night = p[-2] != "False"
                view = p[-1].lower()
                note = f"{'night' if night else 'day'}_{view}_{city}"
                src = os.path.join(src_dir, f"{pano_id}.jpg")
                dst = os.path.join(out_dir, dst_name(lat, lon, pano_id, timestamp, note))
                if os.path.exists(dst):
                    n_done += 1
                    continue
                if not os.path.exists(src):
                    print("MISSING", src, file=sys.stderr)
                    continue
                if a.link:
                    os.link(src, dst)
                else:
                    shutil.copy2(src, dst)
                n_done += 1
            print(f"{city}/{split}: {n_done} images -> {out_dir}")
            total += n_done
    print("total", total)

if __name__ == "__main__":
    main()
