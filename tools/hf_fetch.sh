#!/bin/bash
# Robust parallel downloader for the HF mirror.
#   hf_fetch.sh <repo> <list.txt> <outdir> [parallel] [max_attempts]
# The mirror answers 404/429 with an HTML page under load, so every finished
# transfer is sniffed: an HTML body is discarded and retried with backoff.
set -u
REPO="$1"; LIST="$2"; OUT="$3"; P="${4:-4}"; MAX="${5:-10}"
mkdir -p "$OUT"
fetch_one() {
  local f="$1" dest="$2" repo="$3" url
  url="https://hf-mirror.com/datasets/$repo/resolve/main/$f"
  [ -s "$dest" ] && { echo "skip $f"; return 0; }
  mkdir -p "$(dirname "$dest")"
  local a=1
  while [ "$a" -le "$MAX" ]; do
    curl -sL --retry 0 --connect-timeout 15 --speed-limit 800000 --speed-time 12 -C - \
         -o "$dest.part" "$url" >/dev/null 2>&1
    if [ -s "$dest.part" ]; then
      if head -c 512 "$dest.part" | grep -qiE '<!doctype html|<html'; then
        rm -f "$dest.part"                    # error page, not the file
      else
        mv "$dest.part" "$dest" && { echo "done $f"; return 0; }
      fi
    fi
    a=$((a+1)); sleep $((a*3))
  done
  echo "FAIL $f"
}
export -f fetch_one; export REPO MAX
grep -v '^$' "$LIST" | xargs -P "$P" -I{} bash -c 'fetch_one "$@"' _ {} "$OUT/{}"
echo HF_FETCH_DONE
