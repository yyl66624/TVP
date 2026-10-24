#!/bin/bash
# Usage: hf_list.sh <repo_id> <out_file>  -- lists every file path in a HF dataset repo (paginated)
REPO="$1"; OUT="$2"
URL="https://hf-mirror.com/api/datasets/$REPO/tree/main?recursive=true"
: > "$OUT.part"
while [ -n "$URL" ]; do
  HDR=$(mktemp)
  BODY=$(mktemp)
  curl -s --max-time 60 -D "$HDR" -o "$BODY" "$URL"
  python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
for x in d:
    if x.get('type')=='file': print(x['path'])
" "$BODY" >> "$OUT.part"
  URL=$(grep -i '^link:' "$HDR" | sed -n 's/.*<\([^>]*\)>; rel="next".*/\1/p')
  rm -f "$HDR" "$BODY"
done
mv "$OUT.part" "$OUT"
wc -l "$OUT"
