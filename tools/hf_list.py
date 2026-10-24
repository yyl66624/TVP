"""Paginated listing of a HF dataset repo through the hf-mirror API.

Usage: python tools/hf_list.py <repo_id> <out.txt> [substr ...]
"""
import json
import re
import sys
import time
import urllib.request

repo, out = sys.argv[1], sys.argv[2]
subs = sys.argv[3:]

url = f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true&limit=1000"
paths, pages, sizes = [], 0, {}
while url:
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                link = r.headers.get("Link", "")
            break
        except Exception as e:
            print("retry", attempt, e, file=sys.stderr)
            time.sleep(3)
    else:
        break
    try:
        items = json.loads(body)
    except Exception:
        print("bad page, stopping", file=sys.stderr)
        break
    paths += [x["path"] for x in items if x.get("type") == "file"]
    sizes.update({x["path"]: x.get("size", 0) for x in items if x.get("type") == "file"})
    pages += 1
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    url = m.group(1).replace("https://huggingface.co", "https://hf-mirror.com") if m else None

sel = [p for p in paths if not subs or any(s in p for s in subs)]
with open(out, "w") as f:
    f.write("\n".join(sel) + "\n")
with open(out + ".sizes", "w") as f:
    f.write("\n".join(f"{sizes.get(p,0)}\t{p}" for p in sel) + "\n")
print(f"{pages} pages, {len(paths)} files, {len(sel)} selected")
