#!/usr/bin/env python3
"""Resumable threaded downloader for the HF mirror.

    hf_fetch.py <repo> <list.txt> <outdir> [workers]

The mirror answers with an HTML error page (404/429) under load and throttles
long-lived connections, so every transfer is sniffed for HTML and re-opened
(with a Range resume) after DEADLINE seconds.  Progress is never thrown away.
"""
import os
import sys
import threading
import time
import urllib.request

repo, listfile, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
base = f"https://hf-mirror.com/datasets/{repo}/resolve/main/"
DEADLINE = 25   # seconds per connection
MAX_STALLS = 6  # consecutive attempts without a single new byte

files = [x.strip() for x in open(listfile) if x.strip()]
lock = threading.Lock()
stats = [0, 0]  # ok, failed


def log(msg):
    with lock:
        print(msg, flush=True)


def remote_size(rel):
    try:
        req = urllib.request.Request(base + rel, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length", "0"))
    except Exception:
        return 0


def fetch(rel):
    dest = os.path.join(outdir, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log(f"skip {rel}")
        return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    total = remote_size(rel)
    stalls = 0
    while stalls < MAX_STALLS:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total and have >= total:
            os.replace(part, dest)
            log(f"done {rel} {have//1024//1024}MB")
            return True
        req = urllib.request.Request(base + rel)
        if have:
            req.add_header("Range", f"bytes={have}-")
        got = 0
        finished_early = False
        try:
            deadline = time.time() + DEADLINE
            with urllib.request.urlopen(req, timeout=20) as r:
                head = r.read(512)
                if b"<!doctype html" in head.lower() or b"<html" in head.lower():
                    raise ValueError("html error page")
                mode = "ab" if have and r.status == 206 else "wb"
                with open(part, mode) as f:
                    f.write(head)
                    got += len(head)
                    while time.time() < deadline:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            finished_early = True
                            break
                        f.write(chunk)
                        got += len(chunk)
        except Exception as e:
            if got == 0:
                log(f"retry {rel}: {e}")
        now = os.path.getsize(part) if os.path.exists(part) else 0
        if finished_early and (not total or now >= total):
            os.replace(part, dest)
            log(f"done {rel} {now//1024//1024}MB")
            return True
        if now > have:
            stalls = 0
            if not total:
                total = remote_size(rel)
            continue
        stalls += 1
        time.sleep(min(20, 3 * stalls))
    if os.path.exists(part) and not total:
        os.replace(part, dest)
        log(f"done(unknown-size) {rel}")
        return True
    log(f"FAIL {rel} ({os.path.getsize(part) if os.path.exists(part) else 0}/{total})")
    return False


def worker():
    while True:
        with lock:
            if not files:
                return
            rel = files.pop(0)
        ok = fetch(rel)
        with lock:
            stats[0 if ok else 1] += 1


ts = [threading.Thread(target=worker) for _ in range(workers)]
[t.start() for t in ts]
[t.join() for t in ts]
print(f"HF_FETCH_DONE ok={stats[0]} failed={stats[1]}")
