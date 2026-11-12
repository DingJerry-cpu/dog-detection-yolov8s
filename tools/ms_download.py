"""ModelScope 数据集文件下载器：断点续传 + 进度显示。

用法: python ms_download.py <out_dir> <ns/name> <remote_path> [<remote_path> ...]
"""
import os
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://modelscope.cn/api/v1/datasets"


def url_for(ns_name: str, path: str, rev: str = "master") -> str:
    return f"{BASE}/{ns_name}/repo?" + urllib.parse.urlencode({"Revision": rev, "FilePath": path})


def download(url: str, dest: str, expect: int = 0) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pos = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expect and pos >= expect:
        print(f"[skip] {os.path.basename(dest)} already complete ({pos/1e6:.1f}MB)", flush=True)
        return
    headers = dict(UA)
    if pos:
        headers["Range"] = f"bytes={pos}-"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0)) + pos
        mode = "ab" if pos and r.status == 206 else "wb"
        if mode == "wb":
            pos = 0
        done = pos
        last = 0.0
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 2:
                    last = now
                    speed = (done - pos) / max(now - t0, 1e-6) / 1e6
                    pct = 100 * done / total if total else 0
                    print(
                        f"   {os.path.basename(dest)} {done/1e6:8.1f}/{total/1e6:8.1f}MB "
                        f"{pct:5.1f}% {speed:5.1f}MB/s",
                        flush=True,
                    )
    print(f"[done] {dest} {os.path.getsize(dest)/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    out_dir, ns_name, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    for p in paths:
        dest = os.path.join(out_dir, os.path.basename(p))
        try:
            download(url_for(ns_name, p), dest)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {p}: {type(e).__name__} {e}", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
