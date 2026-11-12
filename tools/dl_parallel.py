"""多线程分段下载器（支持 Range 的服务器）。

    python tools/dl_parallel.py <url> <输出文件> [线程数]

比单连接快很多：GitHub 的 CDN 单连接常被限速到几十 KB/s，8 条并发能跑满带宽。
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

UA = {"User-Agent": "Mozilla/5.0"}
lock = threading.Lock()
done_bytes = 0
total_size = 0


def head(url: str) -> tuple[str, int, bool]:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    r = urllib.request.urlopen(req, timeout=30)
    return r.geturl(), int(r.headers.get("Content-Length", 0)), r.headers.get("Accept-Ranges") == "bytes"


def worker(url: str, path: Path, start: int, end: int, idx: int, errors: list) -> None:
    global done_bytes
    pos = start
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "r+b") as f:
                f.seek(pos)
                while pos <= end:
                    chunk = r.read(min(1 << 20, end - pos + 1))
                    if not chunk:
                        break
                    f.write(chunk)
                    pos += len(chunk)
                    with lock:
                        done_bytes += len(chunk)
            if pos > end:
                return
        except Exception as e:  # noqa: BLE001
            errors.append(f"chunk{idx} {type(e).__name__}: {e}")
            time.sleep(1 + attempt)
    errors.append(f"chunk{idx} 未完成 ({pos}/{end})")


def main() -> None:
    url, out = sys.argv[1], Path(sys.argv[2])
    nthreads = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    final, size, ranges = head(url)
    print(f"文件: {size/1e6:.1f} MB  分段支持: {ranges}", flush=True)
    if size == 0:
        raise SystemExit("拿不到文件大小")

    out.parent.mkdir(parents=True, exist_ok=True)
    part_path = out.with_suffix(out.suffix + ".part")
    if out.exists() and out.stat().st_size == size:
        print("[skip] 已下载完整")
        return
    if part_path.exists():
        part_path.unlink()
    with open(part_path, "wb") as f:
        f.truncate(size)

    part = size // nthreads
    spans = [(i * part, (i + 1) * part - 1) for i in range(nthreads)]
    spans[-1] = (spans[-1][0], size - 1)

    errors: list[str] = []
    threads = [
        threading.Thread(target=worker, args=(final, part_path, s, e, i, errors), daemon=True)
        for i, (s, e) in enumerate(spans)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        el = max(time.time() - t0, 1e-6)
        print(f"   {done_bytes/1e6:7.1f}/{size/1e6:7.1f}MB  {done_bytes/el/1e6:5.2f}MB/s", flush=True)
    for t in threads:
        t.join()

    got = part_path.stat().st_size
    if errors:
        print("警告:", *errors[:3], sep="\n  ")
    if got != size or errors:
        print(f"[失败] {got/1e6:.1f}/{size/1e6:.1f}MB，保留 .part 供重试", flush=True)
        raise SystemExit(1)
    if out.exists():
        out.unlink()
    part_path.rename(out)
    print(f"[done] {got/1e6:.1f}MB in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()

