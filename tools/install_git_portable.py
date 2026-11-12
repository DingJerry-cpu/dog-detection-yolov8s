"""从国内镜像装一个免安装的 PortableGit（不需要管理员权限）。

    python tools/install_git_portable.py [--dest D:\\ds工作文档2\\tools\\PortableGit]

镜像：https://registry.npmmirror.com/-/binary/git-for-windows/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

UA = {"User-Agent": "Mozilla/5.0"}
MIRROR = "https://registry.npmmirror.com/-/binary/git-for-windows/"


def get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def pick_version() -> str:
    dirs = [x["name"].rstrip("/") for x in get_json(MIRROR) if x.get("type") == "dir"]
    stable = [d for d in dirs if "rc" not in d and d.startswith("v2.")]
    stable.sort(key=lambda s: [int(p) for p in s.strip("v/").split(".")[:3] if p.isdigit()])
    return stable[-1]


def pick_asset(version: str) -> tuple[str, int]:
    files = get_json(MIRROR + version + "/")
    cands = [f for f in files if f["name"].startswith("PortableGit-") and f["name"].endswith("64-bit.7z.exe")]
    if not cands:
        raise SystemExit(f"{version} 下没找到 PortableGit")
    f = max(cands, key=lambda x: x.get("size", 0))
    return f["name"], int(f.get("size", 0))


def download(url: str, dest: Path, total: int) -> None:
    if dest.exists() and dest.stat().st_size == total:
        print(f"[skip] 已下载完整 {dest.name}")
        return
    pos = dest.stat().st_size if dest.exists() else 0
    headers = dict(UA)
    if pos:
        headers["Range"] = f"bytes={pos}-"
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        mode = "ab" if pos and r.status == 206 else "wb"
        if mode == "wb":
            pos = 0
        done, last = pos, 0.0
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 1:
                    last = time.time()
                    print(f"   {done/1e6:6.1f}/{total/1e6:6.1f}MB  {done/max(time.time()-t0,1e-6)/1e6:5.1f}MB/s", flush=True)
    print(f"[done] {dest} {dest.stat().st_size/1e6:.1f}MB in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent.parent / "tools" / "PortableGit"))
    args = ap.parse_args()

    ver = pick_version()
    name, size = pick_asset(ver)
    print(f"版本 {ver}  文件 {name}  {size/1e6:.1f}MB")

    cache = Path(os.environ.get("TEMP", ".")) / name
    download(MIRROR + ver + "/" + name, cache, size)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[..] 解压到 {dest}")
    # 7z 自解压包：-o 指定输出目录，-y 全部覆盖
    subprocess.run([str(cache), f"-o{dest}", "-y"], check=True)
    git = dest / "cmd" / "git.exe"
    if not git.exists():
        git = dest / "bin" / "git.exe"
    if not git.exists():
        raise SystemExit("解压后找不到 git.exe")
    out = subprocess.run([str(git), "--version"], capture_output=True, text=True).stdout.strip()
    print(f"[OK] {out}  ->  {git}")


if __name__ == "__main__":
    main()
