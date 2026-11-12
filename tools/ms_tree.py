import json, sys, urllib.parse, urllib.request

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def tree(ns, name, root="", rev="master"):
    url = (
        f"https://modelscope.cn/api/v1/datasets/{ns}/{name}/repo/tree?"
        + urllib.parse.urlencode({"Revision": rev, "Root": root})
    )
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read())
    return (d.get("Data") or {}).get("Files") or []


def show(ns, name, root="", depth=2):
    print(f"\n=== {ns}/{name}  root='{root}'")
    try:
        files = tree(ns, name, root)
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:100])
        return
    for f in files:
        kind = "DIR " if f.get("Type") == "tree" else "FILE"
        print(f"  {kind} {f.get('Path'):<60} {(f.get('Size') or 0)/1e6:9.2f}MB")
    if depth > 1:
        for f in files:
            if f.get("Type") == "tree" and f.get("Path", "").count("/") <= 1:
                show(ns, name, f["Path"], depth - 1)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        ns, name = arg.split("/", 1)
        show(ns, name)
