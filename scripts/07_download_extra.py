"""Downloads extra open datasets with a live progress bar (speed + time left), resumable, MD5-checked, then extracts.
Usage: python 07_download_extra.py PALM"""
import hashlib, json, os, sys, time, zipfile, shutil, urllib.request, datetime
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {"PALM": dict(file="PALM.zip", url="https://ndownloader.figshare.com/files/37786152",
                     md5="28636601413706884ea89ec1be835e94", size=1659253782, extract="PALM",
                     licence="CC0", source="figshare 10.6084/m9.figshare.c.6224616 (Fang et al., PALM)")}
key = sys.argv[1]; d = SETS[key]
arch = os.path.join(ROOT, "raw", "_archives"); os.makedirs(arch, exist_ok=True)
dest = os.path.join(arch, d["file"]); part = dest + ".part"
log = os.path.join(ROOT, "logs", f"download_{key}.json")
rec = dict(d, started=datetime.datetime.now().isoformat(timespec="seconds"))
if not os.path.exists(dest):
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(d["url"], headers={"User-Agent": "Mozilla/5.0 research-download"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as r:
        mode = "ab" if (have and r.status == 206) else "wb"
        have = have if mode == "ab" else 0
        with open(part, mode) as f, tqdm(total=d["size"], initial=have, unit="B", unit_scale=True, unit_divisor=1024,
                                         desc=f"Downloading {d['file']}", dynamic_ncols=True) as bar:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); bar.update(len(b))
    os.replace(part, dest)
h = hashlib.md5()
with open(dest, "rb") as f, tqdm(total=os.path.getsize(dest), unit="B", unit_scale=True, desc="Verifying MD5", dynamic_ncols=True) as bar:
    for c in iter(lambda: f.read(8 << 20), b""):
        h.update(c); bar.update(len(c))
rec["md5"] = h.hexdigest(); rec["md5_ok"] = rec["md5"] == d["md5"]
if not rec["md5_ok"]:
    os.remove(dest); json.dump(rec, open(log, "w"), indent=2); sys.exit("MD5 mismatch - file removed, job will retry")
out = os.path.join(ROOT, "raw", d["extract"]); os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(dest) as z:
    names = [n for n in z.namelist() if "__MACOSX" not in n]
    for n in tqdm(names, desc="Extracting", unit="file", dynamic_ncols=True):
        z.extract(n, out)
rec.update(files=len(names), finished=datetime.datetime.now().isoformat(timespec="seconds"),
           folder_gb=round(sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(out) for f in fs) / 2**30, 2))
# folder overview to inspect the structure
tree = {}
for p, dn, fs in os.walk(out):
    if fs:
        tree[os.path.relpath(p, out)] = [len(fs), sorted(fs)[:3]]
rec["tree"] = tree
json.dump(rec, open(log, "w"), indent=2)
print(json.dumps({k: v for k, v in rec.items() if k != "tree"}, indent=2)); print("folders:", json.dumps(tree, indent=1)[:4000])
