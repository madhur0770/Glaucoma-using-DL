"""Stage 1 / step 1: download the open-access datasets into raw/_archives, verify MD5, extract to raw/<name>.
Standard library only. Resumable: re-running skips finished files and continues partial ones.
Progress and results are written to logs/downloads.json after every file."""
import hashlib, json, os, shutil, sys, time, urllib.request, zipfile, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "raw", "_archives")
LOG = os.path.join(ROOT, "logs", "downloads.json")
os.makedirs(ARCH, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)

# name, file, url, md5, bytes, extract_to (None = keep zip only)
# Chaksu zips are NOT extracted here: their TIFF masks expand ~14x (2.7 GB zip -> 37 GB). 03_chaksu_compact.py extracts them as PNG.
DATASETS = [
    ("SegReady_RIGA_masks", "RIGA_preprocessed.zip",
     "https://zenodo.org/records/23158636/files/RIGA_preprocessed.zip?download=1",
     "34799d72c25a42c32d4cd70b3b775919", 13_600_000, "SegReady_RIGA"),
    ("PAPILA", "PAPILA.zip", "https://ndownloader.figshare.com/files/35013982",
     "7d73fc5157d44b221a62ed5a4baf3779", 590_857_635, "PAPILA"),
    ("Chaksu_readme", "Readme_Chaksu_IMAGE_Database.pdf", "https://ndownloader.figshare.com/files/38944805",
     "47702fba2507bfa11f08fc6d550a7d53", 555_992, None),
    ("Chaksu_Test", "Chaksu_Test.zip", "https://ndownloader.figshare.com/files/37875687",
     "9fdac2596e1e884c7a19dabce1ee7339", 2_704_908_552, None),
    ("Chaksu_Train", "Chaksu_Train.zip", "https://ndownloader.figshare.com/files/37875672",
     "116970c47a89d435523f5c371b8d74a6", 8_629_271_653, None),
]

results = {}
if os.path.exists(LOG):
    try:
        results = json.load(open(LOG))
    except Exception:
        results = {}


def save():
    json.dump(results, open(LOG, "w"), indent=2)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected):
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 glaucoma-research-download"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    t0, last = time.time(), 0
    with urllib.request.urlopen(req, timeout=120) as r:
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        total = expected or int(r.headers.get("Content-Length", 0)) + have
        with open(part, mode) as f:
            while True:
                buf = r.read(4 << 20)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if time.time() - last > 5:
                    last = time.time()
                    speed = (have / 2**20) / max(1e-6, time.time() - t0)
                    pct = 100 * have / total if total else 0
                    print(f"   {os.path.basename(dest)}: {have/2**20:,.0f} MB / {total/2**20:,.0f} MB "
                          f"({pct:4.1f}%)  {speed:.1f} MB/s", flush=True)
    os.replace(part, dest)
    return round(time.time() - t0, 1)


for name, fname, url, md5, size, extract_to in DATASETS:
    dest = os.path.join(ARCH, fname)
    rec = results.get(name, {})
    if os.path.exists(dest) and rec.get("md5_ok") and (rec.get("status") == "ok" or extract_to is None):
        rec["status"] = "ok"; results[name] = rec; save()
        print(f"[skip] {name} already done")
        continue
    free = shutil.disk_usage(ROOT).free
    need = size * (2.2 if extract_to else 1.1) + 5 * 2**30   # archive + extracted copy + 5 GB margin
    rec = {"file": fname, "url": url, "expected_bytes": size, "expected_md5": md5,
           "started": datetime.datetime.now().isoformat(timespec="seconds"),
           "disk_free_gb_before": round(free / 2**30, 1)}
    if free < need:
        rec.update(status="skipped_low_disk", note=f"needs ~{need/2**30:.1f} GB free")
        results[name] = rec; save()
        print(f"[SKIP] {name}: not enough disk ({free/2**30:.1f} GB free, need {need/2**30:.1f})")
        continue
    print(f"[download] {name} ({size/2**20:,.0f} MB)")
    try:
        if not os.path.exists(dest):
            rec["seconds"] = download(url, dest, size)
        rec["bytes"] = os.path.getsize(dest)
        print(f"   verifying MD5 ...")
        rec["md5"] = md5sum(dest)
        rec["md5_ok"] = rec["md5"] == md5
        if not rec["md5_ok"]:
            rec["status"] = "md5_mismatch"
        elif extract_to:
            out = os.path.join(ROOT, "raw", extract_to)
            os.makedirs(out, exist_ok=True)
            print(f"   extracting to {out} ...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(out)
                rec["files_in_zip"] = len(z.namelist())
            rec["status"] = "ok"
        else:
            rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = repr(e)
    rec["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    rec["disk_free_gb_after"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
    results[name] = rec; save()
    print(f"   -> {rec['status']}")

print("\nSummary:")
for k, v in results.items():
    print(f"  {k:22s} {v.get('status')}")
print("Saved", LOG)
