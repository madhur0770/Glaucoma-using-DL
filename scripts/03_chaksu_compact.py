"""Stage 1b: extract Chaksu from its two zips WITHOUT the huge uncompressed TIFF masks.
Every binary TIFF mask is converted on the fly to a compressed PNG (same pixels, ~100x smaller).
Non-TIFF files (original JPGs, CSV/XLS labels, PDFs) are extracted as they are.
Multi-level TIFFs (OD+OC label maps, colour images) are saved as lossless PNG with identical pixel values.
Writes logs/chaksu_compact.json and one sample per folder type into logs/samples/."""
import io, json, os, shutil, sys, zipfile, collections, datetime
import numpy as np
from tqdm import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "raw", "_archives")
OUT = os.path.join(ROOT, "raw", "Chaksu")
SAMP = os.path.join(ROOT, "logs", "samples")
os.makedirs(OUT, exist_ok=True); os.makedirs(SAMP, exist_ok=True)
LOG = os.path.join(ROOT, "logs", "chaksu_compact.json")
stats = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "zips": {}}
sampled = set()


def folder_key(name):
    parts = name.replace("\\", "/").split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


for zname in ["Chaksu_Test.zip", "Chaksu_Train.zip"]:
    zp = os.path.join(ARCH, zname)
    if not os.path.exists(zp):
        stats["zips"][zname] = "missing"; continue
    c = collections.Counter(); per_folder = collections.Counter(); uniq_examples = {}
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and "__MACOSX" not in n and not n.endswith(".DS_Store")]
        print(f"{zname}: {len(names)} files")
        bar = tqdm(names, desc=zname, unit="file", dynamic_ncols=True)
        for i, n in enumerate(bar):
            if shutil.disk_usage(ROOT).free < 10 * 2**30:
                stats["stopped"] = "less than 10 GB free on D:"; print("STOP: disk nearly full"); break
            low = n.lower(); fk = folder_key(n)
            if low.endswith((".tif", ".tiff")):
                dst = os.path.join(OUT, os.path.splitext(n)[0] + ".png")
                if os.path.exists(dst):
                    c["tif_already_done"] += 1; continue
                try:
                    a = np.array(Image.open(io.BytesIO(z.read(n))))
                    if a.ndim == 3:
                        g = a[..., :3].max(axis=2)
                    else:
                        g = a
                    vals = np.unique(g)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if len(vals) <= 2:
                        # binary mask -> clean 0/255 PNG
                        m = (g > (vals.max() / 2)).astype(np.uint8) * 255 if len(vals) > 1 else g.astype(np.uint8)
                        Image.fromarray(m).save(dst, optimize=True)
                        c["tif_to_png_mask"] += 1; per_folder[fk + " [binary mask->png]"] += 1
                    else:
                        # label map / colour image -> PNG with EXACTLY the same pixel values (lossless)
                        Image.fromarray(a).save(dst, optimize=True)
                        c["tif_to_png_exact"] += 1; per_folder[fk + " [multi-level tif->png lossless]"] += 1
                    uniq_examples.setdefault(fk, [str(a.shape), str(a.dtype), f"{len(vals)} unique values: {[int(v) for v in vals[:8]]}"])
                    if fk not in sampled:
                        sampled.add(fk)
                        im = Image.fromarray(a[..., :3] if a.ndim == 3 else a); im.thumbnail((800, 800))
                        im.convert("RGB").save(os.path.join(SAMP, fk.replace("/", "__") + "_" + os.path.basename(n) + ".jpg"), quality=85)
                except Exception as e:
                    c["tif_error"] += 1; uniq_examples.setdefault("errors", []).append(f"{n}: {e}")
            else:
                dst = os.path.join(OUT, n)
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with z.open(n) as s, open(dst, "wb") as d:
                        shutil.copyfileobj(s, d)
                c["copied_as_is"] += 1; per_folder[fk + " [copied]"] += 1
                if fk not in sampled and low.endswith((".jpg", ".jpeg", ".png")):
                    sampled.add(fk)
                    im = Image.open(dst); im.thumbnail((800, 800)); im.convert("RGB").save(os.path.join(SAMP, fk.replace("/", "__") + "_" + os.path.basename(n) + ".jpg"), quality=85)
            if i % 200 == 0:
                bar.set_postfix(masks=c["tif_to_png_mask"], labelmaps=c["tif_to_png_exact"], copied=c["copied_as_is"], free_GB=round(shutil.disk_usage(ROOT).free/2**30, 1))
    stats["zips"][zname] = {"counts": dict(c), "per_folder": dict(per_folder), "examples": uniq_examples}
    json.dump(stats, open(LOG, "w"), indent=2)

size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
stats["chaksu_folder_gb"] = round(size / 2**30, 2)
stats["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
stats["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
json.dump(stats, open(LOG, "w"), indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != "zips"}, indent=2))
