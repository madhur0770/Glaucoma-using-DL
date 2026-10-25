"""Stage 2: build the master dataset from RIGA (RIGA originals in raw/RIGA or $RIGA_ROOT + Segmentation-Ready-RIGA masks), Chaksu and PAPILA.

For every fundus image it produces
  * a consensus OD/OC mask (pixel kept if >= half of the experts marked it)
  * vertical, horizontal and area cup-to-disc ratios (from the consensus mask)
  * per-expert vCDR (-> inter-observer spread) and Dice of each expert vs consensus
  * processed/full512/<uid>.png  : whole image letter-boxed to 512x512  (+ processed/full512_mask)
  * processed/crop512/<uid>.png  : disc-centred crop (3 x disc diameter, room for PPA) at 512x512 (+ _mask)
    mask encoding: 0 = background, 128 = disc rim, 255 = cup
  * labels (clinical/expert) where the dataset has them, patient id, camera, eye
  * patient-level train/val/test split (fixed seed)
Outputs: processed/master.csv, splits/*.csv, results/qc/*.jpg, logs/stage2_stats.json
"""
import os, re, sys, json, glob, math, random, datetime, collections, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIGA_E = os.environ.get("RIGA_ROOT", os.path.join(ROOT, "raw", "RIGA"))
SEGREADY = os.path.join(ROOT, "raw", "SegReady_RIGA", "RIGA_preprocessed")
CHAKSU = os.path.join(ROOT, "raw", "Chaksu")
PAPILA = glob.glob(os.path.join(ROOT, "raw", "PAPILA", "PapilaDB*"))
PAPILA = PAPILA[0] if PAPILA else None
OUT = os.path.join(ROOT, "processed")
for d in ["full512", "full512_mask", "crop512", "crop512_mask"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
QC = os.path.join(ROOT, "results", "qc"); os.makedirs(QC, exist_ok=True)
os.makedirs(os.path.join(ROOT, "splits"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
SEED = 42
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def imread(p, flag=cv2.IMREAD_COLOR):
    a = np.fromfile(p, np.uint8)                      # works with non-ASCII / long Windows paths
    return cv2.imdecode(a, flag)


def imwrite(p, img):
    ok, buf = cv2.imencode(os.path.splitext(p)[1], img)
    buf.tofile(p)


def norm_label(v):
    """Chaksu labels contain typos/double spaces, e.g. 'GLAUCOMA  SUSUPECT'"""
    t = " ".join(str(v).upper().split())
    if "SUS" in t:
        return "GLAUCOMA SUSPECT"
    if "NORMAL" in t:
        return "NORMAL"
    if "GLAUCOMA" in t:
        return "GLAUCOMA"
    return ""


def norm_subset(name):
    n = name.lower().replace("-corrected", "").replace("magrabifemale", "magrabiafemale")
    return n


# ---------------------------------------------------------------- record discovery
def discover_riga():
    """originals: imageNprime.* in E:\\DataSet (BinRushedcorrected preferred); masks: SegReady imageN-k.png"""
    recs, notes = [], collections.Counter()
    masks = collections.defaultdict(list)
    for dp, _, fs in os.walk(SEGREADY):
        if "__MACOSX" in dp:
            continue
        for f in fs:
            m = re.match(r"(?i)image(\d+)-(\d)\.png$", f)
            if m:
                masks[(norm_subset(os.path.basename(dp)), int(m.group(1)))].append(os.path.join(dp, f))
    seen = set()
    roots = [os.path.join(RIGA_E, "BinRushedcorrected"), os.path.join(RIGA_E, "Magrabia"), os.path.join(RIGA_E, "MESSIDOR")]
    for r in roots:
        for dp, _, fs in os.walk(r):
            sub = norm_subset(os.path.basename(dp))
            names = set(f.lower() for f in fs)
            for f in fs:
                m = re.match(r"(?i)image(\d+)prime\.(jpg|jpeg|tif|tiff|png)$", f)
                m2 = re.match(r"(?i)image(\d+)\.(tif|jpg)$", f)          # Magrabia image48.tif has no 'prime'
                if m2 and not any(n.startswith(f"image{m2.group(1)}prime") for n in names):
                    m = m2; notes["original_without_prime_suffix"] += 1
                if not m:
                    continue
                key = (sub, int(m.group(1)))
                if key in seen:
                    continue
                seen.add(key)
                mk = sorted(masks.get(key, []))
                if not mk:
                    notes[f"no_segready_mask:{sub}"] += 1
                    continue
                ds = "RIGA-" + ("BinRushed" if sub.startswith("binrushed") else "Magrabia" if sub.startswith("magrabia") else "MESSIDOR")
                recs.append(dict(dataset=ds, subset=sub, image_id=f"{sub}_{key[1]}", patient_id=f"{sub}_{key[1]}",
                                 img_path=os.path.join(dp, f), mask_kind="segready_rgb", masks=mk, camera=ds.split("-")[1],
                                 eye="", label="", label_source="none"))
    for k in [k for k in masks if k not in seen]:
        notes[f"mask_without_image:{k[0]}"] += 1
    return recs, dict(notes)


def discover_chaksu():
    recs, notes = [], collections.Counter()
    for part in ["Train", "Test"]:
        base = os.path.join(CHAKSU, part)
        labels = {}
        for cam in ["Bosch", "Forus", "Remidio"]:
            lf = glob.glob(os.path.join(base, "6.0_Glaucoma_Decision", f"Glaucoma_Decision_Comparison_{cam}_majority.csv"))
            if lf:
                df = pd.read_csv(lf[0])
                # final decision column is called 'Majority Decision' in some files and 'Glaucoma Decision' in others
                cands = [c for c in df.columns if "majority" in c.lower() or "decision" in c.lower()]
                col = cands[0] if cands else df.columns[-1]
                for _, r in df.iterrows():
                    stem = re.split(r"\.", str(r.iloc[0]))[0]
                    labels[(cam, stem.lower())] = norm_label(r[col])
            for p in glob.glob(os.path.join(base, "1.0_Original_Fundus_Images", cam, "*")):
                if not p.lower().endswith(IMG_EXT):
                    continue
                stem = os.path.splitext(os.path.basename(p))[0]
                mk = {}
                for s in ["Disc", "Cup"]:
                    c = glob.glob(os.path.join(base, "5.0_OD_OC_Mean_Median_Majority_STAPLE", cam, s, "Majority", stem + ".*"))
                    mk[s] = c[0] if c else None
                if not (mk["Disc"] and mk["Cup"]):
                    notes[f"no_majority_mask:{part}/{cam}"] += 1; continue
                lab = labels.get((cam, stem.lower()), "")
                if not lab:
                    notes[f"no_label:{part}/{cam}"] += 1
                pm = re.match(r"(P\d+)_Image\d+", stem)
                pid = f"chaksu_{cam}_{pm.group(1) if pm else stem}"
                recs.append(dict(dataset="Chaksu", subset=f"{part}/{cam}", image_id=f"chaksu_{part}_{cam}_{stem}", patient_id=pid,
                                 img_path=p, mask_kind="chaksu_binary", masks=[mk["Disc"], mk["Cup"]], camera=cam, eye="",
                                 label=lab, label_source="5-expert majority (fundus grading)", official_split=part.lower()))
    return recs, dict(notes)


def discover_papila():
    recs, notes = [], collections.Counter()
    if not PAPILA:
        return recs, {"missing": 1}
    diag = {}
    for eye, fn in [("OD", "patient_data_od.xlsx"), ("OS", "patient_data_os.xlsx")]:
        x = pd.read_excel(os.path.join(PAPILA, "ClinicalData", fn), header=None)
        for _, r in x.iloc[3:].iterrows():
            if isinstance(r[0], str) and r[0].startswith("#"):
                diag[(r[0][1:].zfill(3), eye)] = r[3]
    names = {0: "HEALTHY", 1: "GLAUCOMA", 2: "SUSPECT"}
    for p in sorted(glob.glob(os.path.join(PAPILA, "FundusImages", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]          # RET002OD
        m = re.match(r"RET(\d+)(OD|OS)", stem)
        if not m:
            continue
        cont = {}
        for s in ["disc", "cup"]:
            cont[s] = sorted(glob.glob(os.path.join(PAPILA, "ExpertsSegmentations", "Contours", f"{stem}_{s}_exp*.txt")))
        if not cont["disc"] or not cont["cup"]:
            notes["no_contours"] += 1; continue
        d = diag.get((m.group(1).zfill(3), m.group(2)))
        lab = names.get(int(d), "") if d is not None and not (isinstance(d, float) and math.isnan(d)) else ""
        recs.append(dict(dataset="PAPILA", subset=m.group(2), image_id=f"papila_{stem}", patient_id=f"papila_{m.group(1)}",
                         img_path=p, mask_kind="papila_contours", masks=cont["disc"] + cont["cup"], camera="Topcon",
                         eye=m.group(2), label=lab, label_source="clinical diagnosis (PAPILA)"))
    return recs, dict(notes)


# ---------------------------------------------------------------- per-image processing
def expert_masks(rec, shape):
    """returns list of (disc, cup) boolean masks, one per expert, at image resolution"""
    h, w = shape[:2]
    out = []
    if rec["mask_kind"] == "segready_rgb":
        for mp in rec["masks"]:
            m = imread(mp, cv2.IMREAD_GRAYSCALE)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out.append((m >= 100, m >= 200))
    elif rec["mask_kind"] == "chaksu_binary":
        dm = imread(rec["masks"][0], cv2.IMREAD_GRAYSCALE); cm = imread(rec["masks"][1], cv2.IMREAD_GRAYSCALE)
        if (dm.shape[0] > dm.shape[1]) != (h > w):
            raise ValueError(f"orientation mismatch image {w}x{h} vs mask {dm.shape[1]}x{dm.shape[0]}")
        dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST); cm = cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((dm > 127, cm > 127))
    else:  # papila contours: x y per line
        n = len(rec["masks"]) // 2
        for i in range(n):
            ms = []
            for path in (rec["masks"][i], rec["masks"][n + i]):
                pts = np.loadtxt(path).reshape(-1, 2)
                mm = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mm, [np.round(pts).astype(np.int32)], 1)
                ms.append(mm.astype(bool))
            out.append(tuple(ms))
    return out


def cdr(disc, cup):
    if disc.sum() == 0:
        return dict(vcdr=np.nan, hcdr=np.nan, acdr=np.nan, disc_h=0, disc_w=0)
    ys, xs = np.where(disc); dh = ys.max() - ys.min() + 1; dw = xs.max() - xs.min() + 1
    if cup.sum():
        yc, xc = np.where(cup); ch = yc.max() - yc.min() + 1; cw = xc.max() - xc.min() + 1
    else:
        ch = cw = 0
    return dict(vcdr=ch / dh, hcdr=cw / dw, acdr=cup.sum() / disc.sum(), disc_h=int(dh), disc_w=int(dw))


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 1.0


def letterbox(img, size, interp):
    h, w = img.shape[:2]; s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
    canvas = np.zeros((size, size) + img.shape[2:], img.dtype)
    y0 = (size - r.shape[0]) // 2; x0 = (size - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas, s, x0, y0


def process(rec):
    try:
        img = imread(rec["img_path"])
        if img is None:
            return {**rec, "status": "unreadable"}
        h, w = img.shape[:2]
        exp = expert_masks(rec, img.shape)
        n = len(exp)
        disc = (np.sum([e[0] for e in exp], 0) >= n / 2)
        cup = (np.sum([e[1] for e in exp], 0) >= n / 2) & disc
        if disc.sum() < 100:
            return {**rec, "status": "empty_disc_mask"}
        c = cdr(disc, cup)
        per_v = [cdr(d, cc)["vcdr"] for d, cc in exp]
        per_dd = [dice(d, disc) for d, _ in exp]; per_dc = [dice(cc, cup) for _, cc in exp]
        lab = np.zeros((h, w), np.uint8); lab[disc] = 128; lab[cup] = 255
        uid = rec["image_id"]
        # full image 512
        fi, s, x0, y0 = letterbox(img, 512, cv2.INTER_AREA); fm, _, _, _ = letterbox(lab, 512, cv2.INTER_NEAREST)
        imwrite(os.path.join(OUT, "full512", uid + ".png"), fi); imwrite(os.path.join(OUT, "full512_mask", uid + ".png"), fm)
        # disc-centred crop, side = 3 x disc diameter
        ys, xs = np.where(disc); cy, cx = int(ys.mean()), int(xs.mean())
        side = int(3 * max(c["disc_h"], c["disc_w"])); half = side // 2
        pad = side
        ip = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        lp = cv2.copyMakeBorder(lab, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        y1, x1 = cy + pad - half, cx + pad - half
        ci = cv2.resize(ip[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(lp[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_NEAREST)
        imwrite(os.path.join(OUT, "crop512", uid + ".png"), ci); imwrite(os.path.join(OUT, "crop512_mask", uid + ".png"), cm)
        out = {k: v for k, v in rec.items() if k != "masks"}
        out.update(status="ok", n_experts=n, img_w=w, img_h=h, **c,
                   disc_cx=cx, disc_cy=cy, crop_side_px=side, disc_frac_of_width=round(c["disc_w"] / w, 4),
                   vcdr_expert_mean=float(np.nanmean(per_v)), vcdr_expert_sd=float(np.nanstd(per_v)) if n > 1 else np.nan,
                   vcdr_expert_min=float(np.nanmin(per_v)), vcdr_expert_max=float(np.nanmax(per_v)),
                   dice_disc_expert_vs_consensus=float(np.mean(per_dd)), dice_cup_expert_vs_consensus=float(np.mean(per_dc)),
                   full512=f"processed/full512/{uid}.png", full512_mask=f"processed/full512_mask/{uid}.png",
                   crop512=f"processed/crop512/{uid}.png", crop512_mask=f"processed/crop512_mask/{uid}.png")
        return out
    except Exception as e:
        return {**{k: v for k, v in rec.items() if k != "masks"}, "status": "error", "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- splits
def assign_splits(df):
    rng = random.Random(SEED)
    df["split"] = ""
    for ds, g in df[df.status == "ok"].groupby("dataset"):
        if ds == "Chaksu":
            test_p = set(g[g.official_split == "test"].patient_id)
            df.loc[g.index[g.patient_id.isin(test_p)], "split"] = "test"
            rest = g[~g.patient_id.isin(test_p)]
            fr = {"train": 0.85, "val": 0.15}
        else:
            rest = g; fr = {"train": 0.70, "val": 0.15, "test": 0.15}
        # stratify by the patient's worst label
        sev = {"GLAUCOMA": 3, "GLAUCOMA SUSPECT": 2, "SUSPECT": 2, "NORMAL": 1, "HEALTHY": 1, "": 0}
        pl = rest.groupby("patient_id").label.agg(lambda s: max(s, key=lambda x: sev.get(x, 0)))
        for _, pids in pl.groupby(pl):
            pids = sorted(pids.index); rng.shuffle(pids)
            n = len(pids); cut, start = {}, 0
            names = list(fr)
            for i, k in enumerate(names):
                m = n - start if i == len(names) - 1 else int(round(fr[k] * n))
                for p in pids[start:start + m]:
                    cut[p] = k
                start += m
            idx = rest.index[rest.patient_id.isin(cut)]
            df.loc[idx, "split"] = rest.loc[idx, "patient_id"].map(cut)
    return df


# ---------------------------------------------------------------- QC overlays
def qc_overlays(df, k=12):
    rng = random.Random(SEED)
    for ds, g in df[df.status == "ok"].groupby("dataset"):
        rows = g.sample(min(k, len(g)), random_state=SEED)
        tiles = []
        for _, r in rows.iterrows():
            im = imread(os.path.join(ROOT, r.crop512)); m = imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
            for val, col in [(128, (0, 255, 0)), (255, (255, 0, 255))]:
                cs, _ = cv2.findContours((m >= val).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, col, 2)
            t = cv2.resize(im, (256, 256))
            cv2.putText(t, f"vCDR {r.vcdr:.2f} {r.label[:8]}", (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        imwrite(os.path.join(QC, f"qc_{ds}.jpg"), grid)


def main():
    t0 = datetime.datetime.now()
    print("Discovering images and masks ...")
    recs, notes = [], {}
    for name, fn in [("RIGA", discover_riga), ("Chaksu", discover_chaksu), ("PAPILA", discover_papila)]:
        r, n = fn(); recs += r; notes[name] = n
        print(f"  {name}: {len(r)} images found; notes: {n}")
    results = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2)) as ex:
        futs = [ex.submit(process, r) for r in recs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Processing images", unit="img", dynamic_ncols=True):
            results.append(f.result())
    df = pd.DataFrame(results)
    df["official_split"] = df.get("official_split", "")
    df = assign_splits(df)
    df = df.sort_values(["dataset", "image_id"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "master.csv"), index=False)
    for s in ["train", "val", "test"]:
        df[df.split == s].to_csv(os.path.join(ROOT, "splits", f"{s}.csv"), index=False)
    print("Making QC overlay sheets ...")
    qc_overlays(df)
    ok = df[df.status == "ok"]
    stats = {"started": t0.isoformat(timespec="seconds"), "finished": datetime.datetime.now().isoformat(timespec="seconds"),
             "discovery_notes": notes, "status_counts": df.status.value_counts().to_dict(),
             "errors_sample": df[df.status != "ok"].get("error", pd.Series(dtype=str)).dropna().head(10).tolist(),
             "per_dataset": {}}
    for ds, g in ok.groupby("dataset"):
        stats["per_dataset"][ds] = {
            "images": int(len(g)), "patients": int(g.patient_id.nunique()), "experts": int(g.n_experts.max()),
            "labels": g.label.replace("", "none").value_counts().to_dict(),
            "splits": g.split.value_counts().to_dict(),
            "image_sizes": g.apply(lambda r: f"{r.img_w}x{r.img_h}", axis=1).value_counts().head(5).to_dict(),
            "vcdr_mean_sd": [round(g.vcdr.mean(), 3), round(g.vcdr.std(), 3)],
            "acdr_mean_sd": [round(g.acdr.mean(), 3), round(g.acdr.std(), 3)],
            "vcdr_by_label": g.groupby(g.label.replace("", "none")).vcdr.mean().round(3).to_dict(),
            "acdr_by_label": g.groupby(g.label.replace("", "none")).acdr.mean().round(3).to_dict(),
            "inter_expert_vcdr_sd_mean": round(float(g.vcdr_expert_sd.mean()), 3) if g.n_experts.max() > 1 else None,
            "dice_disc_expert_vs_consensus": round(float(g.dice_disc_expert_vs_consensus.mean()), 3),
            "dice_cup_expert_vs_consensus": round(float(g.dice_cup_expert_vs_consensus.mean()), 3),
            "acdr_vs_vcdr_corr": round(float(g[["acdr", "vcdr"]].corr().iloc[0, 1]), 3),
            "share_acdr_gt_0.25": round(float((g.acdr > 0.25).mean()), 3),
            "share_vcdr_gt_0.6": round(float((g.vcdr > 0.6).mean()), 3),
            "disc_width_frac_mean": round(float(g.disc_frac_of_width.mean()), 4)}
    json.dump(stats, open(os.path.join(ROOT, "logs", "stage2_stats.json"), "w"), indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
