"""Stage 6: learned PPA (parapapillary atrophy) - replaces the paper's blue-channel + fixed temporal mask + K-means.

Data: PALM (Fang et al., CC0) - 1,200 fundus photos with pixel masks of the optic disc and of retinal atrophy.
PPA label = atrophy regions that touch the optic disc (connected components intersecting a thin ring around the disc);
patchy atrophy elsewhere in the retina is ignored. Crops are framed exactly like our glaucoma crops (disc-centred,
side = 3 x disc diameter, 512 px) so the model transfers.
Model: U-Net (ImageNet ResNet-34), 2 sigmoid outputs (disc, PPA), BCE + Dice.
  run A 'supervised'  : PALM labels only
  run B 'meanteacher' : + mean-teacher consistency on our ~unlabelled glaucoma crops (Li et al. 2023 idea):
                        teacher = EMA of student, student sees photometrically perturbed copy, MSE on sigmoid maps
Evaluation: PALM test split (Dice disc / Dice PPA / PPA presence AUC); on glaucoma data (no PPA ground truth):
PPA features per image, distribution by diagnosis, Mann-Whitney test, AUC of PPA alone, agreement with classical PPA.
Outputs: results/tables/ppa_*.csv, results/figures/ppa_*.jpg|png, logs/stage6_stats.json, models/ppa_*.pt
"""
import os, re, json, glob, math, time, argparse, random, datetime, copy
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALM = os.path.join(ROOT, "raw", "PALM"); PC = os.path.join(ROOT, "processed", "palm_crop512")
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures"); MOD = os.path.join(ROOT, "models")
for d in [PC, PC + "_mask", TAB, FIG, MOD]:
    os.makedirs(d, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def imwrite(p, a):
    cv2.imencode(os.path.splitext(p)[1], a)[1].tofile(p)


def binmask(m):
    """PALM-style masks may encode the structure as black on white; take the minority value as foreground."""
    lo = m < 128
    return lo if lo.mean() < 0.5 else ~lo


def split_of(path):
    p = path.lower()
    return "test" if "test" in p else "val" if "valid" in p else "train"


# ----------------------------------------------------------------------------- PALM preparation
def prepare_palm():
    imgs, disc, atro, atro_dirs = {}, {}, {}, set()
    for dp, _, fs in os.walk(PALM):
        low = dp.lower()
        for f in fs:
            stem, ext = os.path.splitext(f); ext = ext.lower(); p = os.path.join(dp, f)
            if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
                continue
            if "atroph" in low:
                atro[(split_of(p), stem.lower())] = p; atro_dirs.add(split_of(p))
            elif "disc" in low or "disk" in low:
                disc[(split_of(p), stem.lower())] = p
            elif "detach" in low or "mask" in low or "lesion" in low:
                continue
            elif ext in (".jpg", ".jpeg"):
                imgs[(split_of(p), stem.lower())] = p
    rows, notes = [], {"images": len(imgs), "disc_masks": len(disc), "atrophy_masks": len(atro), "splits_with_atrophy_folder": sorted(atro_dirs)}
    for key, ip in tqdm(sorted(imgs.items()), desc="Preparing PALM crops", unit="img", dynamic_ncols=True):
        sp, stem = key
        if key not in disc or sp not in atro_dirs:
            continue
        img = imread(ip); h, w = img.shape[:2]
        dm = imread(disc[key], cv2.IMREAD_GRAYSCALE); dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        d = binmask(dm)
        if d.sum() < 200:
            continue
        if key in atro:
            am = cv2.resize(imread(atro[key], cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST); a = binmask(am)
        else:
            a = np.zeros((h, w), bool)
        ys, xs = np.where(d); cy, cx = int(ys.mean()), int(xs.mean()); diam = max(np.ptp(ys), np.ptp(xs)) + 1
        ring = cv2.dilate(d.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.3 * diam) | 1,) * 2)) > 0
        n, lab = cv2.connectedComponents((a & ~d).astype(np.uint8))
        touch = np.unique(lab[ring & (lab > 0)])
        ppa = np.isin(lab, touch[touch > 0])
        side = 3 * diam; pad = side
        m = np.zeros((h, w), np.uint8); m[d] = 128; m[ppa] = 255
        ip_ = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT); mp_ = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        y1, x1 = cy + pad - side // 2, cx + pad - side // 2
        ci = cv2.resize(ip_[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(mp_[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_NEAREST)
        uid = f"palm_{sp}_{stem}"
        imwrite(os.path.join(PC, uid + ".png"), ci); imwrite(os.path.join(PC + "_mask", uid + ".png"), cm)
        rows.append(dict(image_id=uid, split=sp, crop512=os.path.relpath(os.path.join(PC, uid + ".png"), ROOT),
                         crop512_mask=os.path.relpath(os.path.join(PC + "_mask", uid + ".png"), ROOT),
                         has_ppa=bool((cm == 255).sum() > 50), ppa_px=int((cm == 255).sum()), disc_px=int((cm == 128).sum()),
                         any_atrophy=bool(a.any())))
    df = pd.DataFrame(rows)
    if len(df) and (df.split == "test").sum() == 0:            # carve a test split if PALM's test masks are absent
        tr = df[df.split == "train"].sample(frac=0.2, random_state=0).index; df.loc[tr, "split"] = "test"
    if len(df) and (df.split == "val").sum() == 0:
        tr = df[df.split == "train"].sample(frac=0.15, random_state=1).index; df.loc[tr, "split"] = "val"
    df.to_csv(os.path.join(ROOT, "processed", "palm_master.csv"), index=False)
    notes.update(prepared=len(df), by_split=df.split.value_counts().to_dict() if len(df) else {},
                 ppa_prevalence=df.groupby("split").has_ppa.mean().round(3).to_dict() if len(df) else {})
    print("PALM:", json.dumps(notes, indent=1))
    if len(df) < 100:
        raise SystemExit("PALM structure not recognised - see logs (the reader needs adapting)")
    return df, notes


# ----------------------------------------------------------------------------- data
def to_tensor(img):
    return torch.from_numpy(((img[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


class LabSet(Dataset):
    def __init__(s, df, train):
        s.df = df.reset_index(drop=True)
        s.aug = A.Compose([A.HorizontalFlip(p=.5), A.Affine(scale=(.85, 1.15), rotate=(-20, 20), translate_percent=(-.06, .06), p=.8),
                           A.RandomBrightnessContrast(.25, .25, p=.6), A.HueSaturationValue(8, 15, 10, p=.4), A.RandomGamma(p=.3)]) if train else None
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]; im = imread(os.path.join(ROOT, r.crop512)); m = imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
        if s.aug:
            o = s.aug(image=im, mask=m); im, m = o["image"], o["mask"]
        y = np.stack([m >= 128, m >= 255]).astype(np.float32)
        y[0] = (m == 128) | (m == 129)                           # disc only (PPA lies outside the disc)
        return to_tensor(im), torch.from_numpy(y)


class UnlabSet(Dataset):
    def __init__(s, paths): s.p = paths; s.aug = A.Compose([A.HorizontalFlip(p=.5), A.Affine(rotate=(-15, 15), p=.5)])
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        im = s.aug(image=imread(os.path.join(ROOT, s.p[i])))["image"]; return to_tensor(im)


def dice_loss(lo, y, eps=1.):
    p = torch.sigmoid(lo); return 1 - ((2 * (p * y).sum((0, 2, 3)) + eps) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + eps)).mean()


def photometric(x):
    b = x.shape[0]
    gain = 1 + 0.2 * (torch.rand(b, 1, 1, 1, device=x.device) - .5) * 2
    bias = 0.2 * (torch.rand(b, 3, 1, 1, device=x.device) - .5) * 2
    return x * gain + bias + 0.05 * torch.randn_like(x)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def train(df, unl_paths, mode, args, dev):
    ck0 = os.path.join(MOD, f"ppa_{mode}.pt"); hist0 = os.path.join(TAB, f"train_history_ppa_{mode}.csv")
    if args.reuse and os.path.exists(ck0) and os.path.exists(hist0) and not args.smoke:
        model = smp.Unet(args.encoder, encoder_weights=None, classes=2).to(dev)
        model.load_state_dict(torch.load(ck0, map_location=dev)); model.eval()
        h = pd.read_csv(hist0); best = float(((h.val_dice_disc + h.val_dice_ppa) / 2).max())
        if len(h) < 5:      # a 1-epoch smoke-test model must never be re-used as a real result
            print(f"[PPA {mode}] found only a {len(h)}-epoch (smoke-test) model - training properly", flush=True)
            return train(df, unl_paths, mode, argparse.Namespace(**{**vars(args), "reuse": 0}), dev)
        print(f"[PPA {mode}] re-using finished model {ck0} (best val {best:.4f})", flush=True)
        return model, best
    model = smp.Unet(args.encoder, encoder_weights="imagenet", classes=2).to(dev)
    teacher = copy.deepcopy(model).eval() if mode == "meanteacher" else None
    if teacher is not None:
        for p in teacher.parameters():
            p.requires_grad_(False)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dl = DataLoader(LabSet(tr, True), batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    dv = DataLoader(LabSet(va, False), batch_size=args.bs, num_workers=args.workers, persistent_workers=args.workers > 0)
    # one persistent loader with few workers (Windows: every worker process loads torch -> paging-file limits)
    du = cycle(DataLoader(UnlabSet(unl_paths), batch_size=args.bs, shuffle=True, num_workers=min(1, args.workers), drop_last=True,
                          persistent_workers=args.workers > 0)) if teacher is not None and len(unl_paths) >= args.bs else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); bce = nn.BCEWithLogitsLoss()
    best, ck, hist, pat = -1, os.path.join(MOD, f"ppa_{mode}.pt"), [], 0
    for ep in range(1, args.epochs + 1):
        model.train(); w_cons = args.cons_w * min(1.0, ep / max(1, 0.3 * args.epochs)); tl = n = 0
        for it, (x, y) in enumerate(tqdm(dl, desc=f"[PPA {mode}] epoch {ep}/{args.epochs}", unit="batch", leave=False, dynamic_ncols=True)):
            if args.smoke and it > 10:
                break
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = model(x); loss = bce(out, y) + dice_loss(out.float(), y)
                if du is not None:
                    xu = next(du).to(dev)
                    with torch.no_grad():
                        tp = torch.sigmoid(teacher(xu).float())
                    sp = torch.sigmoid(model(photometric(xu)).float())
                    loss = loss + w_cons * ((sp - tp) ** 2).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            if teacher is not None:
                with torch.no_grad():
                    for tp_, sp_ in zip(teacher.parameters(), model.parameters()):
                        tp_.mul_(args.ema).add_(sp_.detach(), alpha=1 - args.ema)
                    for tb, sb in zip(teacher.buffers(), model.buffers()):
                        tb.copy_(sb)
            tl += loss.item() * len(x); n += len(x)
        ev = teacher if teacher is not None else model
        ev.eval(); inter = np.zeros(2); den = np.zeros(2)
        with torch.no_grad():
            for it, (x, y) in enumerate(dv):
                if args.smoke and it > 3:
                    break
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    p = (ev(x.to(dev)).float() > 0).float().cpu()
                inter += (p * y).sum((0, 2, 3)).numpy(); den += (p.sum((0, 2, 3)) + y.sum((0, 2, 3))).numpy()
        vd = 2 * inter / np.maximum(den, 1); score = vd.mean()
        hist.append(dict(epoch=ep, loss=tl / max(1, n), val_dice_disc=vd[0], val_dice_ppa=vd[1], cons_weight=w_cons))
        flag = ""
        if score > best:
            best, pat = score, 0; torch.save(ev.state_dict(), ck); flag = "  * best"
        else:
            pat += 1
        print(f"[PPA {mode}] epoch {ep}/{args.epochs} loss {tl / max(1, n):.4f}  val Dice disc {vd[0]:.4f}  PPA {vd[1]:.4f}{flag}", flush=True)
        if pat >= args.patience:
            print("early stop"); break
    pd.DataFrame(hist).to_csv(os.path.join(TAB, f"train_history_ppa_{mode}.csv"), index=False)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    return model, float(best)


@torch.no_grad()
def predict(model, img, dev):
    x = to_tensor(img)[None].to(dev)
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float()).flip(3)) / 2
    return p[0].cpu().numpy()


def ppa_features(p, temporal_right):
    disc = p[0] > .5; ppa = (p[1] > .5) & ~disc
    dpx = disc.sum(); f = dict(ppa_px=int(ppa.sum()), ppa_to_disc=float(ppa.sum() / dpx) if dpx else np.nan, ppa_maxprob=float(p[1].max()))
    if dpx and ppa.any():
        ys, xs = np.where(disc); cy, cx = ys.mean(), xs.mean()
        py, px = np.where(ppa); ang = (np.degrees(np.arctan2(py - cy, px - cx)) + 360) % 360
        f["ppa_angle_deg"] = float(len(np.unique((ang // 10).astype(int))) * 10)
        temporal = (np.cos(np.radians(ang)) > 0) if temporal_right else (np.cos(np.radians(ang)) < 0)
        f["ppa_temporal_frac"] = float(temporal.mean())
    else:
        f["ppa_angle_deg"] = 0.0; f["ppa_temporal_frac"] = np.nan
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--encoder", default="resnet34"); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cons_w", type=float, default=1.0); ap.add_argument("--ema", type=float, default=0.99); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--feature_model", default="auto", help="auto (best PALM PPA Dice) | supervised | meanteacher")
    ap.add_argument("--reuse", type=int, default=1, help="re-use an already finished model of a variant instead of retraining")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    pm, notes = prepare_palm()
    master = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); master = master[master.status == "ok"]
    unl = master[master.split == "train"].crop512.tolist()
    results, models = {}, {}
    for mode in ["supervised", "meanteacher"]:
        m, best = train(pm, unl, mode, args, dev); models[mode] = m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        te = pm[pm.split == "test"]; rows = []
        for _, r in tqdm(te.iterrows(), total=len(te), desc=f"PALM test ({mode})", unit="img", dynamic_ncols=True):
            p = predict(m, imread(os.path.join(ROOT, r.crop512)), dev); g = imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
            pdisc, pppa = p[0] > .5, (p[1] > .5) & ~(p[0] > .5); gdisc, gppa = g == 128, g == 255
            dd = 2 * (pdisc & gdisc).sum() / max(1, pdisc.sum() + gdisc.sum())
            dp = 2 * (pppa & gppa).sum() / max(1, pppa.sum() + gppa.sum()) if gppa.sum() > 50 else np.nan
            rows.append(dict(image_id=r.image_id, dice_disc=dd, dice_ppa_if_present=dp, gt_has_ppa=r.has_ppa, pred_ppa_px=int(pppa.sum()), maxprob=float(p[1].max())))
        t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, f"ppa_palm_test_{mode}.csv"), index=False)
        from sklearn.metrics import roc_auc_score
        results[mode] = dict(best_val=best, palm_test_images=len(t), dice_disc=float(t.dice_disc.mean()), dice_ppa=float(t.dice_ppa_if_present.mean()),
                             ppa_presence_auc=float(roc_auc_score(t.gt_has_ppa, t.pred_ppa_px)) if t.gt_has_ppa.nunique() == 2 else np.nan)
        print(mode, results[mode], flush=True)
    best_mode = max(results, key=lambda k: results[k]["dice_ppa"] if results[k]["dice_ppa"] == results[k]["dice_ppa"] else -1)
    if args.feature_model != "auto":
        best_mode = args.feature_model
    sfx = "" if args.feature_model == "auto" else f"_{best_mode}"
    model = models[best_mode]
    # ---- apply to every glaucoma image
    feats = []
    for _, r in tqdm(master.iterrows(), total=len(master), desc=f"PPA on glaucoma images ({best_mode})", unit="img", dynamic_ncols=True):
        p = predict(model, imread(os.path.join(ROOT, r.crop512)), dev)
        feats.append(dict(image_id=r.image_id, **ppa_features(p, temporal_right=r.disc_cx < r.img_w / 2)))
    F = pd.DataFrame(feats).merge(master[["image_id", "dataset", "label", "split"]], on="image_id")
    from sklearn.cluster import KMeans
    trv = F[F.split == "train"].ppa_to_disc.fillna(0).values.reshape(-1, 1)
    km = KMeans(2, n_init=10, random_state=0).fit(trv); thr = float(np.sort(km.cluster_centers_.ravel()).mean())
    F["ppa_present"] = F.ppa_to_disc.fillna(0) > thr
    try:
        cl = pd.read_csv(os.path.join(TAB, "baseline_per_image.csv"))[["image_id", "ppa_area_px500"]]; F = F.merge(cl, on="image_id", how="left")
    except Exception:
        pass
    F.to_csv(os.path.join(TAB, f"ppa_features_all{sfx}.csv"), index=False)
    # ---- association with diagnosis
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    lab = F[F.label.isin(["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT", "NORMAL", "HEALTHY"])].copy(); lab["y"] = lab.label.isin(["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT"]).astype(int)
    assoc = []
    for ds, g in list(lab.groupby("dataset")) + [("Pooled", lab)]:
        if g.y.nunique() < 2:
            continue
        a, b = g[g.y == 1].ppa_to_disc.fillna(0), g[g.y == 0].ppa_to_disc.fillna(0)
        assoc.append(dict(dataset=ds, n_pos=len(a), n_neg=len(b), ppa_ratio_mean_pos=a.mean(), ppa_ratio_mean_neg=b.mean(),
                          ppa_present_rate_pos=g[g.y == 1].ppa_present.mean(), ppa_present_rate_neg=g[g.y == 0].ppa_present.mean(),
                          mannwhitney_p=mannwhitneyu(a, b).pvalue, auc_ppa_ratio_alone=roc_auc_score(g.y, g.ppa_to_disc.fillna(0)),
                          corr_with_classical_ppa=g[["ppa_to_disc", "ppa_area_px500"]].corr(method="spearman").iloc[0, 1] if "ppa_area_px500" in g else np.nan))
    assoc = pd.DataFrame(assoc); assoc.to_csv(os.path.join(TAB, f"ppa_association{sfx}.csv"), index=False); print(assoc.to_string())
    pd.DataFrame([dict(mode=k, **v) for k, v in results.items()]).to_csv(os.path.join(TAB, f"ppa_palm_results{sfx}.csv"), index=False)
    # ---- figures
    for name, rows, src in [("ppa_palm_test", pm[pm.split == "test"].sort_values("ppa_px", ascending=False).head(8), "palm"),
                            ("ppa_glaucoma_examples", F.sort_values("ppa_to_disc", ascending=False).head(8).merge(master[["image_id", "crop512"]], on="image_id"), "glau")]:
        tiles = []
        for _, r in rows.iterrows():
            im = imread(os.path.join(ROOT, r.crop512)); p = predict(model, im, dev); vis = im.copy()
            if src == "palm":
                g = imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
                cs, _ = cv2.findContours((g == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, cs, -1, (0, 255, 0), 2)
            for m, col in [(p[0] > .5, (255, 255, 0)), ((p[1] > .5) & ~(p[0] > .5), (0, 0, 255))]:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, cs, -1, col, 2)
            tiles.append(cv2.resize(vis, (256, 256)))
        while tiles and len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        if tiles:
            imwrite(os.path.join(FIG, f"{name}{sfx}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    json.dump(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), minutes=round((time.time() - t0) / 60, 1), palm=notes,
                   results=results, chosen_model=best_mode, ppa_present_threshold_ppa_to_disc=thr, association=assoc.to_dict("records")),
              open(os.path.join(ROOT, "logs", f"stage6_stats{sfx}.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
