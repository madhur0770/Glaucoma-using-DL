"""Stage 12: stronger optic-cup segmentation (closes the gap to the literature on cup Dice / CDR correlation).

Why: every downstream number (CDR, the paper's 0.25 rule, rim, classification) depends on the cup boundary. In stage 4 the
cup is segmented on a 3-disc-diameter crop at 512 px, so the cup spans only ~70 px. Here:
  A  tight crops: 1.5 x disc diameter, cut from the ORIGINAL full-resolution photo -> the cup spans ~2x more pixels
  B  U-Net++ (Zhou et al. 2018, nested skip connections) with an ImageNet ResNet-34 encoder, BCE + Dice, centre-jitter
     augmentation (so it tolerates the automatic centre being a little off)
  C  fully automatic inference: stage-4 localiser -> stage-4 coarse segmenter (disc centre and diameter) -> tight crop from
     the original photo -> U-Net++ -> CDRs; ensemble = mean of stage-4 and stage-12 CDRs
Evaluation on the same test split as stage 4: Dice/IoU on expert-centred tight crops, vCDR/aCDR error and correlation of
the fully automatic CDRs, per dataset; side by side with stage 4.
Outputs: processed/tight512(+_mask), processed/tight_master.csv, models/seg_tight_unetpp.pt,
         results/tables/dl_segmentation_strong.csv, dl_per_image_all_strong.csv, figures/dl_predictions_strong.jpg,
         logs/stage12_stats.json
"""
import os, json, time, argparse, datetime, random, importlib.util
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures"); MOD = os.path.join(ROOT, "models")
TI = os.path.join(ROOT, "processed", "tight512")
for d in [TI, TI + "_mask"]:
    os.makedirs(d, exist_ok=True)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


B = _load("build10", "10_build_master.py"); S = _load("seg30", "30_train_segmentation.py")
TIGHT = 1.5


def tight_crop(img, cx, cy, diam, size=512, interp=cv2.INTER_AREA):
    side = max(16, int(TIGHT * diam)); pad = side
    p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    y1, x1 = int(cy) + pad - side // 2, int(cx) + pad - side // 2
    return cv2.resize(p[y1:y1 + side, x1:x1 + side], (size, size), interpolation=interp)


# ------------------------------------------------------------------ A: tight crops from the original photos
def make_one(rec):
    try:
        uid = rec["image_id"]; fi = os.path.join(TI, uid + ".png"); fm = os.path.join(TI + "_mask", uid + ".png")
        img = B.imread(rec["img_path"]); exp = B.expert_masks(rec, img.shape); n = len(exp)
        disc = np.sum([e[0] for e in exp], 0) >= n / 2; cup = (np.sum([e[1] for e in exp], 0) >= n / 2) & disc
        lab = np.zeros(disc.shape, np.uint8); lab[disc] = 128; lab[cup] = 255
        ys, xs = np.where(disc); diam = max(np.ptp(ys), np.ptp(xs)) + 1
        if not os.path.exists(fi):
            B.imwrite(fi, tight_crop(img, xs.mean(), ys.mean(), diam))
            B.imwrite(fm, tight_crop(lab, xs.mean(), ys.mean(), diam, interp=cv2.INTER_NEAREST))
        return dict(image_id=uid, tight512=os.path.relpath(fi, ROOT), tight512_mask=os.path.relpath(fm, ROOT))
    except Exception as e:
        return dict(image_id=rec.get("image_id"), error=f"{type(e).__name__}: {e}")


def build_tight(master, smoke):
    recs = []
    for fn in [B.discover_riga, B.discover_chaksu, B.discover_papila]:
        recs += fn()[0]
    keep = set(master.image_id); recs = [r for r in recs if r["image_id"] in keep]
    if smoke:
        recs = recs[:: max(1, len(recs) // 60)]
    with ThreadPoolExecutor(8) as ex:
        rows = list(tqdm(ex.map(make_one, recs), total=len(recs), desc="[A] tight crops from original photos", unit="img", dynamic_ncols=True))
    t = pd.DataFrame(rows); bad = t.error.notna().sum() if "error" in t else 0
    print(f"tight crops: {len(t)} ({bad} errors)", flush=True)
    t = t[t.get("error", pd.Series(index=t.index, dtype=object)).isna()] if "error" in t else t
    t.to_csv(os.path.join(ROOT, "processed", "tight_master.csv"), index=False)
    return master.merge(t[["image_id", "tight512", "tight512_mask"]], on="image_id")


# ------------------------------------------------------------------ B: U-Net++
class TightSet(Dataset):
    def __init__(s, df, train):
        s.df = df.reset_index(drop=True)
        s.aug = A.Compose([A.HorizontalFlip(p=.5), A.Affine(scale=(.85, 1.15), rotate=(-20, 20), translate_percent=(-.08, .08), p=.9),
                           A.RandomBrightnessContrast(.25, .25, p=.6), A.HueSaturationValue(8, 15, 10, p=.4), A.RandomGamma(p=.3),
                           A.GaussianBlur(blur_limit=(3, 7), p=.15), A.ImageCompression(quality_range=(40, 95), p=.2)]) if train else None

    def __len__(s): return len(s.df)

    def __getitem__(s, i):
        r = s.df.iloc[i]; img = S.imread(os.path.join(ROOT, r.tight512)); m = S.imread(os.path.join(ROOT, r.tight512_mask), cv2.IMREAD_GRAYSCALE)
        if s.aug is not None:
            o = s.aug(image=img, mask=m); img, m = o["image"], o["mask"]
        return S.to_tensor(img), torch.from_numpy(np.stack([m >= 128, m >= 255]).astype(np.float32))


def train(tr, va, a, dev):
    model = smp.UnetPlusPlus(encoder_name=a.encoder, encoder_weights="imagenet", classes=2).to(dev)
    dl = DataLoader(TightSet(tr, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    dv = DataLoader(TightSet(va, False), batch_size=a.bs, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, a.epochs * len(dl)), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); bce = nn.BCEWithLogitsLoss()
    ck = a.ckpt; best, pat, hist = -1, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); tl = 0
        for it, (x, y) in enumerate(tqdm(dl, desc=f"[B] U-Net++ epoch {ep}/{a.epochs}", unit="batch", leave=False, dynamic_ncols=True)):
            if a.smoke and it >= 10:
                break
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                o = model(x); loss = bce(o, y) + S.dice_loss(o.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        model.eval(); ds = []
        with torch.no_grad():
            for it, (x, y) in enumerate(dv):
                if a.smoke and it >= 3:
                    break
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    ds.append(S.batch_dice(model(x.to(dev)).float(), y.to(dev)).cpu().numpy())
        vd = np.mean(ds, 0); sc = float(vd.mean()); hist.append(dict(epoch=ep, loss=tl, val_dice_disc=float(vd[0]), val_dice_cup=float(vd[1])))
        flag = ""
        if sc > best:
            best, pat = sc, 0; torch.save(model.state_dict(), ck); flag = "  * best"
        else:
            pat += 1
        print(f"[B] U-Net++ epoch {ep}/{a.epochs}  val Dice disc {vd[0]:.4f} cup {vd[1]:.4f}{flag}", flush=True)
        if pat >= a.patience:
            print("[B] early stop", flush=True); break
    pd.DataFrame(hist).to_csv(os.path.join(TAB, f"train_history_tight_unetpp_{a.tag}.csv"), index=False)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval(); return model, best


# ------------------------------------------------------------------ C: evaluation + fully automatic CDR
def evaluate(model, loc, coarse, df, dev):
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="[C] evaluating (tight crops + end-to-end)", unit="img", dynamic_ncols=True):
        out = dict(image_id=r.image_id, dataset=r.dataset, split=r.split)
        img = S.imread(os.path.join(ROOT, r.tight512)); gm = S.imread(os.path.join(ROOT, r.tight512_mask), cv2.IMREAD_GRAYSCALE)
        p = S.predict(model, img, dev); pd_ = S.largest_cc(p[0] > .5); pc = (p[1] > .5) & pd_
        pc = S.largest_cc(pc) if pc.any() else pc
        for nm, a_, b_ in [("disc", pd_, gm >= 128), ("cup", pc, gm >= 255)]:
            s = a_.sum() + b_.sum(); out[f"dice_{nm}"] = 2 * (a_ & b_).sum() / s if s else 1.0
            u = (a_ | b_).sum(); out[f"iou_{nm}"] = (a_ & b_).sum() / u if u else 1.0
        out["vcdr_pred"], out["hcdr_pred"], out["acdr_pred"] = S.masks_to_cdr(pd_, pc)
        # end-to-end: localiser -> coarse disc on 3x crop -> tight crop from the original -> U-Net++
        try:
            full = S.imread(os.path.join(ROOT, r.full512)); pl = S.largest_cc(S.predict(loc, full, dev)[0] > .5)
            ys, xs = np.where(pl); s = 512 / max(r.img_w, r.img_h); x0 = (512 - round(r.img_w * s)) // 2; y0 = (512 - round(r.img_h * s)) // 2
            cx, cy = (xs.mean() - x0) / s, (ys.mean() - y0) / s; dd = (max(np.ptp(ys), np.ptp(xs)) + 1) / s
            orig = S.imread(r.img_path) if os.path.isabs(str(r.img_path)) else S.imread(os.path.join(ROOT, r.img_path))
            side = int(3 * dd); pad = side; op = cv2.copyMakeBorder(orig, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            y1, x1 = int(cy) + pad - side // 2, int(cx) + pad - side // 2
            c3 = cv2.resize(op[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_AREA)
            q = S.predict(coarse, c3, dev); qd = S.largest_cc(q[0] > .5)
            qy, qx = np.where(qd); k = side / 512
            cx2, cy2 = x1 - pad + qx.mean() * k, y1 - pad + qy.mean() * k; dd2 = (max(np.ptp(qy), np.ptp(qx)) + 1) * k
            t = S.predict(model, tight_crop(orig, cx2, cy2, dd2), dev); td = S.largest_cc(t[0] > .5); tc = (t[1] > .5) & td
            tc = S.largest_cc(tc) if tc.any() else tc
            out["vcdr_e2e"], out["hcdr_e2e"], out["acdr_e2e"] = S.masks_to_cdr(td, tc)
            out["acdr_e2e_soft"] = float((t[1] * t[0]).sum() / max(t[0].sum(), 1e-6)); out["e2e_ok"] = True
        except Exception as e:
            out["e2e_ok"] = False; out["e2e_error"] = str(e)[:120]
        rows.append(out)
    return pd.DataFrame(rows)


def finetune_br(a, d, m, dev, loc, coarse):
    """5-fold out-of-fold fine-tuning on BinRushed (as stage 7a2) of the BinRushed-free tight model"""
    from sklearn.model_selection import StratifiedKFold
    br = d[d.dataset == "RIGA-BinRushed"].reset_index(drop=True)
    if a.smoke:
        br = br.head(25)
    strat = (br.acdr > 0.25).astype(int); outs, folds = [], []
    for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(br, strat)):
        model = smp.UnetPlusPlus(encoder_name=a.encoder, encoder_weights=None, classes=2).to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev))
        dl = DataLoader(TightSet(br.iloc[tri], True), batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss(); scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        for ep in range(1, (1 if a.smoke else 15) + 1):
            model.train(); tl = 0
            for x, y in tqdm(dl, desc=f"[tight fine-tune fold {k + 1}/5] epoch {ep}", unit="batch", leave=False, dynamic_ncols=True):
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    o = model(x); loss = bce(o, y) + S.dice_loss(o.float(), y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item()
            print(f"[tight fine-tune fold {k + 1}/5] epoch {ep} loss {tl / max(1, len(dl)):.4f}", flush=True)
        model.eval(); ev = evaluate(model, loc, coarse, br.iloc[tei], dev); outs.append(ev)
        e = ev.merge(br[["image_id", "acdr"]], on="image_id"); e = e.acdr_e2e - e.acdr
        folds.append(dict(fold=k + 1, acdr_bias=float(e.mean()), acdr_mae=float(e.abs().mean()))); print(folds[-1], flush=True)
        del model; torch.cuda.empty_cache()
    ev = pd.concat(outs, ignore_index=True).join(m.set_index("image_id")[["vcdr", "hcdr", "acdr", "label"]], on="image_id")
    ev.to_csv(os.path.join(TAB, f"dl_per_image_all_{a.tag}_ft.csv"), index=False)
    st = dict(folds=folds, acdr_mae=float((ev.acdr_e2e - ev.acdr).abs().mean()), acdr_bias=float((ev.acdr_e2e - ev.acdr).mean()), dice_cup=float(ev.dice_cup.mean()))
    json.dump(st, open(os.path.join(ROOT, "logs", f"stage12_finetune_{a.tag}.json"), "w"), indent=2); print(st, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--patience", type=int, default=8); ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--tag", default="strong")
    ap.add_argument("--exclude_dataset", default="", help="leave this dataset out of training (e.g. RIGA-BinRushed for the paper protocol)")
    ap.add_argument("--loc_tag", default="base", help="stage-4 localiser / coarse segmenter to use (base or noBR)")
    ap.add_argument("--finetune_br", action="store_true", help="only: 5-fold out-of-fold BinRushed fine-tuning of an existing tight model")
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    a.ckpt = os.path.join(MOD, "seg_tight_unetpp.pt" if a.tag == "strong" else f"seg_tight_unetpp_{a.tag}.pt")
    if a.smoke:
        a.epochs = 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", dev, flush=True)
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[m.status == "ok"]
    d = build_tight(m, a.smoke)
    loc = S.build("resnet34", 1, False).to(dev); loc.load_state_dict(torch.load(os.path.join(MOD, f"seg_full512_{a.loc_tag}.pt"), map_location=dev)); loc.eval()
    coarse = S.build("resnet34", 2, False).to(dev); coarse.load_state_dict(torch.load(os.path.join(MOD, f"seg_crop512_{a.loc_tag}.pt"), map_location=dev)); coarse.eval()
    if a.finetune_br:
        finetune_br(a, d, m, dev, loc, coarse); return
    tr, va = d[d.split == "train"], d[d.split == "val"]
    if a.exclude_dataset:
        tr, va = tr[tr.dataset != a.exclude_dataset], va[va.dataset != a.exclude_dataset]; print(f"Leaving {a.exclude_dataset} out of training", flush=True)
    print(f"train {len(tr)} val {len(va)} test {(d.split == 'test').sum()}", flush=True)
    model, best = train(tr, va, a, dev)
    ev = evaluate(model, loc, coarse, d if not a.smoke else d.head(30), dev)
    ev = ev.join(m.set_index("image_id")[["vcdr", "hcdr", "acdr", "label"]], on="image_id")
    base = pd.read_csv(os.path.join(TAB, f"dl_per_image_all_{a.loc_tag}.csv")).set_index("image_id")
    for c in ["vcdr", "hcdr", "acdr"]:
        ev[f"{c}_ens"] = (ev[f"{c}_e2e"].values + base.reindex(ev.image_id)[f"{c}_e2e"].values) / 2
        ev[f"{c}_base"] = base.reindex(ev.image_id)[f"{c}_e2e"].values
    ev.to_csv(os.path.join(TAB, f"dl_per_image_all_{a.tag}.csv"), index=False)
    te = ev[ev.split == "test"]; rows = []
    bseg = pd.read_csv(os.path.join(TAB, f"dl_segmentation_{a.loc_tag}.csv")).set_index("dataset")
    for ds, g in list(te.groupby("dataset")) + [("ALL", te)]:
        r = dict(dataset=ds, images=len(g), dice_disc=g.dice_disc.mean(), dice_cup=g.dice_cup.mean(), iou_disc=g.iou_disc.mean(), iou_cup=g.iou_cup.mean(),
                 dice_cup_stage4=bseg.dice_cup.get(ds, np.nan), dice_disc_stage4=bseg.dice_disc.get(ds, np.nan))
        for k in ["e2e", "ens", "base"]:
            ok = g[[f"vcdr_{k}", "vcdr"]].dropna()
            r[f"vcdr_mae_{k}"] = (g[f"vcdr_{k}"] - g.vcdr).abs().mean(); r[f"acdr_mae_{k}"] = (g[f"acdr_{k}"] - g.acdr).abs().mean()
            r[f"vcdr_corr_{k}"] = ok.corr().iloc[0, 1] if len(ok) > 2 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, f"dl_segmentation_{a.tag}.csv"), index=False); print(t.round(4).to_string(index=False), flush=True)
    try:
        sub = d[d.split == "test"].groupby("dataset").head(3); tiles = []
        for _, r in sub.iterrows():
            img = S.imread(os.path.join(ROOT, r.tight512)); gm = S.imread(os.path.join(ROOT, r.tight512_mask), cv2.IMREAD_GRAYSCALE); p = S.predict(model, img, dev); vis = img.copy()
            for msk, col in [(gm >= 128, (0, 255, 0)), (gm >= 255, (0, 255, 0)), (p[0] > .5, (255, 255, 0)), ((p[1] > .5) & (p[0] > .5), (255, 0, 255))]:
                cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, cs, -1, col, 2)
            tiles.append(cv2.resize(vis, (256, 256)))
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))[1].tofile(os.path.join(FIG, f"dl_predictions_{a.tag}.jpg"))
    except Exception as e:
        print("figure skipped:", e)
    json.dump(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), minutes=round((time.time() - t0) / 60, 1), best_val=best, args={k: str(v) for k, v in vars(a).items()},
                   results=t.to_dict("records")), open(os.path.join(ROOT, "logs", f"stage12_stats_{a.tag}.json" if not a.smoke else "stage12_stats_smoke.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
