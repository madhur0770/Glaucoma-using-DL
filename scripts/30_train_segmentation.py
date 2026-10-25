"""Stage 4: deep-learning replacement of the paper's OD localisation + OD/OC segmentation.

Model A (localiser): U-Net on the whole image (full512)  -> disc mask -> disc centre & size  (replaces 1-D projections)
Model B (segmenter): U-Net on the disc-centred crop (crop512) -> 2 sigmoid channels: disc, cup
                     (replaces K-means + top-hat + polar transform + strip binarisation + ellipse fitting)
Both: ImageNet-pretrained ResNet-34 encoder (segmentation_models_pytorch), BCE + Dice loss, AdamW, cosine LR,
mixed precision on the GPU, augmentation (flip, affine, colour), best checkpoint by validation Dice.
Evaluation on the held-out test split, per dataset: Dice, IoU, vCDR / area-CDR error vs experts, and an
end-to-end run (raw image -> localiser -> crop from the ORIGINAL image -> segmenter -> CDR).
Usage: python 30_train_segmentation.py [--smoke] [--epochs 40] [--encoder resnet34] [--tag base]
"""
import os, sys, json, time, math, argparse, datetime, random
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures"); MOD = os.path.join(ROOT, "models")
for d in [TAB, FIG, MOD]:
    os.makedirs(d, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def to_tensor(img_bgr):
    x = (img_bgr[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SegSet(Dataset):
    def __init__(self, df, kind, train):
        self.df = df.reset_index(drop=True); self.kind = kind; self.train = train
        self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                              A.Affine(scale=(0.85, 1.15), rotate=(-20, 20), translate_percent=(-0.06, 0.06), p=0.8),
                              A.RandomBrightnessContrast(0.25, 0.25, p=0.6), A.HueSaturationValue(8, 15, 10, p=0.4),
                              A.RandomGamma(p=0.3), A.GaussianBlur(blur_limit=(3, 5), p=0.1)]) if train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = imread(os.path.join(ROOT, r[self.kind])); m = imread(os.path.join(ROOT, r[self.kind + "_mask"]), cv2.IMREAD_GRAYSCALE)
        if self.aug is not None:
            o = self.aug(image=img, mask=m); img, m = o["image"], o["mask"]
        y = np.stack([m >= 128, m >= 255]).astype(np.float32)          # disc (incl. cup), cup
        if self.kind == "full512":
            y = y[:1]
        return to_tensor(img), torch.from_numpy(y)


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum((2, 3)) + eps; den = p.sum((2, 3)) + y.sum((2, 3)) + eps
    return 1 - (num / den).mean()


def batch_dice(logits, y):
    p = (logits > 0).float()
    inter = (p * y).sum((2, 3)); s = p.sum((2, 3)) + y.sum((2, 3))
    d = torch.where(s > 0, 2 * inter / s.clamp(min=1e-6), torch.ones_like(s))
    return d.mean(0)                                                    # per channel


def build(encoder, classes, pretrained=True):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=classes)


def train_model(kind, classes, tr, va, args, dev, log):
    model = build(args.encoder, classes).to(dev)
    dl_tr = DataLoader(SegSet(tr, kind, True), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(SegSet(va, kind, False), batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl_tr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(1, steps), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    bce = nn.BCEWithLogitsLoss()
    best, best_ep, hist, patience = -1, -1, [], 0
    ckpt = os.path.join(MOD, f"seg_{kind}_{args.tag}.pt")
    for ep in range(1, args.epochs + 1):
        model.train(); tl, n = 0, 0
        bar = tqdm(dl_tr, desc=f"[{kind}] epoch {ep}/{args.epochs} train", unit="batch", dynamic_ncols=True, leave=False)
        for it, (x, y) in enumerate(bar):
            if args.smoke and it >= 15:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = model(x); loss = bce(out, y) + dice_loss(out.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(x); n += len(x); bar.set_postfix(loss=f"{tl / n:.4f}")
        model.eval(); ds = []
        with torch.no_grad():
            for it, (x, y) in enumerate(dl_va):
                if args.smoke and it >= 5:
                    break
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    out = model(x.to(dev))
                ds.append(batch_dice(out.float(), y.to(dev)).cpu().numpy())
        vd = np.mean(ds, 0); score = float(vd.mean())
        hist.append(dict(epoch=ep, train_loss=tl / max(1, n), val_dice_disc=float(vd[0]), val_dice_cup=float(vd[1]) if len(vd) > 1 else np.nan, lr=sched.get_last_lr()[0]))
        flag = ""
        if score > best:
            best, best_ep, patience = score, ep, 0; torch.save(model.state_dict(), ckpt); flag = "  * best, saved"
        else:
            patience += 1
        msg = f"[{kind}] epoch {ep}/{args.epochs}  loss {tl / max(1, n):.4f}  val Dice disc {vd[0]:.4f}" + (f" cup {vd[1]:.4f}" if len(vd) > 1 else "") + flag
        print(msg, flush=True); log.append(msg)
        if patience >= args.patience:
            print(f"[{kind}] early stop (no improvement for {args.patience} epochs)"); break
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    pd.DataFrame(hist).to_csv(os.path.join(TAB, f"train_history_{kind}_{args.tag}.csv"), index=False)
    return model, dict(best_val=best, best_epoch=best_ep, epochs_run=len(hist), checkpoint=os.path.relpath(ckpt, ROOT))


def masks_to_cdr(disc, cup):
    if disc.sum() == 0:
        return np.nan, np.nan, np.nan
    ys = np.where(disc.any(1))[0]; xs = np.where(disc.any(0))[0]
    dh, dw = np.ptp(ys) + 1, np.ptp(xs) + 1
    cy = np.where(cup.any(1))[0]; cx = np.where(cup.any(0))[0]
    ch = np.ptp(cy) + 1 if len(cy) else 0; cw = np.ptp(cx) + 1 if len(cx) else 0
    return ch / dh, cw / dw, cup.sum() / disc.sum()


def largest_cc(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return m.astype(bool)
    return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


@torch.no_grad()
def predict(model, img_bgr, dev, tta=True):
    x = to_tensor(img_bgr)[None].to(dev)
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        p = torch.sigmoid(model(x).float())
        if tta:
            p = (p + torch.sigmoid(model(torch.flip(x, [3])).float()).flip(3)) / 2
    return p[0].cpu().numpy()


def evaluate(loc, seg, df, dev, smoke):
    rows = []
    sub = df if not smoke else df.groupby("dataset").head(4)
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc="Predicting all images (train/val/test)", unit="img", dynamic_ncols=True):
        out = dict(image_id=r.image_id, dataset=r.dataset, split=r.split)
        # (1) segmentation on the expert-centred crop
        img = imread(os.path.join(ROOT, r.crop512)); gm = imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
        p = predict(seg, img, dev)
        pd_, pc = largest_cc(p[0] > 0.5), (p[1] > 0.5)
        pc = largest_cc(pc & pd_) if (pc & pd_).any() else pc & pd_
        gd, gc = gm >= 128, gm >= 255
        for nm, a, b in [("disc", pd_, gd), ("cup", pc, gc)]:
            s = a.sum() + b.sum(); out[f"dice_{nm}"] = 2 * (a & b).sum() / s if s else 1.0
            u = (a | b).sum(); out[f"iou_{nm}"] = (a & b).sum() / u if u else 1.0
        out["vcdr_pred"], out["hcdr_pred"], out["acdr_pred"] = masks_to_cdr(pd_, pc)
        # (2) end-to-end from the raw image
        full = imread(os.path.join(ROOT, r.full512))
        pl = largest_cc(predict(loc, full, dev)[0] > 0.5)
        gfull = imread(os.path.join(ROOT, r.full512_mask), cv2.IMREAD_GRAYSCALE) >= 128
        if pl.sum() < 10:
            out["e2e_ok"] = False; rows.append(out); continue
        ys, xs = np.where(pl); gys, gxs = np.where(gfull)
        dd = max(np.ptp(ys), np.ptp(xs)) + 1; gdd = max(np.ptp(gys), np.ptp(gxs)) + 1
        out["loc_err_in_disc_diam"] = float(np.hypot(ys.mean() - gys.mean(), xs.mean() - gxs.mean()) / gdd)
        s = 512 / max(r.img_w, r.img_h); x0 = (512 - round(r.img_w * s)) // 2; y0 = (512 - round(r.img_h * s)) // 2
        cx, cy = (xs.mean() - x0) / s, (ys.mean() - y0) / s; side = int(3 * dd / s)
        orig = imread(r.img_path) if os.path.isabs(str(r.img_path)) else imread(os.path.join(ROOT, r.img_path))
        pad = side
        op = cv2.copyMakeBorder(orig, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        y1, x1 = int(cy) + pad - side // 2, int(cx) + pad - side // 2
        crop = cv2.resize(op[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_AREA)
        q = predict(seg, crop, dev)
        qd = largest_cc(q[0] > 0.5); qc = q[1] > 0.5
        qc = largest_cc(qc & qd) if (qc & qd).any() else qc & qd
        out["e2e_ok"] = True
        out["vcdr_e2e"], out["hcdr_e2e"], out["acdr_e2e"] = masks_to_cdr(qd, qc)
        # soft (probability-weighted) area CDR: not quantised by the 0.5 threshold -> more precise near a cut-off
        out["acdr_e2e_soft"] = float((q[1] * q[0]).sum() / max(q[0].sum(), 1e-6))
        out["loc_hit"] = out["loc_err_in_disc_diam"] < 0.5
        rows.append(out)
    return pd.DataFrame(rows)


def overlay_sheet(seg, df, dev, name):
    tiles = []
    for _, r in df.iterrows():
        img = imread(os.path.join(ROOT, r.crop512)); gm = imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
        p = predict(seg, img, dev); vis = img.copy()
        for m, col in [(gm >= 128, (0, 255, 0)), (gm >= 255, (0, 255, 0)), (p[0] > .5, (255, 255, 0)), ((p[1] > .5) & (p[0] > .5), (255, 0, 255))]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, cs, -1, col, 2)
        t = cv2.resize(vis, (256, 256)); cv2.putText(t, r.dataset[:14], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))[1].tofile(os.path.join(FIG, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--encoder", default="resnet34"); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10); ap.add_argument("--tag", default="base")
    ap.add_argument("--loc_from", default="", help="re-use the localiser of this tag instead of training one (saves ~half the time)")
    ap.add_argument("--exclude_dataset", default="", help="leave this dataset out of train/val entirely (still predicted) - gives unbiased predictions on ALL its images")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1; args.tag = "smoke"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''} | torch {torch.__version__} | smp {smp.__version__}")
    df = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); df = df[df.status == "ok"]
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    if args.exclude_dataset:
        tr, va = tr[tr.dataset != args.exclude_dataset], va[va.dataset != args.exclude_dataset]
        print(f"Leaving {args.exclude_dataset} out of training: the model never sees any of its images")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  | per dataset test: {te.dataset.value_counts().to_dict()}")
    log, t0 = [], time.time()
    seg, info_seg = train_model("crop512", 2, tr, va, args, dev, log)
    if args.loc_from:
        loc = build(args.encoder, 1, pretrained=False).to(dev)
        loc.load_state_dict(torch.load(os.path.join(MOD, f"seg_full512_{args.loc_from}.pt"), map_location=dev)); loc.eval()
        info_loc = dict(reused=f"seg_full512_{args.loc_from}.pt")
    else:
        loc, info_loc = train_model("full512", 1, tr, va, args, dev, log)
    ev_all = evaluate(loc, seg, df, dev, args.smoke)
    meta = df.set_index("image_id")[["vcdr", "hcdr", "acdr", "label"]]
    ev_all = ev_all.join(meta, on="image_id")
    ev_all.to_csv(os.path.join(TAB, f"dl_per_image_all_{args.tag}.csv"), index=False)
    ev = ev_all[ev_all.split == "test"]
    ev.to_csv(os.path.join(TAB, f"dl_per_image_test_{args.tag}.csv"), index=False)
    summ = []
    for ds, g in list(ev.groupby("dataset")) + [("ALL", ev)]:
        summ.append(dict(dataset=ds, images=len(g), dice_disc=g.dice_disc.mean(), dice_cup=g.dice_cup.mean(),
                         iou_disc=g.iou_disc.mean(), iou_cup=g.iou_cup.mean(),
                         vcdr_mae=(g.vcdr_pred - g.vcdr).abs().mean(), acdr_mae=(g.acdr_pred - g.acdr).abs().mean(),
                         vcdr_corr=g[["vcdr_pred", "vcdr"]].corr().iloc[0, 1] if len(g) > 2 else np.nan,
                         loc_hit_rate=g.loc_hit.mean() if "loc_hit" in g else np.nan,
                         vcdr_mae_end_to_end=(g.vcdr_e2e - g.vcdr).abs().mean() if "vcdr_e2e" in g else np.nan))
    summ = pd.DataFrame(summ); summ.to_csv(os.path.join(TAB, f"dl_segmentation_{args.tag}.csv"), index=False)
    print(summ.to_string())
    overlay_sheet(seg, te.groupby("dataset").head(4 if not args.smoke else 1), dev, f"dl_predictions_{args.tag}.jpg")
    # training curves
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for i, k in enumerate(["crop512", "full512"]):
            h = pd.read_csv(os.path.join(TAB, f"train_history_{k}_{args.tag}.csv"))
            ax[i].plot(h.epoch, h.val_dice_disc, label="val Dice disc")
            if h.val_dice_cup.notna().any():
                ax[i].plot(h.epoch, h.val_dice_cup, label="val Dice cup")
            ax2 = ax[i].twinx(); ax2.plot(h.epoch, h.train_loss, "k--", alpha=.4, label="train loss")
            ax[i].set_title({"crop512": "Disc/cup segmenter", "full512": "Disc localiser"}[k]); ax[i].set_xlabel("epoch"); ax[i].legend(loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"training_curves_{args.tag}.png"), dpi=150)
    except Exception as e:
        print("plot failed:", e)
    stats = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), minutes=round((time.time() - t0) / 60, 1),
                 args=vars(args), device=str(dev), segmenter=info_seg, localiser=info_loc,
                 n_train=len(tr), n_val=len(va), n_test=len(te), summary=summ.to_dict("records"), epoch_log=log)
    json.dump(stats, open(os.path.join(ROOT, "logs", f"stage4_stats_{args.tag}.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
