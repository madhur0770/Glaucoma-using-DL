"""Stage 8: make PPA work on glaucoma images WITHOUT any hand labelling (fixes the PALM -> glaucoma domain gap).

Problem (stage 6): the PPA U-Net trained on PALM (myopia) reaches PPA Dice 0.85 on PALM but almost never fires on
Chaksu / PAPILA / RIGA crops, so learned PPA adds ~nothing to glaucoma classification.

Three fully automatic remedies, compared on the SAME train/val/test split as stage 5:
  A  FDA  - Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): before prediction, the low-frequency amplitude
            spectrum (colour / illumination 'style') of each glaucoma crop is replaced by that of a PALM crop; the
            image content (phase) is untouched. No retraining.
  B  SELF - self-training: model A's confident predictions on the unlabelled glaucoma TRAIN crops become
            pseudo-labels (uncertain pixels ignored); the PALM model is fine-tuned on PALM labels + pseudo-labels.
  C  RING - peripapillary-ring CNN: a ResNet-18 sees ONLY the ring 0.9-2.2 disc radii around the disc (the disc itself
            and the far retina are blacked out), trained on the glaucoma labels. It learns whatever PPA / peripapillary
            change predicts glaucoma, with no masks at all. (It can also pick up RNFL defects - stated honestly.)
Selection of the best variant uses the VALIDATION split only (no test peeking); all variants are reported on test.
Outputs: results/tables/ppa_adapt_*.csv, ppa_features_all_adapted.csv, figures/ppa_adapt_examples.jpg,
         logs/stage8_stats.json, models/ppa_selftrained.pt, models/cls_ring_resnet18.pt
"""
import os, json, random, argparse, datetime, time
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures"); MOD = os.path.join(ROOT, "models")
POS = ["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT"]; NEG = ["NORMAL", "HEALTHY"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(os.path.join(ROOT, p) if not os.path.isabs(p) else p, np.uint8), flag)


def to_tensor(img):
    return torch.from_numpy(((img[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


# ------------------------------------------------------------------ metrics (same definitions as stage 5)
def youden(y, p):
    f, t, th = roc_curve(y, p); return th[np.argmax(t - f)]


def boot_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5]) if a else (np.nan, np.nan)


def score(y, p, thr):
    y = np.asarray(y).astype(int); yh = (np.asarray(p) >= thr).astype(int)
    tp = ((y == 1) & (yh == 1)).sum(); tn = ((y == 0) & (yh == 0)).sum(); fp = ((y == 0) & (yh == 1)).sum(); fn = ((y == 1) & (yh == 0)).sum()
    se = tp / (tp + fn) if tp + fn else np.nan; sp = tn / (tn + fp) if tn + fp else np.nan; lo, hi = boot_auc(y, p)
    return dict(n=len(y), positives=int(y.sum()), TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn), auc=roc_auc_score(y, p),
                auc_ci_low=lo, auc_ci_high=hi, sensitivity=se, specificity=sp, accuracy=(tp + tn) / len(y), threshold=float(thr))


# ------------------------------------------------------------------ A: Fourier domain adaptation
class FDA:
    def __init__(self, palm_paths, beta=0.03, bank=60, seed=0):
        rng = np.random.default_rng(seed); self.beta = beta
        pick = rng.choice(palm_paths, size=min(bank, len(palm_paths)), replace=False)
        self.amps = [np.abs(np.fft.fft2(imread(p).astype(np.float32), axes=(0, 1))) for p in pick]

    def __call__(self, img, key=0):
        a_trg = self.amps[key % len(self.amps)]
        f = np.fft.fft2(img.astype(np.float32), axes=(0, 1)); amp, pha = np.abs(f), np.angle(f)
        h, w = img.shape[:2]; b = max(1, int(min(h, w) * self.beta))
        amp = np.fft.fftshift(amp, axes=(0, 1)); at = np.fft.fftshift(a_trg, axes=(0, 1)); cy, cx = h // 2, w // 2
        amp[cy - b:cy + b, cx - b:cx + b] = at[cy - b:cy + b, cx - b:cx + b]
        amp = np.fft.ifftshift(amp, axes=(0, 1))
        out = np.real(np.fft.ifft2(amp * np.exp(1j * pha), axes=(0, 1)))
        return np.clip(out, 0, 255).astype(np.uint8)


@torch.no_grad()
def predict(model, img, dev):
    x = to_tensor(img)[None].to(dev)
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float()).flip(3)) / 2
    return p[0].cpu().numpy()


def ppa_features(p, temporal_right):
    disc = p[0] > .5; ppa = (p[1] > .5) & ~disc; dpx = disc.sum()
    f = dict(ppa_px=int(ppa.sum()), ppa_to_disc=float(ppa.sum() / dpx) if dpx else 0.0, ppa_maxprob=float(p[1][~disc].max()) if (~disc).any() else 0.0,
             ppa_meanprob_ring=float(p[1][ring_mask(p.shape[1:])].mean()))
    if dpx and ppa.any():
        ys, xs = np.where(disc); cy, cx = ys.mean(), xs.mean(); py, px = np.where(ppa)
        ang = (np.degrees(np.arctan2(py - cy, px - cx)) + 360) % 360
        f["ppa_angle_deg"] = float(len(np.unique((ang // 10).astype(int))) * 10)
    else:
        f["ppa_angle_deg"] = 0.0
    return f


def ring_mask(shape, r_in=0.9, r_out=2.2):
    h, w = shape; yy, xx = np.mgrid[:h, :w]; r = np.hypot(yy - h / 2, xx - w / 2) / (w / 6)   # crop side = 3 disc diameters
    return (r >= r_in) & (r <= r_out)


# ------------------------------------------------------------------ B: self-training
class MixSet(Dataset):
    """PALM labelled crops (full supervision) + glaucoma crops with pseudo-labels (ignore mask)"""
    def __init__(s, items):
        import albumentations as A
        s.items = items
        s.aug = A.Compose([A.HorizontalFlip(p=.5), A.Affine(scale=(.9, 1.1), rotate=(-15, 15), p=.7),
                           A.RandomBrightnessContrast(.2, .2, p=.5), A.HueSaturationValue(6, 10, 6, p=.3)])

    def __len__(s): return len(s.items)

    def __getitem__(s, i):
        it = s.items[i]; im = imread(it["img"])
        if it["kind"] == "palm":
            m = imread(it["mask"], cv2.IMREAD_GRAYSCALE)
            y = np.stack([(m == 128) | (m == 129), m >= 255]).astype(np.float32); w = np.ones_like(y)
        else:
            z = np.load(it["mask"]); y, w = z["y"].astype(np.float32), z["w"].astype(np.float32)
        o = s.aug(image=im, masks=[y[0], y[1], w[0], w[1]]); im = o["image"]; ms = o["masks"]
        return to_tensor(im), torch.from_numpy(np.stack(ms[:2]).astype(np.float32)), torch.from_numpy(np.stack(ms[2:]).astype(np.float32))


def self_train(base_ck, palm_df, gl_df, fda, dev, args):
    pdir = os.path.join(ROOT, "processed", "ppa_pseudo"); os.makedirs(pdir, exist_ok=True)
    model = smp.Unet("resnet34", encoder_weights=None, classes=2).to(dev); model.load_state_dict(torch.load(base_ck, map_location=dev)); model.eval()
    items, n_pos = [], 0
    for k, (_, r) in enumerate(tqdm(gl_df.iterrows(), total=len(gl_df), desc="[SELF] pseudo-labels on glaucoma train crops", unit="img", dynamic_ncols=True)):
        p = predict(model, fda(imread(r.crop512), k), dev)
        y = (p > 0.5).astype(np.uint8); w = ((p > args.hi) | (p < args.lo)).astype(np.uint8)
        y[1] &= ~y[0]; n_pos += int(y[1][w[1] > 0].sum() > 50)
        f = os.path.join(pdir, r.image_id + ".npz"); np.savez_compressed(f, y=y, w=w)
        items.append(dict(kind="pseudo", img=r.crop512, mask=f))
    items += [dict(kind="palm", img=r.crop512, mask=r.crop512_mask) for _, r in palm_df[palm_df.split == "train"].iterrows()]
    print(f"[SELF] {len(gl_df)} pseudo-labelled glaucoma crops ({n_pos} with confident PPA) + {(palm_df.split == 'train').sum()} PALM crops", flush=True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=args.self_lr, weight_decay=1e-4)
    dl = DataLoader(MixSet(items), batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); bce = nn.BCEWithLogitsLoss(reduction="none")
    val = palm_df[palm_df.split == "val"]; best, ck = -1, os.path.join(MOD, "ppa_selftrained.pt")
    for ep in range(1, args.self_epochs + 1):
        model.train()
        for x, y, w in tqdm(dl, desc=f"[SELF] epoch {ep}/{args.self_epochs}", unit="batch", leave=False, dynamic_ncols=True):
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                lo = model(x).float()
            loss = (bce(lo, y) * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); ds = []
        for _, r in val.iterrows():           # keep PALM performance as the checkpoint criterion (no glaucoma labels used)
            p = predict(model, imread(r.crop512), dev); g = imread(r.crop512_mask, cv2.IMREAD_GRAYSCALE) == 255
            if g.sum() > 50:
                pp = (p[1] > .5) & ~(p[0] > .5); ds.append(2 * (pp & g).sum() / max(1, pp.sum() + g.sum()))
        d = float(np.mean(ds)) if ds else 0.0
        print(f"[SELF] epoch {ep}/{args.self_epochs}  PALM val PPA Dice {d:.4f}" + ("  * best" if d > best else ""), flush=True)
        if d > best:
            best = d; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    return model, best, n_pos


# ------------------------------------------------------------------ C: peripapillary ring CNN
class RingDS(Dataset):
    def __init__(s, d, train):
        import albumentations as A
        s.d = d.reset_index(drop=True); s.t = train; s.m = ring_mask((384, 384))
        s.aug = A.Compose([A.HorizontalFlip(p=.5), A.Affine(scale=(.95, 1.05), rotate=(-15, 15), p=.7),
                           A.RandomBrightnessContrast(.2, .2, p=.5), A.HueSaturationValue(6, 10, 6, p=.3)])

    def __len__(s): return len(s.d)

    def __getitem__(s, i):
        r = s.d.iloc[i]; im = cv2.resize(imread(r.crop512), (384, 384), interpolation=cv2.INTER_AREA)
        if s.t:
            im = s.aug(image=im)["image"]
        im = im * s.m[..., None]
        return to_tensor(im.astype(np.uint8)), torch.tensor(float(r.y))


def train_ring(tr, va, te, dev, args):
    import timm
    model = timm.create_model("resnet18", pretrained=True, num_classes=1).to(dev)
    pw = torch.tensor((tr.y == 0).sum() / max(1, (tr.y == 1).sum()), device=dev); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ring_lr, weight_decay=1e-4)
    dl = DataLoader(RingDS(tr, True), batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.ring_lr, total_steps=args.ring_epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    def infer(d):
        model.eval(); out = []
        with torch.no_grad():
            for x, _ in DataLoader(RingDS(d, False), batch_size=32, num_workers=0):
                x = x.to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float())) / 2
                out.append(p.squeeze(1).cpu().numpy())
        return np.concatenate(out)
    best, ck = -1, os.path.join(MOD, "cls_ring_resnet18.pt")
    for ep in range(1, args.ring_epochs + 1):
        model.train()
        for x, y in tqdm(dl, desc=f"[RING] epoch {ep}/{args.ring_epochs}", unit="batch", leave=False, dynamic_ncols=True):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = lossf(model(x).squeeze(1).float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        auc = roc_auc_score(va.y, infer(va))
        print(f"[RING] epoch {ep}/{args.ring_epochs}  val AUC {auc:.4f}" + ("  * best" if auc > best else ""), flush=True)
        if auc > best:
            best = auc; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=dev))
    return infer(tr), infer(va), infer(te), best


def cnn_scores(d, dev):
    """scores of the stage-5 ResNet-50 crop CNN (same preprocessing), for fusion"""
    import timm
    ck = os.path.join(MOD, "cls_resnet50.pt")
    if not os.path.exists(ck):
        return None
    model = timm.create_model("resnet50", pretrained=False, num_classes=1).to(dev); model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(d), 32), desc="Stage-5 CNN scores", unit="batch", dynamic_ncols=True):
            xs = [to_tensor(cv2.resize(imread(p), (384, 384), interpolation=cv2.INTER_AREA)) for p in d.crop512.iloc[i:i + 32]]
            x = torch.stack(xs).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float())) / 2
            out.append(p.squeeze(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_epochs", type=int, default=8); ap.add_argument("--self_lr", type=float, default=1e-4)
    ap.add_argument("--hi", type=float, default=0.8); ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--ring_epochs", type=int, default=20); ap.add_argument("--ring_lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.self_epochs, args.ring_epochs = 1, 1
    torch.manual_seed(0); np.random.seed(0); random.seed(0); t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", dev, flush=True)
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[m.status == "ok"].copy()
    if args.smoke:
        m = m.groupby(["dataset", "split", "label"], dropna=False).head(10)
    palm = pd.read_csv(os.path.join(ROOT, "processed", "palm_master.csv"))
    fda = FDA(palm[palm.split == "train"].crop512.tolist())
    base = smp.Unet("resnet34", encoder_weights=None, classes=2).to(dev)
    base.load_state_dict(torch.load(os.path.join(MOD, "ppa_supervised.pt"), map_location=dev)); base.eval()
    feats = {}
    # ---- variants on every image: original, A (FDA)
    rows = {"orig": [], "fda": []}
    for k, (_, r) in enumerate(tqdm(m.iterrows(), total=len(m), desc="PPA original + FDA on all images", unit="img", dynamic_ncols=True)):
        im = imread(r.crop512); tr_ = r.disc_cx < r.img_w / 2
        rows["orig"].append(dict(image_id=r.image_id, **ppa_features(predict(base, im, dev), tr_)))
        rows["fda"].append(dict(image_id=r.image_id, **ppa_features(predict(base, fda(im, k), dev), tr_)))
    feats["orig"] = pd.DataFrame(rows["orig"]); feats["fda"] = pd.DataFrame(rows["fda"])
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # ---- B: self-training on glaucoma TRAIN crops (all datasets, labels unused)
    st, st_best, st_pos = self_train(os.path.join(MOD, "ppa_supervised.pt"), palm if not args.smoke else palm.groupby("split").head(16),
                                     m[m.split == "train"], fda, dev, args)
    rr = []
    for k, (_, r) in enumerate(tqdm(m.iterrows(), total=len(m), desc="PPA self-trained on all images", unit="img", dynamic_ncols=True)):
        rr.append(dict(image_id=r.image_id, **ppa_features(predict(st, fda(imread(r.crop512), k), dev), r.disc_cx < r.img_w / 2)))
    feats["self"] = pd.DataFrame(rr); del st
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # ---- labelled glaucoma set (same split as stage 5)
    lab = m[m.label.isin(POS + NEG)].copy(); lab["y"] = lab.label.isin(POS).astype(int)
    dl = pd.read_csv(os.path.join(TAB, "dl_per_image_all_base.csv"))[["image_id", "vcdr_e2e"]]
    lab = lab.merge(dl, on="image_id", how="left"); lab["vcdr_e2e"] = lab.vcdr_e2e.fillna(lab.vcdr_e2e.median())
    tr, va, te = lab[lab.split == "train"].copy(), lab[lab.split == "val"].copy(), lab[lab.split == "test"].copy()
    # ---- C: ring CNN
    rtr, rva, rte, ring_best = train_ring(tr, va, te, dev, args)
    tr["ring"], va["ring"], te["ring"] = rtr, rva, rte
    cs = cnn_scores(pd.concat([va, te]), dev)
    if cs is not None:
        va["cnn"], te["cnn"] = cs[:len(va)], cs[len(va):]
    # ---- association + classification per variant
    assoc, res, val_sel = [], [], {}
    PF = ["ppa_to_disc", "ppa_angle_deg", "ppa_meanprob_ring"]
    for v in ["orig", "fda", "self"]:
        F = feats[v]
        for part in (tr, va, te):
            for c in PF:
                part[f"{v}_{c}"] = part.image_id.map(F.set_index("image_id")[c]).fillna(0).values
        for ds, g in list(te.groupby("dataset")) + [("Pooled", te)]:
            x = g[f"{v}_ppa_meanprob_ring"]; pos, neg = x[g.y == 1], x[g.y == 0]
            assoc.append(dict(variant=v, test_set=ds, ppa_detected_rate_glaucoma=float((g[f"{v}_ppa_to_disc"][g.y == 1] > 0.02).mean()),
                              ppa_detected_rate_normal=float((g[f"{v}_ppa_to_disc"][g.y == 0] > 0.02).mean()),
                              mean_ring_prob_glaucoma=float(pos.mean()), mean_ring_prob_normal=float(neg.mean()),
                              mannwhitney_p=float(mannwhitneyu(pos, neg).pvalue) if len(pos) and len(neg) else np.nan,
                              auc_ppa_alone=float(roc_auc_score(g.y, x)) if g.y.nunique() == 2 else np.nan))
    for ds, g in list(te.groupby("dataset")) + [("Pooled", te)]:
        assoc.append(dict(variant="ring", test_set=ds, auc_ppa_alone=float(roc_auc_score(g.y, g.ring)) if g.y.nunique() == 2 else np.nan,
                          mean_ring_prob_glaucoma=float(g.ring[g.y == 1].mean()), mean_ring_prob_normal=float(g.ring[g.y == 0].mean())))
    models = {}
    for v in ["orig", "fda", "self"]:
        models[f"vCDR + PPA ({v})"] = ["vcdr_e2e"] + [f"{v}_{c}" for c in PF]
        models[f"PPA alone ({v})"] = [f"{v}_{c}" for c in PF]
    models["vCDR + ring CNN (C)"] = ["vcdr_e2e", "ring"]; models["Ring CNN alone (C)"] = ["ring"]
    probs = {}
    for name, f in models.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(tr[f], tr.y)
        probs[name] = (clf.predict_proba(va[f])[:, 1], clf.predict_proba(te[f])[:, 1])
        if name.startswith("vCDR +"):
            val_sel[name] = roc_auc_score(va.y, probs[name][0])
    best_name = max(val_sel, key=val_sel.get); print("Best PPA variant on VALIDATION:", best_name, round(val_sel[best_name], 4), flush=True)
    if cs is not None:      # fusion like M7 but with each PPA variant (trained on val, as in stage 5)
        for v in ["orig", "fda", "self", "ring"]:
            cols = ["cnn", "vcdr_e2e"] + ([f"{v}_{c}" for c in PF] if v != "ring" else ["ring"])
            fu = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(va[cols], va.y)
            probs[f"Fusion CNN + vCDR + PPA ({v})"] = (fu.predict_proba(va[cols])[:, 1], fu.predict_proba(te[cols])[:, 1])
    for name, (pv, pt) in probs.items():
        thr = youden(va.y, pv)
        for scope, mask in [("Pooled (Chaksu+PAPILA)", np.ones(len(te), bool)), ("Chaksu", (te.dataset == "Chaksu").values), ("PAPILA", (te.dataset == "PAPILA").values)]:
            if te.y.values[mask].min() != te.y.values[mask].max():
                res.append(dict(method=name, test_set=scope, val_auc=roc_auc_score(va.y, pv), **score(te.y.values[mask], pt[mask], thr)))
    assoc = pd.DataFrame(assoc); res = pd.DataFrame(res)
    assoc.to_csv(os.path.join(TAB, "ppa_adapt_association.csv"), index=False); res.to_csv(os.path.join(TAB, "ppa_adapt_classification.csv"), index=False)
    print(assoc.round(3).to_string(index=False)); print(res[res.test_set.str.startswith("Pooled")][["method", "val_auc", "auc", "sensitivity", "specificity", "accuracy"]].round(3).to_string(index=False))
    # best variant -> ppa_features_all_adapted.csv (drop-in for stage 5)
    bv = best_name.split("(")[1].split(")")[0]
    if bv in feats:
        F = feats[bv].merge(m[["image_id", "dataset", "label", "split"]], on="image_id")
        F["ppa_present"] = F.ppa_to_disc > 0.02; F["variant"] = bv
        F.to_csv(os.path.join(TAB, "ppa_features_all_adapted.csv"), index=False)
    # example figure: original vs FDA vs self-trained PPA maps on the glaucoma test images with highest ring prob (self)
    try:
        st2 = smp.Unet("resnet34", encoder_weights=None, classes=2).to(dev); st2.load_state_dict(torch.load(os.path.join(MOD, "ppa_selftrained.pt"), map_location=dev)); st2.eval()
        ex = te[te.y == 1].sort_values("self_ppa_meanprob_ring", ascending=False).head(4); tiles = []
        for k, (_, r) in enumerate(ex.iterrows()):
            im = imread(r.crop512); row = []
            for mdl, img in [(base, im), (base, fda(im, k)), (st2, fda(im, k))]:
                p = predict(mdl, img, dev); vis = im.copy()
                for msk, col in [(p[0] > .5, (255, 255, 0)), ((p[1] > .5) & ~(p[0] > .5), (0, 0, 255))]:
                    c, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, c, -1, col, 2)
                row.append(cv2.resize(vis, (256, 256)))
            tiles.append(np.hstack(row))
        if tiles:
            cv2.imencode(".jpg", np.vstack(tiles))[1].tofile(os.path.join(FIG, "ppa_adapt_examples.jpg"))
    except Exception as e:
        print("figure skipped:", e)
    stats = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), minutes=round((time.time() - t0) / 60, 1),
                 selftrain_palm_val_dice=st_best, selftrain_glaucoma_crops_with_confident_ppa=st_pos, ring_val_auc=ring_best,
                 best_variant_on_val=best_name, val_auc=val_sel, association=assoc.to_dict("records"),
                 pooled=res[res.test_set.str.startswith("Pooled")].to_dict("records"))
    json.dump(stats, open(os.path.join(ROOT, "logs", "stage8_stats.json"), "w"), indent=2, default=float)
    print("Saved results/tables/ppa_adapt_*.csv, ppa_features_all_adapted.csv, logs/stage8_stats.json", flush=True)


if __name__ == "__main__":
    main()
