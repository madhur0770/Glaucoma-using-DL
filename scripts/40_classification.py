"""Stage 5: glaucoma classification - classical (paper) vs deep-learning variants, same test images.

Labelled data: Chaksu (5-expert majority: NORMAL vs GLAUCOMA SUSPECT) and PAPILA (clinical: healthy vs glaucoma/suspect).
Methods compared (all trained on train+val, evaluated on test, per dataset and pooled):
  M0  Paper: linear SVM on [classical area-CDR, classical PPA area]              (from Stage 3)
  M1  Logistic regression on DL vertical CDR (end-to-end: raw image -> localiser -> segmenter)
  M2  Logistic regression on DL vCDR + hCDR + area CDR + classical PPA area on the DL disc (hybrid)
  M3  CNN (ImageNet ResNet-50 via timm) fine-tuned on the disc crop, class-weighted loss
  M4  Fusion: M3 probability + DL vCDR (logistic regression)
  M5  DL vCDR + learned PPA features (Stage 6);  M5b learned PPA alone
  M6  The paper's decision table rebuilt from learned parts: vCDR > tau (chosen on val) OR learned PPA present
  M7  Fusion: CNN + DL vCDR + learned PPA
  REF Logistic regression on EXPERT vertical CDR (ceiling for CDR-only methods)
Metrics: AUC with 95% bootstrap CI, sensitivity / specificity / accuracy / balanced accuracy at the threshold chosen
on the validation set (Youden), and sensitivity at 85 % specificity.
Outputs: results/tables/classification_results.csv, classification_per_image.csv, figures/roc_*.png
"""
import os, json, argparse, datetime, random
import numpy as np, pandas as pd, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures")
POS = ["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT"]; NEG = ["NORMAL", "HEALTHY"]


def youden(y, p):
    f, t, th = roc_curve(y, p); return th[np.argmax(t - f)]


def at_spec(y, p, spec=0.85):
    f, t, th = roc_curve(y, p); ok = (1 - f) >= spec
    return float(t[ok].max()) if ok.any() else np.nan


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
    se = tp / (tp + fn) if tp + fn else np.nan; sp = tn / (tn + fp) if tn + fp else np.nan
    lo, hi = boot_auc(y, p)
    return dict(n=len(y), positives=int(y.sum()), TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan, auc_ci_low=lo, auc_ci_high=hi,
                sensitivity=se, specificity=sp, accuracy=(tp + tn) / len(y), balanced_accuracy=np.nanmean([se, sp]),
                sens_at_85spec=at_spec(y, p) if len(np.unique(y)) == 2 else np.nan, threshold=float(thr))


class CropDS:
    """module-level (picklable on Windows) dataset of disc crops for the CNN"""
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(s, d, train):
        import albumentations as A
        s.d = d.reset_index(drop=True); s.t = train
        s.aug = A.Compose([A.HorizontalFlip(p=0.5), A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
                           A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.HueSaturationValue(8, 12, 8, p=0.3)])

    def __len__(s):
        return len(s.d)

    def __getitem__(s, i):
        import torch
        r = s.d.iloc[i]; im = cv2.imdecode(np.fromfile(os.path.join(ROOT, r.crop512), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA)
        if s.t:
            im = s.aug(image=im)["image"]
        x = (im[..., ::-1].astype(np.float32) / 255 - s.MEAN) / s.STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.tensor(float(r.y))


def train_cnn(tr, va, te_all, args):
    import torch, torch.nn as nn, timm
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.cnn, pretrained=True, num_classes=1).to(dev)
    pw = torch.tensor((tr.y == 0).sum() / max(1, (tr.y == 1).sum()), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dl = DataLoader(CropDS(tr, True), batch_size=16, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    def infer(d):
        model.eval(); out = []
        with torch.no_grad():
            for x, _ in DataLoader(CropDS(d, False), batch_size=32, num_workers=0):
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    x = x.to(dev); p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float())) / 2
                out.append(p.squeeze(1).cpu().numpy())
        return np.concatenate(out)
    best, ck = -1, os.path.join(ROOT, "models", f"cls_{args.cnn}.pt"); hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(dl, desc=f"[CNN] epoch {ep}/{args.epochs}", unit="batch", dynamic_ncols=True, leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = lossf(model(x).squeeze(1).float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        pv = infer(va); auc = roc_auc_score(va.y, pv); hist.append(dict(epoch=ep, val_auc=auc))
        print(f"[CNN] epoch {ep}/{args.epochs}  val AUC {auc:.4f}" + ("  * best" if auc > best else ""), flush=True)
        if auc > best:
            best = auc; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=dev))
    pd.DataFrame(hist).to_csv(os.path.join(TAB, "train_history_cnn.csv"), index=False)
    return {k: infer(d) for k, d in te_all.items()}, best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seg_tag", default="base"); ap.add_argument("--cnn", default="resnet50")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--no_cnn", action="store_true")
    args = ap.parse_args(); random.seed(0); np.random.seed(0)
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[(m.status == "ok") & m.label.isin(POS + NEG)].copy()
    m["y"] = m.label.isin(POS).astype(int)
    dl = pd.read_csv(os.path.join(TAB, f"dl_per_image_all_{args.seg_tag}.csv"))[["image_id", "vcdr_e2e", "hcdr_e2e", "acdr_e2e", "vcdr_pred"]]
    cl = pd.read_csv(os.path.join(TAB, "baseline_per_image.csv"))[["image_id", "acdr_classical", "ppa_area_px500"]]
    d = m.merge(dl, on="image_id", how="left").merge(cl, on="image_id", how="left")
    ppa_file = os.path.join(TAB, "ppa_features_all.csv")
    has_ppa = os.path.exists(ppa_file)
    if has_ppa:
        P = pd.read_csv(ppa_file)[["image_id", "ppa_to_disc", "ppa_angle_deg", "ppa_present"]]
        d = d.merge(P, on="image_id", how="left")
        d[["ppa_to_disc", "ppa_angle_deg"]] = d[["ppa_to_disc", "ppa_angle_deg"]].fillna(0); d["ppa_present"] = d.ppa_present.fillna(False).astype(int)
    for c in ["vcdr_e2e", "hcdr_e2e", "acdr_e2e"]:
        d[c] = d[c].fillna(d[c].median())
    d[["acdr_classical", "ppa_area_px500"]] = d[["acdr_classical", "ppa_area_px500"]].fillna(0)
    tr, va, te = d[d.split == "train"], d[d.split == "val"], d[d.split == "test"]
    print(f"labelled: train {len(tr)} (pos {tr.y.sum()}), val {len(va)} (pos {va.y.sum()}), test {len(te)} (pos {te.y.sum()})")
    feats = {"M0 Paper SVM-equivalent: classical CDR + PPA area": ["acdr_classical", "ppa_area_px500"],
             "M1 DL vertical CDR (end-to-end)": ["vcdr_e2e"],
             "M2 DL vCDR + hCDR + area CDR + PPA area (hybrid)": ["vcdr_e2e", "hcdr_e2e", "acdr_e2e", "ppa_area_px500"],
             "REF expert vertical CDR (ceiling)": ["vcdr"]}
    if has_ppa:
        feats["M5 DL vCDR + learned PPA (ratio, angular extent)"] = ["vcdr_e2e", "ppa_to_disc", "ppa_angle_deg"]
        feats["M5b learned PPA alone"] = ["ppa_to_disc", "ppa_angle_deg"]
    probs = {}
    for name, f in feats.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        clf.fit(tr[f], tr.y); probs[name] = (clf.predict_proba(va[f])[:, 1], clf.predict_proba(te[f])[:, 1])
    if has_ppa:
        # M6: the paper's decision table rebuilt from learned parts: glaucoma if vCDR > tau OR learned PPA present
        tau = youden(va.y, va.vcdr_e2e)
        rule = lambda g: np.maximum((g.vcdr_e2e > tau).astype(float), g.ppa_present.astype(float)) * 0.5 + 0.5 * (g.vcdr_e2e > tau)
        probs[f"M6 Paper rule automated: DL vCDR>{tau:.2f} OR learned PPA"] = (rule(va).values, rule(te).values)
    if not args.no_cnn:
        out, best = train_cnn(tr, va, {"va": va, "te": te}, args)
        probs["M3 CNN on disc crop (ResNet-50)"] = (out["va"], out["te"])
        Fva = np.c_[out["va"], va.vcdr_e2e]; Fte = np.c_[out["te"], te.vcdr_e2e]
        fu = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(Fva, va.y)
        probs["M4 Fusion: CNN + DL vCDR"] = (fu.predict_proba(Fva)[:, 1], fu.predict_proba(Fte)[:, 1])
        if has_ppa:
            Fva2 = np.c_[out["va"], va.vcdr_e2e, va.ppa_to_disc, va.ppa_angle_deg]; Fte2 = np.c_[out["te"], te.vcdr_e2e, te.ppa_to_disc, te.ppa_angle_deg]
            fu2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(Fva2, va.y)
            probs["M7 Fusion: CNN + DL vCDR + learned PPA"] = (fu2.predict_proba(Fva2)[:, 1], fu2.predict_proba(Fte2)[:, 1])
    rows, per = [], te[["image_id", "dataset", "label", "y"]].copy()
    for name, (pv, pt) in probs.items():
        thr = youden(va.y, pv); per[name] = pt
        for scope, mask in [("Pooled (Chaksu+PAPILA)", np.ones(len(te), bool)), ("Chaksu", (te.dataset == "Chaksu").values), ("PAPILA", (te.dataset == "PAPILA").values)]:
            yv = te.y.values[mask]
            if len(np.unique(yv)) == 2:
                rows.append(dict(method=name, test_set=scope, **score(yv, pt[mask], thr)))
            else:
                print(f"  skip {scope}: test set has one class only")
    res = pd.DataFrame(rows); res.to_csv(os.path.join(TAB, "classification_results.csv"), index=False)
    per.to_csv(os.path.join(TAB, "classification_per_image.csv"), index=False)
    if len(res):
        print(res[["method", "test_set", "auc", "auc_ci_low", "auc_ci_high", "sensitivity", "specificity", "accuracy"]].to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for scope, mask in [("pooled", np.ones(len(te), bool)), ("Chaksu", (te.dataset == "Chaksu").values), ("PAPILA", (te.dataset == "PAPILA").values)]:
            if len(np.unique(te.y.values[mask])) < 2:
                continue
            plt.figure(figsize=(5.5, 5))
            for name, (_, pt) in probs.items():
                f, t, _ = roc_curve(te.y.values[mask], pt[mask]); plt.plot(f, t, label=f"{name.split(':')[0]} AUC={roc_auc_score(te.y.values[mask], pt[mask]):.3f}")
            plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("1 - specificity"); plt.ylabel("sensitivity"); plt.title(f"ROC - test set ({scope})")
            plt.legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig(os.path.join(FIG, f"roc_{scope}.png"), dpi=150); plt.close()
    except Exception as e:
        print("plot failed", e)
    json.dump(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), n_train=len(tr), n_val=len(va), n_test=len(te),
                   results=res.to_dict("records")), open(os.path.join(ROOT, "logs", "stage5_stats.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
