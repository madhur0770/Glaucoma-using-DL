"""Stage 13: does it work in REAL use (new hospital, new camera, bad photos) - not only on benchmark numbers?

Parts (--parts, default all):
  lodo    - leave-one-dataset-out: the disc/cup segmenter is retrained 4x, each time WITHOUT one dataset (Chaksu, PAPILA,
            MESSIDOR, Magrabia; BinRushed was already done in stage 7), and tested on the unseen dataset (= a new hospital/
            camera). Classification: CNN trained on Chaksu -> tested on PAPILA and vice versa.
  corrupt - test-time corruptions of the test images (blur, dark, bright, low contrast, JPEG q=15, 4x lower resolution,
            noise): disc/cup Dice, vCDR error and glaucoma AUC (CNN, DL vCDR) per corruption and severity
  quality - automatic quality / out-of-distribution flag from the model's own uncertainty (cup/disc entropy, disc-size
            plausibility, localiser confidence); thresholds from the validation set (95th percentile); reports how often it
            fires on clean vs corrupted photos and how much error the flagged images carry
  deploy  - fits the deployable decision model (temperature-scaled CNN + DL vCDR + rim, logistic fusion fitted on validation)
            and saves it to models/deploy.json for scripts/predict.py; reports calibration before/after
Outputs: results/tables/robust_*.csv, figures/robust_*.png, models/deploy.json, logs/stage13_stats.json
"""
import os, json, time, argparse, datetime, random, traceback, importlib.util, io
import numpy as np, pandas as pd, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures"); MOD = os.path.join(ROOT, "models")
POS = ["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT"]; NEG = ["NORMAL", "HEALTHY"]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def dev_():
    import torch; return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def master():
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[m.status == "ok"].copy()
    m["y"] = np.where(m.label.isin(POS), 1, np.where(m.label.isin(NEG), 0, -1)); return m


# ------------------------------------------------------------------ corruptions
def corrupt(img, kind, sev):
    if kind == "clean":
        return img
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), [1.5, 3, 5][sev])
    if kind == "dark":
        return np.clip(img.astype(np.float32) * [0.7, 0.5, 0.35][sev], 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(img.astype(np.float32) * [1.3, 1.6, 1.9][sev], 0, 255).astype(np.uint8)
    if kind == "low_contrast":
        f = [0.7, 0.5, 0.3][sev]; mu = img.mean((0, 1), keepdims=True); return np.clip(mu + (img - mu) * f, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, [40, 20, 10][sev]]); return cv2.imdecode(b, cv2.IMREAD_COLOR)
    if kind == "low_res":
        f = [0.5, 0.25, 0.125][sev]; h, w = img.shape[:2]
        return cv2.resize(cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "noise":
        return np.clip(img + np.random.default_rng(0).normal(0, [8, 16, 28][sev], img.shape), 0, 255).astype(np.uint8)
    raise ValueError(kind)


KINDS = ["clean", "blur", "dark", "bright", "low_contrast", "jpeg", "low_res", "noise"]


def seg_models(dev):
    import torch
    S = _load("seg30", "30_train_segmentation.py")
    seg = S.build("resnet34", 2, False).to(dev); seg.load_state_dict(torch.load(os.path.join(MOD, "seg_crop512_base.pt"), map_location=dev)); seg.eval()
    return S, seg


def cnn_model(dev):
    import torch, timm
    m = timm.create_model("resnet50", pretrained=False, num_classes=1).to(dev)
    m.load_state_dict(torch.load(os.path.join(MOD, "cls_resnet50.pt"), map_location=dev)); m.eval(); return m


def cnn_logit(model, imgs, dev, S):
    import torch
    x = torch.stack([S.to_tensor(cv2.resize(i, (384, 384), interpolation=cv2.INTER_AREA)) for i in imgs]).to(dev)
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        return ((model(x).float() + model(torch.flip(x, [3])).float()) / 2).squeeze(1).cpu().numpy()


def seg_quality(p):
    """uncertainty features from the segmenter's probability maps"""
    eps = 1e-6; disc = p[0] > .5
    ent = lambda q: -(q * np.log(q + eps) + (1 - q) * np.log(1 - q + eps))
    band = cv2.dilate(disc.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return dict(disc_entropy=float(ent(p[0])[band].mean()) if band.any() else 1.0,
                cup_entropy=float(ent(p[1])[band].mean()) if band.any() else 1.0,
                disc_area_frac=float(disc.mean()), n_disc_components=int(cv2.connectedComponents(disc.astype(np.uint8))[0] - 1))


def part_corrupt(st):
    import torch
    dev = dev_(); S, seg = seg_models(dev); cnn = cnn_model(dev); m = master()
    te = m[m.split == "test"].reset_index(drop=True); rows, per = [], []
    for kind in KINDS:
        for sev in ([0] if kind == "clean" else [0, 1, 2]):
            P = []
            for i in tqdm(range(0, len(te), 16), desc=f"[corrupt] {kind} severity {sev + 1}", unit="batch", leave=False, dynamic_ncols=True):
                g = te.iloc[i:i + 16]; imgs = [corrupt(S.imread(os.path.join(ROOT, r.crop512)), kind, sev) for _, r in g.iterrows()]
                lg = cnn_logit(cnn, imgs, dev, S)
                for (_, r), im, l in zip(g.iterrows(), imgs, lg):
                    p = S.predict(seg, im, dev); pd_ = S.largest_cc(p[0] > .5); pc = (p[1] > .5) & pd_; pc = S.largest_cc(pc) if pc.any() else pc
                    gm = S.imread(os.path.join(ROOT, r.crop512_mask), cv2.IMREAD_GRAYSCALE)
                    dd = lambda a_, b_: 2 * (a_ & b_).sum() / max(1, a_.sum() + b_.sum())
                    v, h_, ac = S.masks_to_cdr(pd_, pc)
                    P.append(dict(image_id=r.image_id, dataset=r.dataset, y=r.y, kind=kind, severity=sev + 1, dice_disc=dd(pd_, gm >= 128), dice_cup=dd(pc, gm >= 255),
                                  vcdr_pred=v, vcdr=r.vcdr, cnn_logit=float(l), **seg_quality(p)))
            P = pd.DataFrame(P); per.append(P); lab = P[P.y >= 0]
            rows.append(dict(kind=kind, severity=0 if kind == "clean" else sev + 1, images=len(P), dice_disc=P.dice_disc.mean(), dice_cup=P.dice_cup.mean(),
                             vcdr_mae=(P.vcdr_pred - P.vcdr).abs().mean(), auc_cnn=roc_auc_score(lab.y, lab.cnn_logit), auc_dl_vcdr=roc_auc_score(lab.y, lab.vcdr_pred.fillna(0))))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "robust_corruptions.csv"), index=False)
    pd.concat(per).to_csv(os.path.join(TAB, "robust_corruptions_per_image.csv"), index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for j, (c, t) in enumerate([("dice_cup", "Cup Dice"), ("vcdr_mae", "vCDR error"), ("auc_cnn", "CNN AUC")]):
            for k in KINDS[1:]:
                g = R[R.kind == k]; ax[j].plot([0] + list(g.severity), [R[R.kind == "clean"][c].iloc[0]] + list(g[c]), "o-", label=k)
            ax[j].set_title(t); ax[j].set_xlabel("severity")
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "robust_corruptions.png"), dpi=150); plt.close(fig)
    except Exception as e:
        print("plot skipped", e)
    st["corruptions"] = R.to_dict("records")


def part_quality(st):
    dev = dev_(); S, seg = seg_models(dev); m = master(); va = m[m.split == "val"]; Q = []
    for _, r in tqdm(va.iterrows(), total=len(va), desc="[quality] validation uncertainty", unit="img", dynamic_ncols=True):
        Q.append(seg_quality(S.predict(seg, S.imread(os.path.join(ROOT, r.crop512)), dev)))
    Q = pd.DataFrame(Q)
    thr = dict(disc_entropy=float(Q.disc_entropy.quantile(.95)), cup_entropy=float(Q.cup_entropy.quantile(.95)),
               disc_area_lo=float(Q.disc_area_frac.quantile(.01)), disc_area_hi=float(Q.disc_area_frac.quantile(.99)))
    json.dump(thr, open(os.path.join(MOD, "quality.json"), "w"), indent=2)
    f = os.path.join(TAB, "robust_corruptions_per_image.csv")
    if os.path.exists(f):
        P = pd.read_csv(f)
        P["flag"] = (P.disc_entropy > thr["disc_entropy"]) | (P.cup_entropy > thr["cup_entropy"]) | (P.disc_area_frac < thr["disc_area_lo"]) | \
                    (P.disc_area_frac > thr["disc_area_hi"]) | (P.n_disc_components != 1)
        P["err"] = (P.vcdr_pred - P.vcdr).abs()
        R = P.groupby(["kind", "severity"]).apply(lambda g: pd.Series(dict(flag_rate=g.flag.mean(), vcdr_err_flagged=g.err[g.flag].mean(),
                                                                          vcdr_err_unflagged=g.err[~g.flag].mean(), cup_dice_flagged=g.dice_cup[g.flag].mean(),
                                                                          cup_dice_unflagged=g.dice_cup[~g.flag].mean()))).reset_index()
        R.to_csv(os.path.join(TAB, "robust_quality_flag.csv"), index=False); print(R.round(3).to_string(index=False), flush=True)
        st["quality_flag"] = R.to_dict("records")
    st["quality_thresholds"] = thr


def part_deploy(st):
    """deployable model: temperature-scaled CNN logit + DL vCDR + rim features (if stage 9 ran) -> logistic, fitted on validation"""
    import torch
    dev = dev_(); S, _ = seg_models(dev); cnn = cnn_model(dev); m = master(); lab = m[m.y >= 0]
    dl = pd.read_csv(os.path.join(TAB, "dl_per_image_all_base.csv"))[["image_id", "vcdr_e2e"]]; lab = lab.merge(dl, on="image_id", how="left")
    rimf = os.path.join(TAB, "extras_isnt_per_image.csv"); RF = []
    if os.path.exists(rimf):
        RF = ["rim_I", "rim_S", "rim_N", "rim_T", "rim_min"]; lab = lab.merge(pd.read_csv(rimf)[["image_id"] + RF], on="image_id", how="left")
    for c in ["vcdr_e2e"] + RF:
        lab[c] = lab[c].fillna(lab[c].median())
    va, te = lab[lab.split == "val"].copy(), lab[lab.split == "test"].copy()
    for part in (va, te):
        L = []
        for i in tqdm(range(0, len(part), 16), desc="[deploy] CNN logits", unit="batch", leave=False, dynamic_ncols=True):
            L.append(cnn_logit(cnn, [S.imread(os.path.join(ROOT, p)) for p in part.crop512.iloc[i:i + 16]], dev, S))
        part["cnn_logit"] = np.concatenate(L)
    # temperature scaling of the CNN alone (Guo et al. 2017)
    ts = np.linspace(0.3, 5, 95); nll = [np.mean(np.log1p(np.exp(-(2 * va.y - 1) * va.cnn_logit / t))) for t in ts]; T = float(ts[int(np.argmin(nll))])
    F = ["cnn_logit", "vcdr_e2e"] + RF
    sc = StandardScaler().fit(va[F]); lr = LogisticRegression(max_iter=3000).fit(sc.transform(va[F]), va.y)   # unweighted -> calibrated probabilities
    pt = lr.predict_proba(sc.transform(te[F]))[:, 1]; pv = lr.predict_proba(sc.transform(va[F]))[:, 1]
    f, t, th = roc_curve(va.y, pv); thr = float(th[np.argmax(t - f)])
    ece = lambda y, p: sum(((p >= a) & (p < a + .1)).mean() * abs(y[(p >= a) & (p < a + .1)].mean() - p[(p >= a) & (p < a + .1)].mean()) for a in np.arange(0, 1, .1) if ((p >= a) & (p < a + .1)).any())
    sig = lambda z: 1 / (1 + np.exp(-z))
    yv = te.y.values.astype(float)
    res = dict(temperature=T, test_auc=roc_auc_score(te.y, pt), test_ece_fusion=float(ece(yv, pt)),
               test_ece_cnn_raw=float(ece(yv, sig(te.cnn_logit.values))), test_ece_cnn_temperature=float(ece(yv, sig(te.cnn_logit.values / T))),
               threshold_youden_val=thr, sens=float(((pt >= thr) & (yv == 1)).sum() / max(1, (yv == 1).sum())),
               spec=float(((pt < thr) & (yv == 0)).sum() / max(1, (yv == 0).sum())), features=F)
    json.dump(dict(features=F, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                   threshold=thr, temperature=T, cnn="models/cls_resnet50.pt", seg="models/seg_crop512_base.pt", loc="models/seg_full512_base.pt",
                   ppa="models/ppa_supervised.pt", note="fitted on the validation split by scripts/85_robustness.py"), open(os.path.join(MOD, "deploy.json"), "w"), indent=2)
    pd.DataFrame([res]).to_csv(os.path.join(TAB, "robust_deploy_model.csv"), index=False); print(res, flush=True); st["deploy"] = res


def part_lodo(st, a):
    import torch, subprocess, sys
    rows = []
    for ds in ["Chaksu", "PAPILA", "RIGA-MESSIDOR", "RIGA-Magrabia"]:
        tag = "lodo_" + ds.replace("RIGA-", "").lower()
        out = os.path.join(TAB, f"dl_segmentation_{tag}.csv")
        if not os.path.exists(out):
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "30_train_segmentation.py"), "--epochs", str(a.lodo_epochs), "--workers", "2",
                   "--tag", tag, "--exclude_dataset", ds, "--loc_from", "base"] + (["--smoke"] if a.smoke else [])
            print(f"[lodo] training without {ds}: {' '.join(cmd[2:])}", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise RuntimeError(f"LODO training without {ds} failed")
            if a.smoke:
                out = os.path.join(TAB, "dl_segmentation_smoke.csv")
        t = pd.read_csv(out); seen = pd.read_csv(os.path.join(TAB, "dl_segmentation_base.csv"))
        g = t[t.dataset == ds]; b = seen[seen.dataset == ds]
        if len(g):
            rows.append(dict(held_out_dataset=ds, dice_disc_unseen=g.dice_disc.iloc[0], dice_cup_unseen=g.dice_cup.iloc[0], vcdr_mae_unseen=g.vcdr_mae.iloc[0],
                             dice_disc_seen=b.dice_disc.iloc[0], dice_cup_seen=b.dice_cup.iloc[0], vcdr_mae_seen=b.vcdr_mae.iloc[0]))
    br = os.path.join(TAB, "dl_segmentation_noBR.csv")
    if os.path.exists(br):
        t = pd.read_csv(br); g = t[t.dataset == "RIGA-BinRushed"]; b = pd.read_csv(os.path.join(TAB, "dl_segmentation_base.csv")); b = b[b.dataset == "RIGA-BinRushed"]
        rows.append(dict(held_out_dataset="RIGA-BinRushed", dice_disc_unseen=g.dice_disc.iloc[0], dice_cup_unseen=g.dice_cup.iloc[0], vcdr_mae_unseen=g.vcdr_mae.iloc[0],
                         dice_disc_seen=b.dice_disc.iloc[0], dice_cup_seen=b.dice_cup.iloc[0], vcdr_mae_seen=b.vcdr_mae.iloc[0]))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "robust_lodo_segmentation.csv"), index=False); print(R.round(4).to_string(index=False), flush=True)
    # classification across datasets: CNN trained on one labelled dataset, tested on the other
    X = _load("extras80", "80_reviewer_extras.py"); m = X.labelled(a.smoke); cr = []
    for src, dst in [("Chaksu", "PAPILA"), ("PAPILA", "Chaksu")]:
        tr, te = m[m.dataset == src], m[m.dataset == dst]
        p = X.cv_cnn(tr, te, 1 if a.smoke else 20, 0)
        cr.append(dict(train_on=src, test_on=dst, n_test=len(te), auc_cnn=roc_auc_score(te.y, p), auc_dl_vcdr=roc_auc_score(te.y, te.vcdr_e2e)))
    C = pd.DataFrame(cr); C.to_csv(os.path.join(TAB, "robust_cross_dataset_classification.csv"), index=False); print(C.round(4).to_string(index=False), flush=True)
    st["lodo_segmentation"] = R.to_dict("records"); st["cross_dataset_classification"] = C.to_dict("records")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="corrupt,quality,deploy,lodo"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lodo_epochs", type=int, default=20)
    a = ap.parse_args(); random.seed(0); np.random.seed(0); t0 = time.time(); st = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), parts={}, errors={})
    if a.smoke:
        global master
        _m = master; master = lambda: _m().groupby(["dataset", "split", "y"]).head(6)
    fns = dict(corrupt=lambda: part_corrupt(st), quality=lambda: part_quality(st), deploy=lambda: part_deploy(st), lodo=lambda: part_lodo(st, a))
    for p in a.parts.split(","):
        t1 = time.time(); print(f"\n===== part {p} =====", flush=True)
        try:
            fns[p](); st["parts"][p] = f"ok ({(time.time() - t1) / 60:.1f} min)"
        except Exception:
            st["parts"][p] = "FAILED"; st["errors"][p] = traceback.format_exc()[-2000:]; print(st["errors"][p], flush=True)
    st["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(st, open(os.path.join(ROOT, "logs", "stage13_stats.json" if not a.smoke else "stage13_stats_smoke.json"), "w"), indent=2, default=float)
    print("\nParts:", st["parts"], flush=True)
    if st["errors"]:
        raise SystemExit(f"failed parts: {list(st['errors'])}")


if __name__ == "__main__":
    main()
