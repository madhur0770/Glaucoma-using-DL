"""Stage 9: what reviewers of a 2024-26 glaucoma deep-learning paper expect, on top of stages 2-8 (all automatic).

Parts (run all, or pick with --parts):
  delong  - DeLong test (Sun & Xu 2014 fast algorithm) for every method vs the paper method (M0) and vs our best (M7),
            on the stage-5 test set (pooled, Chaksu, PAPILA)                                              [CPU, seconds]
  calib   - calibration of the stage-5 test probabilities: Brier score, expected calibration error (10 bins),
            reliability diagram                                                                            [CPU, seconds]
  isnt    - neuroretinal rim width in the Inferior / Superior / Nasal / Temporal sectors (ISNT rule) from the DL
            disc/cup masks; rim features alone, with vCDR, and fused with the CNN                          [GPU, ~5 min]
  gradcam - Grad-CAM of the stage-5 ResNet-50 CNN; share of heat inside 1.5 disc radii (does it look at the disc?)
            + figure of glaucoma / normal test cases                                                       [GPU, ~2 min]
  dino    - foundation-model comparison: frozen DINOv2 ViT-B/14 (Oquab et al. 2023, open weights) + linear probe
            (logistic regression), same split                                                              [GPU, ~10 min]
  cv      - 5-fold patient-level cross-validation on all labelled eyes (Chaksu + PAPILA): feature models
            (paper M0, DL vCDR, hybrid, + rim) and the ResNet-50 CNN retrained per fold                     [GPU, ~60 min]
A part that fails is recorded in logs/stage9_stats.json and the others still run; the job exits non-zero so the
the runner retries / the script is fixed. Outputs: results/tables/extras_*.csv, results/figures/extras_*.png|jpg.
"""
import os, json, argparse, datetime, time, random, traceback
import numpy as np, pandas as pd, cv2
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures"); MOD = os.path.join(ROOT, "models")
POS = ["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT"]; NEG = ["NORMAL", "HEALTHY"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SCOPES = lambda te: [("Pooled (Chaksu+PAPILA)", np.ones(len(te), bool)), ("Chaksu", (te.dataset == "Chaksu").values), ("PAPILA", (te.dataset == "PAPILA").values)]


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(os.path.join(ROOT, p), np.uint8), flag)


def youden(y, p):
    f, t, th = roc_curve(y, p); return th[np.argmax(t - f)]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))


def cls_metrics(y, p, thr):
    y = np.asarray(y).astype(int); yh = (np.asarray(p) >= thr).astype(int)
    tp = ((y == 1) & (yh == 1)).sum(); tn = ((y == 0) & (yh == 0)).sum(); fp = ((y == 0) & (yh == 1)).sum(); fn = ((y == 1) & (yh == 0)).sum()
    return dict(n=len(y), positives=int(y.sum()), auc=roc_auc_score(y, p), sensitivity=tp / max(1, tp + fn), specificity=tn / max(1, tn + fp),
                accuracy=(tp + tn) / len(y))


# ------------------------------------------------------------------ DeLong (Sun & Xu 2014)
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t; return out


def delong(y, p1, p2):
    """two correlated ROC AUCs on the same cases -> (auc1, auc2, z, two-sided p)"""
    y = np.asarray(y).astype(int); order = np.argsort(-y); y = y[order]; P = np.vstack([p1, p2])[:, order]
    m = int(y.sum()); n = len(y) - m
    tx = np.array([_midrank(P[r, :m]) for r in range(2)]); ty = np.array([_midrank(P[r, m:]) for r in range(2)])
    tz = np.array([_midrank(P[r]) for r in range(2)])
    auc = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(auc[0]), float(auc[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def part_delong(S):
    per = pd.read_csv(os.path.join(TAB, "classification_per_image.csv"))
    meth = [c for c in per.columns if c[:2] in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7") or c.startswith("REF")]
    ref_paper = next(c for c in meth if c.startswith("M0")); best = next((c for c in meth if c.startswith("M7")), meth[-1])
    rows = []
    for ref in [ref_paper, best]:
        for c in meth:
            if c == ref:
                continue
            for scope, mk in SCOPES(per):
                g = per[mk]
                if g.y.nunique() < 2:
                    continue
                a1, a2, z, p = delong(g.y.values, g[c].values, g[ref].values)
                rows.append(dict(comparison=f"{c.split(':')[0]}  vs  {ref.split(':')[0]}", method=c, reference=ref, test_set=scope, n=len(g),
                                 auc_method=a1, auc_reference=a2, auc_difference=a1 - a2, z=z, p_value=p, significant_0_05=p < 0.05))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, "extras_delong.csv"), index=False)
    print(t[["comparison", "test_set", "auc_method", "auc_reference", "p_value"]].round(4).to_string(index=False), flush=True)
    S["delong_vs_paper_pooled"] = t[(t.reference == ref_paper) & t.test_set.str.startswith("Pooled")][["method", "auc_difference", "p_value"]].to_dict("records")


# ------------------------------------------------------------------ calibration
def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def part_calib(S):
    per = pd.read_csv(os.path.join(TAB, "classification_per_image.csv"))
    meth = [c for c in per.columns if c[:2] in ("M0", "M1", "M2", "M3", "M4", "M5", "M7")]
    rows = []
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5)); ax.plot([0, 1], [0, 1], "k:")
    for c in meth:
        y, p = per.y.values.astype(float), np.clip(per[c].values.astype(float), 0, 1)
        rows.append(dict(method=c, brier=float(np.mean((p - y) ** 2)), ece_10bins=ece(y, p), mean_predicted=float(p.mean()), observed_rate=float(y.mean())))
        edges = np.linspace(0, 1, 11); mids, obs = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p < b)
            if m.sum() >= 5:
                mids.append(p[m].mean()); obs.append(y[m].mean())
        if c[:2] in ("M0", "M1", "M3", "M7"):
            ax.plot(mids, obs, "o-", label=c.split(":")[0])
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed glaucoma rate"); ax.set_title("Reliability (test set, pooled)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "extras_calibration.png"), dpi=150); plt.close(fig)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, "extras_calibration.csv"), index=False); print(t.round(4).to_string(index=False), flush=True)
    S["calibration"] = rows


# ------------------------------------------------------------------ shared: labelled table + DL features
def labelled(smoke):
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[(m.status == "ok") & m.label.isin(POS + NEG)].copy()
    m["y"] = m.label.isin(POS).astype(int)
    dl = pd.read_csv(os.path.join(TAB, "dl_per_image_all_base.csv"))[["image_id", "vcdr_e2e", "hcdr_e2e", "acdr_e2e"]]
    cl = pd.read_csv(os.path.join(TAB, "baseline_per_image.csv"))[["image_id", "acdr_classical", "ppa_area_px500"]]
    m = m.merge(dl, on="image_id", how="left").merge(cl, on="image_id", how="left")
    for c in ["vcdr_e2e", "hcdr_e2e", "acdr_e2e", "acdr_classical", "ppa_area_px500"]:
        m[c] = m[c].fillna(m[c].median())
    if smoke:
        m = m.groupby(["dataset", "split", "y"]).head(15)
    return m.reset_index(drop=True)


def to_tensor(img, size=None):
    import torch
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(((img[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


# ------------------------------------------------------------------ ISNT rim
def rim_profile(disc, cup, temporal_right):
    ys, xs = np.where(disc)
    if len(ys) < 50:
        return None
    cy, cx = ys.mean(), xs.mean(); h, w = disc.shape
    rr = np.arange(0, max(h, w), 0.5)[None, :]; t = np.radians(np.arange(360))[:, None]
    py = np.clip((cy + rr * np.sin(t)).astype(int), 0, h - 1); px = np.clip((cx + rr * np.cos(t)).astype(int), 0, w - 1)
    din, cin = disc[py, px], cup[py, px]                       # 360 rays x samples
    last = lambda b: np.where(b.any(1), rr[0][b.shape[1] - 1 - np.argmax(b[:, ::-1], 1)], 0.0)
    rd, rc = last(din), last(cin)
    diam = 2 * rd.mean(); rim = np.clip(rd - rc, 0, None) / max(diam, 1)
    ang = np.arange(360)          # image coordinates: y points down -> 90 deg = inferior
    sec = {"I": (ang >= 45) & (ang < 135), "S": (ang >= 225) & (ang < 315)}
    right = (ang < 45) | (ang >= 315); left = (ang >= 135) & (ang < 225)
    sec["T"], sec["N"] = (right, left) if temporal_right else (left, right)
    f = {f"rim_{k}": float(rim[v].mean()) for k, v in sec.items()}
    f["rim_min"] = float(rim.min()); f["rim_min_angle_inferior_or_superior"] = int(sec["I"][rim.argmin()] or sec["S"][rim.argmin()])
    f["isnt_rule_ok"] = int(f["rim_I"] >= f["rim_S"] >= f["rim_N"] >= f["rim_T"])
    f["is_rule_ok"] = int(f["rim_I"] >= f["rim_S"]); f["rim_to_disc_area"] = float((disc & ~cup).sum() / max(1, disc.sum()))
    return f


def part_isnt(S, d, smoke):
    import torch, segmentation_models_pytorch as smp
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = smp.Unet("resnet34", encoder_weights=None, classes=2).to(dev)
    seg.load_state_dict(torch.load(os.path.join(MOD, "seg_crop512_base.pt"), map_location=dev)); seg.eval()
    rows = []
    with torch.no_grad():
        for _, r in tqdm(d.iterrows(), total=len(d), desc="[ISNT] rim widths from DL masks", unit="img", dynamic_ncols=True):
            x = to_tensor(imread(r.crop512))[None].to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                p = (torch.sigmoid(seg(x).float()) + torch.sigmoid(seg(torch.flip(x, [3])).float()).flip(3)) / 2
            p = p[0].cpu().numpy(); disc = p[0] > .5; cup = (p[1] > .5) & disc
            f = rim_profile(disc, cup, r.disc_cx < r.img_w / 2) or {}
            rows.append(dict(image_id=r.image_id, **f))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "extras_isnt_per_image.csv"), index=False)
    d2 = d.merge(R, on="image_id", how="left"); RF = ["rim_I", "rim_S", "rim_N", "rim_T", "rim_min", "rim_to_disc_area"]
    for c in RF + ["isnt_rule_ok", "is_rule_ok"]:
        d2[c] = d2[c].fillna(d2[c].median())
    tr, va, te = d2[d2.split == "train"], d2[d2.split == "val"], d2[d2.split == "test"]
    # descriptive: rim by diagnosis, ISNT violation rates
    desc = []
    for ds, g in list(d2.groupby("dataset")) + [("Pooled", d2)]:
        for c in RF:
            a, b = g[c][g.y == 1], g[c][g.y == 0]
            desc.append(dict(dataset=ds, feature=c, mean_glaucoma=a.mean(), mean_normal=b.mean(), mannwhitney_p=stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan,
                             auc_alone=roc_auc_score(g.y, -g[c]) if g.y.nunique() == 2 else np.nan))
        desc.append(dict(dataset=ds, feature="ISNT rule violated (%)", mean_glaucoma=100 * (1 - g.isnt_rule_ok[g.y == 1].mean()), mean_normal=100 * (1 - g.isnt_rule_ok[g.y == 0].mean())))
    pd.DataFrame(desc).to_csv(os.path.join(TAB, "extras_isnt_by_diagnosis.csv"), index=False)
    models = {"R1 rim features alone (ISNT widths, min rim, rim/disc area)": RF,
              "R2 DL vCDR + rim features": ["vcdr_e2e"] + RF,
              "R3 DL vCDR + hCDR + aCDR + rim features": ["vcdr_e2e", "hcdr_e2e", "acdr_e2e"] + RF}
    res, probs = [], {}
    for name, f in models.items():
        c = lr().fit(tr[f], tr.y); probs[name] = (c.predict_proba(va[f])[:, 1], c.predict_proba(te[f])[:, 1])
    per = pd.read_csv(os.path.join(TAB, "classification_per_image.csv")).set_index("image_id")
    cnn_col = next((c for c in per.columns if c.startswith("M3")), None)
    if cnn_col is not None and os.path.exists(os.path.join(MOD, "cls_resnet50.pt")):
        cva = cnn_scores(va, dev); cte = cnn_scores(te, dev)
        F = ["vcdr_e2e"] + RF
        fu = lr().fit(np.c_[cva, va[F]], va.y)
        probs["R4 Fusion: CNN + DL vCDR + rim features"] = (fu.predict_proba(np.c_[cva, va[F]])[:, 1], fu.predict_proba(np.c_[cte, te[F]])[:, 1])
    for name, (pv, pt) in probs.items():
        thr = youden(va.y, pv)
        for scope, mk in SCOPES(te):
            if te.y.values[mk].min() != te.y.values[mk].max():
                res.append(dict(method=name, test_set=scope, **cls_metrics(te.y.values[mk], pt[mk], thr)))
    res = pd.DataFrame(res); res.to_csv(os.path.join(TAB, "extras_isnt_classification.csv"), index=False)
    print(res.round(3).to_string(index=False), flush=True)
    S["isnt"] = res[res.test_set.str.startswith("Pooled")].to_dict("records")


def cnn_model(dev):
    import torch, timm
    m = timm.create_model("resnet50", pretrained=False, num_classes=1).to(dev)
    m.load_state_dict(torch.load(os.path.join(MOD, "cls_resnet50.pt"), map_location=dev)); m.eval(); return m


def cnn_scores(d, dev, model=None):
    import torch
    model = model or cnn_model(dev); out = []
    with torch.no_grad():
        for i in range(0, len(d), 32):
            x = torch.stack([to_tensor(imread(p), 384) for p in d.crop512.iloc[i:i + 32]]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float())) / 2
            out.append(p.squeeze(1).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ Grad-CAM
def part_gradcam(S, d):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = cnn_model(dev).float()
    acts, grads = {}, {}
    layer = model.layer4
    def fwd(m, i, o):            # tensor hook (safe with timm's in-place ReLUs, unlike module backward hooks)
        acts["a"] = o; o.register_hook(lambda g: grads.__setitem__("g", g))
    layer.register_forward_hook(fwd)
    te = d[d.split == "test"].reset_index(drop=True); rows, cams = [], {}
    yy, xx = np.mgrid[:384, :384]; rad = np.hypot(yy - 192, xx - 192) / (384 / 6)        # crop side = 3 disc diameters
    for _, r in tqdm(te.iterrows(), total=len(te), desc="[Grad-CAM] test images", unit="img", dynamic_ncols=True):
        x = to_tensor(imread(r.crop512), 384)[None].to(dev).requires_grad_(True)
        model.zero_grad(); out = model(x); out[0, 0].backward()
        w = grads["g"].mean((2, 3), keepdim=True); cam = torch.relu((w * acts["a"]).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (384, 384)); cam = cam / cam.max() if cam.max() > 0 else cam
        rows.append(dict(image_id=r.image_id, dataset=r.dataset, y=r.y, prob=float(torch.sigmoid(out)[0, 0]),
                         heat_inside_1disc_radius=float(cam[rad <= 1.0].sum() / max(cam.sum(), 1e-9)),
                         heat_inside_1_5disc_radii=float(cam[rad <= 1.5].sum() / max(cam.sum(), 1e-9)),
                         peak_distance_disc_radii=float(rad.ravel()[cam.argmax()])))
        cams[r.image_id] = cam
    G = pd.DataFrame(rows); G.to_csv(os.path.join(TAB, "extras_gradcam_per_image.csv"), index=False)
    area_1_5 = float((rad <= 1.5).mean())
    summ = G.groupby("y")[["heat_inside_1disc_radius", "heat_inside_1_5disc_radii", "peak_distance_disc_radii"]].mean().reset_index()
    summ["share_of_image_area_inside_1_5_radii"] = area_1_5
    summ.to_csv(os.path.join(TAB, "extras_gradcam_summary.csv"), index=False); print(summ.round(3).to_string(index=False), flush=True)
    tiles = []
    for yv in [1, 0]:
        pick = G[G.y == yv].sort_values("prob", ascending=(yv == 0)).head(4)
        row = []
        for _, r in pick.iterrows():
            im = cv2.resize(imread(te.set_index("image_id").loc[r.image_id, "crop512"]), (384, 384))
            hm = cv2.applyColorMap((cams[r.image_id] * 255).astype(np.uint8), cv2.COLORMAP_JET); v = cv2.addWeighted(im, 0.55, hm, 0.45, 0)
            cv2.putText(v, f"{'glaucoma' if yv else 'normal'} p={r.prob:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            row.append(cv2.resize(v, (256, 256)))
        while len(row) < 4:
            row.append(np.zeros((256, 256, 3), np.uint8))
        tiles.append(np.hstack(row))
    cv2.imencode(".jpg", np.vstack(tiles))[1].tofile(os.path.join(FIG, "extras_gradcam.jpg"))
    S["gradcam"] = summ.to_dict("records")


# ------------------------------------------------------------------ DINOv2 foundation model (frozen) + linear probe
def part_dino(S, d):
    import torch, timm
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "vit_base_patch14_dinov2.lvd142m"
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(d), 32), desc="[DINOv2] frozen features", unit="batch", dynamic_ncols=True):
            x = torch.stack([to_tensor(imread(p), 224) for p in d.crop512.iloc[i:i + 32]]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                f = (model(x).float() + model(torch.flip(x, [3])).float()) / 2
            feats.append(f.cpu().numpy())
    X = np.concatenate(feats); np.save(os.path.join(TAB, "extras_dinov2_features.npy"), X)
    tr, va, te = (d.split == "train").values, (d.split == "val").values, (d.split == "test").values
    best, bc = -1, None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        c = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[tr], d.y[tr])
        a = roc_auc_score(d.y[va], c.predict_proba(X[va])[:, 1])
        if a > best:
            best, bc = a, c
    pv, pt = bc.predict_proba(X[va])[:, 1], bc.predict_proba(X[te])[:, 1]
    T = d[te].copy(); res = []
    probs = {"D1 DINOv2 ViT-B/14 frozen + linear probe": (pv, pt)}
    F = np.c_[pv, d.vcdr_e2e[va]]; Ft = np.c_[pt, d.vcdr_e2e[te]]
    fu = lr().fit(F, d.y[va]); probs["D2 Fusion: DINOv2 probe + DL vCDR"] = (fu.predict_proba(F)[:, 1], fu.predict_proba(Ft)[:, 1])
    for nm, (a, b) in probs.items():
        thr = youden(d.y[va], a)
        for scope, mk in SCOPES(T):
            if T.y.values[mk].min() != T.y.values[mk].max():
                res.append(dict(method=nm, test_set=scope, val_auc=roc_auc_score(d.y[va], a), **cls_metrics(T.y.values[mk], b[mk], thr)))
    res = pd.DataFrame(res); res.to_csv(os.path.join(TAB, "extras_foundation_model.csv"), index=False); print(res.round(3).to_string(index=False), flush=True)
    S["dinov2"] = res[res.test_set.str.startswith("Pooled")].to_dict("records")


# ------------------------------------------------------------------ 5-fold patient-level CV
def part_cv(S, d, smoke, cnn_epochs):
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    isnt_f = os.path.join(TAB, "extras_isnt_per_image.csv")
    RF = ["rim_I", "rim_S", "rim_N", "rim_T", "rim_min", "rim_to_disc_area"]
    if os.path.exists(isnt_f):
        d = d.merge(pd.read_csv(isnt_f)[["image_id"] + RF], on="image_id", how="left")
        for c in RF:
            d[c] = d[c].fillna(d[c].median())
    feats = {"M0 Paper (classical CDR + PPA area)": ["acdr_classical", "ppa_area_px500"], "M1 DL vCDR": ["vcdr_e2e"],
             "M2 DL CDRs + PPA area": ["vcdr_e2e", "hcdr_e2e", "acdr_e2e", "ppa_area_px500"]}
    if os.path.exists(isnt_f):
        feats["R2 DL vCDR + rim features"] = ["vcdr_e2e"] + RF
    rows, oof = [], {k: np.zeros(len(d)) for k in feats}
    groups = d.dataset + "_" + d.patient_id.astype(str)
    folds = list(sgk.split(d, d.y, groups))
    for k, (tri, tei) in enumerate(folds):
        tr, te = d.iloc[tri], d.iloc[tei]
        for name, f in feats.items():
            c = lr().fit(tr[f], tr.y); p = c.predict_proba(te[f])[:, 1]; oof[name][tei] = p
            rows.append(dict(fold=k + 1, method=name, n=len(te), auc=roc_auc_score(te.y, p)))
    if not smoke or cnn_epochs:
        oof["M3 CNN (ResNet-50, retrained per fold)"] = np.zeros(len(d))
        for k, (tri, tei) in enumerate(folds):
            p = cv_cnn(d.iloc[tri], d.iloc[tei], cnn_epochs if cnn_epochs else 25, k)
            oof["M3 CNN (ResNet-50, retrained per fold)"][tei] = p
            rows.append(dict(fold=k + 1, method="M3 CNN (ResNet-50, retrained per fold)", n=len(tei), auc=roc_auc_score(d.y.iloc[tei], p)))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "extras_cv5_folds.csv"), index=False)
    summ = R.groupby("method", sort=False).auc.agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"mean": "auc_mean", "std": "auc_sd", "min": "auc_min", "max": "auc_max"})
    summ["auc_pooled_out_of_fold"] = [roc_auc_score(d.y, oof[m]) for m in summ.method]
    for ds in ["Chaksu", "PAPILA"]:
        mk = (d.dataset == ds).values
        summ[f"auc_oof_{ds}"] = [roc_auc_score(d.y[mk], oof[m][mk]) for m in summ.method]
    summ.to_csv(os.path.join(TAB, "extras_cv5_summary.csv"), index=False); print(summ.round(4).to_string(index=False), flush=True)
    S["cv5"] = summ.to_dict("records")


def cv_cnn(tr, te, epochs, fold):
    import torch, torch.nn as nn, timm
    from torch.utils.data import DataLoader, TensorDataset
    import albumentations as A
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("resnet50", pretrained=True, num_classes=1).to(dev)
    aug = A.Compose([A.HorizontalFlip(p=.5), A.Affine(scale=(.9, 1.1), rotate=(-15, 15), p=.7), A.RandomBrightnessContrast(.2, .2, p=.5), A.HueSaturationValue(8, 12, 8, p=.3)])
    imgs = {p: cv2.resize(imread(p), (384, 384), interpolation=cv2.INTER_AREA) for p in tqdm(tr.crop512.tolist() + te.crop512.tolist(), desc=f"[CV-CNN fold {fold + 1}] loading", unit="img", leave=False, dynamic_ncols=True)}
    pw = torch.tensor((tr.y == 0).sum() / max(1, (tr.y == 1).sum()), device=dev); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); bs = 16; steps = max(1, len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=epochs * steps, pct_start=0.1); scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    P, Y = tr.crop512.values, tr.y.values.astype(np.float32)
    for ep in range(epochs):
        model.train(); idx = np.random.permutation(len(P))
        for b in tqdm(range(steps), desc=f"[CV-CNN fold {fold + 1}/5] epoch {ep + 1}/{epochs}", unit="batch", leave=False, dynamic_ncols=True):
            j = idx[b * bs:(b + 1) * bs]
            x = torch.stack([to_tensor(aug(image=imgs[P[i]])["image"]) for i in j]).to(dev); y = torch.from_numpy(Y[j]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = lossf(model(x).squeeze(1).float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(te), 32):
            x = torch.stack([to_tensor(imgs[p]) for p in te.crop512.iloc[i:i + 32]]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out.append(((torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, [3])).float())) / 2).squeeze(1).cpu().numpy())
    p = np.concatenate(out); print(f"[CV-CNN] fold {fold + 1}/5 AUC {roc_auc_score(te.y, p):.4f}", flush=True)
    del model; torch.cuda.empty_cache(); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="delong,calib,isnt,gradcam,dino,cv"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cv_cnn_epochs", type=int, default=25)
    a = ap.parse_args(); random.seed(0); np.random.seed(0); t0 = time.time()
    S = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), parts={}, errors={})
    d = labelled(a.smoke); print(f"labelled eyes: {len(d)} (glaucoma {d.y.sum()})", flush=True)
    jobs = {"delong": lambda: part_delong(S), "calib": lambda: part_calib(S), "isnt": lambda: part_isnt(S, d, a.smoke),
            "gradcam": lambda: part_gradcam(S, d), "dino": lambda: part_dino(S, d),
            "cv": lambda: part_cv(S, d, a.smoke, 1 if a.smoke else a.cv_cnn_epochs)}
    for p in a.parts.split(","):
        t1 = time.time(); print(f"\n===== part {p} =====", flush=True)
        try:
            jobs[p](); S["parts"][p] = f"ok ({(time.time() - t1) / 60:.1f} min)"
        except Exception as e:
            S["parts"][p] = "FAILED"; S["errors"][p] = traceback.format_exc()[-2000:]; print(S["errors"][p], flush=True)
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    S["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(S, open(os.path.join(ROOT, "logs", "stage9_stats.json" if not a.smoke else "stage9_stats_smoke.json"), "w"), indent=2, default=float)
    print("\nParts:", S["parts"], flush=True)
    if S["errors"]:
        raise SystemExit(f"failed parts: {list(S['errors'])}")


if __name__ == "__main__":
    main()
