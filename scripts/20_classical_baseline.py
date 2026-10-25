"""Stage 3: re-implementation of Kumar & Kumar (2022), Multimedia Tools & Applications 81:13513-13535.
Every step follows the paper's Section 3; where the paper is silent, the choice is written in PARAMS and logged.

  3.1 ROI: image resized to 500 px height; horizontal OD location = peak of (|vertical edges| - |horizontal edges|)
      inside a full-height window of width 20 sliding left->right; vertical location = peak of (edges + brightness)
      inside a square window (~OD diameter) sliding top->bottom at that column; 140x140 crop around the centre.
  OC: green channel, standardisation I2 = I1 - (mean + std), K-means (k=2), largest bright component, direct
      least-squares ellipse fit (Fitzgibbon = cv2.fitEllipseDirect).
  OD: red channel, top-hat, closing, histogram equalisation, polar transform, strip-wise Otsu binarisation,
      inverse polar transform, morphological edge, direct least-squares ellipse fit.
  CDR = area(OC ellipse) / area(OD ellipse)  (+ we also report vertical CDR from the ellipses).
  PPA: blue channel, OD ellipse filled black, knowledge-based temporal mask, binarisation, white-pixel area;
      K-means (k=2) on areas separates PPA from residue (paper: centres 1252 / 144 px).
  Classifier: linear SVM, 5-fold CV on train, evaluated on the held-out test split.
Evaluated against the expert consensus masks and the datasets' own labels (Chaksu, PAPILA).
Outputs: results/tables/baseline_*.csv, results/figures/baseline_examples_*.jpg, logs/stage3_stats.json
"""
import os, json, math, random, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures")
for d in [TAB, FIG, os.path.join(ROOT, "logs")]:
    os.makedirs(d, exist_ok=True)
PARAMS = dict(resize_height=500, h_window_width=20, roi=140, kmeans_attempts=5, polar_strips=16,
              tophat_se=25, close_se=5, ppa_ring=(1.0, 1.6), ppa_wedge_deg=120,
              cup_kmeans_inside_detected_disc=True, std_instead_of_variance="paper writes 'variance' but subtracting the variance of 8-bit data blanks the image; std used",
              temporal_side="side of the disc facing the image centre (macula side)")


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def fov_mask(img):
    m = (img[..., 2] > 20).astype(np.uint8)
    return cv2.erode(m, np.ones((15, 15), np.uint8))


def locate_od(img500):
    g = cv2.cvtColor(img500, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fm = fov_mask(img500).astype(np.float32)
    k = np.array([[1, 0, -1]], np.float32)
    ev = np.abs(cv2.filter2D(g, -1, k)) * fm          # vertical edges
    eh = np.abs(cv2.filter2D(g, -1, k.T)) * fm        # horizontal edges
    W = PARAMS["h_window_width"]
    col_score = np.convolve(ev.sum(0) - eh.sum(0), np.ones(W), "same")
    hx = int(np.argmax(col_score))
    D = 70                                             # ~OD diameter at 500 px height
    x0, x1 = max(0, hx - D // 2), min(g.shape[1], hx + D // 2)
    bright = (g * fm)[:, x0:x1].sum(1); edges = (ev + eh)[:, x0:x1].sum(1)
    norm = lambda v: (v - v.min()) / (np.ptp(v) + 1e-6)
    row_score = np.convolve(norm(bright) + norm(edges), np.ones(D), "same")
    vy = int(np.argmax(row_score))
    return hx, vy


def fit_ellipse_mask(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, np.zeros_like(mask, np.uint8)
    c = max(cs, key=cv2.contourArea)
    if len(c) < 6:
        return None, np.zeros_like(mask, np.uint8)
    e = cv2.fitEllipseDirect(c)
    out = np.zeros(mask.shape, np.uint8); cv2.ellipse(out, e, 1, -1)
    return e, out


def kmeans2(values):
    Z = values.reshape(-1, 1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, lab, cen = cv2.kmeans(Z, 2, None, crit, PARAMS["kmeans_attempts"], cv2.KMEANS_PP_CENTERS)
    return lab.reshape(values.shape), cen.ravel()


def keep_central_component(mask):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(np.uint8)
    h, w = mask.shape; c = np.array([w / 2, h / 2])
    best = min(range(1, n), key=lambda i: np.linalg.norm(cen[i] - c) - 0.02 * st[i, cv2.CC_STAT_AREA])
    return (lab == best).astype(np.uint8)


def segment_cup(roi, disc_fill):
    """Paper: standardise green channel, K-means (k=2), crop around the OD, binarise, ellipse fit.
    The clustering is done on the pixels inside the detected OD (otherwise k=2 separates disc from background)."""
    g = roi[..., 1].astype(np.float32)
    I2 = g - (g.mean() + g.std())
    inside = disc_fill > 0
    if inside.sum() < 30:
        return None, np.zeros_like(disc_fill)
    lab_in, cen = kmeans2(I2[inside])
    cup = np.zeros(g.shape, np.uint8); cup[inside] = (lab_in == int(np.argmax(cen))).astype(np.uint8)
    cup = cv2.morphologyEx(cup, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cup = keep_central_component(cup)
    return fit_ellipse_mask(cup)


def segment_disc(roi):
    r = roi[..., 2]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PARAMS["tophat_se"],) * 2)
    th = cv2.morphologyEx(r, cv2.MORPH_TOPHAT, se)
    enh = cv2.add(r, th)                                # top-hat used to boost foreground/background contrast
    enh = cv2.morphologyEx(enh, cv2.MORPH_CLOSE, np.ones((PARAMS["close_se"],) * 2, np.uint8))   # removes dark vessels
    enh = cv2.equalizeHist(enh)
    h, w = enh.shape; c = (w / 2, h / 2); R = w / 2
    polar = cv2.warpPolar(enh, (int(R), 360), c, R, cv2.WARP_POLAR_LINEAR)     # rows = angle, cols = radius
    binp = np.zeros_like(polar)
    S = PARAMS["polar_strips"]
    for i in range(S):
        a, b = i * 360 // S, (i + 1) * 360 // S
        strip = polar[a:b]
        _, t = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binp[a:b] = t
    back = cv2.warpPolar(binp, (w, h), c, R, cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
    disc = keep_central_component(back > 127)
    edge = disc - cv2.erode(disc, np.ones((3, 3), np.uint8))
    return fit_ellipse_mask(edge | disc)


def ppa_area(roi, disc_e, disc_fill, temporal_right):
    b = roi[..., 0].copy()
    b[disc_fill > 0] = 0
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    (cx, cy), (MA, ma), ang = disc_e
    r_d = max(MA, ma) / 2
    rr = np.hypot(xx - cx, yy - cy)
    theta = np.degrees(np.arctan2(yy - cy, xx - cx))           # 0 deg = right
    centre = 0 if temporal_right else 180
    dth = np.abs((theta - centre + 180) % 360 - 180)
    lo, hi = PARAMS["ppa_ring"]
    mask = (rr >= r_d * lo) & (rr <= r_d * hi) & (dth <= PARAMS["ppa_wedge_deg"] / 2) & (disc_fill == 0)
    if mask.sum() < 20:
        return 0, mask
    vals = b[mask]
    t, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = (b > t) & mask
    return int(white.sum()), white


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def ellipse_vcdr(cup_fill, disc_fill):
    def height(m):
        ys = np.where(m.any(1))[0]
        return (ys.max() - ys.min() + 1) if len(ys) else 0
    hd = height(disc_fill)
    return height(cup_fill) / hd if hd else np.nan


def run_one(row):
    try:
        img = imread(os.path.join(ROOT, row["full512"]))           # letter-boxed 512 image (same for all methods)
        gt = imread(os.path.join(ROOT, row["full512_mask"]), cv2.IMREAD_GRAYSCALE)
        # undo letterbox padding so the aspect ratio is the original one, then resize to 500 px height
        ys, xs = np.where(img.max(2) > 5)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        img, gt = img[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        s = PARAMS["resize_height"] / img.shape[0]
        img500 = cv2.resize(img, (round(img.shape[1] * s), PARAMS["resize_height"]), interpolation=cv2.INTER_AREA)
        gt500 = cv2.resize(gt, (img500.shape[1], img500.shape[0]), interpolation=cv2.INTER_NEAREST)
        hx, vy = locate_od(img500)
        gys, gxs = np.where(gt500 >= 128)
        gcx, gcy = gxs.mean(), gys.mean()
        loc_err = float(np.hypot(hx - gcx, vy - gcy))
        disc_r_gt = (gxs.max() - gxs.min() + 1) / 2
        R = PARAMS["roi"] // 2
        pad = cv2.copyMakeBorder(img500, R, R, R, R, cv2.BORDER_CONSTANT); gpad = cv2.copyMakeBorder(gt500, R, R, R, R, cv2.BORDER_CONSTANT)
        roi = pad[vy:vy + 2 * R, hx:hx + 2 * R]; groi = gpad[vy:vy + 2 * R, hx:hx + 2 * R]
        de, dfill = segment_disc(roi)
        ce, cfill = segment_cup(roi, dfill) if de is not None else (None, None)
        if de is None or ce is None:
            raise ValueError("ellipse fit failed")
        cfill = cfill & dfill
        acdr = cfill.sum() / max(1, dfill.sum()); vcdr = ellipse_vcdr(cfill, dfill)
        temporal_right = hx < img500.shape[1] / 2          # disc left of centre -> macula/temporal side is to its right
        ppa, _ = ppa_area(roi, de, dfill, temporal_right)
        gdisc, gcup = groi >= 128, groi >= 255
        return dict(image_id=row["image_id"], ok=True, loc_err_px500=loc_err, loc_err_in_disc_radii=loc_err / disc_r_gt,
                    loc_hit=bool(loc_err < disc_r_gt), dice_disc=dice(dfill > 0, gdisc), dice_cup=dice(cfill > 0, gcup),
                    acdr_classical=acdr, vcdr_classical=vcdr, ppa_area_px500=ppa, temporal_right=temporal_right)
    except Exception as e:
        return dict(image_id=row["image_id"], ok=False, error=f"{type(e).__name__}: {e}")


def metrics(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    sen = tp / (tp + fn) if tp + fn else np.nan; spe = tn / (tn + fp) if tn + fp else np.nan
    return dict(n=len(y), TP=tp, TN=tn, FP=fp, FN=fn, sensitivity=sen, specificity=spe,
                accuracy=(tp + tn) / len(y) if len(y) else np.nan, balanced_accuracy=np.nanmean([sen, spe]))


def main():
    t0 = datetime.datetime.now()
    df = pd.read_csv(os.path.join(ROOT, "processed", "master.csv"))
    df = df[df.status == "ok"].copy()
    rows = df.to_dict("records")
    out = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2)) as ex:
        futs = [ex.submit(run_one, r) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Classical pipeline", unit="img", dynamic_ncols=True):
            out.append(f.result())
    res = df.merge(pd.DataFrame(out), on="image_id", how="left")
    res.to_csv(os.path.join(TAB, "baseline_per_image.csv"), index=False)
    ok = res[res.ok == True].copy()

    # --- segmentation / localisation quality per dataset
    seg = ok.groupby("dataset").agg(images=("image_id", "count"), od_localisation_hit_rate=("loc_hit", "mean"),
                                    dice_disc_mean=("dice_disc", "mean"), dice_cup_mean=("dice_cup", "mean"),
                                    vcdr_mae=("vcdr_classical", lambda s: np.nan), acdr_mae=("acdr_classical", lambda s: np.nan))
    for ds, g in ok.groupby("dataset"):
        seg.loc[ds, "vcdr_mae"] = (g.vcdr_classical - g.vcdr).abs().mean()
        seg.loc[ds, "acdr_mae"] = (g.acdr_classical - g.acdr).abs().mean()
    seg["failures"] = res[res.ok != True].groupby("dataset").size().reindex(seg.index).fillna(0).astype(int)
    seg = seg.reset_index(); seg.to_csv(os.path.join(TAB, "baseline_segmentation.csv"), index=False)

    # --- PPA K-means on areas (paper: residue vs PPA) using train split
    tr = ok[ok.split == "train"]
    labk, cen = kmeans2(tr.ppa_area_px500.values.astype(np.float32))
    cen = np.sort(cen); ppa_thr = float(cen.mean())
    ok["ppa_present"] = ok.ppa_area_px500 > ppa_thr

    # --- classification on datasets with labels
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    lab = ok[ok.label.isin(["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT", "NORMAL", "HEALTHY"])].copy()
    lab["y"] = lab.label.isin(["GLAUCOMA", "GLAUCOMA SUSPECT", "SUSPECT"]).astype(int)
    rows_cls = []
    for scope, g in [("Chaksu", lab[lab.dataset == "Chaksu"]), ("PAPILA", lab[lab.dataset == "PAPILA"]), ("Chaksu+PAPILA", lab)]:
        trn, tst = g[g.split.isin(["train", "val"])], g[g.split == "test"]
        if trn.y.nunique() < 2 or tst.y.nunique() < 2:
            continue
        X = ["acdr_classical", "ppa_area_px500"]
        # (a) paper decision table, threshold 0.25 on area CDR, PPA by K-means
        p_rule = ((tst.acdr_classical > 0.25) | tst.ppa_present).astype(int)
        rows_cls.append(dict(scope=scope, method="Paper decision table (CDR>0.25 or PPA)", **metrics(tst.y, p_rule), auc=np.nan))
        # (b) paper SVM: linear kernel on [CDR, PPA area], 5-fold CV on train
        svm = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=True, random_state=0))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        cvp = cross_val_predict(svm, trn[X], trn.y, cv=cv)
        svm.fit(trn[X], trn.y); pt = svm.predict(tst[X]); pr = svm.predict_proba(tst[X])[:, 1]
        rows_cls.append(dict(scope=scope, method="Paper SVM (linear, CDR + PPA area)", **metrics(tst.y, pt),
                             auc=roc_auc_score(tst.y, pr), cv5_train_accuracy=float((cvp == trn.y).mean())))
        # (c) same SVM but with the EXPERT CDR (upper bound for any CDR-based method)
        svm2 = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=True, random_state=0))
        svm2.fit(trn[["vcdr"]], trn.y); pr2 = svm2.predict_proba(tst[["vcdr"]])[:, 1]
        rows_cls.append(dict(scope=scope, method="Reference: SVM on expert vertical CDR", **metrics(tst.y, svm2.predict(tst[["vcdr"]])),
                             auc=roc_auc_score(tst.y, pr2)))
    cls = pd.DataFrame(rows_cls); cls.to_csv(os.path.join(TAB, "baseline_classification.csv"), index=False)
    ok.to_csv(os.path.join(TAB, "baseline_per_image.csv"), index=False)

    # --- example figure per dataset
    for ds, g in ok.groupby("dataset"):
        ex = g.sample(min(8, len(g)), random_state=1)
        tiles = []
        for _, r in ex.iterrows():
            im = imread(os.path.join(ROOT, r.full512))
            cv2.putText(im, f"Dice OD {r.dice_disc:.2f} OC {r.dice_cup:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(im, f"CDR cls {r.acdr_classical:.2f} / expert {r.acdr:.2f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(cv2.resize(im, (256, 256)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))[1].tofile(
            os.path.join(FIG, f"baseline_examples_{ds}.jpg"))
    stats = dict(started=t0.isoformat(timespec="seconds"), finished=datetime.datetime.now().isoformat(timespec="seconds"),
                 params=PARAMS, images=len(res), succeeded=int(res.ok.eq(True).sum()),
                 failed=int(res.ok.ne(True).sum()), error_examples=res[res.ok != True].get("error", pd.Series(dtype=str)).dropna().value_counts().head(8).to_dict(),
                 ppa_kmeans_centres_px500=[float(c) for c in cen], ppa_threshold=ppa_thr,
                 paper_ppa_centres=[144, 1252], segmentation=seg.to_dict("records"), classification=cls.to_dict("records"))
    json.dump(stats, open(os.path.join(ROOT, "logs", "stage3_stats.json"), "w"), indent=2, default=float)
    print(seg.to_string()); print(cls.to_string())


if __name__ == "__main__":
    main()
