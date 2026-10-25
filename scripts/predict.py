"""Glaucoma screening tool - run on ANY fundus photo(s).

Usage:  python scripts/predict.py           (processes every image in predict_in/)
        python scripts/predict.py <image or folder> [...]
For each photo: optic disc found automatically -> disc / cup / PPA outlined -> vertical, horizontal, area and soft CDR,
neuroretinal rim widths (ISNT), PPA extent -> glaucoma probability (calibrated fusion of CNN + CDR + rim, fitted on the
validation split) -> Grad-CAM heat map -> image-quality / out-of-distribution warning.
Outputs in predict_out/: <name>_report.html, <name>_overlay.png, <name>_gradcam.png, and summary.csv
NOT a medical device: research prototype; a clinician must confirm any finding.
"""
import os, sys, glob, json, html, datetime, importlib.util, base64
import numpy as np, pandas as pd, cv2, torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "models"); IN = os.path.join(ROOT, "predict_in"); OUT = os.path.join(ROOT, "predict_out")
os.makedirs(IN, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S = _load("seg30", "30_train_segmentation.py")
X = _load("extras80", "80_reviewer_extras.py")


class Models:
    def __init__(s, dev):
        import segmentation_models_pytorch as smp, timm
        s.dev = dev
        def ld(m, f):
            m.load_state_dict(torch.load(os.path.join(MOD, f), map_location=dev)); return m.to(dev).eval()
        s.loc = ld(S.build("resnet34", 1, False), "seg_full512_base.pt")
        s.seg = ld(S.build("resnet34", 2, False), "seg_crop512_base.pt")
        s.ppa = ld(smp.Unet("resnet34", encoder_weights=None, classes=2), "ppa_supervised.pt") if os.path.exists(os.path.join(MOD, "ppa_supervised.pt")) else None
        s.strong = ld(smp.UnetPlusPlus("resnet34", encoder_weights=None, classes=2), "seg_tight_unetpp.pt") if os.path.exists(os.path.join(MOD, "seg_tight_unetpp.pt")) else None
        s.cnn = ld(timm.create_model("resnet50", pretrained=False, num_classes=1), "cls_resnet50.pt")
        # peripapillary-ring CNN (stage 8): sees only the band 0.9-2.2 disc radii around the disc; learned from the diagnosis
        s.ring = ld(timm.create_model("resnet18", pretrained=False, num_classes=1), "cls_ring_resnet18.pt") if os.path.exists(os.path.join(MOD, "cls_ring_resnet18.pt")) else None
        s.deploy = json.load(open(os.path.join(MOD, "deploy.json"))) if os.path.exists(os.path.join(MOD, "deploy.json")) else None
        s.quality = json.load(open(os.path.join(MOD, "quality.json"))) if os.path.exists(os.path.join(MOD, "quality.json")) else None


def gradcam(model, img384, dev):
    acts, grads = {}, {}
    def fwd(m, i, o):
        acts["a"] = o; o.register_hook(lambda g: grads.__setitem__("g", g))
    h = model.layer4.register_forward_hook(fwd)
    x = S.to_tensor(img384)[None].to(dev).requires_grad_(True); model.zero_grad(); out = model.float()(x); out[0, 0].backward(); h.remove()
    cam = torch.relu((grads["g"].mean((2, 3), keepdim=True) * acts["a"]).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (384, 384)); return cam / cam.max() if cam.max() > 0 else cam, float(out[0, 0].detach())


def crop(img, cx, cy, side, size=512):
    pad = int(side); p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    y1, x1 = int(cy) + pad - int(side) // 2, int(cx) + pad - int(side) // 2
    return cv2.resize(p[y1:y1 + int(side), x1:x1 + int(side)], (size, size), interpolation=cv2.INTER_AREA)


def analyse(path, M):
    img = S.imread(path)
    if img is None:
        raise ValueError("cannot read image")
    h, w = img.shape[:2]; r = {"file": os.path.basename(path), "image_size": f"{w}x{h}"}
    # 1 locate the disc
    s = 512 / max(h, w); lb = np.zeros((512, 512, 3), np.uint8); rs = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (512 - rs.shape[0]) // 2, (512 - rs.shape[1]) // 2; lb[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    pl = S.predict(M.loc, lb, M.dev)[0]; r["localiser_confidence"] = float(pl.max()); m = S.largest_cc(pl > .5)
    if m.sum() < 10:
        r["warning"] = "optic disc not found - image may not be a fundus photo or is of very poor quality"; return r, None
    ys, xs = np.where(m); cx, cy = (xs.mean() - x0) / s, (ys.mean() - y0) / s; dd = (max(np.ptp(ys), np.ptp(xs)) + 1) / s
    side = 3 * dd; c3 = crop(img, cx, cy, side)
    # 2 disc / cup
    p = S.predict(M.seg, c3, M.dev); disc = S.largest_cc(p[0] > .5); cup = (p[1] > .5) & disc; cup = S.largest_cc(cup) if cup.any() else cup
    r["vCDR"], r["hCDR"], r["area_CDR"] = [round(float(v), 3) for v in S.masks_to_cdr(disc, cup)]
    vcdr_base = float(S.masks_to_cdr(disc, cup)[0])      # the decision model was fitted on this (stage-4) vCDR
    r["soft_area_CDR"] = round(float((p[1] * p[0]).sum() / max(p[0].sum(), 1e-6)), 3)
    if M.strong is not None:
        qy, qx = np.where(disc); k = side / 512; cx2, cy2 = cx - side / 2 + qx.mean() * k, cy - side / 2 + qy.mean() * k
        t = S.predict(M.strong, crop(img, cx2, cy2, 1.5 * (max(np.ptp(qy), np.ptp(qx)) + 1) * k), M.dev); td = S.largest_cc(t[0] > .5); tc = (t[1] > .5) & td
        v2 = S.masks_to_cdr(td, S.largest_cc(tc) if tc.any() else tc)
        r["vCDR_strong"] = round(float(v2[0]), 3); r["vCDR"] = round((r["vCDR"] + float(v2[0])) / 2, 3); r["area_CDR"] = round((r["area_CDR"] + float(v2[2])) / 2, 3)
    temporal_right = cx < w / 2
    rim = X.rim_profile(disc, cup, temporal_right) or {}
    r.update({k: round(v, 3) if isinstance(v, float) else v for k, v in rim.items()})
    # 3 PPA
    if M.ppa is not None:
        q = S.predict(M.ppa, c3, M.dev); raw = (q[1] > .5) & ~(q[0] > .5)
        r["PPA_to_disc_area"] = round(float(raw.sum() / max(1, (q[0] > .5).sum())), 3)
        ppa = S.largest_cc(raw) if raw.sum() > 50 else np.zeros_like(disc)      # outline only a real region, not specks
        r["PPA_outline"] = "drawn (red)" if ppa.any() else "no atrophy region found by the pixel model"
    else:
        ppa = np.zeros_like(disc)
    if getattr(M, "ring", None) is not None:      # ring CNN: atrophy-related appearance around the disc (works without an outline)
        i3 = cv2.resize(c3, (384, 384), interpolation=cv2.INTER_AREA); yy, xx = np.mgrid[:384, :384]; rr = np.hypot(yy - 192, xx - 192) / (384 / 6)
        x = S.to_tensor((i3 * ((rr >= .9) & (rr <= 2.2))[..., None]).astype(np.uint8))[None].to(M.dev)
        with torch.no_grad():
            pr = float((torch.sigmoid(M.ring(x)) + torch.sigmoid(M.ring(torch.flip(x, [3])))).item() / 2)
        r["peripapillary_ring_score"] = round(pr, 3)
        r["peripapillary_changes"] = "suggestive of glaucoma-related changes" if pr >= 0.631 else "not suggestive"

    # 4 CNN + Grad-CAM
    i384 = cv2.resize(c3, (384, 384), interpolation=cv2.INTER_AREA); cam, logit = gradcam(M.cnn, i384, M.dev)
    with torch.no_grad():                                 # mirror-image prediction, averaged as when the decision model was fitted
        logit = (logit + float(M.cnn(torch.flip(S.to_tensor(i384)[None].to(M.dev), [3])).float().item())) / 2
    r["cnn_logit"] = round(logit, 3)
    # 5 decision
    if M.deploy:
        D = M.deploy; feats = {"cnn_logit": logit, "vcdr_e2e": vcdr_base, **{k: r.get(k, np.nan) for k in D["features"] if k.startswith("rim_")}}
        x = (np.array([feats[f] for f in D["features"]], float) - np.array(D["mean"])) / np.array(D["scale"])
        z = float(np.dot(np.nan_to_num(x), D["coef"]) + D["intercept"]); prob = 1 / (1 + np.exp(-z))
        r["glaucoma_probability"] = round(prob, 3); r["decision"] = "REFER (glaucoma suspected)" if prob >= D["threshold"] else "no glaucoma signs detected"
        r["decision_threshold"] = round(D["threshold"], 3)
    else:
        prob = 1 / (1 + np.exp(-logit)); r["glaucoma_probability"] = round(float(prob), 3); r["decision"] = "REFER" if prob >= 0.5 else "no glaucoma signs detected"
    r["paper_rule_area_CDR_gt_0.25"] = bool(r["area_CDR"] > 0.25)
    # 6 quality / out-of-distribution
    qf = X_quality(p); r.update({k: round(v, 4) if isinstance(v, float) else v for k, v in qf.items()})
    warn = []
    if M.quality:
        Q = M.quality
        if qf["disc_entropy"] > Q["disc_entropy"] or qf["cup_entropy"] > Q["cup_entropy"]:
            warn.append("model is uncertain about the disc/cup boundary (blur, glare, poor focus or unusual image)")
        if not (Q["disc_area_lo"] <= qf["disc_area_frac"] <= Q["disc_area_hi"]) or qf["n_disc_components"] != 1:
            warn.append("disc size/shape outside the range seen in training")
    if r["localiser_confidence"] < 0.8:
        warn.append("disc localisation confidence is low")
    r["quality_warning"] = "; ".join(warn) if warn else "none"
    vis = c3.copy()
    for msk, col in [(disc, (255, 255, 0)), (cup, (255, 0, 255)), (ppa, (0, 0, 255))]:
        cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, cs, -1, col, 2)
    heat = cv2.addWeighted(i384, 0.55, cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.45, 0)
    return r, (vis, heat)


def X_quality(p):
    eps = 1e-6; disc = p[0] > .5; band = cv2.dilate(disc.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    ent = lambda q: -(q * np.log(q + eps) + (1 - q) * np.log(1 - q + eps))
    return dict(disc_entropy=float(ent(p[0])[band].mean()) if band.any() else 1.0, cup_entropy=float(ent(p[1])[band].mean()) if band.any() else 1.0,
                disc_area_frac=float(disc.mean()), n_disc_components=int(cv2.connectedComponents(disc.astype(np.uint8))[0] - 1))


def b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def report(r, ims, stem):
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in r.items())
    pics = "" if ims is None else f'<img src="data:image/png;base64,{b64(ims[0])}" width="384"> <img src="data:image/png;base64,{b64(ims[1])}" width="384">'
    col = "#b00020" if str(r.get("decision", "")).startswith("REFER") else "#1b5e20"
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(stem)}</title>
<style>body{{font-family:Segoe UI,Arial;margin:24px;max-width:900px}}td{{padding:3px 10px;border-bottom:1px solid #ddd}}h2{{color:{col}}}</style>
<h1>Glaucoma screening report</h1><p>{html.escape(r['file'])} &middot; {datetime.datetime.now():%Y-%m-%d %H:%M}</p>
<h2>{html.escape(str(r.get('decision', r.get('warning', ''))))} &mdash; probability {r.get('glaucoma_probability', 'n/a')}</h2>
<p><b>Quality warning:</b> {html.escape(str(r.get('quality_warning', r.get('warning', ''))))}</p>{pics}
<p>Left: disc (cyan), cup (magenta), PPA (red; drawn only when the pixel model finds an atrophy region). Right: where the CNN looked (Grad-CAM).</p><table>{rows}</table>
<p style="color:#777">Research prototype (MNNIT Allahabad). Not a medical device - findings must be confirmed by an ophthalmologist.</p>"""
    open(os.path.join(OUT, stem + "_report.html"), "w", encoding="utf-8").write(page)


def main():
    args = sys.argv[1:] or [IN]; files = []
    for a in args:
        files += sorted(f for f in glob.glob(os.path.join(a, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))) if os.path.isdir(a) else [a]
    if not files:
        print(f"No images found. Put fundus photos into {IN} and run again."); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); M = Models(dev); rows = []
    for f in tqdm(files, desc="Analysing fundus photos", unit="img", dynamic_ncols=True):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            r, ims = analyse(f, M)
            if ims is not None:
                cv2.imencode(".png", ims[0])[1].tofile(os.path.join(OUT, stem + "_overlay.png")); cv2.imencode(".png", ims[1])[1].tofile(os.path.join(OUT, stem + "_gradcam.png"))
            report(r, ims, stem)
        except Exception as e:
            r = dict(file=os.path.basename(f), error=f"{type(e).__name__}: {e}")
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(f"\nDone: {len(rows)} image(s). Reports in {OUT}")
    for r in rows:
        print(f"  {r['file']}: {r.get('decision', r.get('error', r.get('warning')))}  p={r.get('glaucoma_probability', '-')}  vCDR={r.get('vCDR', '-')}  quality: {r.get('quality_warning', '-')}")


if __name__ == "__main__":
    main()
