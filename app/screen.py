"""Command-line glaucoma screening with the exported ONNX models (no PyTorch needed).

    pip install -r app/requirements.txt
    python app/screen.py photo.jpg [more photos or folders]  ->  prints a summary and writes <photo>_result.json / _overlay.jpg

Same pipeline as the browser demo (docs/): disc localisation, disc/cup segmentation with tight-crop cup refinement,
CDRs, rim widths (ISNT), PPA, peripapillary ring CNN, disc CNN and the calibrated decision model.
Research prototype - not a medical device.
"""
import numpy as np, cv2, json, onnxruntime as ort, sys, glob, os
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
MD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "models")
CFG = json.load(open(os.path.join(MD, "config.json"))); D, Q = CFG["deploy"], CFG["quality"]
def sess(n, suffix=""):
    o = ort.SessionOptions(); o.enable_cpu_mem_arena = False; o.enable_mem_pattern = False
    return ort.InferenceSession(os.path.join(MD, f"{n}{suffix}.onnx"), o, providers=["CPUExecutionProvider"])
def tens(bgr): return ((bgr[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None].copy()
def lcc(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return m.astype(bool) if n <= 1 else lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
def cdr(d, c):
    if d.sum() == 0: return np.nan, np.nan, np.nan
    ys = np.where(d.any(1))[0]; xs = np.where(d.any(0))[0]; cy = np.where(c.any(1))[0]; cx = np.where(c.any(0))[0]
    return ((np.ptp(cy) + 1) if len(cy) else 0) / (np.ptp(ys) + 1), ((np.ptp(cx) + 1) if len(cx) else 0) / (np.ptp(xs) + 1), c.sum() / d.sum()
def crop(img, cx, cy, side, size=512):
    side = int(side); pad = side; p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    y1, x1 = int(cy) + pad - side // 2, int(cx) + pad - side // 2
    return cv2.resize(p[y1:y1 + side, x1:x1 + side], (size, size), interpolation=cv2.INTER_AREA)
def rim(disc, cup, temporal_right):
    ys, xs = np.where(disc)
    if len(ys) < 50: return {}
    cy, cx = ys.mean(), xs.mean(); h, w = disc.shape
    rr = np.arange(0, max(h, w), 0.5)[None, :]; t = np.radians(np.arange(360))[:, None]
    py = np.clip((cy + rr * np.sin(t)).astype(int), 0, h - 1); px = np.clip((cx + rr * np.cos(t)).astype(int), 0, w - 1)
    din, cin = disc[py, px], cup[py, px]
    last = lambda b: np.where(b.any(1), rr[0][b.shape[1] - 1 - np.argmax(b[:, ::-1], 1)], 0.0)
    rd, rc = last(din), last(cin); diam = 2 * rd.mean(); r = np.clip(rd - rc, 0, None) / max(diam, 1); ang = np.arange(360)
    sec = {"I": (ang >= 45) & (ang < 135), "S": (ang >= 225) & (ang < 315)}; right = (ang < 45) | (ang >= 315); left = (ang >= 135) & (ang < 225)
    sec["T"], sec["N"] = (right, left) if temporal_right else (left, right)
    f = {f"rim_{k}": float(r[v].mean()) for k, v in sec.items()}; f["rim_min"] = float(r.min()); f["isnt_rule_ok"] = int(f["rim_I"] >= f["rim_S"] >= f["rim_N"] >= f["rim_T"])
    return f
class Pipe:
    def __init__(s, suffix=""): s.m = {n: sess(n, suffix) for n in ["localiser", "discup", "tight", "ppa", "cnn", "ring"]}
    def run(s, n, x): return s.m[n].run(None, {"input": x})
    def analyse(s, img):
        h, w = img.shape[:2]; r = {}
        sc = 512 / max(h, w); lb = np.zeros((512, 512, 3), np.uint8); rs = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        y0, x0 = (512 - rs.shape[0]) // 2, (512 - rs.shape[1]) // 2; lb[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
        pl = s.run("localiser", tens(lb))[0][0, 0]; r["localiser_confidence"] = float(pl.max()); m = lcc(pl > .5)
        if m.sum() < 10: return {"warning": "optic disc not found"}
        ys, xs = np.where(m); cx, cy = (xs.mean() - x0) / sc, (ys.mean() - y0) / sc; dd = (max(np.ptp(ys), np.ptp(xs)) + 1) / sc; side = 3 * dd
        c3 = crop(img, cx, cy, side); p = s.run("discup", tens(c3))[0][0]
        disc = lcc(p[0] > .5); cup = (p[1] > .5) & disc; cup = lcc(cup) if cup.any() else cup
        v, hc, a = cdr(disc, cup); r["vCDR_base"] = float(v); r["hCDR"] = float(hc); r["area_CDR_base"] = float(a)
        qy, qx = np.where(disc); k = side / 512; cx2, cy2 = cx - side / 2 + qx.mean() * k, cy - side / 2 + qy.mean() * k
        t = s.run("tight", tens(crop(img, cx2, cy2, 1.5 * (max(np.ptp(qy), np.ptp(qx)) + 1) * k)))[0][0]; td = lcc(t[0] > .5); tc = (t[1] > .5) & td
        v2 = cdr(td, lcc(tc) if tc.any() else tc); r["vCDR"] = (v + float(v2[0])) / 2; r["area_CDR"] = (a + float(v2[2])) / 2
        r.update(rim(disc, cup, cx < w / 2))
        q = s.run("ppa", tens(c3))[0][0]; raw = (q[1] > .5) & ~(q[0] > .5); r["PPA_to_disc_area"] = float(raw.sum() / max(1, (q[0] > .5).sum()))
        i3 = cv2.resize(c3, (384, 384), interpolation=cv2.INTER_AREA); yy, xx = np.mgrid[:384, :384]; rr = np.hypot(yy - 192, xx - 192) / 64
        r["ring"] = float(s.run("ring", tens((i3 * ((rr >= .9) & (rr <= 2.2))[..., None]).astype(np.uint8)))[0].ravel()[0])
        lg, feat = s.run("cnn", tens(i3)); r["cnn_logit"] = float(lg.ravel()[0])
        f = {"cnn_logit": r["cnn_logit"], "vcdr_e2e": r["vCDR_base"], **{k: r.get(k, np.nan) for k in D["features"] if k.startswith("rim_")}}
        z = float(np.dot(np.nan_to_num((np.array([f[k] for k in D["features"]]) - D["mean"]) / D["scale"]), D["coef"]) + D["intercept"]); r["prob"] = 1 / (1 + np.exp(-z))
        eps = 1e-6; dpred0 = p[0] > .5; band = cv2.dilate(dpred0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        ent = lambda x: -(x * np.log(x + eps) + (1 - x) * np.log(1 - x + eps)); dpred = p[0] > .5
        r["disc_entropy"] = float(ent(p[0])[band].mean()); r["cup_entropy"] = float(ent(p[1])[band].mean()); r["disc_area_frac"] = float(dpred.mean())
        r["n_disc_components"] = int(cv2.connectedComponents(dpred.astype(np.uint8))[0] - 1)
        return r

def main(paths):
    files = []
    for a in paths:
        files += sorted(f for f in glob.glob(os.path.join(a, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))) if os.path.isdir(a) else [a]
    if not files:
        print(__doc__); return
    p = Pipe()
    for f in files:
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{f}: cannot read image"); continue
        r = p.analyse(img); r["file"] = os.path.basename(f)
        if "warning" not in r:
            r["decision"] = "REFER (glaucoma suspected)" if r["prob"] >= D["threshold"] else "no glaucoma signs detected"
            w = []
            if r["disc_entropy"] > Q["disc_entropy"] or r["cup_entropy"] > Q["cup_entropy"]: w.append("uncertain disc/cup boundary")
            if not (Q["disc_area_lo"] <= r["disc_area_frac"] <= Q["disc_area_hi"]) or r["n_disc_components"] != 1: w.append("disc size/shape unusual")
            if r["localiser_confidence"] < 0.8: w.append("low localisation confidence")
            r["quality_warning"] = "; ".join(w) or "none"
            print(f"{r['file']}: {r['decision']}  p={r['prob']:.3f}  vCDR={r['vCDR']:.3f}  area CDR={r['area_CDR']:.3f}  ring={r['ring']:.3f}  quality: {r['quality_warning']}")
        else:
            print(f"{r['file']}: {r['warning']}")
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}, open(os.path.splitext(f)[0] + "_result.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
