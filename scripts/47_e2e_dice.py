"""Stage 13c (verification): FULLY END-TO-END disc/cup Dice and IoU in the ORIGINAL photo coordinates.

Why: the Dice/IoU reported in stages 4 and 12 are measured on crops centred on the EXPERT disc (common practice, but it
hides localisation error). The classical baseline Dice is end-to-end (its own localisation). For a like-for-like
comparison this script runs the complete automatic pipeline on every TEST image:
  localiser (whole photo) -> coarse disc on a 3x crop -> tight 1.5x crop from the original -> U-Net++ -> masks pasted back
into the original image -> Dice / IoU against the expert consensus masks at full resolution.
Models: base localiser + base coarse segmenter + seg_tight_unetpp.pt (all trained without the test split).
Outputs: results/tables/e2e_dice_per_image.csv, results/tables/e2e_dice.csv, logs/stage13c_stats.json
"""
import os, json, datetime, importlib.util
import numpy as np, pandas as pd, cv2, torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); MOD = os.path.join(ROOT, "models")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


B = _load("build10", "10_build_master.py"); S = _load("seg30", "30_train_segmentation.py"); G = _load("seg45", "45_seg_strong.py")


def paste(mask512, cx, cy, side, shape):
    """inverse of a square crop of `side` px centred at (cx, cy), resized to 512: put the mask back into the original frame"""
    side = int(side); full = np.zeros(shape[:2], np.float32)
    m = cv2.resize(mask512.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    x1, y1 = int(cx) - side // 2, int(cy) - side // 2
    sx0, sy0 = max(0, -x1), max(0, -y1); dx0, dy0 = max(0, x1), max(0, y1)
    w = min(side - sx0, shape[1] - dx0); h = min(side - sy0, shape[0] - dy0)
    if w > 0 and h > 0:
        full[dy0:dy0 + h, dx0:dx0 + w] = m[sy0:sy0 + h, sx0:sx0 + w]
    return full > 0.5


def dice_iou(a, b):
    s = a.sum() + b.sum(); u = (a | b).sum(); i = (a & b).sum()
    return (2 * i / s if s else 1.0), (i / u if u else 1.0)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", dev, flush=True)
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[(m.status == "ok") & (m.split == "test")]
    recs = []
    for fn in [B.discover_riga, B.discover_chaksu, B.discover_papila]:
        recs += fn()[0]
    keep = set(m.image_id); recs = {r["image_id"]: r for r in recs if r["image_id"] in keep}
    loc = S.build("resnet34", 1, False).to(dev); loc.load_state_dict(torch.load(os.path.join(MOD, "seg_full512_base.pt"), map_location=dev)); loc.eval()
    coarse = S.build("resnet34", 2, False).to(dev); coarse.load_state_dict(torch.load(os.path.join(MOD, "seg_crop512_base.pt"), map_location=dev)); coarse.eval()
    tight = smp.UnetPlusPlus("resnet34", encoder_weights=None, classes=2).to(dev); tight.load_state_dict(torch.load(os.path.join(MOD, "seg_tight_unetpp.pt"), map_location=dev)); tight.eval()
    rows = []
    for _, r in tqdm(m.iterrows(), total=len(m), desc="End-to-end Dice at full resolution (test images)", unit="img", dynamic_ncols=True):
        out = dict(image_id=r.image_id, dataset=r.dataset)
        try:
            rec = recs[r.image_id]; img = B.imread(rec["img_path"]); ex = B.expert_masks(rec, img.shape); n = len(ex)
            gd = np.sum([e[0] for e in ex], 0) >= n / 2; gc = (np.sum([e[1] for e in ex], 0) >= n / 2) & gd
            h, w = img.shape[:2]; s = 512 / max(h, w); lb = np.zeros((512, 512, 3), np.uint8)
            rs = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA); y0, x0 = (512 - rs.shape[0]) // 2, (512 - rs.shape[1]) // 2
            lb[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
            pl = S.largest_cc(S.predict(loc, lb, dev)[0] > .5); ys, xs = np.where(pl)
            cx, cy = (xs.mean() - x0) / s, (ys.mean() - y0) / s; dd = (max(np.ptp(ys), np.ptp(xs)) + 1) / s; side = int(3 * dd)
            pad = side; op = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            y1, x1 = int(cy) + pad - side // 2, int(cx) + pad - side // 2
            c3 = cv2.resize(op[y1:y1 + side, x1:x1 + side], (512, 512), interpolation=cv2.INTER_AREA)
            qd = S.largest_cc(S.predict(coarse, c3, dev)[0] > .5); qy, qx = np.where(qd); k = side / 512
            cx2, cy2 = x1 - pad + qx.mean() * k, y1 - pad + qy.mean() * k; dd2 = (max(np.ptp(qy), np.ptp(qx)) + 1) * k
            t = S.predict(tight, G.tight_crop(img, cx2, cy2, dd2), dev); td = S.largest_cc(t[0] > .5); tc = (t[1] > .5) & td
            tc = S.largest_cc(tc) if tc.any() else tc
            side2 = max(16, int(G.TIGHT * dd2))
            pdisc, pcup = paste(td, cx2, cy2, side2, img.shape), paste(tc, cx2, cy2, side2, img.shape)
            out["dice_disc_e2e"], out["iou_disc_e2e"] = dice_iou(pdisc, gd); out["dice_cup_e2e"], out["iou_cup_e2e"] = dice_iou(pcup, gc)
            gy, gx = np.where(gd); out["centre_err_disc_diam"] = float(np.hypot(cx2 - gx.mean(), cy2 - gy.mean()) / (max(np.ptp(gy), np.ptp(gx)) + 1))
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:150]
        rows.append(out)
    P = pd.DataFrame(rows); P.to_csv(os.path.join(TAB, "e2e_dice_per_image.csv"), index=False)
    ok = P[P.get("error").isna()] if "error" in P else P
    cols = ["dice_disc_e2e", "dice_cup_e2e", "iou_disc_e2e", "iou_cup_e2e", "centre_err_disc_diam"]
    T = pd.concat([ok.groupby("dataset")[cols].mean(), ok[cols].mean().to_frame("ALL").T]); T["images"] = list(ok.groupby("dataset").size()) + [len(ok)]
    T["failures"] = list(P.groupby("dataset").apply(lambda g: g.get("error", pd.Series(dtype=object)).notna().sum() if "error" in g else 0)) + [int(P.get("error", pd.Series(dtype=object)).notna().sum()) if "error" in P else 0]
    T.index.name = "dataset"; T.reset_index().to_csv(os.path.join(TAB, "e2e_dice.csv"), index=False); print(T.round(4).to_string(), flush=True)
    json.dump(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), table=T.reset_index().to_dict("records")), open(os.path.join(ROOT, "logs", "stage13c_stats.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
