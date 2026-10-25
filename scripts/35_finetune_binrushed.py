"""Stage 7a2: in-domain adaptation for the base paper's protocol, still without any test leakage.

The BinRushed-free U-Net (tag noBR) over-estimates area CDR on BinRushed by ~+0.02 (annotation-style / camera shift),
and 36 of 191 eyes lie within +/-0.03 of the paper's 0.25 cut-off, so a small bias flips labels.
The paper itself trains on BinRushed images, so we may too - as long as an image is never predicted by a model that
saw it. 5-fold out-of-fold fine-tuning: for each fold, the noBR segmenter is fine-tuned on the BinRushed images of the
other 4 folds (expert disc/cup masks), then predicts the held-out fold (localiser: noBR, unchanged).
Output: results/tables/dl_per_image_all_noBR_ft.csv (same columns as stage 4; BinRushed rows only, out-of-fold),
        logs/stage7a2_stats.json
"""
import os, sys, json, time, argparse, datetime, importlib.util, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("seg30", os.path.join(ROOT, "scripts", "30_train_segmentation.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
TAB, MOD = S.TAB, S.MOD
BR = "RIGA-BinRushed"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--encoder", default="resnet34"); ap.add_argument("--tag", default="noBR", help="which BinRushed-free model to fine-tune")
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0); t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", dev, flush=True)
    df = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); df = df[df.status == "ok"]
    br = df[df.dataset == BR].reset_index(drop=True)
    if a.smoke:
        br = br.head(25); a.epochs = 1
    loc = S.build("resnet34", 1, pretrained=False).to(dev); loc.load_state_dict(torch.load(os.path.join(MOD, "seg_full512_noBR.pt"), map_location=dev)); loc.eval()
    strat = (br.acdr > 0.25).astype(int)
    out, fold_log = [], []
    for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(br, strat)):
        seg = S.build(a.encoder, 2, pretrained=False).to(dev); seg.load_state_dict(torch.load(os.path.join(MOD, f"seg_crop512_{a.tag}.pt"), map_location=dev))
        dl = DataLoader(S.SegSet(br.iloc[tri], "crop512", True), batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
        opt = torch.optim.AdamW(seg.parameters(), lr=a.lr, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        for ep in range(1, a.epochs + 1):
            seg.train(); tl = 0
            for x, y in tqdm(dl, desc=f"[fine-tune fold {k + 1}/5] epoch {ep}/{a.epochs}", unit="batch", leave=False, dynamic_ncols=True):
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    o = seg(x); loss = bce(o, y) + S.dice_loss(o.float(), y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item()
            print(f"[fine-tune fold {k + 1}/5] epoch {ep}/{a.epochs}  loss {tl / max(1, len(dl)):.4f}", flush=True)
        seg.eval()
        ev = S.evaluate(loc, seg, br.iloc[tei], dev, False)
        e = (ev.merge(br[["image_id", "acdr"]], on="image_id").pipe(lambda g: (g.acdr_e2e - g.acdr)))
        fold_log.append(dict(fold=k + 1, n_train=len(tri), n_test=len(tei), acdr_bias=float(e.mean()), acdr_mae=float(e.abs().mean())))
        print(f"[fold {k + 1}/5] held-out area-CDR bias {e.mean():+.4f}  MAE {e.abs().mean():.4f}", flush=True)
        out.append(ev); del seg; torch.cuda.empty_cache()
    ev = pd.concat(out, ignore_index=True)
    ev = ev.join(df.set_index("image_id")[["vcdr", "hcdr", "acdr", "label"]], on="image_id")
    ev.to_csv(os.path.join(TAB, f"dl_per_image_all_{a.tag}_ft.csv"), index=False)
    base = pd.read_csv(os.path.join(TAB, f"dl_per_image_all_{a.tag}.csv")); base = base[base.dataset == BR]
    st = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), minutes=round((time.time() - t0) / 60, 1), folds=fold_log,
              acdr_mae_before=float((base.acdr_e2e - base.acdr).abs().mean()), acdr_bias_before=float((base.acdr_e2e - base.acdr).mean()),
              acdr_mae_after=float((ev.acdr_e2e - ev.acdr).abs().mean()), acdr_bias_after=float((ev.acdr_e2e - ev.acdr).mean()),
              dice_cup_after=float(ev.dice_cup.mean()))
    json.dump(st, open(os.path.join(ROOT, "logs", f"stage7a2_stats_{a.tag}.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in st.items() if k != "folds"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
