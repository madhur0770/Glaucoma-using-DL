"""Stage 7d: human-expert ceiling for the base paper's protocol (label = consensus area CDR > 0.25).

RIGA has 6 ophthalmologists per image. For every doctor, apply the paper's rule to HIS/HER OWN area CDR and compare
with the consensus label (pixel kept if >= half of the doctors marked it). This is what a single human expert achieves on
the paper's task - an upper bound for any automatic method (and a favourable one for the doctors, since each doctor
contributed to the consensus). Our automatic CDR (BinRushed fine-tuned, out-of-fold) is scored the same way.
Outputs: results/tables/expert_ceiling.csv, logs/stage7d_stats.json
"""
import os, json, datetime, importlib.util, glob
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAB = os.path.join(ROOT, "results", "tables")
spec = importlib.util.spec_from_file_location("build10", os.path.join(ROOT, "scripts", "10_build_master.py")); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


def per_expert(rec):
    try:
        img = B.imread(rec["img_path"]); ex = B.expert_masks(rec, img.shape); n = len(ex)
        disc = np.sum([e[0] for e in ex], 0) >= n / 2; cup = (np.sum([e[1] for e in ex], 0) >= n / 2) & disc
        out = dict(image_id=rec["image_id"], dataset=rec["dataset"], consensus_acdr=float(cup.sum() / max(1, disc.sum())))
        for k, (d, c) in enumerate(ex):
            out[f"expert{k + 1}_acdr"] = float((c & d).sum() / max(1, d.sum()))
        return out
    except Exception as e:
        return dict(image_id=rec.get("image_id"), error=str(e))


def score(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return dict(n=len(y), TP=tp, TN=tn, FP=fp, FN=fn, sensitivity=tp / max(1, tp + fn), specificity=tn / max(1, tn + fp), accuracy=(tp + tn) / len(y))


def main():
    recs, _ = B.discover_riga()
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); ok = set(m[m.status == "ok"].image_id)
    recs = [r for r in recs if r["image_id"] in ok]
    with ThreadPoolExecutor(8) as ex:
        rows = list(tqdm(ex.map(per_expert, recs), total=len(recs), desc="Per-expert area CDR (RIGA)", unit="img", dynamic_ncols=True))
    E = pd.DataFrame(rows); E = E[E.get("error").isna()] if "error" in E else E
    E.to_csv(os.path.join(TAB, "expert_ceiling_per_image.csv"), index=False)
    res = []
    for ds, g in E.groupby("dataset"):
        y = (g.consensus_acdr > 0.25).astype(int).values
        accs = []
        for c in sorted(c for c in g.columns if c.startswith("expert")):
            r = score(y, (g[c] > 0.25).astype(int).values); accs.append(r["accuracy"]); res.append(dict(dataset=ds, rater=c.replace("_acdr", ""), **r))
        res.append(dict(dataset=ds, rater="mean of the 6 doctors", accuracy=float(np.mean(accs)), n=len(g)))
        if ds == "RIGA-BinRushed":
            for f, nm in [("dl_per_image_all_noBR_ft.csv", "ours: ResNet-34 fine-tuned (out-of-fold)"), ("dl_per_image_all_noBR_r50_ft.csv", "ours: ResNet-50 fine-tuned (out-of-fold)")]:
                p = os.path.join(TAB, f)
                if os.path.exists(p):
                    D = g.merge(pd.read_csv(p)[["image_id", "acdr_e2e"]], on="image_id")
                    res.append(dict(dataset=ds, rater=nm, **score((D.consensus_acdr > 0.25).astype(int).values, (D.acdr_e2e > 0.25).astype(int).values)))
            fts = [f for f in glob.glob(os.path.join(TAB, "dl_per_image_all_noBR*_ft.csv"))]
            if len(fts) > 1:
                P = pd.concat([pd.read_csv(f)[["image_id", "acdr_e2e"]] for f in fts]).groupby("image_id").mean().reset_index()
                D = g.merge(P, on="image_id")
                res.append(dict(dataset=ds, rater="ours: ensemble of fine-tuned models (out-of-fold)", **score((D.consensus_acdr > 0.25).astype(int).values, (D.acdr_e2e > 0.25).astype(int).values)))
    R = pd.DataFrame(res); R.to_csv(os.path.join(TAB, "expert_ceiling.csv"), index=False); print(R.round(4).to_string(index=False))
    json.dump(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), results=R.to_dict("records")), open(os.path.join(ROOT, "logs", "stage7d_stats.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
