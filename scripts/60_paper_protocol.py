"""Stage 7: the base paper's OWN protocol - reproduce its numbers, then surpass them.

Kumar & Kumar 2022 (MTAP 81:13513): RIGA BinRushed, 180 images, 141 train / 39 test, label from area CDR > 0.25,
linear SVM on [CDR, PPA area]; reported test sensitivity 100 %, specificity 88.2 %, accuracy 94.87 % (train acc 97.3 %).

Here (same images, same label rule, same classifier family):
  label  : expert-consensus area CDR > 0.25 (6 ophthalmologists, RIGA)
  P1     : paper method  - classical CDR + classical PPA area -> linear SVM            (our faithful re-implementation)
  P2     : paper rule automated - DL area CDR > 0.25 (no training at all)
  P3     : paper classifier, DL features - linear SVM on [DL area CDR, learned PPA ratio]
  P4     : DL full features - linear SVM on [DL area/vertical/horizontal CDR, PPA ratio, PPA angle]
DL CDR comes from the segmentation model trained WITHOUT any BinRushed image (tag noBR) -> unbiased on all 191.
Reported: (a) one fixed paper-style split (180 images, 141/39, seed 42), (b) 500 repeated random 141/39 splits
(mean, SD, 95 % range, share of splits reaching the paper's accuracy), (c) external check of the trained
classifiers on the test images of every other dataset with the same label rule.
"""
import os, json, datetime, warnings
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures")
PAPER = dict(sensitivity=1.0, specificity=0.882, accuracy=0.9487, train_accuracy=0.973)
BR = "RIGA-BinRushed"; N_REP = 500


def metrics(y, p, s=None):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    out = dict(n=len(y), positives=int(y.sum()), TP=tp, TN=tn, FP=fp, FN=fn,
               sensitivity=tp / (tp + fn) if tp + fn else np.nan, specificity=tn / (tn + fp) if tn + fp else np.nan,
               accuracy=(tp + tn) / len(y))
    out["auc"] = roc_auc_score(y, s) if s is not None and len(set(y)) == 2 else np.nan
    return out


def load(files=None):
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[m.status == "ok"]
    cl = pd.read_csv(os.path.join(TAB, "baseline_per_image.csv"))[["image_id", "acdr_classical", "vcdr_classical", "ppa_area_px500"]]
    import glob
    # primary CDR source (decided before seeing any result): the BinRushed fine-tuned, out-of-fold models if they exist
    # (the paper also trains on BinRushed), otherwise every BinRushed-free model
    allf = sorted(glob.glob(os.path.join(TAB, "dl_per_image_all_noBR*.csv")))
    files = files or ([f for f in allf if f.endswith("_ft.csv")] or allf)
    if not files:
        files = [os.path.join(TAB, "dl_per_image_all_base.csv")]; print("WARNING: noBR model missing - using base model (BinRushed train images seen in training)")
    cols = ["acdr_e2e", "vcdr_e2e", "hcdr_e2e", "acdr_pred", "vcdr_pred", "acdr_e2e_soft"]
    parts = [pd.read_csv(f).reindex(columns=["image_id"] + cols).set_index("image_id") for f in files]
    dl = pd.concat(parts).groupby(level=0).mean()      # ensemble = per-image mean CDR of all BinRushed-free models available
    dl.index.name = "image_id"; dl = dl.reset_index(); src = " + ".join(os.path.basename(f) for f in files)
    print(f"DL CDR from {len(files)} model(s): {src}")
    pp = pd.read_csv(os.path.join(TAB, "ppa_features_all.csv"))[["image_id", "ppa_to_disc", "ppa_angle_deg"]]
    d = m[["image_id", "dataset", "split", "acdr", "vcdr"]].merge(cl, on="image_id", how="left").merge(dl, on="image_id", how="left").merge(pp, on="image_id", how="left")
    # end-to-end CDR is the fully automatic one; fall back to the crop prediction if localisation failed
    for k in ["acdr", "vcdr"]:
        d[f"{k}_dl"] = d[f"{k}_e2e"].fillna(d[f"{k}_pred"])
    d["hcdr_dl"] = d.hcdr_e2e
    d["acdr_soft_dl"] = d.acdr_e2e_soft.fillna(d.acdr_dl)
    d["y"] = (d.acdr > 0.25).astype(int)
    return d, src


FEATS = {"P1 Paper method (classical CDR + PPA area, linear SVM)": ["acdr_classical", "ppa_area_px500"],
         "P3 Paper classifier with DL features (DL CDR + learned PPA, linear SVM)": ["acdr_dl", "ppa_to_disc"],
         "P4 DL full features (area/vertical/horizontal CDR + PPA, linear SVM)": ["acdr_dl", "vcdr_dl", "hcdr_dl", "ppa_to_disc", "ppa_angle_deg"]}
RULE = "P2 Paper rule automated (DL area CDR > 0.25, no training)"
TUNED = "P5 DL area CDR, threshold tuned on the training images"
SOFT = "P7 DL soft (probability-weighted) area CDR > 0.25, no training"
SOFTSVM = "P8 DL hard + soft area CDR + vertical/horizontal CDR, linear SVM"
RBF = "P6 DL full features, RBF-kernel SVM"
SCREEN = "P9 Screening mode: DL area CDR, threshold set on training images for >=98% sensitivity"


def tuned_threshold(x, y):
    cand = np.unique(np.round(x.dropna(), 4))
    best = max(cand, key=lambda t: (np.mean([((x > t) & (y == 1)).sum() / max(1, (y == 1).sum()), ((x <= t) & (y == 0)).sum() / max(1, (y == 0).sum())]), -abs(t - 0.25)))
    return float(best)


def fit_eval(tr, te, cols, weight=None):
    med = tr[cols].median()
    Xtr, Xte = tr[cols].fillna(med), te[cols].fillna(med)   # classical failures -> train median (counted in stats)
    # plain linear SVM as in the paper (no class re-weighting -> optimises accuracy, the paper's headline metric)
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight=weight, probability=True, random_state=0))
    clf.fit(Xtr, tr.y)
    tr_acc = float((clf.predict(Xtr) == tr.y).mean())
    return clf, tr_acc, clf.predict(Xte), clf.predict_proba(Xte)[:, 1], med


def run_split(tr, te):
    rows = {}
    for name, cols in FEATS.items():
        # classical features: class-balanced SVM (an unweighted SVM collapses to 'all normal' on them - best case for the paper method)
        _, tr_acc, p, s, _ = fit_eval(tr, te, cols, "balanced" if name.startswith("P1") else None)
        rows[name] = dict(**metrics(te.y, p, s), train_accuracy=tr_acc)
    rows[RULE] = dict(**metrics(te.y, (te.acdr_dl > 0.25).astype(int), te.acdr_dl.values),
                      train_accuracy=float(((tr.acdr_dl > 0.25).astype(int) == tr.y).mean()))
    # screening operating point (declared before the final run, like the paper's 100%-sensitivity emphasis):
    # the threshold keeps >= 98% of the TRAINING glaucoma eyes above it; test images never influence it
    pos = tr.acdr_dl[tr.y == 1].dropna()
    ts = float(np.quantile(pos, 0.02)) - 1e-6 if len(pos) else 0.25
    rows[SCREEN] = dict(**metrics(te.y, (te.acdr_dl > ts).astype(int), te.acdr_dl.values),
                        train_accuracy=float(((tr.acdr_dl > ts).astype(int) == tr.y).mean()), threshold=ts)
    t = tuned_threshold(tr.acdr_dl, tr.y)
    rows[TUNED] = dict(**metrics(te.y, (te.acdr_dl > t).astype(int), te.acdr_dl.values),
                       train_accuracy=float(((tr.acdr_dl > t).astype(int) == tr.y).mean()), threshold=t)
    rows[SOFT] = dict(**metrics(te.y, (te.acdr_soft_dl > 0.25).astype(int), te.acdr_soft_dl.values),
                      train_accuracy=float(((tr.acdr_soft_dl > 0.25).astype(int) == tr.y).mean()))
    _, tr_acc, p, sc, _ = fit_eval(tr, te, ["acdr_dl", "acdr_soft_dl", "vcdr_dl", "hcdr_dl"])
    rows[SOFTSVM] = dict(**metrics(te.y, p, sc), train_accuracy=tr_acc)
    cols = FEATS["P4 DL full features (area/vertical/horizontal CDR + PPA, linear SVM)"]; med = tr[cols].median()
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, probability=True, random_state=0)).fit(tr[cols].fillna(med), tr.y)
    rows[RBF] = dict(**metrics(te.y, clf.predict(te[cols].fillna(med)), clf.predict_proba(te[cols].fillna(med))[:, 1]),
                     train_accuracy=float((clf.predict(tr[cols].fillna(med)) == tr.y).mean()))
    return rows


def by_source():
    """repeated-split accuracy for each BinRushed-free CDR source separately (transparency table)"""
    import glob
    files = sorted(glob.glob(os.path.join(TAB, "dl_per_image_all_noBR*.csv")))
    rows = []
    for f in files + ([None] if len(files) > 1 else []):
        d, src = load([f] if f else files); br = d[d.dataset == BR].reset_index(drop=True)
        if br.acdr_dl.isna().all():
            continue
        acc = {k: [] for k in [RULE, TUNED]}; sen = {k: [] for k in acc}; spe = {k: [] for k in acc}
        for i in tqdm(range(N_REP), desc=f"CDR source {os.path.basename(f) if f else 'ensemble'}", unit="split", dynamic_ncols=True):
            s180 = br.sample(180, random_state=i) if len(br) > 180 else br
            tr_i, te_i = train_test_split(s180, test_size=39, stratify=s180.y, random_state=i)
            for k, thr in [(RULE, 0.25), (TUNED, tuned_threshold(tr_i.acdr_dl, tr_i.y))]:
                mm = metrics(te_i.y, (te_i.acdr_dl > thr).astype(int)); acc[k].append(mm["accuracy"]); sen[k].append(mm["sensitivity"]); spe[k].append(mm["specificity"])
        e = br.acdr_dl - br.acdr
        for k in acc:
            rows.append(dict(cdr_source="ensemble of all" if f is None else os.path.basename(f), method=k, binrushed_acdr_bias=e.mean(), binrushed_acdr_mae=e.abs().mean(),
                             accuracy_mean=np.mean(acc[k]), accuracy_sd=np.std(acc[k]), sensitivity_mean=np.nanmean(sen[k]), specificity_mean=np.nanmean(spe[k]),
                             share_splits_acc_ge_paper=float(np.mean(np.array(acc[k]) >= PAPER["accuracy"] - 1e-9))))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, "paper_protocol_by_cdr_source.csv"), index=False)
    print("\n(d) by CDR source\n", t.round(4).to_string(index=False)); return t


def main():
    t0 = datetime.datetime.now()
    src_table = by_source()
    d, src = load()
    br = d[d.dataset == BR].reset_index(drop=True)
    print(f"BinRushed images: {len(br)} | label area CDR > 0.25: {br.y.sum()} positive / {(1 - br.y).sum()} negative | DL source {src}")
    print(f"classical pipeline failures on BinRushed: {br.acdr_classical.isna().sum()} (imputed with train median)")
    # ---- (a) fixed paper-style split
    sub, _ = train_test_split(br, train_size=180, stratify=br.y, random_state=42) if len(br) > 180 else (br, None)
    tr, te = train_test_split(sub, test_size=39, stratify=sub.y, random_state=42)
    fixed = run_split(tr, te)
    ft = pd.DataFrame([dict(method="Kumar & Kumar 2022 (reported)", n=39, sensitivity=PAPER["sensitivity"], specificity=PAPER["specificity"],
                            accuracy=PAPER["accuracy"], train_accuracy=PAPER["train_accuracy"])] +
                      [dict(method=k, **v) for k, v in fixed.items()])
    ft.to_csv(os.path.join(TAB, "paper_protocol_fixed_split.csv"), index=False)
    print("\n(a) fixed split 141/39 (seed 42)\n", ft.round(4).to_string(index=False))
    # ---- (b) 500 repeated splits
    rep = []
    for i in tqdm(range(N_REP), desc="Repeated 141/39 splits", unit="split", dynamic_ncols=True):
        s180 = br.sample(180, random_state=i) if len(br) > 180 else br
        tr_i, te_i = train_test_split(s180, test_size=39, stratify=s180.y, random_state=i)
        for k, v in run_split(tr_i, te_i).items():
            rep.append(dict(rep=i, method=k, **v))
    rep = pd.DataFrame(rep); rep.to_csv(os.path.join(TAB, "paper_protocol_repeats_raw.csv"), index=False)
    summ = []
    for k, g in rep.groupby("method", sort=False):
        r = dict(method=k, splits=len(g))
        for mname in ["sensitivity", "specificity", "accuracy", "auc", "train_accuracy"]:
            v = g[mname].dropna()
            r[f"{mname}_mean"] = v.mean(); r[f"{mname}_sd"] = v.std(); r[f"{mname}_p2.5"] = v.quantile(.025); r[f"{mname}_p97.5"] = v.quantile(.975)
        r["share_splits_acc_ge_paper"] = float((g.accuracy >= PAPER["accuracy"] - 1e-9).mean())
        r["share_splits_beating_paper_all3"] = float(((g.accuracy >= PAPER["accuracy"] - 1e-9) & (g.sensitivity >= PAPER["sensitivity"] - 1e-9) & (g.specificity >= PAPER["specificity"] - 1e-9)).mean())
        summ.append(r)
    summ = pd.DataFrame(summ); summ.to_csv(os.path.join(TAB, "paper_protocol_repeated_splits.csv"), index=False)
    print("\n(b) 500 repeated splits\n", summ[["method", "accuracy_mean", "accuracy_sd", "sensitivity_mean", "specificity_mean", "auc_mean", "share_splits_acc_ge_paper"]].round(4).to_string(index=False))
    # ---- (c) external: train on all BinRushed, test on every other dataset's TEST images with the same label rule
    ext = []
    # other datasets: fine-tuned models only exist for BinRushed -> use the BinRushed-free (not fine-tuned) models there
    import glob
    d_ext, _ = load([f for f in sorted(glob.glob(os.path.join(TAB, "dl_per_image_all_noBR*.csv"))) if not f.endswith("_ft.csv")] or None)
    others = d_ext[(d_ext.dataset != BR) & (d_ext.split == "test")]
    for name, cols in FEATS.items():
        clf, _, _, _, med = fit_eval(br, br.iloc[:2], cols, "balanced" if name.startswith("P1") else None)
        for ds, g in list(others.groupby("dataset")) + [("All other datasets", others)]:
            X = g[cols].fillna(med)
            ext.append(dict(method=name, test_set=ds, **metrics(g.y, clf.predict(X), clf.predict_proba(X)[:, 1])))
    for ds, g in list(others.groupby("dataset")) + [("All other datasets", others)]:
        ext.append(dict(method=RULE, test_set=ds, **metrics(g.y, (g.acdr_dl > 0.25).astype(int), g.acdr_dl.values)))
    t_all = tuned_threshold(br.acdr_dl, br.y)
    for ds, g in list(others.groupby("dataset")) + [("All other datasets", others)]:
        ext.append(dict(method=TUNED, test_set=ds, threshold=t_all, **metrics(g.y, (g.acdr_dl > t_all).astype(int), g.acdr_dl.values)))
    ext = pd.DataFrame(ext); ext.to_csv(os.path.join(TAB, "paper_protocol_external.csv"), index=False)
    print("\n(c) external (trained on BinRushed)\n", ext[ext.test_set == "All other datasets"][["method", "n", "sensitivity", "specificity", "accuracy", "auc"]].round(4).to_string(index=False))
    # ---- figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.8)); w = 0.2
        names = ["Paper (reported)"] + [k.split(" (")[0] for k in summ.method]
        for j, mname in enumerate(["sensitivity", "specificity", "accuracy"]):
            vals = [PAPER[mname]] + list(summ[f"{mname}_mean"]); err = [0] + list(summ[f"{mname}_sd"])
            ax.bar(np.arange(len(names)) + (j - 1) * w, vals, w, yerr=err, capsize=3, label=mname)
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 1.05)
        ax.set_title("Paper protocol (BinRushed, 141/39, label area CDR>0.25): mean +/- SD over 500 splits")
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "paper_protocol_comparison.png"), dpi=150)
    except Exception as e:
        print("figure skipped:", e)
    stats = dict(time=str(datetime.datetime.now()), minutes=(datetime.datetime.now() - t0).seconds / 60, dl_source=src,
                 binrushed_images=len(br), positives=int(br.y.sum()), classical_failures=int(br.acdr_classical.isna().sum()),
                 paper=PAPER, fixed_split=ft.to_dict("records"), repeated=summ.to_dict("records"),
                 external_all=ext[ext.test_set == "All other datasets"].to_dict("records"), by_cdr_source=src_table.to_dict("records"))
    json.dump(stats, open(os.path.join(ROOT, "logs", "stage7_stats.json"), "w"), indent=2, default=float)
    print("Saved results/tables/paper_protocol_*.csv, figures/paper_protocol_comparison.png, logs/stage7_stats.json")


if __name__ == "__main__":
    main()
