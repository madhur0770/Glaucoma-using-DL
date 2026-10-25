"""Stage 14: (a) end-to-end test of the deployable tool (scripts/predict.py) on held-out TEST photos - including a degraded
copy of each - exactly as a user would run it; (b) the scorecard: every number of the base paper and of the cited recent
literature next to ours, with ahead / tied / behind.
Outputs: predict_in/demo/*, predict_out/*, results/tables/demo_predict_check.csv, results/tables/scorecard.csv, logs/stage14_stats.json
"""
import os, sys, json, shutil, subprocess, datetime
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAB = os.path.join(ROOT, "results", "tables")


PRIMARY = "P2"      # paper's own decision rule (area CDR > 0.25) with our fully automatic CDR - declared before the results


def rd(f):
    p = os.path.join(TAB, f); return pd.read_csv(p) if os.path.exists(p) else None


def demo():
    m = pd.read_csv(os.path.join(ROOT, "processed", "master.csv")); m = m[(m.status == "ok") & (m.split == "test")]
    pick = m.groupby("dataset").head(2); dd = os.path.join(ROOT, "predict_in", "demo"); os.makedirs(dd, exist_ok=True)
    for f in os.listdir(dd):
        os.remove(os.path.join(dd, f))
    for _, r in pick.iterrows():
        src = r.img_path if os.path.isabs(str(r.img_path)) else os.path.join(ROOT, r.img_path)
        img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode(".jpg", img)[1].tofile(os.path.join(dd, f"{r.image_id}.jpg"))
        bad = cv2.GaussianBlur(cv2.convertScaleAbs(img, alpha=0.45), (0, 0), 6)                       # dark + blurred copy
        cv2.imencode(".jpg", bad, [cv2.IMWRITE_JPEG_QUALITY, 15])[1].tofile(os.path.join(dd, f"{r.image_id}__degraded.jpg"))
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "predict.py"), dd]).returncode
    if rc != 0:
        raise SystemExit("predict.py failed")
    s = pd.read_csv(os.path.join(ROOT, "predict_out", "summary.csv"))
    s["image_id"] = s.file.str.replace("__degraded", "").str.replace(".jpg", "", regex=False); s["degraded"] = s.file.str.contains("__degraded")
    s = s.merge(m[["image_id", "dataset", "label", "vcdr"]], on="image_id", how="left")
    s["vcdr_error"] = (s.get("vCDR") - s.vcdr).abs()
    s.to_csv(os.path.join(TAB, "demo_predict_check.csv"), index=False)
    cl = s[~s.degraded]; dg = s[s.degraded]
    return dict(images=len(s), clean_vcdr_mae=float(cl.vcdr_error.mean()), degraded_vcdr_mae=float(dg.vcdr_error.mean()),
                quality_warning_rate_clean=float((cl.get("quality_warning", "none") != "none").mean()),
                quality_warning_rate_degraded=float((dg.get("quality_warning", "none") != "none").mean()), reports=int(len(os.listdir(os.path.join(ROOT, "predict_out")))))


def status(ours, theirs, higher_better=True, tol=0.005):
    if ours is None or theirs is None or (isinstance(ours, float) and np.isnan(ours)):
        return "pending"
    d = (ours - theirs) if higher_better else (theirs - ours)
    return "AHEAD" if d > tol else ("TIED" if d >= -tol else "BEHIND")


def scorecard():
    R = []
    add = lambda area, metric, src, theirs, tdata, ours, odata, hb=True: R.append(dict(area=area, metric=metric, reference=src, reference_value=theirs, reference_data=tdata,
                                                                                     ours=ours, our_data=odata, status=status(ours, theirs, hb)))
    seg = rd("dl_segmentation_base.csv"); strong = rd("dl_segmentation_strong.csv")
    allb = seg[seg.dataset == "ALL"].iloc[0] if seg is not None else None
    alls = strong[strong.dataset == "ALL"].iloc[0] if strong is not None and (strong.dataset == "ALL").any() else None
    # pre-declared (18:10 IST, before computing): Dice/IoU from the strong U-Net++ model; vCDR from the mean of ALL
    # trained segmenters whose training excluded the test split (no max-over-models selection on test)
    src = alls if alls is not None else allb
    cup_dice, disc_dice, disc_iou, cup_iou = float(src.dice_cup), float(src.dice_disc), float(src.iou_disc), float(src.iou_cup)
    # verification (19:20 IST, decided before the numbers exist): if the fully end-to-end, full-resolution Dice/IoU of
    # stage 13c is available, the scorecard uses it (stricter: includes localisation error), for every comparison
    e2e = rd("e2e_dice.csv"); seg_basis = "expert-centred crop"
    if e2e is not None and (e2e.dataset == "ALL").any():
        a_ = e2e[e2e.dataset == "ALL"].iloc[0]
        cup_dice, disc_dice, disc_iou, cup_iou = float(a_.dice_cup_e2e), float(a_.dice_disc_e2e), float(a_.iou_disc_e2e), float(a_.iou_cup_e2e); seg_basis = "end-to-end, full resolution"
    vs = []
    for t in ["base", "strong", "noBR", "noBR_r50", "noBR_tight"]:
        d = rd(f"dl_per_image_all_{t}.csv")
        if d is not None and "vcdr_e2e" in d:
            vs.append(d[d.split == "test"].set_index("image_id")[["vcdr_e2e", "vcdr"]])
    if vs:
        ids = sorted(set.intersection(*[set(v.index) for v in vs]))
        E = pd.concat([v.loc[ids, "vcdr_e2e"] for v in vs], axis=1).mean(axis=1); g = vs[0].loc[ids, "vcdr"]
        vmae, vr = float((E - g).abs().mean()), float(np.corrcoef(E, g)[0, 1])
    else:
        vmae, vr = float(allb.vcdr_mae), float(allb.vcdr_corr)
    # base paper, own protocol
    fx = rd("paper_protocol_fixed_split.csv"); rp = rd("paper_protocol_repeated_splits.csv")
    if fx is not None:
        ours = fx[fx.method.str.startswith(PRIMARY)].iloc[0]          # primary method fixed in advance (no picking on test)
        for k, v in [("accuracy", 0.9487), ("sensitivity", 1.0), ("specificity", 0.882), ("train_accuracy", 0.973)]:
            add("Base paper - own protocol (fixed 39-image split)", k, "Kumar & Kumar 2022 (results section)", v, "GLAUCOMA-PY + BinRushed, 39 test", float(ours[k]), f"BinRushed 39 test ({ours.method.split(' (')[0]})")
        # the paper's abstract and Table 4 quote 97.3% as "accuracy" (= its training accuracy in the results section)
        add("Base paper - own protocol (fixed 39-image split)", "accuracy as quoted in the abstract / Table 4", "Kumar & Kumar 2022 (abstract)", 0.973, "GLAUCOMA-PY + BinRushed", float(ours["accuracy"]), f"BinRushed 39 test ({ours.method.split(' (')[0]})")
    if rp is not None:
        ours = rp[rp.method.str.startswith(PRIMARY)].iloc[0]
        add("Base paper - own protocol (mean of 500 random 141/39 splits)", "accuracy", "Kumar & Kumar 2022 (single split)", 0.9487, "BinRushed 39 test", float(ours.accuracy_mean), f"BinRushed, 500 splits ({ours.method.split(' (')[0]})")
        p9 = rp[rp.method.str.startswith("P9")]
        if len(p9):     # screening operating point, declared in advance (paper itself favours sensitivity)
            for k, v in [("sensitivity", 1.0), ("specificity", 0.882), ("accuracy", 0.9487)]:
                add("Base paper - own protocol, screening mode (mean of 500 splits)", k, "Kumar & Kumar 2022 (single split)", v, "BinRushed 39 test",
                    float(p9[f"{k}_mean"].iloc[0]), "BinRushed, 500 splits (P9 screening threshold)")
        p1 = rp[rp.method.str.startswith("P1")]
        if len(p1):
            add("Reproduction check (not a contest)", "accuracy of the paper's own method, re-implemented by us", "Kumar & Kumar 2022 (reported)", 0.9487, "BinRushed", float(p1.accuracy_mean.iloc[0]), "BinRushed, 500 splits")
            R[-1]["status"] = "does not reproduce" if R[-1]["ours"] < 0.9487 - 0.05 else "reproduces"
    bp = rd("baseline_per_image.csv")
    if bp is not None:      # classical pipeline on exactly the same 522 test images (it is end-to-end by construction)
        bt = bp[bp.split == "test"]; tag = f"same 522 test images; DL Dice {seg_basis}"
        add("Base paper - pipeline steps", "OD localisation hit rate", "paper method re-implemented", float(bt.loc_hit.mean()), "522 test", float(allb.loc_hit_rate), tag)
        add("Base paper - pipeline steps", "disc Dice", "paper method re-implemented", float(bt.dice_disc.mean()), "522 test", float(disc_dice), tag)
        add("Base paper - pipeline steps", "cup Dice", "paper method re-implemented", float(bt.dice_cup.mean()), "522 test", float(cup_dice), tag)
        add("Base paper - pipeline steps", "vCDR MAE", "paper method re-implemented", float((bt.vcdr_classical - bt.vcdr).abs().mean()), "522 test", float(vmae), "522 test", False)
    cls = rd("classification_results.csv")
    if cls is not None:
        p = cls[cls.test_set.str.startswith("Pooled")]
        add("Glaucoma detection (real labels)", "AUC", "paper method re-implemented (M0)", float(p[p.method.str.startswith("M0")].auc.iloc[0]), "Chaksu+PAPILA 411 test", float(p[p.method.str.startswith("M7")].auc.iloc[0]), "Chaksu+PAPILA 411 test (M7, our pre-defined final model)")
    # literature (different datasets - approximate comparison, stated)
    add("Literature - segmentation", "cup Dice", "Wang, Kim, Eom 2025 (Drishti-GS)", 0.9127, "Drishti-GS", float(cup_dice), f"Chaksu/PAPILA/RIGA test ({seg_basis})")
    add("Literature - segmentation", "cup Dice", "Wang, Kim, Eom 2025 (REFUGE)", 0.9014, "REFUGE", float(cup_dice), f"Chaksu/PAPILA/RIGA test ({seg_basis})")
    add("Literature - segmentation", "mean Dice (disc+cup)", "Zhao, Mookiah, Trucco 2025 (internal test)", 0.9051, "GoDARTS/IDRiD/Drishti/RIM-ONE/REFUGE", float((cup_dice + disc_dice) / 2), f"Chaksu/PAPILA/RIGA test ({seg_basis})")
    add("Literature - segmentation", "IoU (theirs: overall multi-label OD/OC/vessels/PPA; ours: mean of disc and cup)", "Kako et al. 2024", 0.87, "HRF (45 images)", float((disc_iou + cup_iou) / 2), f"Chaksu/PAPILA/RIGA test ({seg_basis})")
    add("Literature - CDR", "vCDR MAE", "Lv et al. 2025", 0.0387, "Drishti-GS/REFUGE", float(vmae), "Chaksu/PAPILA/RIGA test", False)
    add("Literature - CDR", "vCDR correlation r", "Lv et al. 2025", 0.919, "Drishti-GS/REFUGE", float(vr), "Chaksu/PAPILA/RIGA test")
    pp = rd("ppa_palm_results.csv")
    if pp is not None:
        add("Literature - PPA", "PPA Dice", "Li et al. 2023 (semi-supervised)", 0.815, "Beijing Tongren 150+407", float(pp[pp["mode"] == "supervised"].dice_ppa.iloc[0]), "PALM test 384")
    if cls is not None:
        from sklearn.metrics import roc_curve
        per = rd("classification_per_image.csv")
        if per is not None:
            best = p[p.method.str.startswith("M7")].method.iloc[0]
            f, t, _ = roc_curve(per.y, per[best]); ok = (1 - f) >= 0.821
            add("Literature - detection", "sensitivity at specificity 0.821", "Chung et al. 2026", 0.903, "NTUH: trained on 43,152, tested on 749", float(t[ok].max()), f"Chaksu+PAPILA 411 test ({best.split(':')[0]})")
    T = pd.DataFrame(R); T.to_csv(os.path.join(TAB, "scorecard.csv"), index=False); print(T.to_string(index=False)); return T


def main():
    st = dict(time=datetime.datetime.now().isoformat(timespec="seconds"))
    try:
        st["demo"] = demo()
    except Exception as e:
        st["demo_error"] = str(e)
    T = scorecard(); st["scorecard_counts"] = T.status.value_counts().to_dict(); st["behind"] = T[T.status == "BEHIND"][["metric", "reference"]].to_dict("records")
    json.dump(st, open(os.path.join(ROOT, "logs", "stage14_stats.json"), "w"), indent=2, default=float); print(json.dumps(st, indent=1, default=float))
    if "demo_error" in st:
        raise SystemExit(st["demo_error"])


if __name__ == "__main__":
    main()
