# =====================================================================================
#  EXPORT FOR THE WEB APP  -  run this in a NEW cell AFTER the v3 pitch cell has finished
#  (it reuses the variables v3 left in memory: T1, PET, BM, REGIONS, FEATS, COV, OOF_ADJ, ...)
#
#  Writes /kaggle/working/imaging_xai_assets.zip  ->  unzip it into the website's  assets/  folder.
#  Contents: fitted model parameters, atlas regions, healthy-norm maps, cohort distributions,
#            and a few example subjects (OpenNeuro ds004054, CC0) already in MNI space.
# =====================================================================================
import json, os, shutil, numpy as np, nibabel as nib
from sklearn.preprocessing import StandardScaler

SITE = "/kaggle/working/imaging_xai_assets"
if os.path.exists(SITE): shutil.rmtree(SITE)
os.makedirs(f"{SITE}/samples", exist_ok=True)

def fidx(mask):
    """Voxel indices as flat FORTRAN-order offsets (the order NIfTI stores voxels on disk),
    listed in numpy's C-order enumeration so they line up with V[:, mask] arrays."""
    return np.ravel_multi_index(np.nonzero(mask), SHAPE, order="F").astype(np.int32)

# ---- brain mask + atlas regions + temporal ROI ---------------------------------------
bm_idx = fidx(BM); bm_idx.tofile(f"{SITE}/bm.bin")
reg_meta, chunks, off = [], [], 0
for name, m in REGIONS:
    ix = fidx(m); chunks.append(ix); reg_meta.append({"name": name, "off": off, "len": int(len(ix))}); off += len(ix)
np.concatenate(chunks).astype(np.int32).tofile(f"{SITE}/regions.bin")
roiL, roiR = fidx(L), fidx(R)
np.concatenate([roiL, roiR]).astype(np.int32).tofile(f"{SITE}/roi.bin")

# ---- models: the same age/sex adjustment + scaler + logistic regression as fit_explain() -----
def export_model(F):
    D = np.c_[np.ones(len(F)), COV]
    beta, *_ = np.linalg.lstsq(D, F, rcond=None) if HAVE_COV else (np.zeros((3, F.shape[1])), None)
    Fa = F - COV @ beta[1:] if HAVE_COV else F
    sc = StandardScaler().fit(Fa); clf = model().fit(sc.transform(Fa), Y)
    prob = clf.predict_proba(sc.transform(Fa))[:, 1]
    return ({"b_age": beta[1].tolist(), "b_sex": beta[2].tolist(), "mean": sc.mean_.tolist(),
             "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}, prob)
MODELS, INSAMPLE = {}, {}
for key, m in [("MRI", "MRI"), ("PET", "PET"), ("FUSED", "MRI + PET")]:
    MODELS[key], INSAMPLE[key] = export_model(FEATS[m])

# ---- healthy norm per voxel (controls only): value ~ b0 + b_age*age + b_sex*sex, residual sd ----
ctrl = np.where(Y == 0)[0]
Dc = np.c_[np.ones(len(ctrl)), AGE[ctrl], SEX[ctrl]] if HAVE_COV else np.ones((len(ctrl), 1))
ZFRAC = {}
for key, V in [("MRI", T1), ("PET", PET)]:
    Yc = V[ctrl][:, BM]; beta, *_ = np.linalg.lstsq(Dc, Yc, rcond=None)
    sd = (Yc - Dc @ beta).std(0) * np.sqrt(len(ctrl) / max(len(ctrl) - Dc.shape[1], 1)) + 1e-3
    if beta.shape[0] == 1: beta = np.vstack([beta, np.zeros_like(beta), np.zeros_like(beta)])
    np.vstack([beta, sd[None]]).astype(np.float32).tofile(f"{SITE}/norm_{key.lower()}.bin")
    Dall = np.c_[np.ones(N), AGE, SEX]
    Z = (V[:, BM] - Dall @ beta) / sd
    ZFRAC[key] = ((np.abs(Z) >= 2.5).mean(1) * 100).round(3).tolist()   # % of brain with |z| >= 2.5

# ---- example subjects (already registered to MNI) ------------------------------------
pat, con = np.where(Y == 1)[0], np.where(Y == 0)[0]
fo = OOF_ADJ["MRI + PET"]
picks = [case, int(pat[np.argmax(fo[pat])]), int(pat[np.argsort(fo[pat])[len(pat) // 2]]),
         int(con[np.argmin(fo[con])]), int(con[np.argsort(fo[con])[len(con) // 2]]), int(con[np.argmax(fo[con])])]
picks = [p for k, p in enumerate(picks) if p not in picks[:k] and not bad[p]]
samples = []
for i in picks:
    sid = IDS[i]
    for mod, V in [("T1w", T1), ("pet", PET)]:
        img = nib.Nifti1Image(V[i].astype(np.float32), AFF); img.set_data_dtype(np.int16)   # nibabel scales to int16
        nib.save(img, f"{SITE}/samples/{sid}_{mod}_MNI.nii.gz")
    samples.append({"id": sid, "group": "patient" if Y[i] else "control", "age": float(AGE[i]),
                    "sex": "M" if SEX[i] == 1 else "F",
                    "t1": f"samples/{sid}_T1w_MNI.nii.gz", "pet": f"samples/{sid}_pet_MNI.nii.gz",
                    "oof": {k: float(OOF_ADJ[m][i]) for k, m in [("MRI", "MRI"), ("PET", "PET"), ("FUSED", "MRI + PET")]},
                    "ref": {k: float(INSAMPLE[k][i]) for k in INSAMPLE}})

# ---- cohort context + model performance ----------------------------------------------
auc_rows = []
for s in SETTINGS:
    row = {"setting": s.replace("\n", ", "), "n": int(len(SETTINGS[s][0]))}
    for k, m in [("AGE", "Age + sex only"), ("MRI", "MRI"), ("PET", "PET"), ("FUSED", "MRI + PET")]:
        r = RES.get((m, s)); row[k] = None if r is None else [round(float(r.mean()), 3), round(float(r.std()), 3)]
    auc_rows.append(row)
meta = {
    "version": 1, "dataset": "OpenNeuro ds004054 (adult TLE vs controls, hybrid FDG-PET/MRI)",
    "template": "MNI152 2 mm (nilearn), RAS", "shape": [int(s) for s in SHAPE], "affine": AFF.tolist(),
    "registration": REG, "atlas": ATLAS,
    "n": {"controls": int(NC), "patients": int(NP)},
    "age_range_controls": [float(AGE[ctrl].min()), float(AGE[ctrl].max())],
    "bm_len": int(len(bm_idx)), "regions": reg_meta, "roi": {"L": int(len(roiL)), "R": int(len(roiR))},
    "models": MODELS, "have_cov": bool(HAVE_COV),
    "cohort": {"y": Y.astype(int).tolist(), "age": AGE.round(1).tolist(),
               "oof": {k: np.round(OOF_ADJ[m], 4).tolist() for k, m in [("MRI", "MRI"), ("PET", "PET"), ("FUSED", "MRI + PET")]},
               "ai": {"MRI": np.round(aiM, 3).tolist(), "PET": np.round(aiP, 3).tolist()},
               "zfrac": ZFRAC},
    "auc": auc_rows, "asym_p": {k: float(v) for k, v in P_ASYM.items()},
    "qc_flagged": [IDS[i] for i in np.where(bad)[0]],
    "samples": samples,
}
json.dump(meta, open(f"{SITE}/meta.json", "w"), separators=(",", ":"))
shutil.make_archive("/kaggle/working/imaging_xai_assets", "zip", SITE)
tot = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(SITE) for f in fs)
print(f"Exported {len(reg_meta)} regions, {len(samples)} example subjects, {tot / 1e6:.1f} MB")
print("Download /kaggle/working/imaging_xai_assets.zip and unzip it into the site's assets/ folder.")
