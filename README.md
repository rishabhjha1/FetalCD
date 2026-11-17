# EpiFusion-XAI

Explainable PET + MRI analysis that runs entirely in the browser. Choose an example subject or upload an
MNI-registered T1 MRI (and FDG-PET). The page shows:

- model scores (MRI, PET, MRI+PET) against the training cohort's cross-validated scores
- a three-plane brain viewer with PET, normative z-maps and SHAP overlays
- exact SHAP explanations by brain region, and each modality's share of the evidence
- deviation from an age- and sex-matched healthy norm
- temporal-lobe left–right asymmetry

The model was trained on OpenNeuro ds004054 (adults with temporal lobe epilepsy and healthy controls, hybrid FDG-PET/MRI).
It is a research prototype, not a diagnostic tool. Files you upload are never sent anywhere.

## Folder layout

```
index.html               the whole app (HTML + CSS + JS, no build step)
prepare_scan_cell.py     Kaggle/Colab cell: register a raw T1 (+PET) to MNI 2 mm for upload
webapp_export_cell.py    Kaggle cell: exports the trained model into assets/
assets/                  produced by webapp_export_cell.py (meta.json, *.bin, samples/)
.nojekyll                tells GitHub Pages to serve files as-is
```

## Build the assets (once, on Kaggle)

1. Run the v3 pitch cell (`pet_mri_xai_pitch_cell_v3.py`) so the model is in memory.
2. In a new cell below it, run `webapp_export_cell.py`.
3. Download `/kaggle/working/imaging_xai_assets.zip` and unzip its contents into `assets/`
   (so that `assets/meta.json` exists).

## Publish on GitHub Pages

1. Create a new public repository, e.g. `EpiFusion-XAI`.
2. Upload everything in this folder, including `assets/` and `.nojekyll`
   (GitHub web: "Add file" → "Upload files"; drag the folder contents in).
3. Repository → Settings → Pages → Source: "Deploy from a branch", Branch: `main`, folder `/ (root)` → Save.
4. After a minute the site is live at `https://<your-username>.github.io/EpiFusion-XAI/`.

Link straight to an example subject with `?sample=<id>`, e.g. `.../EpiFusion-XAI/?sample=sub-patient17`.

## Test locally

`index.html` loads its assets with `fetch`, so opening the file directly does not work. From this folder run
`python -m http.server 8000` and open http://localhost:8000.

## Analysing a new scan

Run `prepare_scan_cell.py` with your T1 (and PET) paths. It applies the same skull-strip and ANTsPy registration
as training and writes `*_T1w_MNI.nii.gz` and `*_pet_MNI.nii.gz`. Upload those, and enter age and sex
(the model regresses them out). Scans not on the MNI152 2 mm grid are rejected with a message.
