# =====================================================================================
#  PREPARE A NEW SCAN FOR THE WEB APP  (Kaggle / Colab cell)
#  Registers a raw T1-weighted MRI (+ FDG-PET from the same session, if you have it) to the
#  MNI152 2 mm grid with exactly the preprocessing the model was trained with.
#  Upload the two output files on the web page.
#
#  Works best for hybrid PET-MRI (PET already in the MRI's scanner space, like ds004054).
#  PET from a separate scanner needs its own PET->T1 registration first (not handled here).
# =====================================================================================
T1_PATH  = "/kaggle/input/YOUR_DATASET/sub-XX_T1w.nii.gz"     # <- edit
PET_PATH = "/kaggle/input/YOUR_DATASET/sub-XX_pet.nii.gz"     # <- edit, or None for MRI only
OUT_DIR  = "/kaggle/working/scan_for_webapp"
REG      = "Affine"                                           # must match the model ("Affine" in v3)

import subprocess, sys, os, tempfile
for pkg, mod in [("antspyx", "ants"), ("nilearn", "nilearn")]:
    try: __import__(mod)
    except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np, nibabel as nib, ants
from nibabel.processing import resample_to_output, resample_from_to
from nilearn import datasets as nds
from scipy import ndimage

os.makedirs(OUT_DIR, exist_ok=True); tmp = tempfile.mkdtemp()
TMPL = nib.as_closest_canonical(nds.load_mni152_template(resolution=2)); nib.save(TMPL, f"{tmp}/tmpl.nii.gz")
FIXED = ants.image_read(f"{tmp}/tmpl.nii.gz")

def load_t1(path):
    img = nib.load(path); d = np.asarray(img.dataobj, dtype=np.float32)
    if d.ndim == 4: d = d[..., 0]
    return nib.as_closest_canonical(nib.Nifti1Image(np.nan_to_num(d), img.affine))
def load_pet3d(path):
    img = nib.load(path)
    if len(img.shape) == 4 and img.shape[3] > 1:                   # dynamic PET: average the late frames
        n = img.shape[3]; s = n - max(1, n // 3)
        d = np.mean([np.asarray(img.dataobj[..., k], dtype=np.float32) for k in range(s, n)], 0)
    else: d = np.asarray(img.dataobj, dtype=np.float32).reshape(img.shape[:3])
    return nib.as_closest_canonical(nib.Nifti1Image(np.nan_to_num(d), img.affine))
def brain_mask(vol):
    p = ndimage.gaussian_filter(vol, 1.5); m = p > 0.35 * np.percentile(p[p > 0], 99)
    lab, n = ndimage.label(m)
    if n > 1: m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    return ndimage.binary_dilation(m, iterations=1)

t1i = resample_to_output(load_t1(T1_PATH), voxel_sizes=2.0, order=1)
if PET_PATH:
    peti = resample_from_to(load_pet3d(PET_PATH), t1i, order=1)
    mask = brain_mask(peti.get_fdata())                            # same PET-based skull strip as training
else:
    peti = None
    print("No PET: skull-stripping from the T1 itself (less reliable) - check the overlay on the web page.")
    mask = brain_mask(t1i.get_fdata())
nib.save(nib.Nifti1Image(t1i.get_fdata().astype(np.float32) * mask, t1i.affine), f"{tmp}/t1.nii.gz")
reg = ants.registration(fixed=FIXED, moving=ants.image_read(f"{tmp}/t1.nii.gz"), type_of_transform=REG)

name = os.path.basename(T1_PATH).split(".nii")[0].replace("_T1w", "")
def save(arr, suffix):
    img = nib.Nifti1Image(arr.astype(np.float32), TMPL.affine); img.set_data_dtype(np.int16)
    nib.save(img, f"{OUT_DIR}/{name}_{suffix}_MNI.nii.gz"); print("saved", f"{OUT_DIR}/{name}_{suffix}_MNI.nii.gz")
save(reg["warpedmovout"].numpy(), "T1w")
if peti is not None:
    nib.save(nib.Nifti1Image(peti.get_fdata().astype(np.float32), peti.affine), f"{tmp}/pet.nii.gz")
    pet_w = ants.apply_transforms(fixed=FIXED, moving=ants.image_read(f"{tmp}/pet.nii.gz"), transformlist=reg["fwdtransforms"])
    save(pet_w.numpy(), "pet")
w = reg["warpedmovout"].numpy(); t = TMPL.get_fdata(); m = t > 0
print(f"Registration check: correlation with the MNI template = {np.corrcoef(w[m], t[m])[0, 1]:.2f} "
      "(compare across your scans; an outlier means check the overlay on the web page)")
