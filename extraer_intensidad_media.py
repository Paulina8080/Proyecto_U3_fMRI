import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path("sujetos_46.csv")
out_dir = Path("derivatives_unidad3")
glm_dir = out_dir / "glm_primer_nivel"

df = pd.read_csv(csv_path)

filas = []
for idx, r in df.iterrows():
    sub = r["sub"]
    ses = r["ses"]
    run = int(r["run"])
    nombre = f"{sub}_ses-{ses}_task-Motor2Class_run-{run}"
    
    f_mean_mni = out_dir / sub / f"{nombre}_mean_mni.nii.gz"
    f_bold_orig = out_dir / sub / "bold_mean.nii.gz"
    f_mask = glm_dir / sub / f"{sub}_space-MNI152_desc-brain_mask.nii.gz"
    
    # 1. En espacio MNI dentro de la máscara funcional
    val_mni = np.nan
    std_mni = np.nan
    if f_mean_mni.exists() and f_mask.exists():
        data_mni = nib.load(str(f_mean_mni)).get_fdata()
        mask = nib.load(str(f_mask)).get_fdata() > 0
        vox_brain = data_mni[mask]
        val_mni = float(np.mean(vox_brain))
        std_mni = float(np.std(vox_brain))
    elif f_mean_mni.exists():
        data_mni = nib.load(str(f_mean_mni)).get_fdata()
        vox_brain = data_mni[data_mni > 0]
        val_mni = float(np.mean(vox_brain))
        std_mni = float(np.std(vox_brain))
        
    # 2. En espacio nativo (bold_mean.nii.gz)
    val_nativo = np.nan
    std_nativo = np.nan
    if f_bold_orig.exists():
        data_orig = nib.load(str(f_bold_orig)).get_fdata()
        vox_orig = data_orig[data_orig > 0]
        val_nativo = float(np.mean(vox_orig))
        std_nativo = float(np.std(vox_orig))
        
    filas.append({
        "sub": sub,
        "grupo": r["grupo"],
        "sex": r["sex"],
        "age": r["age"],
        "TR": r["TR"],
        "Seq": r["Seq"],
        "intensidad_media_cerebro_mni": round(val_mni, 2),
        "std_cerebro_mni": round(std_mni, 2),
        "intensidad_media_nativo": round(val_nativo, 2),
        "std_nativo": round(std_nativo, 2)
    })

res_df = pd.DataFrame(filas)
csv_salida = out_dir / "intensidad_media_sujetos.csv"
res_df.to_csv(csv_salida, index=False)
print(f"Archivo guardado exitosamente en: {csv_salida}")
print(f"Total sujetos procesados: {len(res_df)}")
