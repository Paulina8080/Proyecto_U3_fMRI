import nibabel as nib
import numpy as np

# 1. Supongamos que tu máscara de grupo en memoria es un arreglo numpy 3D (binario: 1 para los vóxeles válidos, 0 para los demás)
# mascara_grupo_arr = ... (tu matriz resultante de la Parte E)

# 2. Cargas una imagen de referencia (puede ser la plantilla MNI o el header de cualquier sujeto)
ref_img = nib.load(str(MNI_PATH))

# 3. Creas la imagen NIfTI conservando la matriz afín (affine) y el espacio de coordenadas original
mask_img = nib.Nifti1Image(
    mascara_grupo_arr.astype(np.uint8), ref_img.affine, ref_img.header
)

# 4. La guardas en tu carpeta de derivatives (por ejemplo, en una nueva subcarpeta o junto al primer nivel)
mask_output_path = (
    OUT_DIR / "mascara_grupo_cobertura.nii.gz"
)  # O en la ruta que prefieras
nib.save(mask_img, str(mask_output_path))

print(f"Máscara de grupo guardada exitosamente en: {mask_output_path}")