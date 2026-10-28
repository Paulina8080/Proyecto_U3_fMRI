"""
Pipeline de Análisis Estadístico de Primer Nivel en fMRI (GLM)
Basado en: instrucciones.md y Proyecto Funcional 2026_2.pdf
Cohorte de 10 sujetos: 5 Controles Sanos y 5 Pacientes con Epilepsia
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

# Configurar directorio temporal de matplotlib para evitar advertencias de permisos
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

import nilearn.glm.first_level.first_level as fl_module
import nilearn.maskers.nifti_masker as n_nifti_masker
import nilearn.masking as n_masking
from nilearn import plotting
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_epi_mask

# ==============================================================================
# OPTIMIZACIÓN DE MEMORIA PARA SISTEMAS CON RECURSOS LIMITADOS (Windows / 4GB RAM)
# Permite procesar secuencias PRESTO (TR 0.61s con 442+ volúmenes) sin MemoryError
# ==============================================================================

def safe_create_all_designs(self, run_imgs, events, confounds, sample_masks=None):
    if getattr(self, "design_matrices_", None) is not None:
        return self.design_matrices_
    design_matrices = []
    for run_idx, run_img in enumerate(run_imgs):
        if isinstance(run_img, fl_module.SurfaceImage):
            n_scans = run_img.shape[1]
        else:
            img = nib.load(str(run_img)) if isinstance(run_img, (str, Path)) else run_img
            n_scans = img.shape[3]
        design = self._create_single_design(n_scans, events, confounds, run_idx)
        design_matrices.append(design)
    return design_matrices

fl_module.FirstLevelModel._create_all_designs = safe_create_all_designs

orig_apply_mask_fmri = n_masking.apply_mask_fmri

def chunked_apply_mask_fmri(imgs, mask_img, dtype="f", smoothing_fwhm=None, ensure_finite=True):
    if isinstance(imgs, fl_module.SurfaceImage):
        return orig_apply_mask_fmri(imgs, mask_img, dtype=dtype, smoothing_fwhm=smoothing_fwhm, ensure_finite=ensure_finite)
    
    mask_img = n_masking.check_niimg_3d(mask_img)
    mask_data = n_masking.as_ndarray(n_masking.get_data(mask_img), dtype=bool)
    imgs_img = n_masking.check_niimg(imgs)
    
    n_vol = imgs_img.shape[3]
    n_vox = int(mask_data.sum())
    
    # Para pocos volúmenes (SENSE, <= 170 vols) se puede usar el estándar si smoothing != None
    if n_vol <= 170 and smoothing_fwhm is not None:
        return orig_apply_mask_fmri(imgs, mask_img, dtype=dtype, smoothing_fwhm=smoothing_fwhm, ensure_finite=ensure_finite)
    
    # Lectura por bloques contiguos para evitar duplicar en RAM el array completo 4D
    out = np.empty((n_vol, n_vox), dtype=np.float32)
    chunk_size = 50
    for start in range(0, n_vol, chunk_size):
        end = min(start + chunk_size, n_vol)
        chunk = imgs_img.dataobj[..., start:end]
        out[start:end, :] = chunk[mask_data].T
    
    return out

n_masking.apply_mask_fmri = chunked_apply_mask_fmri
n_nifti_masker.apply_mask_fmri = chunked_apply_mask_fmri

print("[INFO] Parches de optimización de memoria instalados exitosamente.")

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS METODOLÓGICOS (instrucciones.md)
# ==============================================================================

RAIZ        = Path.cwd().resolve()
CSV_SUJETOS = RAIZ / "sujetos_46.csv"
DATA_DIR    = RAIZ / "ds005366_subset"
OUT_DIR     = RAIZ / "derivatives_unidad3"
GLM_DIR     = OUT_DIR / "glm_primer_nivel"
MNI_PATH    = Path(r"C:\fMRI\Worskshop-fMRI-preprocessing-basics\data\external\templates\MNI152_T1_2mm.nii.gz")

GLM_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros GLM de primer nivel (siguiendo instrucciones.md)
HRF_MODEL     = "glover"      # HRF canónica de Glover
DRIFT_MODEL   = "cosine"      # Deriva de baja frecuencia por coseno
HIGH_PASS_HZ  = 0.00625       # Corte paso-altas especificado en instrucciones.md (~160s)
NOISE_MODEL   = "ar1"         # Ruido autoregresivo AR(1)
SMOOTHING_FWHM = None         # Los datos ya fueron suavizados a 6 mm en el preprocesamiento
FDR_ALPHA     = 0.05          # Corrección por tasa de falsos descubrimientos (q < 0.05)
CLUSTER_TH    = 10            # Tamaño mínimo de clúster espacial k >= 10 vóxeles

# Selección de los 10 sujetos: 5 Controles Sanos y 5 Pacientes con Epilepsia
# Balanceados por sexo y seleccionados con bajo movimiento relativo dentro de su grupo
COHORTE_10 = [
    # --- 5 Controles Sanos ---
    "sub-083",  # M, 25 años, TR 2.5s, NVol 120
    "sub-084",  # M, 20 años, TR 2.5s, NVol 120
    "sub-118",  # F, 20 años, TR 2.5s, NVol 120
    "sub-111",  # F, 30 años, TR 2.5s, NVol 120
    "sub-132",  # M, 44 años, TR 2.5s, NVol 120 (menor FD de toda la cohorte)
    # --- 5 Pacientes con Epilepsia ---
    "sub-019",  # F, 14 años, TR 1.85s, NVol 145
    "sub-076",  # M, 25 años, TR 1.85s, NVol 145
    "sub-069",  # M, 22 años, TR 0.61s, NVol 442 (menor FD entre epilepsia PRESTO)
    "sub-022",  # F, 42 años, TR 0.61s, NVol 442
    "sub-020",  # F, 15 años, TR 0.61s, NVol 442
]

# ==============================================================================
# FUNCIÓN DE PRIMER NIVEL POR SUJETO
# ==============================================================================

def procesar_primer_nivel(sub_id: str, df_meta: pd.DataFrame, sobrescribir: bool = False):
    """
    Ejecuta el GLM de primer nivel para un sujeto individual según instrucciones.md.
    Calcula:
      1. Matriz de diseño (eventos 'move' con línea base 'rest' implícita + regresores de ruido FD).
      2. Máscara funcional MNI específica del sujeto.
      3. Ajuste GLM (Glover HRF, drift cosine, filtro paso-altas 0.00625 Hz, ruido AR(1)).
      4. Contraste 'move' (movimiento de mano derecha > línea base).
      5. Mapas estadísticos (Z-score, Effect Size, Variance).
      6. Umbralización FDR (q < 0.05, cluster k >= 10).
      7. Figuras de diagnóstico y control de calidad.
    """
    t0 = time.time()
    r = df_meta[df_meta["sub"] == sub_id].iloc[0]
    grupo = r["grupo"]
    sexo = r["sex"]
    edad = r["age"]
    tr = float(r["TR"])
    nombre_base = f"{sub_id}_ses-{r['ses']}_task-Motor2Class_run-{int(r['run'])}"

    d_pre = OUT_DIR / sub_id
    d_glm = GLM_DIR / sub_id
    d_glm.mkdir(parents=True, exist_ok=True)

    # Rutas de salida
    f_z        = d_glm / f"{sub_id}_contrast-move_stat-z.nii.gz"
    f_effect   = d_glm / f"{sub_id}_contrast-move_stat-effect.nii.gz"
    f_var      = d_glm / f"{sub_id}_contrast-move_stat-variance.nii.gz"
    f_fdr      = d_glm / f"{sub_id}_task-Motor2Class_desc-fdr05_statmap.nii.gz"
    f_mask     = d_glm / f"{sub_id}_space-MNI152_desc-brain_mask.nii.gz"
    f_dm_csv   = d_glm / f"{sub_id}_design_matrix.csv"
    f_dm_png   = d_glm / f"{sub_id}_design_matrix.png"
    f_png_z    = d_glm / f"{sub_id}_zmap.png"
    f_png_fdr  = d_glm / f"{sub_id}_zmap_fdr.png"

    # Verificar si ya existe
    if not sobrescribir and f_fdr.exists() and f_z.exists() and f_png_fdr.exists():
        print(f"[{sub_id}] ({grupo}) ya procesado completamente. Se omite re-cálculo.")
        # Cargar métricas existentes
        img_z = nib.load(str(f_z)).get_fdata()
        img_fdr = nib.load(str(f_fdr)).get_fdata()
        return {
            "sub": sub_id,
            "grupo": grupo,
            "sex": sexo,
            "age": edad,
            "TR_s": tr,
            "n_vol": int(r["NVol"]),
            "z_max": float(np.nanmax(img_z)),
            "voxeles_fdr_sig": int((img_fdr > 0).sum()),
            "tiempo_s": 0.0,
            "estado": "ya existia",
        }

    print(f"\n=======================================================")
    print(f" Procesando {sub_id} | Grupo: {grupo.upper()} | Sex: {sexo} | Edad: {edad} | TR: {tr}s")
    print(f"=======================================================")

    bold_path = d_pre / f"{nombre_base}_space-MNI152_desc-preproc_bold.nii.gz"
    media_path = d_pre / f"{nombre_base}_mean_mni.nii.gz"
    ev_path = DATA_DIR / sub_id / f"ses-{r['ses']}" / r["events_file"]
    mot_path = d_pre / "motion.csv"

    if not bold_path.exists():
        raise FileNotFoundError(f"No existe BOLD preprocesado: {bold_path}")
    if not ev_path.exists():
        raise FileNotFoundError(f"No existe archivo de eventos: {ev_path}")

    # 1. Cargar y preparar eventos (bloques de 'move', 'rest' actúa como línea base implícita)
    events_raw = pd.read_csv(ev_path, sep="\t")
    events_df = events_raw[events_raw["trial_type"] == "move"][["onset", "duration", "trial_type"]].reset_index(drop=True)

    # 2. Cargar regresores de confusión de movimiento (FD)
    confounds_df = None
    n_vol = nib.load(str(bold_path)).shape[3]
    if mot_path.exists():
        mot_data = pd.read_csv(mot_path)
        if "FD" in mot_data.columns:
            fd_vals = mot_data["FD"].fillna(0.0).to_numpy(float)
            if len(fd_vals) == n_vol:
                confounds_df = pd.DataFrame({"FD": fd_vals})
                print(f"  [Confounds] FD incluido ({len(fd_vals)} volumenes, FD medio = {fd_vals.mean():.3f} mm)")
            else:
                print(f"  [Aviso] Longitud de FD ({len(fd_vals)}) no coincide con n_vol ({n_vol})")

    # 3. Calcular y guardar máscara funcional en espacio MNI
    print("  [Máscara] Calculando máscara cerebral funcional...")
    mascara_img = compute_epi_mask(str(media_path))
    mascara_img.to_filename(str(f_mask))

    # 4. Configurar e Instanciar FirstLevelModel
    print("  [GLM] Configurando FirstLevelModel (Glover, Cosine, 0.00625 Hz, AR1)...")
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model=HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=HIGH_PASS_HZ,
        mask_img=mascara_img,
        smoothing_fwhm=SMOOTHING_FWHM,
        noise_model=NOISE_MODEL,
        standardize=False,
        minimize_memory=True,
        n_jobs=2,
        verbose=0,
    )

    # 5. Ajuste del modelo
    t_fit_start = time.time()
    print("  [GLM] Ajustando modelo a la serie funcional...")
    glm.fit(str(bold_path), events=events_df, confounds=confounds_df)
    t_fit = time.time() - t_fit_start
    print(f"  [GLM] Modelo ajustado exitosamente en {t_fit:.1f} s.")

    # Guardar matriz de diseño
    design_matrix = glm.design_matrices_[0]
    design_matrix.to_csv(f_dm_csv, index=False)

    # Visualizar y guardar figura de la matriz de diseño
    fig_dm = plotting.plot_design_matrix(design_matrix, output_file=str(f_dm_png))
    plt.close("all")

    # 6. Estimación del Contraste 'move' (mano derecha > reposo)
    print("  [Contraste] Estimando contraste 'move' vs. reposo...")
    contrast_res = glm.compute_contrast("move", output_type="all")
    mapa_z = contrast_res["z_score"]
    mapa_effect = contrast_res["effect_size"]
    mapa_var = contrast_res["effect_variance"]

    mapa_z.to_filename(str(f_z))
    mapa_effect.to_filename(str(f_effect))
    mapa_var.to_filename(str(f_var))

    z_vals = mapa_z.get_fdata()
    z_max = float(np.nanmax(z_vals))
    print(f"  [Estadística] Z máximo = {z_max:.2f}")

    # 7. Umbralización Estadística con Corrección por Múltiples Comparaciones FDR (q < 0.05, k >= 10)
    print("  [Umbralización] Aplicando FDR (q < 0.05, cluster_threshold >= 10)...")
    mapa_z_fdr, umbral_fdr = threshold_stats_img(
        stat_img=mapa_z,
        alpha=FDR_ALPHA,
        height_control="fdr",
        cluster_threshold=CLUSTER_TH,
    )
    print(f"  [FDR] Umbral Z equivalente: {umbral_fdr:.3f}")
    mapa_z_fdr.to_filename(str(f_fdr))

    fdr_vals = mapa_z_fdr.get_fdata()
    vox_sig = int((fdr_vals > 0).sum())
    print(f"  [FDR] Vóxeles significativos activos: {vox_sig}")

    # 8. Visualización de los Mapas Estadísticos sobre MNI152
    bg_template = str(MNI_PATH) if MNI_PATH.exists() else None

    # Figura A: Mapa Z no umbralizado (umbral visual z > 3.1)
    fig_z = plt.figure(figsize=(11, 3.2))
    plotting.plot_stat_map(
        mapa_z,
        bg_img=bg_template,
        threshold=3.1,
        display_mode="z",
        cut_coords=[46, 52, 58, 64],
        figure=fig_z,
        title=f"{sub_id} ({grupo}, {edad}a) move > rest, Z > 3.1",
    )
    fig_z.savefig(f_png_z, dpi=120)
    plt.close(fig_z)

    # Figura B: Mapa corregido por FDR (q < 0.05, k >= 10)
    fig_fdr = plt.figure(figsize=(11, 3.2))
    plotting.plot_stat_map(
        mapa_z_fdr,
        bg_img=bg_template,
        threshold=umbral_fdr,
        display_mode="z",
        cut_coords=[46, 52, 58, 64],
        figure=fig_fdr,
        title=f"{sub_id} ({grupo}) FDR q<0.05 (Z_thr={umbral_fdr:.2f}, k>={CLUSTER_TH}, {vox_sig} vox)",
    )
    fig_fdr.savefig(f_png_fdr, dpi=120)
    plt.close(fig_fdr)

    # Liberar memoria de variables pesadas
    del glm, contrast_res, mapa_z, mapa_effect, mapa_var, mapa_z_fdr, z_vals, fdr_vals
    gc.collect()

    t_total = time.time() - t0
    print(f"  [Completado] {sub_id} procesado exitosamente en {t_total:.1f} s.")

    return {
        "sub": sub_id,
        "grupo": grupo,
        "sex": sexo,
        "age": edad,
        "TR_s": tr,
        "n_vol": n_vol,
        "z_max": round(z_max, 2),
        "umbral_fdr": round(float(umbral_fdr), 3),
        "voxeles_fdr_sig": vox_sig,
        "tiempo_s": round(t_total, 1),
        "estado": "ok",
    }

# ==============================================================================
# EJECUCIÓN POR LOTE DE LA COHORTE DE 10 SUJETOS
# ==============================================================================

def main():
    print("=" * 70)
    print(" INICIANDO ANÁLISIS DE PRIMER NIVEL fMRI - COHORTE DE 10 SUJETOS")
    print(f" 5 Controles Sanos y 5 Pacientes con Epilepsia")
    print("=" * 70)

    df_meta = pd.read_csv(CSV_SUJETOS)
    resultados = []

    for sub_id in COHORTE_10:
        try:
            res = procesar_primer_nivel(sub_id, df_meta, sobrescribir=False)
            resultados.append(res)
        except Exception as e:
            traceback.print_exc()
            resultados.append({
                "sub": sub_id,
                "grupo": df_meta[df_meta["sub"] == sub_id]["grupo"].iloc[0],
                "sex": df_meta[df_meta["sub"] == sub_id]["sex"].iloc[0],
                "age": df_meta[df_meta["sub"] == sub_id]["age"].iloc[0],
                "TR_s": df_meta[df_meta["sub"] == sub_id]["TR"].iloc[0],
                "n_vol": df_meta[df_meta["sub"] == sub_id]["NVol"].iloc[0],
                "z_max": np.nan,
                "umbral_fdr": np.nan,
                "voxeles_fdr_sig": 0,
                "tiempo_s": 0.0,
                "estado": f"ERROR: {e}",
            })

    # Guardar resumen en CSV
    df_res = pd.DataFrame(resultados)
    resumen_path = GLM_DIR / "resumen_primer_nivel_cohorte10.csv"
    df_res.to_csv(resumen_path, index=False)
    print("\n" + "=" * 70)
    print(" RESUMEN ESTADÍSTICO DE PRIMER NIVEL (COHORTE DE 10 SUJETOS)")
    print("=" * 70)
    print(df_res[["sub", "grupo", "sex", "age", "z_max", "umbral_fdr", "voxeles_fdr_sig", "estado"]])
    print("\n[PROMEDIOS POR GRUPO]")
    print(df_res.groupby("grupo")[["z_max", "voxeles_fdr_sig"]].describe().T)

    # ==============================================================================
    # FIGURA COMPARATIVA MULTI-SUJETO: CONTROLES SANOS VS. PACIENTES CON EPILEPSIA
    # ==============================================================================
    print("\n[INFO] Generando figura comparativa de mapas de activación FDR...")
    try:
        import matplotlib.image as mpimg
        fig, axes = plt.subplots(len(COHORTE_10), 1, figsize=(14, 2.8 * len(COHORTE_10)))
        for idx, sub_id in enumerate(COHORTE_10):
            ax = axes[idx]
            f_png_fdr = GLM_DIR / sub_id / f"{sub_id}_zmap_fdr.png"
            if f_png_fdr.exists():
                img_arr = mpimg.imread(str(f_png_fdr))
                ax.imshow(img_arr)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, f"{sub_id}: No disponible", ha="center", va="center")
                ax.axis("off")
        fig.tight_layout()
        fig_comp_path = GLM_DIR / "comparativa_primer_nivel_sanos_vs_epilepsia.png"
        fig.savefig(fig_comp_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] Figura comparativa guardada en: {fig_comp_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo generar la figura compuesta: {e}")
    print("\n[SUCCESS] Procesamiento de primer nivel finalizado exitosamente.")

if __name__ == "__main__":
    main()
