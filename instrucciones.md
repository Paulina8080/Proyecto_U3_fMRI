# Guía Paso a Paso: Análisis de Primer Nivel en fMRI (Task-fMRI / GLM)

Esta guía detalla el procedimiento metodológico y de programación en Python (utilizando `Nilearn`) para ejecutar el análisis estadístico de primer nivel (modelo lineal general a nivel de sujeto individual) en una cohorte de al menos 10 sujetos (5 controles sanos y 5 pacientes con epilepsia).

---

## 1. Requisitos Previos y Datos de Entrada
Antes de iniciar el modelo lineal general (GLM) de primer nivel, cada sujeto debe haber superado exitosamente el preprocesamiento anatómico y funcional[cite: 10]:
* **Imagen BOLD preprocesada y normalizada** al espacio estándar MNI (`*_desc-preproc_bold.nii.gz`).
* **Máscara cerebral funcional** en espacio MNI (`*_desc-brain_mask.nii.gz`)[cite: 8].
* **Template anatómico MNI152** de referencia para visualización[cite: 8].
* **Archivos de eventos BIDS** (`*_events.tsv`) que especifican los tiempos de inicio (*onset*), duración y condición de los estímulos o tareas para cada sujeto[cite: 8].
* **Regresores de confusión** (matriz de movimiento de 24 parámetros HMP y componentes de ruido fisiológico aCompCor)[cite: 8].

---

## 2. Configuración del Entorno y Carga de Datos por Sujeto
Para cada uno de los 10 sujetos (sanos y con epilepsia), se cargan las rutas correspondientes, los parámetros de adquisición (TR) y la tabla de eventos experimentales[cite: 8]:

```python
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting

# Ejemplo iterando sobre la tabla de sujetos preprocesados
# (Asegurarse de recorrer tanto los 5 sanos como los 5 con epilepsia)
for idx, fila_bold in resultados_bold_mni.iterrows():
    sujeto = fila_bold["subject"]
    tr = float(fila_bold["TR_s"])
    bold_preproc = Path(fila_bold["normalized_bold_path"])
    mascara_mni = Path(fila_bold["normalized_mask_path"])
    t1_mni = Path(fila_bold["mni_template_path"])
    
    # Cargar archivo de eventos BIDS del sujeto
    events_file = PATHS["bids"] / f"sub-{sujeto}" / "func" / f"sub-{sujeto}_task-Paradigma_events.tsv"
    events_df = pd.read_csv(events_file, sep=",")

# Cargar parámetros de movimiento
confounds_mov = pd.read_csv(fila_bold["motion_path"], sep="\t")
cols_mov = ["trans_x_mm", "trans_y_mm", "trans_z_mm", "rot_x_rad", "rot_y_rad", "rot_z_rad"]
base_mov = confounds_mov[cols_mov].fillna(0.0).values
deriv_mov = np.vstack([np.zeros((1, 6)), np.diff(base_mov, axis=0)])

# Matriz de 24 HMP (Motion parameters: base, derivadas y potencias al cuadrado)
mov_24 = np.column_stack([base_mov, deriv_mov, base_mov**2, deriv_mov**2])
nombres_mov = cols_mov + [f"{c}_deriv" for c in cols_mov] + [f"{c}_sq" for c in cols_mov] + [f"{c}_deriv_sq" for c in cols_mov]
confounds_total = pd.DataFrame(mov_24, columns=nombres_mov)

# Opcional: Agregar componentes aCompCor de sustancia blanca y LCR si están disponibles

glm_primer_nivel = FirstLevelModel(
    t_r=tr,
    hrf_model="glover",                  # HRF canónica de Glover[cite: 8]
    drift_model="cosine",                # Modelo de deriva de baja frecuencia[cite: 8]
    high_pass=0.00625,                   # Período de corte estándar para el filtro paso alto[cite: 8]
    mask_img=str(mascara_mni),           # Máscara cerebral en espacio MNI[cite: 8]
    smoothing_fwhm=4.0,                  # Suavizado espacial gaussiano (ej. 4-6 mm)[cite: 8]
    noise_model="ar1",                   # Modelo de ruido autocorrelacionado temporal[cite: 8]
    standardize=False,
    verbose=1
)

# Ajuste del modelo con las imágenes BOLD, eventos y confounds
glm_primer_nivel = glm_primer_nivel.fit(
    run_imgs=str(bold_preproc),
    events=events_df,
    confounds=confounds_total
)


from nilearn.glm import threshold_stats_img

# Aplicar umbralización y corrección FDR (q < 0.05) con un tamaño mínimo de clúster (k >= 10)
mapa_z_fdr, umbral_fdr = threshold_stats_img(
    stat_img=mapa_z,
    alpha=0.05,
    height_control="fdr",
    cluster_threshold=10
)
print(f"Umbral Z equivalente para FDR (q < 0.05): {umbral_fdr:.3f}")

dir_salida = PATHS["derivatives"] / "first_level" / f"sub-{sujeto}" / "func"
dir_salida.mkdir(parents=True, exist_ok=True)

ruta_salida = dir_salida / f"sub-{sujeto}_task-Paradigma_desc-fdr05_statmap.nii.gz"
mapa_z_fdr.to_filename(str(ruta_salida))

Consideraciones Especiales para el Grupo con Epilepsia
Análisis comparativo: Asegúrate de aplicar exactamente el mismo pipeline de primer nivel (mismo HRF, mismo filtrado, mismos parámetros de confusión) tanto a los 5 sujetos sanos como a los 5 pacientes con epilepsia para evitar sesgos sistemáticos.

Control de artefactos por interictal epileptiform discharges (IEDs) o movimientos: Los pacientes con epilepsia pueden presentar mayor propensión al movimiento; verifica detalladamente las métricas de desplazamiento (FD) y considera un censurado (scrubbing) o la inclusión estricta de regresores adicionales si se detectan picos severos de ruido.