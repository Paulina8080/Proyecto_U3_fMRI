"""Analisis de segundo nivel (grupo) - Motor2Class mano derecha, ds005366.
Uso: python segundo_nivel.py <carpeta glm_primer_nivel> <mascara_grupo.nii.gz> <carpeta salida>
Entradas: mapas de EFECTO (beta) del contraste move de cada sujeto + resumen_primer_nivel_46sujetos.csv
"""
import sys, os, numpy as np, pandas as pd, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn import plotting, image, datasets
from nilearn.maskers import NiftiSpheresMasker

GLM, MASK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NPERM = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(GLM, "resumen_primer_nivel_46sujetos.csv"))
maps = [f"{GLM}/{s}/{s}_contrast-move_stat-effect.nii.gz" for s in df["sub"]]
mask = nib.load(MASK)
bg = datasets.load_mni152_template(resolution=1)

# Matriz de diseno: una columna por grupo + edad centrada + sexo centrado (F=+0.5, M=-0.5)
X = pd.DataFrame({
    "sano": (df.grupo == "sano").astype(float),
    "epilepsia": (df.grupo == "epilepsia").astype(float),
    "edad": df.age - df.age.mean(),
    "sexo": np.where(df.sex == "F", 0.5, -0.5),
})
X.to_csv(f"{OUT}/matriz_diseno_segundo_nivel.csv", index=False)
slm = SecondLevelModel(mask_img=mask, smoothing_fwhm=None).fit(maps, design_matrix=X)
gl = len(df) - X.shape[1]

contrastes = {
    "media_sanos": [1, 0, 0, 0],
    "media_epilepsia": [0, 1, 0, 0],
    "sanos_menos_epilepsia": [1, -1, 0, 0],
}
resumen = []
for nombre, c in contrastes.items():
    out = slm.compute_contrast(c, output_type="all")
    z, eff = out["z_score"], out["effect_size"]
    z.to_filename(f"{OUT}/{nombre}_z.nii.gz"); eff.to_filename(f"{OUT}/{nombre}_efecto.nii.gz")
    dos_colas = nombre.startswith("sanos_menos")
    for ctrl, kw in {"fdr": dict(height_control="fdr", alpha=0.05, cluster_threshold=20),
                     "bonferroni": dict(height_control="bonferroni", alpha=0.05, cluster_threshold=0)}.items():
        zt, thr = threshold_stats_img(z, mask_img=mask, two_sided=dos_colas, **kw)
        zt.to_filename(f"{OUT}/{nombre}_z_{ctrl}.nii.gz")
        d = zt.get_fdata()
        resumen.append(dict(contraste=nombre, control=ctrl, umbral_z=round(float(thr), 3),
                            vox_pos=int((d > 0).sum()), vox_neg=int((d < 0).sum()),
                            z_max=round(float(np.nanmax(z.get_fdata())), 2), z_min=round(float(np.nanmin(z.get_fdata())), 2)))
        if ctrl == "fdr":
            if np.isfinite(thr) and (d != 0).any():
                tab = get_clusters_table(z, stat_threshold=thr, cluster_threshold=20, two_sided=dos_colas, min_distance=16)
                tab.to_csv(f"{OUT}/{nombre}_clusters_fdr.csv", index=False)
            fig = plt.figure(figsize=(12, 3.3))
            plotting.plot_stat_map(zt if (d != 0).any() else z, bg_img=bg, threshold=thr if np.isfinite(thr) else 3.1,
                                   display_mode="z", cut_coords=[-10, 6, 22, 38, 48, 56, 64], figure=fig, cmap="cold_hot",
                                   title=f"{nombre}  FDR q<0.05 (Z>{thr:.2f}, k>=20)")
            fig.savefig(f"{OUT}/{nombre}_fdr.png", dpi=150); plt.close(fig)
    fig = plt.figure(figsize=(12, 3.3))
    plotting.plot_stat_map(z, bg_img=bg, threshold=3.1, display_mode="z", cut_coords=[-10, 6, 22, 38, 48, 56, 64],
                           figure=fig, cmap="cold_hot", title=f"{nombre}  |Z|>3.1 sin corregir")
    fig.savefig(f"{OUT}/{nombre}_z31.png", dpi=150); plt.close(fig)

# Permutaciones (max-T, FWE) para la diferencia entre grupos, dos colas
neglog = non_parametric_inference(maps, design_matrix=X, second_level_contrast=[1, -1, 0, 0], mask=mask,
                                  model_intercept=False, n_perm=NPERM, two_sided_test=True, n_jobs=2, random_state=0, verbose=0)
neglog.to_filename(f"{OUT}/sanos_menos_epilepsia_neglog10p_fwe_perm.nii.gz")
p = neglog.get_fdata()
resumen.append(dict(contraste="sanos_menos_epilepsia", control=f"FWE permutaciones (n={NPERM})", umbral_z=np.nan,
                    vox_pos=int((p > -np.log10(0.05)).sum()), vox_neg=0, z_max=round(float(p.max()), 3), z_min=np.nan))
pd.DataFrame(resumen).to_csv(f"{OUT}/resumen_segundo_nivel.csv", index=False)

# ROI: esferas de 8 mm. M1 mano izq/der y SMA
rois = {"M1_izq": (-38, -22, 56), "M1_der": (38, -22, 56), "SMA": (-4, -6, 58)}
vals = NiftiSpheresMasker(list(rois.values()), radius=8, mask_img=mask).fit_transform(maps)
roi = df[["sub", "grupo", "sex", "age", "TR", "NVol"]].copy()
for i, k in enumerate(rois): roi[k] = np.asarray(vals)[:, i]
L, R = roi.M1_izq.clip(lower=0), roi.M1_der.clip(lower=0)
roi["LI_M1"] = (L - R) / (L + R).replace(0, np.nan)
roi.to_csv(f"{OUT}/roi_por_sujeto.csv", index=False)
filas = []
Xm = X.values
for k in list(rois) + ["LI_M1"]:
    y = roi[k].values; ok = ~np.isnan(y)
    b, *_ = np.linalg.lstsq(Xm[ok], y[ok], rcond=None)
    res = y[ok] - Xm[ok] @ b; dof = ok.sum() - Xm.shape[1]
    cov = res @ res / dof * np.linalg.inv(Xm[ok].T @ Xm[ok])
    c = np.array([1, -1, 0, 0]); t = c @ b / np.sqrt(c @ cov @ c)
    s, e = y[(df.grupo == "sano") & ok], y[(df.grupo == "epilepsia") & ok]
    filas.append(dict(roi=k, sanos_media=s.mean(), sanos_de=s.std(ddof=1), epi_media=e.mean(), epi_de=e.std(ddof=1),
                      dif_ajustada=c @ b, t=t, gl=dof, p=2 * stats.t.sf(abs(t), dof),
                      p_mannwhitney=stats.mannwhitneyu(s, e).pvalue,
                      p_sanos_vs0=stats.ttest_1samp(s, 0).pvalue, p_epi_vs0=stats.ttest_1samp(e, 0).pvalue))
pd.DataFrame(filas).round(4).to_csv(f"{OUT}/roi_estadisticas.csv", index=False)

fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, k in zip(axs, list(rois) + ["LI_M1"]):
    for j, (g, col) in enumerate([("sano", "#2a78d6"), ("epilepsia", "#d6532a")]):
        y = roi.loc[roi.grupo == g, k].dropna()
        ax.scatter(np.random.RandomState(1).normal(j, 0.06, len(y)), y, s=22, color=col, alpha=0.75)
        ax.hlines(y.mean(), j - 0.25, j + 0.25, color="k", lw=2)
    ax.axhline(0, color="gray", lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["Sanos", "Epilepsia"])
    ax.set_title(k.replace("_", " ")); ax.set_ylabel("Indice de lateralidad" if k == "LI_M1" else "Efecto (beta), u.a.")
fig.tight_layout(); fig.savefig(f"{OUT}/roi_grupos.png", dpi=150); plt.close(fig)
print(pd.DataFrame(resumen).to_string()); print(pd.DataFrame(filas).round(4).to_string()); print("gl", gl)
