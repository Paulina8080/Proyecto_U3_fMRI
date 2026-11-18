import sys, os, numpy as np, pandas as pd, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from PIL import Image
from nilearn import plotting, datasets
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn.maskers import NiftiSpheresMasker
GLM, MASK, OUT, FIG, MEDIA, REPO = sys.argv[1:7]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
bg = datasets.load_mni152_template(resolution=1)
df = pd.read_csv(f"{GLM}/resumen_primer_nivel_46sujetos.csv")
suj = pd.read_csv(f"{REPO}/sujetos_46.csv")[["sub", "handedness", "Seq"]]
fd = pd.read_csv(f"{REPO}/derivatives_unidad3/qc_movimiento/resumen_FD_46.csv")[["sub", "FD_medio_mm", "FD_max_mm", "pct_vol_FD_gt_0_5"]]
df = df.merge(suj, on="sub").merge(fd, on="sub")
df.to_csv(f"{OUT}/tabla_46_sujetos.csv", index=False)
maps = [f"{GLM}/{s}/{s}_contrast-move_stat-effect.nii.gz" for s in df["sub"]]
mask = nib.load(MASK)
CUTS = [-10, 6, 22, 38, 48, 56, 64]

# ---- Fig: grupo (a) sanos (b) epilepsia (c) diferencia sin corregir
fig, axs = plt.subplots(3, 1, figsize=(10, 6.9))
for ax, nombre, tit in zip(axs[:2], ["media_sanos", "media_epilepsia"], ["(a) Sanos (n = 30)", "(b) Epilepsia (n = 16)"]):
    z = nib.load(f"{OUT}/{nombre}_z.nii.gz")
    zt, thr = threshold_stats_img(z, mask_img=mask, alpha=0.05, height_control="fdr", cluster_threshold=20, two_sided=False)
    plotting.plot_stat_map(zt, bg_img=bg, threshold=thr, display_mode="z", cut_coords=CUTS, axes=ax, cmap="inferno",
                           symmetric_cbar=False, vmin=0, vmax=9, black_bg=True, annotate=True)
    ax.set_title(f"{tit}: movimiento > reposo, FDR q < 0,05 (Z > {thr:.2f}; k ≥ 20)".replace(".", ","), fontsize=9, loc="left")
plotting.plot_stat_map(nib.load(f"{OUT}/sanos_menos_epilepsia_z.nii.gz"), bg_img=bg, threshold=3.1, display_mode="z", cut_coords=CUTS,
                       axes=axs[2], cmap="cold_hot", vmax=5, black_bg=True)
axs[2].set_title("(c) Sanos − epilepsia, |Z| > 3,1 sin corregir (ningún vóxel sobrevive la corrección)", fontsize=9, loc="left")
fig.savefig(f"{FIG}/fig_grupo.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- tablas de picos (umbral Bonferroni para separar regiones)
for nombre in ["media_sanos", "media_epilepsia"]:
    z = nib.load(f"{OUT}/{nombre}_z.nii.gz")
    t = get_clusters_table(z, stat_threshold=4.96, cluster_threshold=10, min_distance=20)
    t.to_csv(f"{OUT}/{nombre}_picos_bonferroni.csv", index=False); print(nombre); print(t.to_string())
t = get_clusters_table(nib.load(f"{OUT}/sanos_menos_epilepsia_z.nii.gz"), stat_threshold=3.1, cluster_threshold=10, two_sided=True)
t.to_csv(f"{OUT}/sanos_menos_epilepsia_picos_sin_corregir.csv", index=False); print(t.to_string())

# ---- ROI + sensibilidad
rois = {"M1_izq": (-38, -22, 56), "M1_der": (38, -22, 56), "SMA": (-4, -6, 58)}
vals = np.asarray(NiftiSpheresMasker(list(rois.values()), radius=8, mask_img=mask).fit_transform(maps))
for i, k in enumerate(rois): df[k] = vals[:, i]
L, R = df.M1_izq.clip(lower=0), df.M1_der.clip(lower=0); df["LI_M1"] = (L - R) / (L + R)
def glm_roi(d, y):
    X = np.c_[(d.grupo == "sano"), (d.grupo == "epilepsia"), d.age - d.age.mean(), np.where(d.sex == "F", .5, -.5)].astype(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b; dof = len(y) - 4
    c = np.array([1, -1, 0, 0]); se = np.sqrt(c @ (r @ r / dof * np.linalg.inv(X.T @ X)) @ c)
    return c @ b, c @ b / se, dof, 2 * stats.t.sf(abs(c @ b / se), dof)
sets = {"Todos (30 vs. 16)": df,
        "Sin sub-025 y sub-032 (30 vs. 14)": df[~df["sub"].isin(["sub-025", "sub-032"])],
        "Pacientes adultos diestros (30 vs. 8)": df[(df.grupo == "sano") | ((df.age >= 18) & (df.handedness == "R"))],
        "Pacientes TR 0,61 s (30 vs. 14)": df[(df.grupo == "sano") | (df.TR == 0.61)]}
filas = []
for nom, d in sets.items():
    X = pd.DataFrame({"sano": (d.grupo == "sano").astype(float), "epilepsia": (d.grupo == "epilepsia").astype(float),
                      "edad": d.age - d.age.mean(), "sexo": np.where(d.sex == "F", .5, -.5)})
    z = SecondLevelModel(mask_img=mask).fit([maps[i] for i in d.index], design_matrix=X).compute_contrast([1, -1, 0, 0])
    zt, thr = threshold_stats_img(z, mask_img=mask, alpha=0.05, height_control="fdr", two_sided=True)
    fila = dict(analisis=nom, n=len(d), vox_fdr=int((zt.get_fdata() != 0).sum()), z_max=float(z.get_fdata().max()), z_min=float(z.get_fdata().min()))
    for k in ["M1_izq", "SMA", "LI_M1"]:
        dif, t_, dof, p = glm_roi(d, d[k].values); fila[f"{k}_dif"] = dif; fila[f"{k}_t"] = t_; fila[f"{k}_p"] = p; fila["gl"] = dof
    filas.append(fila)
# --- sensibilidad adicional: (i) solo voxeles con datos en los 46 sujetos, (ii) FD medio como covariable
datos = np.stack([nib.load(m).get_fdata() != 0 for m in maps]).all(0) & (mask.get_fdata() > 0)
m_int = nib.Nifti1Image(datos.astype(np.uint8), mask.affine); m_int.to_filename(f"{OUT}/mascara_interseccion_46.nii.gz")
Xb = pd.DataFrame({"sano": (df.grupo == "sano").astype(float), "epilepsia": (df.grupo == "epilepsia").astype(float),
                   "edad": df.age - df.age.mean(), "sexo": np.where(df.sex == "F", .5, -.5)})
def fila_extra(nom, X, m):
    z = SecondLevelModel(mask_img=m).fit(maps, design_matrix=X).compute_contrast([1, -1] + [0] * (X.shape[1] - 2))
    zt, thr = threshold_stats_img(z, mask_img=m, alpha=0.05, height_control="fdr", two_sided=True)
    fila = dict(analisis=nom, n=len(df), vox_fdr=int((zt.get_fdata() != 0).sum()), z_max=float(z.get_fdata().max()), z_min=float(z.get_fdata().min()))
    Xm = X.values
    for k in ["M1_izq", "SMA", "LI_M1"]:
        y = df[k].values; b = np.linalg.lstsq(Xm, y, rcond=None)[0]; r = y - Xm @ b; dof = len(y) - Xm.shape[1]
        c = np.zeros(Xm.shape[1]); c[0], c[1] = 1, -1; se = np.sqrt(c @ (r @ r / dof * np.linalg.inv(Xm.T @ Xm)) @ c)
        fila[f"{k}_dif"] = c @ b; fila[f"{k}_t"] = c @ b / se; fila[f"{k}_p"] = 2 * stats.t.sf(abs(c @ b / se), dof); fila["gl"] = dof
    return fila
filas.append(fila_extra(f"Solo voxeles con datos en los 46 ({int(datos.sum())} vox)", Xb, m_int))
# El FD medio no se usa como covariable de grupo: sus rangos no se solapan entre grupos.
print("voxeles interseccion", int(datos.sum()), "de", int((mask.get_fdata() > 0).sum()))
sens = pd.DataFrame(filas).round(3); sens.to_csv(f"{OUT}/sensibilidad.csv", index=False); print(sens.to_string())
# correlaciones de M1 con edad y FD
for g in ["sano", "epilepsia"]:
    d = df[df.grupo == g]; print(g, "M1~edad", stats.spearmanr(d.age, d.M1_izq), "M1~FD", stats.spearmanr(d.FD_medio_mm, d.M1_izq))
print(df.groupby("grupo")[["age", "FD_medio_mm", "FD_max_mm", "pct_vol_FD_gt_0_5", "z_max", "voxeles_fdr_sig"]].agg(["mean", "std", "median", "min", "max"]).round(2).T.to_string())
print("edad t-test", stats.ttest_ind(df[df.grupo=="sano"].age, df[df.grupo=="epilepsia"].age, equal_var=False))
print("LI<0.5:", df[df.LI_M1 < 0.5][["sub", "grupo", "age", "handedness", "LI_M1"]].round(2).to_string())
df.round(4).to_csv(f"{OUT}/roi_por_sujeto.csv", index=False)

fig, axs = plt.subplots(2, 2, figsize=(5.2, 4.8))
tit = {"M1_izq": "M1 izquierda", "M1_der": "M1 derecha", "SMA": "SMA", "LI_M1": "Índice de lateralidad (M1)"}
for ax, k in zip(axs.ravel(), tit):
    for j, (g, col) in enumerate([("sano", "#2a78d6"), ("epilepsia", "#d6532a")]):
        y = df.loc[df.grupo == g, k]
        ax.scatter(np.random.RandomState(1).normal(j, 0.07, len(y)), y, s=14, color=col, alpha=0.75, linewidths=0)
        ax.hlines(y.mean(), j - 0.28, j + 0.28, color="k", lw=1.8)
    ax.axhline(0, color="gray", lw=0.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["Sanos", "Epilepsia"]); ax.set_xlim(-.6, 1.6)
    ax.set_title(tit[k], fontsize=9); ax.set_ylabel("LI" if k == "LI_M1" else "Efecto β (% de señal)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_roi.png", dpi=220); plt.close(fig)

# ---- cobertura
fig = plt.figure(figsize=(5.2, 2.0))
plotting.plot_roi(mask, bg_img=bg, display_mode="ortho", cut_coords=(-38, -22, 56), figure=fig, cmap="autumn", alpha=0.45, black_bg=True, draw_cross=True)
fig.savefig(f"{FIG}/fig_cobertura.png", dpi=220, facecolor="black"); plt.close(fig)

# ---- diagrama de seleccion
fig, ax = plt.subplots(figsize=(5.2, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def caja(x, y, w, h, txt, fc="#eef3fa"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec="#33475b", lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.6)
def fl(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#33475b", lw=0.9))
caja(2.2, 8.8, 5.6, 1.0, "Conjunto de datos público\n155 participantes (92 sanos, 63 pacientes)")
caja(0.6, 6.3, 8.8, 1.7, "Filtros comunes: Motor2Class, dedos de la mano derecha;\n1,5 T o 3 T; sin movimiento excesivo (Tablas 1 y 2 de [5]);\ncobertura completa de la corteza sensoriomotora; T1w en la sesión;\nsin tumor", fc="#fff7e6")
caja(0.2, 3.9, 4.4, 1.5, "Sanos adultos diestros:\n70 candidatos")
caja(5.4, 3.9, 4.4, 1.5, "Pacientes con epilepsia adultos,\ndiestros: 8 candidatos")
caja(5.4, 1.9, 4.4, 1.3, "Se admiten menores (≥ 11 años),\n1 zurdo y 1 ambidiestro", fc="#fff7e6")
caja(0.2, 0.1, 4.4, 1.2, "30 sanos (15 M, 15 H)\nedad pareada con pacientes", fc="#e6f4ea")
caja(5.4, 0.1, 4.4, 1.2, "16 pacientes (8 M, 8 H)", fc="#e6f4ea")
fl(5, 8.72, 5, 8.1); fl(3, 6.22, 2.4, 5.5); fl(7, 6.22, 7.6, 5.5); fl(2.4, 3.82, 2.4, 1.4); fl(7.6, 3.82, 7.6, 3.3); fl(7.6, 1.82, 7.6, 1.4)
fig.savefig(f"{FIG}/fig_seleccion.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---- composiciones de imagenes existentes
def apilar(paths, cols, out, pad=12):
    ims = [Image.open(p).convert("RGB") for p in paths]; w = max(i.width for i in ims)
    ims = [i.resize((w, int(i.height * w / i.width))) for i in ims]; rows = -(-len(ims) // cols)
    hs = [max(i.height for i in ims[r * cols:(r + 1) * cols]) for r in range(rows)]
    lienzo = Image.new("RGB", (cols * w + (cols - 1) * pad, sum(hs) + (rows - 1) * pad), "white")
    for n, im in enumerate(ims):
        r, c = divmod(n, cols); lienzo.paste(im, (c * (w + pad), sum(hs[:r]) + r * pad))
    lienzo.save(out)
apilar([f"{MEDIA}/image3.png", f"{MEDIA}/image2.png"], 2, f"{FIG}/fig_movimiento.png")
apilar([f"{MEDIA}/image5.png", f"{MEDIA}/image4.png"], 1, f"{FIG}/fig_qc.png")
ej = ["sub-083", "sub-076", "sub-084", "sub-069", "sub-118", "sub-022"]
apilar([f"{GLM}/{s}/{s}_zmap_fdr.png" for s in ej], 2, f"{FIG}/fig_primer_nivel.png")
Image.open(f"{REPO}/derivatives_unidad3/qc_movimiento/FD_por_grupo_y_TR.png").convert("RGB").save(f"{FIG}/fig_fd.png")
