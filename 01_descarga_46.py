"""
01_descarga_46.py
Descarga SOLO lo necesario de OpenNeuro ds005366 para los 46 sujetos elegidos:
  - BOLD (.nii) + JSON + events.tsv de la corrida elegida
  - T1w de la misma sesion
Es reanudable: si un archivo ya existe con el tamano correcto, lo salta.

Uso (desde terminal o notebook):
    python 01_descarga_46.py
    python 01_descarga_46.py --destino D:/fMRI/ds005366 --csv sujetos_46.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

S3 = "https://s3.amazonaws.com/openneuro.org/ds005366"


def descargar(url: str, destino: Path, chunk=1 << 20) -> bool:
    """Descarga con reanudacion simple. Devuelve True si el archivo queda listo."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    if head.status_code != 200:
        return False
    total = int(head.headers.get("Content-Length", 0))
    if destino.exists() and total and destino.stat().st_size == total:
        print(f"   ya existe: {destino.name}")
        return True
    tmp = destino.with_suffix(destino.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        hecho = 0
        with open(tmp, "wb") as f:
            for bloque in r.iter_content(chunk):
                f.write(bloque)
                hecho += len(bloque)
                if total:
                    print(f"\r   {destino.name}: {hecho / total:6.1%}", end="")
    print()
    tmp.replace(destino)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sujetos_46.csv")
    ap.add_argument("--destino", default="ds005366_subset")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    raiz = Path(args.destino)
    fallidos = []

    for i, r in df.iterrows():
        sub, ses = r["sub"], r["ses"]
        print(f"[{i + 1}/{len(df)}] {sub} ({r['grupo']})")
        for col in ["t1_file", "bold_file", "json_file", "events_file"]:
            rel = f"{sub}/ses-{ses}/{r[col]}"
            ok = False
            # el dataset lista .nii; por si acaso se prueba tambien .nii.gz
            for cand in (rel, rel + ".gz") if rel.endswith(".nii") else (rel,):
                if descargar(f"{S3}/{cand}", raiz / cand):
                    ok = True
                    break
            if not ok:
                print(f"   !! no se pudo descargar {rel}")
                fallidos.append(rel)

    # archivos raiz utiles
    for f in ["participants.tsv", "participants.json", "dataset_description.json"]:
        descargar(f"{S3}/{f}", raiz / f)

    print("\nListo." if not fallidos else f"\nTerminado con {len(fallidos)} fallos:")
    for f in fallidos:
        print("  -", f)


if __name__ == "__main__":
    main()
