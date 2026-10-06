"""
Export the real single-generator field from the MuMax3 maps as a compact table.

Run in the same folder as run_mumax_study.py, after the study has finished:
    python export_field_table.py

Writes study_results/field_tables.json: Bz(rho, z) for each generator type,
averaged around the generator axis (the field of a round perpendicular magnet is
axially symmetric). Send that file back: it replaces the point-dipole law in the
web model with the field MuMax3 actually computed.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "study_results")
CELL = 3e-9
tables = {}
for d in sorted(glob.glob(os.path.join(OUT, "*.out"))):
    name = os.path.basename(d)[:-4]
    f = glob.glob(os.path.join(d, "[bB]_demag*.npy"))
    if not f:
        continue
    bz = np.load(f[0])[2]                     # [z, y, x]
    n = bz.shape[-1]
    c = (n - 1) / 2
    ax = (np.arange(n) - c) * CELL
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    rho = np.sqrt(xx ** 2 + yy ** 2)
    rbins = np.arange(0, c * CELL + CELL / 2, CELL)
    idx = np.clip(np.round(rho / CELL).astype(int), 0, len(rbins) - 1)
    table = np.zeros((bz.shape[0], len(rbins)))
    for k in range(bz.shape[0]):
        s = np.bincount(idx.ravel(), weights=bz[k].ravel(), minlength=len(rbins))
        cnt = np.bincount(idx.ravel(), minlength=len(rbins))
        table[k] = s / np.maximum(cnt, 1)
    zs = (np.arange(bz.shape[0]) - (bz.shape[0] - 1) / 2) * CELL
    tables[name] = dict(rho_nm=[round(r * 1e9, 3) for r in rbins], z_nm=[round(z * 1e9, 3) for z in zs],
                        bz_mT=[[round(v * 1e3, 6) for v in row] for row in table])
    print(f"{name}: table {table.shape[0]} x {table.shape[1]}, Bz 21 nm above the centre {table[int(round(c + 7)), 0] * 1e3:.2f} mT")
with open(os.path.join(OUT, "field_tables.json"), "w", encoding="utf-8") as fh:
    json.dump(tables, fh)
print("Saved study_results/field_tables.json")
