"""
Analyse the MuMax3 run of hexcell_C_coupling.mx3.

Before running:
    mumax3-convert -numpy hexcell_C_coupling.out/b_demag*.ovf
Then:
    python analyze_coupling.py hexcell_C_coupling.out

What it reports:
  1. How strongly each generator acts on each sensor position (in millitesla).
  2. How many generators a sensor effectively "hears" (the model says about 9).
  3. How the coupling falls off with distance (the model assumes 1/r^3).
  4. The field contrast a real sensor would have to resolve.
"""
import glob
import os
import sys

import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "hexcell_C_coupling.out"
files = sorted(glob.glob(os.path.join(out_dir, "[bB]_demag*.npy")))
if len(files) != 31:
    sys.exit(f"Expected 31 .npy files (baseline + 30 flips), found {len(files)}. Run mumax3-convert -numpy first.")

# geometry, must match the .mx3 script
CELL_Z = 3e-9  # in-plane cell size and grid are read from the data below
R, RC = 150e-9, 45e-9
LAYER_Z_INDEX = [0, 7, 14]  # replaced below from the data


def idx(x, y):
    return int(round(x / CELL_XY + (NXY - 1) / 2)), int(round(y / CELL_XY + (NXY - 1) / 2))


def perimeter_positions():
    pts = []
    for k in range(6):
        a0, a1 = k * np.pi / 3, (k + 1) * np.pi / 3
        for t in (1 / 3, 2 / 3):
            pts.append((R * (np.cos(a0) + (np.cos(a1) - np.cos(a0)) * t), R * (np.sin(a0) + (np.sin(a1) - np.sin(a0)) * t)))
    return pts


def column_positions():
    return [(RC * (np.cos(k * np.pi / 3) + np.cos((k + 1) * np.pi / 3)) / 2,
             RC * (np.sin(k * np.pi / 3) + np.sin((k + 1) * np.pi / 3)) / 2) for k in range(6)]


# generators in region order 1..30: layer1 perimeter (12), layer2 column (6), layer3 perimeter (12)
gens = [(x, y, 0) for x, y in perimeter_positions()] + [(x, y, 1) for x, y in column_positions()] + [(x, y, 2) for x, y in perimeter_positions()]
# sensors in variant C: column positions on layers 1 and 3, perimeter positions on layer 2
sens = [(x, y, 0) for x, y in column_positions()] + [(x, y, 1) for x, y in perimeter_positions()] + [(x, y, 2) for x, y in column_positions()]


def bz_at(arr, x, y, layer):
    # mumax3-convert -numpy layout: [component, z, y, x]
    i, j = idx(x, y)
    return arr[2, LAYER_Z_INDEX[layer], j, i]


base = np.load(files[0])
NXY = base.shape[-1]
CELL_XY = 384e-9 / NXY
NZ = base.shape[1]
LAYER_Z_INDEX = [0, NZ // 2, NZ - 1]
print("array shape", base.shape, "-> grid", NXY, "cells, cell size", CELL_XY * 1e9, "nm, layers at z index", LAYER_Z_INDEX)
J = np.zeros((len(sens), len(gens)))
for g in range(30):
    flip = np.load(files[g + 1])
    for si, (x, y, l) in enumerate(sens):
        # flipping a generator changes its moment by -2, so coupling per unit moment = dB / -2
        J[si, g] = (bz_at(flip, x, y, l) - bz_at(base, x, y, l)) / -2.0

Jm = J * 1e3  # millitesla
print("\n1) Coupling generator -> sensor, |B_z| per generator, mT")
print(f"   strongest {np.abs(Jm).max():.3f}   median {np.median(np.abs(Jm)):.4f}   weakest {np.abs(Jm).min():.5f}")

absJ = np.abs(J)
share = np.sort(absJ, axis=1)[:, ::-1]
tot = share.sum(axis=1, keepdims=True)
pr = (absJ.sum(axis=1) ** 2) / (absJ ** 2).sum(axis=1)
print("\n2) How many generators a sensor effectively hears (model: about 9, from all cells and layers)")
print(f"   share of the strongest 1 / 3 / 10: {np.mean(share[:, 0] / tot[:, 0]):.2f} / {np.mean(share[:, :3].sum(1) / tot[:, 0]):.2f} / {np.mean(share[:, :10].sum(1) / tot[:, 0]):.2f}")
print(f"   effective number of sources: {pr.mean():.1f}  (only 30 generators of one cell exist here)")

d, v = [], []
for si, (sx, sy, sl) in enumerate(sens):
    for g, (gx, gy, gl) in enumerate(gens):
        r = np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2 + ((LAYER_Z_INDEX[sl] - LAYER_Z_INDEX[gl]) * CELL_Z) ** 2)
        if r > 0:
            d.append(r)
            v.append(absJ[si, g])
d, v = np.array(d), np.array(v)
ok = v > 0
slope = np.polyfit(np.log(d[ok]), np.log(v[ok]), 1)[0]
print("\n3) Fall-off with distance")
print(f"   fitted exponent: {slope:.2f}  (pure point dipole: -3; finite-size dots usually come out a bit flatter up close)")

rng = np.random.default_rng(0)
pats = rng.choice([-1, 1], size=(200, 30))
fields = pats @ J.T * 1e3
diffs = np.abs(np.diff(fields, axis=0))
print("\n4) What a sensor must resolve (random generator patterns, linear superposition)")
print(f"   typical field at a sensor: {np.mean(np.abs(fields)):.3f} mT, typical change between two patterns: {np.median(diffs):.3f} mT")
print("   Compare with the resolution of the sensor you have in mind (MTJ or Hall).")
np.save("coupling_matrix_mT.npy", Jm)
print("\nSaved coupling_matrix_mT.npy (rows = sensors, columns = generators).")
