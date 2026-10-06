#!/usr/bin/env python3
"""
One-command MuMax3 study of the magnetic honeycomb architecture.

Put this file next to mumax3.exe and mumax3-convert.exe, then run:
    python run_mumax_study.py

What it does:
  1. Runs MuMax3 once per generator type (a single perpendicular nanomagnet in a
     400 x 400 x 400 nm box) and saves the full stray-field map around it.
  2. Validates superposition: if hexcell_C_coupling.out from the earlier 30-generator
     run is present, it rebuilds that cell from single-generator maps and compares
     with what MuMax3 computed directly.
  3. Builds every configuration in the sweep (1, 7 or 19 cells, 6 or 10 layers,
     several layer spacings and gaps, architectures base and C) by superposition
     and measures what matters for the computing model.
  4. Writes study_results/summary.txt, results.json and results.csv.
     Send summary.txt and results.json back for analysis.

Why superposition is valid: the stray field is linear in magnetisation, and the
earlier run showed generators stay saturated (|mz| = 0.99999) next to their
neighbours. The validation step measures how accurate this is.

Requires: numpy. Optional flag --fake builds an analytic dipole map instead of
running MuMax3 (only for testing the script without a GPU).
"""
import csv
import glob
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""
MUMAX = os.path.join(HERE, "mumax3" + EXE)
CONVERT = os.path.join(HERE, "mumax3-convert" + EXE)
CACHE = os.path.join(HERE, "kcache")
OUT = os.path.join(HERE, "study_results")
FAKE = "--fake" in sys.argv

MU0 = 4e-7 * np.pi
KB = 1.380649e-23
CELL = 3e-9          # map cell size, m
NMAP = 135           # odd, so the generator sits exactly on the central cell
HALF = (NMAP - 1) / 2 * CELL   # 201 nm

# Generator types (perpendicular magnets). Diameters are whole numbers of cells.
DOTS = [
    dict(name="d30_t3", diam=30e-9, thick=3e-9, Msat=1.0e6, Ku=1.0e6, Aex=15e-12),
    dict(name="d21_t3", diam=21e-9, thick=3e-9, Msat=1.0e6, Ku=1.0e6, Aex=15e-12),
    dict(name="d39_t3", diam=39e-9, thick=3e-9, Msat=1.0e6, Ku=1.0e6, Aex=15e-12),
]

# Sweep (physical units). R = hexagon circumradius, rc = pillar radius.
R = 150e-9
RC = 0.3 * R
SWEEP = dict(
    arch=["base", "C"],
    rings=[1, 2],               # 7 or 19 cells
    layers=[6, 10],
    spacing=[30e-9, 45e-9, 60e-9],
    gap=[0.3 * R, 0.5 * R],
)


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------- MuMax3 part
def mx3_text(d):
    return f"""// single generator field map: {d['name']}
SetGridSize({NMAP}, {NMAP}, {NMAP})
SetCellSize({CELL}, {CELL}, {CELL})
Msat  = {d['Msat']}
Aex   = {d['Aex']}
Ku1   = {d['Ku']}
AnisU = vector(0, 0, 1)
alpha = 0.5
SetGeom(Cylinder({d['diam']}, {d['thick']}))
m = Uniform(0, 0, 1)
Relax()
print("relaxed, average m:", m.Average())
Save(B_demag)
"""


def run_mumax(d):
    name = d["name"]
    mx3 = os.path.join(OUT, name + ".mx3")
    outdir = os.path.join(OUT, name + ".out")
    npy = os.path.join(outdir, "B_demag000000.npy")
    if os.path.exists(npy):
        log(f"[{name}] field map already computed, reusing it")
        return np.load(npy)
    with open(mx3, "w", encoding="utf-8") as f:
        f.write(mx3_text(d))
    os.makedirs(CACHE, exist_ok=True)
    log(f"[{name}] running MuMax3 (first run computes the demag kernel, may take a few minutes)...")
    t0 = time.time()
    with open(os.path.join(OUT, name + "_mumax.txt"), "w", encoding="utf-8") as lf:
        p = subprocess.run([MUMAX, "-http=", "-cache=" + CACHE, mx3], cwd=OUT, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        sys.exit(f"MuMax3 failed for {name}, see {name}_mumax.txt in study_results")
    ovf = [f for f in glob.glob(os.path.join(outdir, "*.ovf")) if "demag" in os.path.basename(f).lower()]
    if not ovf:
        sys.exit(f"No B_demag file produced for {name}")
    subprocess.run([CONVERT, "-numpy", ovf[0]], cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    npys = glob.glob(os.path.join(outdir, "*.npy"))
    if not npys:
        sys.exit(f"Conversion to numpy failed for {name}")
    log(f"[{name}] done in {time.time() - t0:.0f} s")
    return np.load(npys[0])


def fake_map(d):
    """Analytic point-dipole map, only for testing the script without MuMax3."""
    ax = (np.arange(NMAP) - (NMAP - 1) / 2) * CELL
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z) + 1e-12
    mom = d["Msat"] * np.pi * (d["diam"] / 2) ** 2 * d["thick"]
    bz = MU0 / (4 * np.pi) * mom * (3 * z * z / r ** 2 - 1) / r ** 3
    bz[r < d["diam"] / 2] = 0
    return np.stack([bz * 0, bz * 0, bz])


# ------------------------------------------------------- field of one generator
class DotField:
    """Bz of one generator pointing +z, at offsets (dx, dy, dz) in metres."""

    def __init__(self, arr, d):
        self.bz = arr[2]                         # layout [component, z, y, x]
        self.d = d
        # effective dipole moment fitted on the axis at 150 nm, used beyond the map
        zc = 150e-9
        b_axis = self.interp(np.array([0.0]), np.array([0.0]), np.array([zc]))[0]
        self.m_eff = b_axis * zc ** 3 * 4 * np.pi / (2 * MU0)
        self.m_geo = d["Msat"] * np.pi * (d["diam"] / 2) ** 2 * d["thick"]

    def interp(self, dx, dy, dz):
        g = (NMAP - 1) / 2
        fx, fy, fz = dx / CELL + g, dy / CELL + g, dz / CELL + g
        x0, y0, z0 = np.floor(fx).astype(int), np.floor(fy).astype(int), np.floor(fz).astype(int)
        x0, y0, z0 = np.clip(x0, 0, NMAP - 2), np.clip(y0, 0, NMAP - 2), np.clip(z0, 0, NMAP - 2)
        tx, ty, tz = fx - x0, fy - y0, fz - z0
        b = self.bz
        out = np.zeros_like(fx, dtype=float)
        for ix, wx in ((0, 1 - tx), (1, tx)):
            for iy, wy in ((0, 1 - ty), (1, ty)):
                for iz, wz in ((0, 1 - tz), (1, tz)):
                    out += wx * wy * wz * b[z0 + iz, y0 + iy, x0 + ix]
        return out

    def bz_at(self, dx, dy, dz):
        lim = HALF - 2 * CELL
        inside = (np.abs(dx) < lim) & (np.abs(dy) < lim) & (np.abs(dz) < lim)
        out = np.zeros(np.shape(dx))
        if inside.any():
            out[inside] = self.interp(dx[inside], dy[inside], dz[inside])
        far = ~inside
        if far.any():
            x, y, z = dx[far], dy[far], dz[far]
            r2 = x * x + y * y + z * z
            r = np.sqrt(r2)
            out[far] = MU0 / (4 * np.pi) * self.m_eff * (3 * z * z / r2 - 1) / r ** 3
        return out


# ------------------------------------------------------------------ geometry
def build(arch, rings, layers, spacing, gap, dot):
    S = np.sqrt(3) * R + gap
    cells = []
    for q in range(-rings, rings + 1):
        for r_ in range(-rings, rings + 1):
            if max(abs(q), abs(r_), abs(q + r_)) <= rings:
                cells.append((q * S * np.sqrt(3) / 2, q * S * 0.5 + r_ * S, max(abs(q), abs(r_), abs(q + r_))))
    local = []
    for k in range(6):
        a0, a1 = k * np.pi / 3, (k + 1) * np.pi / 3
        v0, v1 = np.array([np.cos(a0), np.sin(a0)]), np.array([np.cos(a1), np.sin(a1)])
        for t in (1 / 3, 2 / 3):
            p = R * (v0 + (v1 - v0) * t)
            local.append((p[0], p[1], 0))
    for k in range(6):
        a0, a1 = k * np.pi / 3, (k + 1) * np.pi / 3
        local.append((RC * (np.cos(a0) + np.cos(a1)) / 2, RC * (np.sin(a0) + np.sin(a1)) / 2, 1))
    el = []   # x, y, z, cell, layer, is_pillar
    for ci, (cx, cy, _) in enumerate(cells):
        for l in range(layers):
            z = (l - (layers - 1) / 2) * spacing
            for (lx, ly, pil) in local:
                el.append((cx + lx, cy + ly, z, ci, l, pil))
    el = np.array(el)
    pil = el[:, 5].astype(bool)
    lay = el[:, 4].astype(int)
    if arch == "base":
        emit = np.ones(len(el), bool)
    elif arch == "C":
        emit = np.where(lay % 2 == 0, ~pil, pil)
    elif arch == "A":
        emit = ~pil
    else:
        emit = pil
    gens = el[emit]
    if arch == "base":
        # each generator has its own sensor beside it, pushed off the magnet towards the cell centre
        sens = gens.copy()
        off = dot["diam"] / 2 + 8e-9
        for i, (x, y, z, ci, l, p) in enumerate(gens):
            cx, cy, _ = cells[int(ci)]
            v = np.array([cx - x, cy - y])
            n = np.linalg.norm(v)
            v = v / n if n > 0 else np.array([1.0, 0.0])
            if p:
                v = -v
            sens[i, 0] += v[0] * off
            sens[i, 1] += v[1] * off
    else:
        sens = el[~emit]
    return gens, sens, cells


def coupling(gens, sens, field, skip_self):
    S, G = len(sens), len(gens)
    J = np.zeros((S, G))
    for s0 in range(0, S, 400):
        sl = slice(s0, min(S, s0 + 400))
        dx = sens[sl, 0][:, None] - gens[None, :, 0]
        dy = sens[sl, 1][:, None] - gens[None, :, 1]
        dz = sens[sl, 2][:, None] - gens[None, :, 2]
        J[sl] = field.bz_at(dx, dy, dz)
    if skip_self:
        np.fill_diagonal(J, 0.0)   # base: a sensor does not read its own generator
    return J


def metrics(arch, rings, layers, spacing, gap, dot, field):
    gens, sens, cells = build(arch, rings, layers, spacing, gap, dot)
    J = coupling(gens, sens, field, skip_self=(arch == "base"))
    A = np.abs(J)
    tot = A.sum(1)
    ok = tot > 0
    srt = -np.sort(-A, axis=1)
    pr = tot[ok] ** 2 / (A[ok] ** 2).sum(1)
    same_cell = sens[:, 3][:, None] == gens[None, :, 3]
    same_layer = sens[:, 4][:, None] == gens[None, :, 4]
    other_cell_share = float(np.mean((A * ~same_cell).sum(1)[ok] / tot[ok]))
    other_layer_share = float(np.mean((A * ~same_layer).sum(1)[ok] / tot[ok]))
    # neighbouring cells only (excluding the generator's own cell) for the central cell's sensors
    centre = [i for i, c in enumerate(cells) if c[2] == 0][0]
    cs = sens[:, 3] == centre
    neigh = A[cs][:, gens[:, 3] != centre]
    rng = np.random.default_rng(0)
    pats = rng.choice([-1.0, 1.0], size=(200, len(gens)))
    F = pats @ J.T * 1e3
    change = np.abs(np.diff(F, axis=0))
    # stability: worst-case opposing field at every generator from all other generators
    Jg = coupling(gens, gens, field, skip_self=True)
    worst = np.abs(Jg).sum(1).max() * 1e3
    typical_g = np.median(np.abs(pats[:50] @ Jg.T)) * 1e3
    Nzx = 0.8   # thin-disk demag factor difference, approximate
    keff = dot["Ku"] - 0.5 * MU0 * dot["Msat"] ** 2 * Nzx
    bk = 2 * keff / dot["Msat"] * 1e3      # effective anisotropy field, mT (upper bound on switching)
    vol = np.pi * (dot["diam"] / 2) ** 2 * dot["thick"]
    return dict(
        dot=dot["name"], arch=arch, cells=len(cells), layers=layers,
        spacing_nm=round(spacing * 1e9), gap_nm=round(gap * 1e9),
        generators=int(len(gens)), sensors=int(len(sens)),
        strongest_mT=round(float(A.max() * 1e3), 3),
        median_mT=round(float(np.median(A[A > 0]) * 1e3), 5),
        eff_sources=round(float(pr.mean()), 2),
        top1_share=round(float(np.mean(srt[ok, 0] / tot[ok])), 3),
        top3_share=round(float(np.mean(srt[ok, :3].sum(1) / tot[ok])), 3),
        other_layer_share=round(other_layer_share, 3),
        other_cell_share=round(other_cell_share, 4),
        centre_cell_max_from_neighbours_mT=round(float(neigh.max() * 1e3) if neigh.size else 0.0, 4),
        typical_sensor_field_mT=round(float(np.mean(np.abs(F))), 3),
        typical_pattern_change_mT=round(float(np.median(change)), 3),
        worst_case_field_on_generator_mT=round(float(worst), 2),
        typical_field_on_generator_mT=round(float(typical_g), 2),
        anisotropy_field_mT=round(float(bk), 0),
        stability_margin=round(float(bk / max(worst, 1e-9)), 1),
        thermal_delta_300K=round(float(keff * vol / (KB * 300)), 0),
    )


# --------------------------------------------------------------- validation
def validate(field30):
    d = os.path.join(HERE, "hexcell_C_coupling.out")
    files = sorted(glob.glob(os.path.join(d, "[bB]_demag*.npy")))
    if len(files) != 31:
        log("Validation skipped: hexcell_C_coupling.out with 31 .npy files not found next to the script.")
        return None
    base = np.load(files[0])
    nxy, nz = base.shape[-1], base.shape[1]
    cxy, cz = 384e-9 / nxy, 3e-9
    zl = [0, 7, 14]
    per, col = [], []
    for k in range(6):
        a0, a1 = k * np.pi / 3, (k + 1) * np.pi / 3
        for t in (1 / 3, 2 / 3):
            per.append((R * (np.cos(a0) + (np.cos(a1) - np.cos(a0)) * t), R * (np.sin(a0) + (np.sin(a1) - np.sin(a0)) * t)))
        col.append((RC * (np.cos(a0) + np.cos(a1)) / 2, RC * (np.sin(a0) + np.sin(a1)) / 2))
    gens = [(x, y, 0) for x, y in per] + [(x, y, 1) for x, y in col] + [(x, y, 2) for x, y in per]
    sens = [(x, y, 0) for x, y in col] + [(x, y, 1) for x, y in per] + [(x, y, 2) for x, y in col]

    def at(arr, x, y, l):
        i, j = int(round(x / cxy + (nxy - 1) / 2)), int(round(y / cxy + (nxy - 1) / 2))
        return arr[2, zl[l], j, i]

    Jd = np.zeros((len(sens), 30))
    for g in range(30):
        fl = np.load(files[g + 1])
        for si, (x, y, l) in enumerate(sens):
            Jd[si, g] = (at(fl, x, y, l) - at(base, x, y, l)) / -2.0
    Js = np.zeros_like(Jd)
    for si, (x, y, l) in enumerate(sens):
        for g, (gx, gy, gl) in enumerate(gens):
            Js[si, g] = field30.bz_at(np.array([x - gx]), np.array([y - gy]), np.array([(zl[l] - zl[gl]) * cz]))[0]
    big = np.abs(Jd) > 0.01 * np.abs(Jd).max()
    rel = np.abs(Js[big] - Jd[big]) / np.abs(Jd[big])
    corr = float(np.corrcoef(Js.ravel(), Jd.ravel())[0, 1])
    res = dict(pairs_compared=int(big.sum()), median_relative_error=round(float(np.median(rel)), 4),
               p90_relative_error=round(float(np.percentile(rel, 90)), 4), correlation=round(corr, 5),
               strongest_direct_mT=round(float(np.abs(Jd).max() * 1e3), 3), strongest_superposed_mT=round(float(np.abs(Js).max() * 1e3), 3))
    log(f"Validation against the direct 30-generator run: {res}")
    return res


# --------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    if not FAKE and not os.path.exists(MUMAX):
        sys.exit(f"mumax3 not found at {MUMAX}. Put this script next to mumax3{EXE}.")
    log(f"=== study started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    fields = {}
    for d in DOTS:
        arr = fake_map(d) if FAKE else run_mumax(d)
        fields[d["name"]] = DotField(arr, d)
        f = fields[d["name"]]
        log(f"[{d['name']}] effective moment {f.m_eff:.3e} A*m^2 (geometric {f.m_geo:.3e}); "
            f"Bz 21 nm above: {f.bz_at(np.array([0.0]), np.array([0.0]), np.array([21e-9]))[0]*1e3:.2f} mT")
    val = validate(fields["d30_t3"])
    rows = []
    combos = list(itertools.product(DOTS, SWEEP["arch"], SWEEP["rings"], SWEEP["layers"], SWEEP["spacing"], SWEEP["gap"]))
    log(f"Evaluating {len(combos)} configurations by superposition...")
    t0 = time.time()
    for n, (dot, arch, rings, layers, spacing, gap) in enumerate(combos, 1):
        rows.append(metrics(arch, rings, layers, spacing, gap, dot, fields[dot["name"]]))
        if n % 12 == 0:
            log(f"  {n}/{len(combos)} done ({time.time() - t0:.0f} s)")
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(validation=val, fake_maps=FAKE, cell_radius_nm=R * 1e9, pillar_radius_nm=RC * 1e9, results=rows), f, indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["dot", "arch", "cells", "layers", "spacing_nm", "gap_nm", "eff_sources", "top1_share", "other_layer_share",
            "other_cell_share", "typical_pattern_change_mT", "worst_case_field_on_generator_mT", "stability_margin"]
    lines = ["Validation: " + json.dumps(val), "", "  ".join(c[:14].rjust(14) for c in cols)]
    for r in rows:
        lines.append("  ".join(str(r[c])[:14].rjust(14) for c in cols))
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"=== finished. Send study_results/summary.txt and study_results/results.json ===")


if __name__ == "__main__":
    main()
