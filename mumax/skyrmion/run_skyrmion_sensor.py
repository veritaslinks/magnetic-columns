#!/usr/bin/env python3
"""
Skyrmion film as a field sensor under the magnetic-columns generators (MuMax3).

Put this file in the same folder as mumax3.exe, mumax3-convert.exe and the
study_results folder from run_mumax_study.py, then run:
    python run_skyrmion_sensor.py

It writes skyrmion_results/summary.txt and skyrmion_results/results.json.
Send those two files back for analysis.

What it tests, for two DMI strengths:
  A. Transfer curve: skyrmion size while a uniform field is swept 0 -> +20 -> -20 -> 0 mT.
  B. Memory: field raised until the skyrmion collapses, then removed. Does it come back?
  C. One generator 21 nm or 45 nm above the film, pointing up or down.
  D. Three generators 45 nm above the film with four up/down patterns:
     does the skyrmion respond to the SUM of their fields, like a sensor in the model?

The generator field is the real MuMax3 field of a 30 nm generator, taken from
study_results/d30_t3.out/B_demag000000.npy (full vector). If that file is missing,
the axial component from field_tables.json is used instead.

Requires numpy.
"""
import glob
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
OUT = os.path.join(HERE, "skyrmion_results")

# film: Pt/Co-like ultrathin layer with interfacial DMI (parameters widely used in the literature)
N = 128                 # cells per side, 2 nm cells -> 256 x 256 nm film (1 nm cells made the dynamics far too slow)
DX, DZ = 2e-9, 0.4e-9
RUN_NS = 2.0            # nanoseconds of real dynamics after a generator field is switched on
FILM = dict(Msat=580e3, Aex=15e-12, Ku=0.8e6, alpha=0.3)
DMI = [3.0e-3, 3.5e-3]  # J/m^2, the second is closer to the stripe instability -> larger skyrmion
SENSOR_R = 20e-9        # radius of the read-out disk (an MTJ above the film centre)
TIMEOUT = 900           # seconds per MuMax3 run; longer runs are stopped and recorded

# generator field source (from the earlier study)
MAP_CELL = 3e-9
MAP_N = 135


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ------------------------------------------------------------ generator field
def load_generator_field():
    npy = os.path.join(HERE, "study_results", "d30_t3.out", "B_demag000000.npy")
    if os.path.exists(npy):
        arr = np.load(npy)                      # [component, z, y, x], tesla, generator pointing +z at the centre
        log("Generator field: full vector map from " + npy)
        return ("vector", arr)
    for p in [os.path.join(HERE, "study_results", "field_tables.json"),
              os.path.join(HERE, "field_tables.json"),
              os.path.join(HERE, "magnetic-columns", "data", "field_tables.json")]:
        if os.path.exists(p):
            t = json.load(open(p))["d30_t3"]
            log("Generator field: axial component only, from " + p + " (lateral components not available)")
            return ("table", t)
    sys.exit("No generator field found. Run run_mumax_study.py first (it creates study_results).")


def field_at(src, x, y, h):
    """Field (Bx, By, Bz) in tesla at lateral offsets x, y (arrays, m) and height h (m) below a +z generator."""
    if src[0] == "vector":
        arr = src[1]
        g = (MAP_N - 1) / 2
        k = int(round(g - h / MAP_CELL))         # film is below the generator: negative z offset
        fx, fy = x / MAP_CELL + g, y / MAP_CELL + g
        x0 = np.clip(np.floor(fx).astype(int), 0, MAP_N - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, MAP_N - 2)
        tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        out = []
        for c in range(3):
            b = arr[c, k]
            v = (1 - tx) * (1 - ty) * b[y0, x0] + tx * (1 - ty) * b[y0, x0 + 1] + (1 - tx) * ty * b[y0 + 1, x0] + tx * ty * b[y0 + 1, x0 + 1]
            out.append(v)
        return np.array(out)
    t = src[1]
    rho_nm = np.hypot(x, y) * 1e9
    z_nm = h * 1e9
    bz = np.array(t["bz_mT"]) * 1e-3
    zs = np.array(t["z_nm"])
    rs = np.array(t["rho_nm"])
    kz = int(np.argmin(np.abs(zs - z_nm)))
    val = np.interp(rho_nm, rs, bz[kz])
    return np.array([np.zeros_like(val), np.zeros_like(val), val])


def write_ovf(path, B):
    """B: array (3, N, N) in tesla, layout [component, y, x]. OVF 2.0 binary-4 file, the same format MuMax3 writes."""
    head = ["# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header", "# Title: B_generators",
            "# meshtype: rectangular", "# meshunit: m", "# xmin: 0", "# ymin: 0", "# zmin: 0",
            f"# xmax: {N * DX:.6e}", f"# ymax: {N * DX:.6e}", f"# zmax: {DZ:.6e}", "# valuedim: 3",
            "# valuelabels: B_x B_y B_z", "# valueunits: T T T", "# Desc: generator stray field",
            f"# xbase: {DX / 2:.6e}", f"# ybase: {DX / 2:.6e}", f"# zbase: {DZ / 2:.6e}",
            f"# xnodes: {N}", f"# ynodes: {N}", "# znodes: 1",
            f"# xstepsize: {DX:.6e}", f"# ystepsize: {DX:.6e}", f"# zstepsize: {DZ:.6e}",
            "# End: Header", "# Begin: Data Binary 4"]
    data = np.stack([B[0], B[1], B[2]], axis=-1).astype("<f4")   # [y, x, component], x fastest
    # binary mode: the header must end with "\n" only, MuMax3 rejects Windows "\r\n" line endings
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.array([1234567.0], dtype="<f4").tobytes())
        f.write(data.tobytes())
        f.write(b"\n# End: Data Binary 4\n# End: Segment\n")


def generator_map(src, gens):
    """gens: list of (x_offset, y_offset, height, sign). Returns (3, N, N) field on the film grid."""
    c = (np.arange(N) - (N - 1) / 2) * DX
    X, Y = np.meshgrid(c, c)                    # [y, x]
    B = np.zeros((3, N, N))
    for gx, gy, h, s in gens:
        B += s * field_at(src, X - gx, Y - gy, h)
    return B


# ------------------------------------------------------------------- MuMax3
def header(D):
    return f"""SetGridSize({N}, {N}, 1)
SetCellSize({DX}, {DX}, {DZ})
Msat  = {FILM['Msat']}
Aex   = {FILM['Aex']}
Ku1   = {FILM['Ku']}
AnisU = vector(0, 0, 1)
Dind  = {D}
alpha = {FILM['alpha']}
m = Uniform(0, 0, 1)
m.SetInShape(Circle(30e-9), Uniform(0, 0, -1))
TableAdd(ext_topologicalcharge)
TableAdd(B_ext)
"""


def run(name, text):
    mx3 = os.path.join(OUT, name + ".mx3")
    outdir = os.path.join(OUT, name + ".out")
    flag = os.path.join(outdir, "done.flag")
    if os.path.exists(flag):
        log(f"[{name}] already done, reusing")
        return outdir
    with open(mx3, "w", encoding="utf-8") as f:
        f.write(text)
    t0 = time.time()
    table = os.path.join(outdir, "table.txt")
    with open(os.path.join(OUT, name + "_mumax.txt"), "w", encoding="utf-8") as lf:
        p = subprocess.Popen([MUMAX, "-http=", "-cache=" + CACHE, mx3], cwd=OUT, stdout=lf, stderr=subprocess.STDOUT)
        last = 0.0
        while p.poll() is None:
            time.sleep(2)
            el = time.time() - t0
            if el > TIMEOUT:
                p.kill()
                p.wait()
                log(f"[{name}] stopped after {TIMEOUT // 60} min without finishing, recorded as timeout")
                return None
            if el - last >= 30:
                last = el
                rows = 0
                tsim = 0.0
                if os.path.exists(table):
                    with open(table, encoding="utf-8") as tf:
                        lines = [l for l in tf if l.strip() and not l.startswith("#")]
                    rows = len(lines)
                    if lines:
                        tsim = float(lines[-1].split()[0]) * 1e9
                print(f"   [{name}] running {el:.0f} s, simulated time {tsim:.2f} ns, table rows {rows}", flush=True)
    if p.returncode != 0 or not os.path.exists(os.path.join(outdir, "table.txt")):
        sys.exit(f"MuMax3 failed for {name}. See skyrmion_results/{name}_mumax.txt and send me its last lines.")
    for ovf in glob.glob(os.path.join(outdir, "m*.ovf")):
        subprocess.run([CONVERT, "-numpy", ovf], cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finals = sorted(glob.glob(os.path.join(outdir, "m*.ovf")))
    if finals:
        subprocess.run([CONVERT, "-png", finals[-1]], cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    open(flag, "w").close()
    log(f"[{name}] done in {time.time() - t0:.0f} s")
    return outdir


def read_table(outdir):
    with open(os.path.join(outdir, "table.txt"), encoding="utf-8") as f:
        head = f.readline().lstrip("#").split("\t")
        cols = [h.strip().split(" ")[0] for h in head]
        rows = [list(map(float, l.split())) for l in f if l.strip() and not l.startswith("#")]
    return cols, np.array(rows)


def analyse_m(npy):
    mz = np.load(npy)[2, 0]                      # [y, x]
    c = (np.arange(N) - (N - 1) / 2) * DX
    X, Y = np.meshgrid(c, c)
    core = mz < 0
    area = core.sum() * DX * DX
    r = np.sqrt(area / np.pi) * 1e9
    if core.any():
        cx, cy = X[core].mean() * 1e9, Y[core].mean() * 1e9
    else:
        cx = cy = float("nan")
    disk = (X ** 2 + Y ** 2) < SENSOR_R ** 2
    return dict(radius_nm=round(float(r), 2), centre_shift_nm=round(float(np.hypot(cx, cy)), 2) if core.any() else None,
                sensor_mz=round(float(mz[disk].mean()), 4), exists=bool(core.sum() > 10))


# ----------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(MUMAX):
        sys.exit(f"mumax3 not found at {MUMAX}. Put this script next to mumax3{EXE}.")
    os.makedirs(CACHE, exist_ok=True)
    log(f"=== skyrmion sensor study started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    src = load_generator_field()
    res = dict(film=FILM, dmi=DMI, generator_field_source=src[0], runs={})

    # field maps for parts C and D
    maps = {}
    for h in (21e-9, 45e-9):
        for s, tag in ((1, "up"), (-1, "down")):
            maps[f"one_h{int(h * 1e9)}_{tag}"] = [(0.0, 0.0, h, s)]
    pats = {"ppp": (1, 1, 1), "ppm": (1, 1, -1), "pmm": (1, -1, -1), "mmm": (-1, -1, -1)}
    for k, sg in pats.items():
        maps["three_" + k] = [(0.0, 0.0, 45e-9, sg[0]), (45e-9, 0.0, 45e-9, sg[1]), (-22.5e-9, 39e-9, 45e-9, sg[2])]
    centre_field = {}
    for k, g in maps.items():
        B = generator_map(src, g)
        write_ovf(os.path.join(OUT, k + ".ovf"), B)
        centre_field[k] = round(float(B[2, N // 2, N // 2] * 1e3), 3)
    res["field_at_film_centre_mT"] = centre_field
    log("Generator field at the film centre (mT): " + json.dumps(centre_field))

    for D in DMI:
        dtag = f"D{D * 1e3:.1f}".replace(".", "p")
        # A. reversible transfer curve, uniform field
        steps = list(range(0, 21, 2)) + list(range(18, -21, -2)) + list(range(-18, 1, 2))
        # dynamics per step (0.3 ns), never the minimiser: it can loop forever on this film
        body = "".join(f"B_ext = vector(0, 0, {b * 1e-3:.4e})\nRun(3e-10)\nTableSave()\nSave(m)\n" for b in steps)
        od = run(f"{dtag}_A_transfer_r", header(D) + "Run(1e-9)\n" + body)
        npys = sorted(glob.glob(os.path.join(od, "m*.npy"))) if od else []
        res["runs"][f"{dtag}_A_transfer"] = [dict(field_mT=b, **analyse_m(p)) for b, p in zip(steps, npys)]
        # B. collapse and memory
        up = list(range(0, 601, 40))
        down = list(range(560, -1, -80))
        # real dynamics per field step: the minimiser never converges while the skyrmion is collapsing
        body = "".join(f"B_ext = vector(0, 0, {b * 1e-3:.4e})\nRun(3e-10)\nTableSave()\n" for b in up + down) + "Save(m)\n"
        od = run(f"{dtag}_B_memory_r", header(D) + "Run(1e-9)\n" + body)
        if od:
            cols, tab = read_table(od)
            q = tab[:, cols.index("ext_topologicalcharge")]
            seq = up + down
            collapse = next((seq[i] for i in range(min(len(up), len(q))) if abs(q[i]) < 0.5), None)
            res["runs"][f"{dtag}_B_memory"] = dict(collapse_field_mT=collapse, charge_at_start=round(float(q[0]), 3),
                                                   charge_after_field_removed=round(float(q[-1]), 3))
        else:
            res["runs"][f"{dtag}_B_memory"] = dict(collapse_field_mT=None, charge_at_start=None, charge_after_field_removed=None, timeout=True)
        # C and D. generators: switch the generator field on and follow 5 ns of real dynamics (a sensor read-out window)
        for k in maps:
            text = header(D) + "Run(1e-9)\n" + f'B_ext.Add(LoadFile("{os.path.join(OUT, k + ".ovf").replace(chr(92), "/")}"), 1)\nTableAutosave({RUN_NS / 10 * 1e-9:.3e})\nRun({RUN_NS * 1e-9:.3e})\nSave(m)\n'
            od = run(f"{dtag}_{k}_r", text)   # "_r": settled by dynamics, not by the minimiser
            npys = sorted(glob.glob(os.path.join(od, "m*.npy"))) if od else []
            if not npys:
                res["runs"][f"{dtag}_{k}"] = dict(field_centre_mT=centre_field[k], radius_nm=None, centre_shift_nm=None, sensor_mz=None, exists=None, timeout=True)
                continue
            cols, tab = read_table(od)
            qt = tab[:, cols.index("ext_topologicalcharge")]
            res["runs"][f"{dtag}_{k}"] = dict(field_centre_mT=centre_field[k], charge_end=round(float(qt[-1]), 3), **analyse_m(npys[-1]))

    # ------------------------------------------------------------- summary
    L = ["Skyrmion film as a field sensor under the magnetic-columns generators", ""]
    for D in DMI:
        dtag = f"D{D * 1e3:.1f}".replace(".", "p")
        L.append(f"=== DMI {D * 1e3:.1f} mJ/m^2 ===")
        curve = res["runs"][f"{dtag}_A_transfer"]
        z = [c for c in curve if c["field_mT"] == 0] or [dict(radius_nm=None)]
        L.append(f"A. Radius at 0 mT: {z[0]['radius_nm']} nm (start) and {z[-1]['radius_nm']} nm (after the sweep)")
        ok = [c for c in curve if c["exists"]]
        if len(ok) > 3:
            f = np.array([c["field_mT"] for c in ok])
            r = np.array([c["radius_nm"] for c in ok])
            s = np.array([c["sensor_mz"] for c in ok])
            L.append(f"   Radius slope: {np.polyfit(f, r, 1)[0]:.3f} nm per mT, sensor signal slope: {np.polyfit(f, s, 1)[0]:.4f} per mT")
            L.append("   Curve (field mT -> radius nm, sensor mz): " + ", ".join(f"{c['field_mT']}:{c['radius_nm']}/{c['sensor_mz']}" for c in curve))
        mem = res["runs"][f"{dtag}_B_memory"]
        L.append(f"B. Collapse field: {mem['collapse_field_mT']} mT; topological charge at start {mem['charge_at_start']}, after the field is removed {mem['charge_after_field_removed']}"
                 + ("  -> the skyrmion does NOT come back: the film remembers the event" if mem['collapse_field_mT'] is not None and abs(mem['charge_after_field_removed']) < 0.5 else ""))
        for h in (21, 45):
            u, d = res["runs"][f"{dtag}_one_h{h}_up"], res["runs"][f"{dtag}_one_h{h}_down"]
            L.append(f"C. One generator {h} nm above: field at centre {u['field_centre_mT']} / {d['field_centre_mT']} mT; "
                     f"radius {u['radius_nm']} / {d['radius_nm']} nm; sensor {u['sensor_mz']} / {d['sensor_mz']}; "
                     f"skyrmion centre moved {u['centre_shift_nm']} / {d['centre_shift_nm']} nm (up / down)")
        three = [(k, res["runs"][f"{dtag}_three_{k}"]) for k in pats]
        L.append("D. Three generators 45 nm above (pattern: field at centre mT -> radius nm, sensor, shift nm): " +
                 "; ".join(f"{k}: {v['field_centre_mT']} -> {v['radius_nm']}, {v['sensor_mz']}, {v['centre_shift_nm']}" for k, v in three))
        good = [(v["field_centre_mT"], v["sensor_mz"]) for _, v in three if v.get("sensor_mz") is not None]
        fs = np.array([g[0] for g in good]); ss = np.array([g[1] for g in good])
        if len(good) > 2 and np.ptp(fs) > 0 and np.ptp(ss) > 0:
            L.append(f"   Correlation between summed field and sensor signal: {np.corrcoef(fs, ss)[0, 1]:.3f}")
        L.append("")
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log("\n".join(L))
    log("=== finished. Send skyrmion_results/summary.txt and skyrmion_results/results.json ===")


if __name__ == "__main__":
    main()
