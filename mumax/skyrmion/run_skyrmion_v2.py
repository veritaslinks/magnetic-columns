#!/usr/bin/env python3
"""
Skyrmion sensor study, part 2 (MuMax3). One command:
    python run_skyrmion_v2.py              # all three parts
    python run_skyrmion_v2.py --only S     # just one part: S, T or W

Put it in the same folder as mumax3.exe, mumax3-convert.exe and study_results.
Writes skyrmion2_results/summary.txt and results.json. Finished runs are reused
if the script is started again, so it can be stopped and resumed.

S. Sensor configurations at 0 K (clean reference):
   skyrmion size (DMI 3.4-3.7 mJ/m^2), generator-to-film distance 30 and 45 nm,
   generator 30 and 39 nm. Which configuration gives the strongest signal?
T. Real operating temperatures 300 K and 350 K:
   signal-to-noise of the sensor for generator up / down, and whether the skyrmion survives.
W. Writing and erasing with a current pulse through a 30 nm contact:
   threshold current for both current directions, two pulse lengths, with and
   without the generator field helping. Current and energy per write.

No minimiser is used anywhere: it can loop forever on this film. Everything is real dynamics.
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
OUT = os.path.join(HERE, "skyrmion2_results")

# ---------------------------------------------------------------- settings
N, DX, DZ = 128, 2e-9, 0.4e-9                 # 256 x 256 nm film, 2 nm cells
FILM = dict(Msat=580e3, Aex=15e-12, Ku=0.8e6, alpha=0.3)
SENSOR_R = 20e-9                              # read-out disk radius
TIMEOUT = 1200                                # seconds per run

S_DMI = [3.4e-3, 3.5e-3, 3.6e-3, 3.7e-3]
S_HEIGHTS = [30e-9, 45e-9]
T_DMI = [3.5e-3, 3.6e-3]
T_TEMPS = [300, 350]                          # real operating temperatures, kelvin
T_HEIGHTS = [30e-9, 45e-9]
T_SETTLE, T_MEASURE, T_DT = 0.5e-9, 1.5e-9, 3e-14
W_DMI = 3.5e-3
W_J = [0.1e12, 0.3e12, 1e12, 3e12]            # A/m^2
W_TAU = [0.3e-9, 1e-9]                        # pulse length, s
W_CONTACT_D = 30e-9
W_VOLT = 0.3                                  # assumed voltage across the contact, V (for the energy estimate)

MAP_CELL, MAP_N = 3e-9, 135


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ------------------------------------------------------ generator field maps
_cache = {}


def load_map(dot):
    if dot in _cache:
        return _cache[dot]
    npy = os.path.join(HERE, "study_results", f"{dot}_t3.out", "B_demag000000.npy")
    if os.path.exists(npy):
        _cache[dot] = ("vector", np.load(npy))
        return _cache[dot]
    for p in [os.path.join(HERE, "study_results", "field_tables.json"), os.path.join(HERE, "field_tables.json")]:
        if os.path.exists(p):
            _cache[dot] = ("table", json.load(open(p))[f"{dot}_t3"])
            log(f"Generator {dot}: axial field only (vector map not found)")
            return _cache[dot]
    sys.exit("No generator field found. Run run_mumax_study.py first (it creates study_results).")


def field_at(src, x, y, h):
    if src[0] == "vector":
        arr = src[1]
        g = (MAP_N - 1) / 2
        k = int(round(g - h / MAP_CELL))
        fx, fy = x / MAP_CELL + g, y / MAP_CELL + g
        x0 = np.clip(np.floor(fx).astype(int), 0, MAP_N - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, MAP_N - 2)
        tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        return np.array([(1 - tx) * (1 - ty) * arr[c, k][y0, x0] + tx * (1 - ty) * arr[c, k][y0, x0 + 1]
                         + (1 - tx) * ty * arr[c, k][y0 + 1, x0] + tx * ty * arr[c, k][y0 + 1, x0 + 1] for c in range(3)])
    t = src[1]
    bz, zs, rs = np.array(t["bz_mT"]) * 1e-3, np.array(t["z_nm"]), np.array(t["rho_nm"])
    kz = int(np.argmin(np.abs(zs - h * 1e9)))
    val = np.interp(np.hypot(x, y) * 1e9, rs, bz[kz])
    return np.array([0 * val, 0 * val, val])


def write_ovf(path, B):
    head = ["# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header", "# Title: B_generators",
            "# meshtype: rectangular", "# meshunit: m", "# xmin: 0", "# ymin: 0", "# zmin: 0",
            f"# xmax: {N * DX:.6e}", f"# ymax: {N * DX:.6e}", f"# zmax: {DZ:.6e}", "# valuedim: 3",
            "# valuelabels: B_x B_y B_z", "# valueunits: T T T", "# Desc: generator stray field",
            f"# xbase: {DX / 2:.6e}", f"# ybase: {DX / 2:.6e}", f"# zbase: {DZ / 2:.6e}",
            f"# xnodes: {N}", f"# ynodes: {N}", "# znodes: 1",
            f"# xstepsize: {DX:.6e}", f"# ystepsize: {DX:.6e}", f"# zstepsize: {DZ:.6e}",
            "# End: Header", "# Begin: Data Binary 4"]
    data = np.stack([B[0], B[1], B[2]], axis=-1).astype("<f4")
    with open(path, "wb") as f:                        # binary: "\n" line endings, as MuMax3 requires
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.array([1234567.0], dtype="<f4").tobytes())
        f.write(data.tobytes())
        f.write(b"\n# End: Data Binary 4\n# End: Segment\n")


def make_map(name, dot, h, sign):
    path = os.path.join(OUT, name + ".ovf")
    c = (np.arange(N) - (N - 1) / 2) * DX
    X, Y = np.meshgrid(c, c)
    B = sign * field_at(load_map(dot), X, Y, h)
    write_ovf(path, B)
    return path.replace("\\", "/"), float(B[2, N // 2, N // 2] * 1e3)


# ------------------------------------------------------------------ MuMax3
def header(D, temp=0, skyrmion=True, sensor=False, contact=False):
    L = [f"SetGridSize({N}, {N}, 1)", f"SetCellSize({DX}, {DX}, {DZ})",
         f"Msat  = {FILM['Msat']}", f"Aex   = {FILM['Aex']}", f"Ku1   = {FILM['Ku']}",
         "AnisU = vector(0, 0, 1)", f"Dind  = {D}", f"alpha = {FILM['alpha']}"]
    if sensor:
        L.append(f"DefRegion(1, Circle({2 * SENSOR_R}))")
    if contact:
        L += [f"DefRegion(2, Circle({W_CONTACT_D}))", "Pol = 0.4", "Lambda = 1", "EpsilonPrime = 0",
              "FixedLayer = vector(0, 0, 1)"]
    if temp > 0:
        L += ["SetSolver(2)", f"FixDt = {T_DT}", f"Temp = {temp}", "ThermSeed(1)"]
    L.append("m = Uniform(0, 0, 1)")
    if skyrmion:
        L.append("m.SetInShape(Circle(30e-9), Uniform(0, 0, -1))")
    L += ["TableAdd(ext_topologicalcharge)", "TableAdd(B_ext)"]
    if sensor:
        L.append("TableAdd(m.Region(1))")
    return "\n".join(L) + "\n"


def run(name, text):
    mx3 = os.path.join(OUT, name + ".mx3")
    outdir = os.path.join(OUT, name + ".out")
    flag = os.path.join(outdir, "done.flag")
    if os.path.exists(flag):
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
                log(f"[{name}] stopped after {TIMEOUT // 60} min, recorded as timeout")
                return None
            if el - last >= 30:
                last = el
                tsim, rows = 0.0, 0
                if os.path.exists(table):
                    with open(table, encoding="utf-8") as tf:
                        lines = [l for l in tf if l.strip() and not l.startswith("#")]
                    rows = len(lines)
                    if lines:
                        tsim = float(lines[-1].split()[0]) * 1e9
                print(f"   [{name}] running {el:.0f} s, simulated time {tsim:.2f} ns, table rows {rows}", flush=True)
    if p.returncode != 0 or not os.path.exists(table):
        log(f"[{name}] MuMax3 failed, see skyrmion2_results/{name}_mumax.txt (recorded as failed)")
        return None
    for ovf in glob.glob(os.path.join(outdir, "m*.ovf")):
        subprocess.run([CONVERT, "-numpy", ovf], cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    open(flag, "w").close()
    log(f"[{name}] done in {time.time() - t0:.0f} s")
    return outdir


def read_table(outdir):
    with open(os.path.join(outdir, "table.txt"), encoding="utf-8") as f:
        head = f.readline().lstrip("#").split("\t")
        cols = [h.strip().split(" ")[0] for h in head]
        rows = [list(map(float, l.split())) for l in f if l.strip() and not l.startswith("#")]
    return cols, np.array(rows)


def col(cols, *needles):
    for i, c in enumerate(cols):
        lc = c.lower()
        if all(n in lc for n in needles):
            return i
    return None


def analyse_m(npy):
    mz = np.load(npy)[2, 0]
    c = (np.arange(N) - (N - 1) / 2) * DX
    X, Y = np.meshgrid(c, c)
    core = mz < 0
    r = np.sqrt(core.sum() * DX * DX / np.pi) * 1e9
    disk = X ** 2 + Y ** 2 < SENSOR_R ** 2
    return dict(radius_nm=round(float(r), 2), sensor_mz=round(float(mz[disk].mean()), 4), exists=bool(core.sum() > 3))


def last_m(od):
    npys = sorted(glob.glob(os.path.join(od, "m*.npy"))) if od else []
    return npys


# ---------------------------------------------------------------- part S
def part_S(res):
    log("--- Part S: sensor configurations at 0 K ---")
    out = {}
    steps = [0, 3, 6, 3, 0, -3, -6, -3, 0]
    gens = [("d30", h) for h in S_HEIGHTS] + [("d39", 45e-9)]
    for D in S_DMI:
        tag = f"S_D{D * 1e3:.1f}".replace(".", "p")
        body = "".join(f"B_ext = vector(0, 0, {b * 1e-3:.4e})\nRun(5e-10)\nTableSave()\nSave(m)\n" for b in steps)
        od = run(tag + "_transfer", header(D) + "Run(1e-9)\n" + body)
        pts = [dict(field_mT=b, **analyse_m(p)) for b, p in zip(steps, last_m(od))]
        slope = float(np.polyfit([p["field_mT"] for p in pts], [p["sensor_mz"] for p in pts], 1)[0]) if len(pts) == len(steps) else None
        entry = dict(radius_at_0_nm=pts[0]["radius_nm"] if pts else None, sensor_at_0=pts[0]["sensor_mz"] if pts else None,
                     slope_per_mT=round(slope, 5) if slope is not None else None,
                     stripes_or_lost=(not all(p["exists"] for p in pts)) if pts else None, generators={})
        for dot, h in gens:
            vals = {}
            for s, st in ((1, "up"), (-1, "down")):
                name = f"{tag}_{dot}_h{int(round(h * 1e9))}_{st}"
                path, bc = make_map(name, dot, h, s)
                od = run(name, header(D) + "Run(1e-9)\n" + f'B_ext.Add(LoadFile("{path}"), 1)\nRun(2e-9)\nTableSave()\nSave(m)\n')
                npys = last_m(od)
                vals[st] = dict(field_centre_mT=round(bc, 3), **(analyse_m(npys[-1]) if npys else dict(radius_nm=None, sensor_mz=None, exists=None)))
            u, d = vals["up"]["sensor_mz"], vals["down"]["sensor_mz"]
            entry["generators"][f"{dot}_h{int(round(h * 1e9))}"] = dict(up=vals["up"], down=vals["down"],
                                                                 contrast=round(abs(u - d), 4) if u is not None and d is not None else None)
        out[f"{D * 1e3:.1f}"] = entry
    res["S"] = out


# ---------------------------------------------------------------- part T
def part_T(res):
    log("--- Part T: real operating temperatures ---")
    out = {}
    nset = int(round(T_SETTLE / 1e-11))
    for D in T_DMI:
        for temp in T_TEMPS:
            for h in T_HEIGHTS:
                key = f"D{D * 1e3:.1f}_T{temp}_h{int(round(h * 1e9))}"
                series = {}
                for s, st in ((1, "up"), (-1, "down")):
                    name = "T_" + key.replace(".", "p") + "_" + st
                    path, bc = make_map(name, "d30", h, s)
                    text = (header(D, temp=temp, sensor=True) + f'B_ext.Add(LoadFile("{path}"), 1)\n'
                            + f"Run({T_SETTLE})\nTableAutosave(1e-11)\nRun({T_MEASURE})\nSave(m)\n")
                    od = run(name, text)
                    if not od:
                        series[st] = None
                        continue
                    cols, tab = read_table(od)
                    iz = col(cols, "region", "z")
                    iq = col(cols, "topologicalcharge")
                    if iz is None or len(tab) < 20:
                        series[st] = dict(columns=cols)
                        continue
                    z = tab[:, iz]
                    q = tab[:, iq]
                    blocks = z[: len(z) // 25 * 25].reshape(-1, 25).mean(axis=1)   # 0.25 ns blocks
                    series[st] = dict(field_centre_mT=round(bc, 3), mean=float(z.mean()), std=float(z.std()),
                                      block_std=float(blocks.std()) if len(blocks) > 1 else None,
                                      survived_fraction=float(np.mean(np.abs(q) > 0.5)))
                e = dict(up=series.get("up"), down=series.get("down"))
                u, d = series.get("up"), series.get("down")
                if u and d and "mean" in u and "mean" in d:
                    delta = abs(u["mean"] - d["mean"])
                    sig = np.sqrt((u["std"] ** 2 + d["std"] ** 2) / 2)
                    e["contrast"] = round(delta, 4)
                    e["snr_instant"] = round(delta / sig, 2) if sig > 0 else None
                    if u["block_std"] and d["block_std"]:
                        sb = np.sqrt((u["block_std"] ** 2 + d["block_std"] ** 2) / 2)
                        e["snr_0p25ns"] = round(delta / sb, 2) if sb > 0 else None
                out[key] = e
    res["T"] = out


# ---------------------------------------------------------------- part W
def part_W(res):
    log("--- Part W: writing and erasing with a current pulse ---")
    area = np.pi * (W_CONTACT_D / 2) ** 2
    assists = {"none": None, "gen21": 21e-9}
    out = []
    for op in ("write", "erase"):
        for an, ah in assists.items():
            # the generator helps: pointing down favours a skyrmion (write), pointing up pushes it out (erase)
            path = None
            if ah:
                path, _ = make_map(f"W_{op}_{an}", "d30", ah, -1 if op == "write" else 1)
            for sign in (1, -1):
                for tau in W_TAU:
                    for J in W_J:
                        name = f"W_{op}_{an}_{'pos' if sign > 0 else 'neg'}_t{tau * 1e9:.1f}_J{J / 1e12:.1f}".replace(".", "p")
                        text = header(W_DMI, skyrmion=(op == "erase"), contact=True) + "Run(1e-9)\n"
                        if path:
                            text += f'B_ext.Add(LoadFile("{path}"), 1)\nRun(3e-10)\n'
                        text += (f"J.SetRegion(2, vector(0, 0, {sign * J:.4e}))\nRun({tau})\n"
                                 f"J.SetRegion(2, vector(0, 0, 0))\nRun(1e-9)\nTableSave()\nSave(m)\n")
                        od = run(name, text)
                        q = None
                        if od:
                            cols, tab = read_table(od)
                            q = float(tab[-1, col(cols, "topologicalcharge")])
                        ok = None if q is None else (abs(q + 1) < 0.4 if op == "write" else abs(q) < 0.3)
                        out.append(dict(op=op, assist=an, current_sign=sign, pulse_ns=tau * 1e9, J=J, charge_end=q, success=ok,
                                        current_uA=round(J * area * 1e6, 1),
                                        energy_fJ=round(J * area * tau * W_VOLT * 1e15, 2)))
    res["W"] = out
    best = {}
    for r in out:
        if r["success"]:
            k = (r["op"], r["assist"])
            if k not in best or r["energy_fJ"] < best[k]["energy_fJ"]:
                best[k] = r
    res["W_best"] = {f"{k[0]}_{k[1]}": v for k, v in best.items()}


# ----------------------------------------------------------------- summary
def summary(res):
    L = ["Skyrmion sensor study, part 2", ""]
    if "S" in res:
        L.append("S. Sensor configurations at 0 K (contrast = |sensor(up) - sensor(down)|, full scale is 2)")
        for D, e in res["S"].items():
            L.append(f"  DMI {D}: skyrmion radius {e['radius_at_0_nm']} nm, slope {e['slope_per_mT']} per mT"
                     + ("  [stripes or skyrmion lost]" if e["stripes_or_lost"] else ""))
            for g, v in e["generators"].items():
                L.append(f"     generator {g}: field {v['up']['field_centre_mT']} mT, contrast {v['contrast']}")
        L.append("")
    if "T" in res:
        L.append(f"T. Real temperatures (SNR = contrast / noise; 0.25 ns = averaged over a 0.25 ns read window)")
        for k, e in res["T"].items():
            sv = [x.get("survived_fraction") for x in (e.get("up"), e.get("down")) if x and "survived_fraction" in x]
            L.append(f"  {k}: contrast {e.get('contrast')}, SNR instant {e.get('snr_instant')}, SNR 0.25 ns {e.get('snr_0p25ns')}, "
                     f"skyrmion present {min(sv) * 100:.0f}% of the time" if sv else f"  {k}: no data")
        L.append("")
    if "W" in res:
        L.append(f"W. Current pulse through a {W_CONTACT_D * 1e9:.0f} nm contact (energy assumes {W_VOLT} V across the contact)")
        for k, v in res.get("W_best", {}).items():
            L.append(f"  best {k}: J {v['J']:.1e} A/m^2 ({v['current_uA']} uA), {v['pulse_ns']} ns, current sign {v['current_sign']}, "
                     f"energy {v['energy_fJ']} fJ")
        for op in ("write", "erase"):
            for an in ("none", "gen21"):
                if f"{op}_{an}" not in res.get("W_best", {}):
                    L.append(f"  {op} with assist '{an}': no success in the tested range")
        L.append("  All runs: " + "; ".join(f"{r['op']}/{r['assist']}/{'+' if r['current_sign'] > 0 else '-'}/{r['pulse_ns']}ns/"
                                           f"{r['J']:.1e}: Q={None if r['charge_end'] is None else round(r['charge_end'], 2)}" for r in res["W"]))
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log("\n".join(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(MUMAX):
        sys.exit(f"mumax3 not found at {MUMAX}. Put this script next to mumax3{EXE}.")
    only = sys.argv[sys.argv.index("--only") + 1].upper() if "--only" in sys.argv else "STW"
    log(f"=== skyrmion study part 2 started {time.strftime('%Y-%m-%d %H:%M:%S')}, parts {only} ===")
    rpath = os.path.join(OUT, "results.json")
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}
    if "S" in only:
        part_S(res)
        summary(res)
    if "T" in only:
        part_T(res)
        summary(res)
    if "W" in only:
        part_W(res)
        summary(res)
    log("=== finished. Send skyrmion2_results/summary.txt and skyrmion2_results/results.json ===")


if __name__ == "__main__":
    main()
