#!/usr/bin/env python3
"""
Skyrmion study, part 4 (MuMax3). Needs run_skyrmion_v2.py in the same folder.
    python run_skyrmion_v4.py              # M, then V in the best geometry found by M
    python run_skyrmion_v4.py --only M     # or --only V (V needs M results)

Writes skyrmion4_results/summary.txt and results.json. Resumable.

Part 3 showed that in an open 0.4 nm film at 300 K the skyrmion wanders: the sensor is right on
average but not in every read-out. Real devices fix this with thicker magnetic stacks and by
confining each skyrmion in a patterned nanodisk. This script tests exactly that.

M. Sensor reliability at 300 K: open film vs 60 nm and 80 nm nanodisks, magnetic layer 0.4 nm and
   1.2 nm, 39 nm generator 45 nm above, three thermal seeds, 3 ns read-out. Key number: the share
   of 0.25 ns read windows in which the sensor names the generator state correctly.
V. Voltage-gate write/erase at 300 K in the best geometry from M, with no-pulse controls,
   so that writing by the gate can be told apart from random thermal events.
"""
import json
import os
import sys
import time

import numpy as np

import run_skyrmion_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "skyrmion4_results")
v2.OUT = OUT

D_MAT = 3.6e-3
TEMP = 300
SEEDS = [1, 2, 3]
GEOMS = {"open": None, "disk60": 60e-9, "disk80": 80e-9}
THICK = [0.4e-9, 1.2e-9]
GEN_DOT, GEN_H = "d39", 45e-9
READ = 3e-9
WIN = 25                                          # table rows per read window (0.25 ns at 10 ps)

V_FRACS = {"write": [0.0, 0.5, 0.8, 0.95], "erase": [0.0, 0.5, 1.0, 1.5]}   # 0.0 = control, no gate pulse
V_ASSIST = {"none": None, "gen45": 45e-9, "gen21": 21e-9}
V_SEEDS = [1, 2]
V_TAU = 1e-9
GATE_D = 30e-9
XI, D_OX, EPS_R, EPS0 = 100e-15, 1e-9, 9.8, 8.854e-12


def write_ovf(path, B, dz):
    head = ["# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header", "# Title: B_generators",
            "# meshtype: rectangular", "# meshunit: m", "# xmin: 0", "# ymin: 0", "# zmin: 0",
            f"# xmax: {v2.N * v2.DX:.6e}", f"# ymax: {v2.N * v2.DX:.6e}", f"# zmax: {dz:.6e}", "# valuedim: 3",
            "# valuelabels: B_x B_y B_z", "# valueunits: T T T", "# Desc: generator stray field",
            f"# xbase: {v2.DX / 2:.6e}", f"# ybase: {v2.DX / 2:.6e}", f"# zbase: {dz / 2:.6e}",
            f"# xnodes: {v2.N}", f"# ynodes: {v2.N}", "# znodes: 1",
            f"# xstepsize: {v2.DX:.6e}", f"# ystepsize: {v2.DX:.6e}", f"# zstepsize: {dz:.6e}",
            "# End: Header", "# Begin: Data Binary 4"]
    data = np.stack([B[0], B[1], B[2]], axis=-1).astype("<f4")
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.array([1234567.0], dtype="<f4").tobytes())
        f.write(data.tobytes())
        f.write(b"\n# End: Data Binary 4\n# End: Segment\n")


def make_map(name, dot, h, sign, dz):
    path = os.path.join(OUT, name + ".ovf")
    c = (np.arange(v2.N) - (v2.N - 1) / 2) * v2.DX
    X, Y = np.meshgrid(c, c)
    write_ovf(path, sign * v2.field_at(v2.load_map(dot), X, Y, h), dz)
    return path.replace("\\", "/")


def head(geom, dz, seed, skyrmion=True, gate=False):
    L = [f"SetGridSize({v2.N}, {v2.N}, 1)", f"SetCellSize({v2.DX}, {v2.DX}, {dz:.4e})"]
    if GEOMS[geom]:
        L.append(f"SetGeom(Circle({GEOMS[geom]}))")
    L += [f"Msat  = {v2.FILM['Msat']}", f"Aex   = {v2.FILM['Aex']}", f"Ku1   = {v2.FILM['Ku']}",
          "AnisU = vector(0, 0, 1)", f"Dind  = {D_MAT}", f"alpha = {v2.FILM['alpha']}",
          f"DefRegion(1, Circle({2 * v2.SENSOR_R}))"]
    if gate:
        L += [f"DefRegion(2, Circle({GATE_D}))", f"Ku1.SetRegion(2, {v2.FILM['Ku']})"]
    L += ["SetSolver(2)", f"FixDt = {v2.T_DT}", f"Temp = {TEMP}", f"ThermSeed({seed})", "m = Uniform(0, 0, 1)"]
    if skyrmion:
        L.append("m.SetInShape(Circle(30e-9), Uniform(0, 0, -1))")
    L += ["TableAdd(ext_topologicalcharge)", "TableAdd(B_ext)", "TableAdd(m.Region(1))"]
    return "\n".join(L) + "\n"


def tag(geom, dz):
    return f"{geom}_t{dz * 1e9:.1f}".replace(".", "p")


# ------------------------------------------------------------------ part M
def part_M(res):
    v2.log(f"--- Part M: sensor reliability at {TEMP} K ---")
    out = {}
    for geom in GEOMS:
        for dz in THICK:
            key = tag(geom, dz)
            wins = {"up": [], "down": []}
            seeds = []
            for s in SEEDS:
                sm = {}
                for sg, st in ((1, "up"), (-1, "down")):
                    name = f"M_{key}_s{s}_{st}"
                    path = make_map(name, GEN_DOT, GEN_H, sg, dz)
                    text = head(geom, dz, s) + f'B_ext.Add(LoadFile("{path}"), 1)\n' \
                        + f"Run({v2.T_SETTLE})\nTableAutosave(1e-11)\nRun({READ})\nSave(m)\n"
                    od = v2.run(name, text)
                    if not od:
                        continue
                    cols, tab = v2.read_table(od)
                    iz, iq = v2.col(cols, "region", "z"), v2.col(cols, "topologicalcharge")
                    if iz is None:
                        continue
                    z = tab[:, iz]
                    w = z[: len(z) // WIN * WIN].reshape(-1, WIN).mean(axis=1)
                    wins[st] += list(w)
                    sm[st] = dict(mean=float(z.mean()), alive=float(np.mean(np.abs(tab[:, iq]) > 0.5)))
                if "up" in sm and "down" in sm:
                    seeds.append(dict(seed=s, up=sm["up"]["mean"], down=sm["down"]["mean"],
                                      contrast=sm["up"]["mean"] - sm["down"]["mean"],
                                      alive=min(sm["up"]["alive"], sm["down"]["alive"])))
            if not wins["up"] or not wins["down"]:
                out[key] = dict(error="no data")
                continue
            u, d = np.array(wins["up"]), np.array(wins["down"])
            thr = (u.mean() + d.mean()) / 2          # one fixed decision threshold for all seeds
            acc = (np.sum(u > thr) + np.sum(d <= thr)) / (len(u) + len(d))
            c = np.array([x["contrast"] for x in seeds])
            out[key] = dict(read_accuracy=round(float(acc), 3), contrast_mean=round(float(c.mean()), 3),
                            contrast_min=round(float(c.min()), 3), wrong_sign_seeds=int(np.sum(c <= 0)),
                            skyrmion_alive_min=round(float(min(x["alive"] for x in seeds)), 3), seeds=seeds)
    res["M"] = out
    ok = {k: v for k, v in out.items() if "read_accuracy" in v}
    if ok:
        best = max(ok, key=lambda k: (ok[k]["read_accuracy"], ok[k]["contrast_min"]))
        res["M_best"] = best


# ------------------------------------------------------------------ part V
def gate_cost(frac, dz):
    volt = frac * v2.FILM["Ku"] * dz * D_OX / XI          # anisotropy change over the whole layer, supplied by the interface
    cap = EPS0 * EPS_R * np.pi * (GATE_D / 2) ** 2 / D_OX
    return round(float(volt), 2), round(float(cap * volt ** 2 * 1e15), 4)


def part_V(res):
    best = res.get("M_best")
    if not best:
        v2.log("Part V skipped: run part M first.")
        return
    geom, dz = best.rsplit("_t", 1)[0], round(float(best.rsplit("_t", 1)[1].replace("p", ".")), 3) * 1e-9
    v2.log(f"--- Part V: voltage-gate write/erase at {TEMP} K in geometry {best} ---")
    rows = []
    for op in ("write", "erase"):
        for an, ah in V_ASSIST.items():
            for f in V_FRACS[op]:
                for s in V_SEEDS:
                    name = f"V_{best}_{op}_{an}_f{int(f * 100)}_s{s}"
                    text = head(geom, dz, s, skyrmion=(op == "erase"), gate=True) + "Run(1e-9)\n"
                    if ah:
                        path = make_map(f"V_{best}_{op}_h{int(round(ah * 1e9))}", "d30", ah, -1 if op == "write" else 1, dz)
                        text += f'B_ext.Add(LoadFile("{path}"), 1)\nRun(3e-10)\n'
                    if f > 0:
                        kp = v2.FILM["Ku"] * (1 - f if op == "write" else 1 + f)
                        text += f"Ku1.SetRegion(2, {kp:.6e})\nRun({V_TAU})\nKu1.SetRegion(2, {v2.FILM['Ku']})\n"
                    else:
                        text += f"Run({V_TAU})\n"
                    text += "Run(1e-9)\nTableSave()\nSave(m)\n"
                    od = v2.run(name, text)
                    q = None
                    if od:
                        cols, tab = v2.read_table(od)
                        q = float(tab[-1, v2.col(cols, "topologicalcharge")])
                    ok = None if q is None else (abs(q + 1) < 0.4 if op == "write" else abs(q) < 0.3)
                    volt, en = gate_cost(f, dz)
                    rows.append(dict(op=op, assist=an, frac=f, seed=s, charge_end=q, success=ok, gate_V=volt, energy_fJ=en))
    res["V"] = rows


def summary(res):
    L = ["Skyrmion study, part 4", ""]
    if "M" in res:
        L.append(f"M. Sensor at {TEMP} K, generator {GEN_DOT} at {GEN_H * 1e9:.0f} nm, {len(SEEDS)} seeds x {READ * 1e9:.0f} ns")
        L.append("   read accuracy = share of 0.25 ns windows where the sensor names the generator state correctly (0.5 = guessing)")
        for k, v in res["M"].items():
            if "read_accuracy" in v:
                L.append(f"  {k}: read accuracy {v['read_accuracy']}, contrast mean {v['contrast_mean']} (min {v['contrast_min']}), "
                         f"wrong-sign seeds {v['wrong_sign_seeds']}, skyrmion alive (worst) {v['skyrmion_alive_min']}")
            else:
                L.append(f"  {k}: {v.get('error')}")
        L.append(f"  best geometry: {res.get('M_best')}")
        L.append("")
    if "V" in res:
        L.append(f"V. Voltage gate at {TEMP} K in {res.get('M_best')}, pulse {V_TAU * 1e9:.0f} ns, {len(V_SEEDS)} seeds (0% = control, no pulse)")
        for op in ("write", "erase"):
            for an in V_ASSIST:
                parts = []
                for f in V_FRACS[op]:
                    rr = [r for r in res["V"] if r["op"] == op and r["assist"] == an and r["frac"] == f]
                    n_ok = sum(1 for r in rr if r["success"])
                    parts.append(f"{int(f * 100)}%: {n_ok}/{len(rr)}" + ("" if f == 0 else f" ({rr[0]['gate_V']} V, {rr[0]['energy_fJ']} fJ)"))
                L.append(f"  {op}, generator {an}: " + "; ".join(parts))
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    v2.log("\n".join(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(v2.CACHE, exist_ok=True)
    if not os.path.exists(v2.MUMAX):
        sys.exit(f"mumax3 not found at {v2.MUMAX}.")
    only = sys.argv[sys.argv.index("--only") + 1].upper() if "--only" in sys.argv else "MV"
    v2.log(f"=== skyrmion study part 4 started {time.strftime('%Y-%m-%d %H:%M:%S')}, parts {only} ===")
    rpath = os.path.join(OUT, "results.json")
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}
    if "M" in only:
        part_M(res)
        summary(res)
    if "V" in only:
        part_V(res)
        summary(res)
    v2.log("=== finished. Send skyrmion4_results/summary.txt and skyrmion4_results/results.json ===")


if __name__ == "__main__":
    main()
