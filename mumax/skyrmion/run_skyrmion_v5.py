#!/usr/bin/env python3
"""
Skyrmion study, part 5 (MuMax3). Needs run_skyrmion_v2.py and run_skyrmion_v4.py in the same folder.
    python run_skyrmion_v5.py              # A, then B in the best stable configuration found by A
    python run_skyrmion_v5.py --only A     # or --only B (B needs A results)

Writes skyrmion5_results/summary.txt and results.json. Resumable.

Part 4 showed: a 0.4 nm layer gives signal but the skyrmion lives only part of the time; a 1.2 nm
layer in an 80 nm disk keeps it alive 99% of the time but the material (tuned for 0.4 nm) gives a
weak signal. And the gate test used an unstable geometry and a one-instant success test.

A. 80 nm disk at 300 K: layer 1.2 nm with DMI 3.2/3.4/3.8 and layer 2.0 nm with DMI 3.4/3.6/3.8.
   Read accuracy for 0.25 ns and 1 ns windows, contrast, skyrmion survival.
B. In the best configuration from A with survival >= 95%: measured reference levels (skyrmion present /
   absent), then gate write/erase with and without a generator 21 nm above, three seeds, no-pulse
   controls. Success = sensor signal averaged over the last 0.5 ns crosses the midpoint between the
   two reference levels.
"""
import json
import os
import sys
import time

import numpy as np

import run_skyrmion_v2 as v2
import run_skyrmion_v4 as v4

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "skyrmion5_results")
v2.OUT = OUT
v4.OUT = OUT

TEMP = 300
DISK = 80e-9
SEEDS = [1, 2, 3]
A_CONFIGS = [(1.2e-9, 3.2e-3), (1.2e-9, 3.4e-3), (1.2e-9, 3.8e-3), (2.0e-9, 3.4e-3), (2.0e-9, 3.6e-3), (2.0e-9, 3.8e-3)]
GEN_DOT, GEN_H = "d39", 45e-9
READ = 3e-9
B_FRACS = {"write": [0.0, 0.5, 0.8, 0.95], "erase": [0.0, 0.5, 1.0, 1.5]}
B_ASSIST = {"none": None, "gen21": 21e-9}
TAU = 1e-9
GATE_D = 30e-9


def head(dz, D, seed, skyrmion=True, gate=False):
    L = [f"SetGridSize({v2.N}, {v2.N}, 1)", f"SetCellSize({v2.DX}, {v2.DX}, {dz:.4e})", f"SetGeom(Circle({DISK}))",
         f"Msat  = {v2.FILM['Msat']}", f"Aex   = {v2.FILM['Aex']}", f"Ku1   = {v2.FILM['Ku']}",
         "AnisU = vector(0, 0, 1)", f"Dind  = {D}", f"alpha = {v2.FILM['alpha']}",
         f"DefRegion(1, Circle({2 * v2.SENSOR_R}))"]
    if gate:
        L += [f"DefRegion(2, Circle({GATE_D}))", f"Ku1.SetRegion(2, {v2.FILM['Ku']})"]
    L += ["SetSolver(2)", f"FixDt = {v2.T_DT}", f"Temp = {TEMP}", f"ThermSeed({seed})", "m = Uniform(0, 0, 1)"]
    if skyrmion:
        L.append("m.SetInShape(Circle(30e-9), Uniform(0, 0, -1))")
    L += ["TableAdd(ext_topologicalcharge)", "TableAdd(B_ext)", "TableAdd(m.Region(1))"]
    return "\n".join(L) + "\n"


def key_of(dz, D):
    return f"t{dz * 1e9:.1f}_D{D * 1e3:.1f}".replace(".", "p")


def sensor_series(od):
    cols, tab = v2.read_table(od)
    iz, iq = v2.col(cols, "region", "z"), v2.col(cols, "topologicalcharge")
    return tab[:, iz], tab[:, iq]


def accuracy(u, d):
    thr = (u.mean() + d.mean()) / 2
    return float((np.sum(u > thr) + np.sum(d <= thr)) / (len(u) + len(d)))


# ------------------------------------------------------------------ part A
def part_A(res):
    v2.log(f"--- Part A: material for stable skyrmions, 80 nm disk, {TEMP} K ---")
    out = {}
    for dz, D in A_CONFIGS:
        k = key_of(dz, D)
        w25 = {"up": [], "down": []}
        w100 = {"up": [], "down": []}
        seeds = []
        for s in SEEDS:
            sm = {}
            for sg, st in ((1, "up"), (-1, "down")):
                name = f"A_{k}_s{s}_{st}"
                path = v4.make_map(name, GEN_DOT, GEN_H, sg, dz)
                text = head(dz, D, s) + f'B_ext.Add(LoadFile("{path}"), 1)\n' \
                    + f"Run({v2.T_SETTLE})\nTableAutosave(1e-11)\nRun({READ})\nSave(m)\n"
                od = v2.run(name, text)
                if not od:
                    continue
                z, q = sensor_series(od)
                w25[st] += list(z[: len(z) // 25 * 25].reshape(-1, 25).mean(axis=1))
                w100[st] += list(z[: len(z) // 100 * 100].reshape(-1, 100).mean(axis=1))
                sm[st] = dict(mean=float(z.mean()), alive=float(np.mean(np.abs(q) > 0.5)))
            if "up" in sm and "down" in sm:
                seeds.append(dict(seed=s, up=sm["up"]["mean"], down=sm["down"]["mean"],
                                  contrast=sm["up"]["mean"] - sm["down"]["mean"],
                                  alive=min(sm["up"]["alive"], sm["down"]["alive"])))
        if not seeds:
            out[k] = dict(error="no data")
            continue
        c = np.array([x["contrast"] for x in seeds])
        out[k] = dict(thickness_nm=dz * 1e9, dmi=D * 1e3,
                      read_accuracy_0p25ns=round(accuracy(np.array(w25["up"]), np.array(w25["down"])), 3),
                      read_accuracy_1ns=round(accuracy(np.array(w100["up"]), np.array(w100["down"])), 3),
                      contrast_mean=round(float(c.mean()), 3), contrast_min=round(float(c.min()), 3),
                      wrong_sign_seeds=int(np.sum(c <= 0)), skyrmion_alive_min=round(float(min(x["alive"] for x in seeds)), 3),
                      sensor_level=round(float(np.mean([(x["up"] + x["down"]) / 2 for x in seeds])), 3), seeds=seeds)
    res["A"] = out
    stable = {k: v for k, v in out.items() if v.get("skyrmion_alive_min", 0) >= 0.95}
    if stable:
        res["A_best"] = max(stable, key=lambda k: (stable[k]["read_accuracy_1ns"], stable[k]["read_accuracy_0p25ns"], stable[k]["contrast_min"]))


# ------------------------------------------------------------------ part B
def part_B(res):
    best = res.get("A_best")
    if not best:
        v2.log("Part B skipped: no configuration in part A kept the skyrmion alive >= 95% of the time.")
        return
    cfg = res["A"][best]
    dz, D = cfg["thickness_nm"] * 1e-9, cfg["dmi"] * 1e-3
    v2.log(f"--- Part B: gate write/erase at {TEMP} K in {best} ---")
    # reference levels: skyrmion present / absent, no generator, no gate pulse
    refs = {"present": [], "absent": []}
    for s in SEEDS:
        for lvl, sk in (("present", True), ("absent", False)):
            name = f"B_{best}_ref_{lvl}_s{s}"
            od = v2.run(name, head(dz, D, s, skyrmion=sk) + f"Run(1e-9)\nTableAutosave(1e-11)\nRun(1e-9)\nSave(m)\n")
            if od:
                z, _ = sensor_series(od)
                refs[lvl].append(float(z.mean()))
    if not refs["present"] or not refs["absent"]:
        v2.log("Part B stopped: reference runs failed.")
        return
    lp, la = float(np.mean(refs["present"])), float(np.mean(refs["absent"]))
    thr = (lp + la) / 2
    res["B_refs"] = dict(present=round(lp, 3), absent=round(la, 3), threshold=round(thr, 3))
    rows = []
    for op in ("write", "erase"):
        for an, ah in B_ASSIST.items():
            for f in B_FRACS[op]:
                for s in SEEDS:
                    name = f"B_{best}_{op}_{an}_f{int(f * 100)}_s{s}"
                    text = head(dz, D, s, skyrmion=(op == "erase"), gate=True) + "Run(1e-9)\n"
                    if ah:
                        path = v4.make_map(f"B_{best}_{op}_h{int(round(ah * 1e9))}", "d30", ah, -1 if op == "write" else 1, dz)
                        text += f'B_ext.Add(LoadFile("{path}"), 1)\nRun(3e-10)\n'
                    if f > 0:
                        kp = v2.FILM["Ku"] * (1 - f if op == "write" else 1 + f)
                        text += f"Ku1.SetRegion(2, {kp:.6e})\nRun({TAU})\nKu1.SetRegion(2, {v2.FILM['Ku']})\n"
                    else:
                        text += f"Run({TAU})\n"
                    text += "Run(5e-10)\nTableAutosave(1e-11)\nRun(5e-10)\nSave(m)\n"
                    od = v2.run(name, text)
                    level = None
                    if od:
                        z, _ = sensor_series(od)
                        n = max(1, int(round(5e-10 / 1e-11)))
                        level = float(z[-n:].mean())
                    present = None if level is None else level < thr
                    ok = None if present is None else (present if op == "write" else not present)
                    volt, en = v4.gate_cost(f, dz)
                    rows.append(dict(op=op, assist=an, frac=f, seed=s, sensor_level=level, success=ok, gate_V=volt, energy_fJ=en))
    res["B"] = rows


def summary(res):
    L = ["Skyrmion study, part 5", ""]
    if "A" in res:
        L.append(f"A. 80 nm disk, {TEMP} K, generator {GEN_DOT} at {GEN_H * 1e9:.0f} nm, {len(SEEDS)} seeds x {READ * 1e9:.0f} ns "
                 f"(read accuracy: 0.5 = guessing, 1.0 = always right)")
        for k, v in res["A"].items():
            if "error" in v:
                L.append(f"  {k}: {v['error']}")
                continue
            L.append(f"  layer {v['thickness_nm']:.1f} nm, DMI {v['dmi']:.1f}: accuracy 0.25 ns {v['read_accuracy_0p25ns']}, 1 ns {v['read_accuracy_1ns']}; "
                     f"contrast {v['contrast_mean']} (min {v['contrast_min']}); wrong-sign seeds {v['wrong_sign_seeds']}; "
                     f"alive (worst) {v['skyrmion_alive_min']}; sensor level {v['sensor_level']}")
        L.append(f"  best stable configuration: {res.get('A_best')}")
        L.append("")
    if "B" in res:
        r = res["B_refs"]
        L.append(f"B. Gate write/erase at {TEMP} K in {res.get('A_best')}, pulse {TAU * 1e9:.0f} ns, {len(SEEDS)} seeds (0% = control)")
        L.append(f"   reference sensor levels: skyrmion present {r['present']}, absent {r['absent']}, threshold {r['threshold']}")
        for op in ("write", "erase"):
            for an in B_ASSIST:
                parts = []
                for f in B_FRACS[op]:
                    rr = [x for x in res["B"] if x["op"] == op and x["assist"] == an and x["frac"] == f]
                    n_ok = sum(1 for x in rr if x["success"])
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
    only = sys.argv[sys.argv.index("--only") + 1].upper() if "--only" in sys.argv else "AB"
    v2.log(f"=== skyrmion study part 5 started {time.strftime('%Y-%m-%d %H:%M:%S')}, parts {only} ===")
    rpath = os.path.join(OUT, "results.json")
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}
    if "A" in only:
        part_A(res)
        summary(res)
    if "B" in only:
        part_B(res)
        summary(res)
    v2.log("=== finished. Send skyrmion5_results/summary.txt and skyrmion5_results/results.json ===")


if __name__ == "__main__":
    main()
