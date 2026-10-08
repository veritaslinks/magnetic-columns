#!/usr/bin/env python3
"""
Skyrmion study, part 3 (MuMax3). Needs run_skyrmion_v2.py in the same folder (reuses its tools).
    python run_skyrmion_v3.py              # all parts
    python run_skyrmion_v3.py --only V     # V, C or R

Writes skyrmion3_results/summary.txt and results.json. Resumable.

V. Writing and erasing with a VOLTAGE gate (voltage-controlled magnetic anisotropy, VCMA), 0 K grid:
   a 30 nm gate lowers the anisotropy (write) or raises it (erase) for a short pulse, with the
   generator pointing down (write) or up (erase) at 21 nm, at 45 nm, or no generator.
   Minimal anisotropy change and the gate voltage and energy it would need.
C. Confirmation of the best V settings at 300 K, three thermal seeds each.
R. Sensor at 300 K repeated with three thermal seeds and 3 ns of read-out, for solid signal-to-noise numbers.
"""
import json
import os
import sys
import time

import numpy as np

import run_skyrmion_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "skyrmion3_results")
v2.OUT = OUT                                      # v2 tools write into this folder

D_GATE = 3.6e-3                                   # best sensor material from part 2
GATE_D = 30e-9
# relative anisotropy change during the pulse. At 0 K nothing nucleates (a uniform state has nothing to break its
# symmetry), so part V runs at the real operating temperature: thermal fluctuations do that job, as in a real chip.
FRACS = {"write": [0.5, 0.65, 0.8, 0.95], "erase": [0.5, 1.0, 1.5]}
V_TEMP = 300
TAUS = [0.5e-9, 2e-9]
ASSIST = {"none": None, "gen45": 45e-9, "gen21": 21e-9}
SEEDS = [1, 2, 3]
C_SEEDS = [2, 3, 4]                               # new seeds for the confirmation (seed 1 is used in part V)
# gate model for the voltage/energy estimate (typical literature values, stated as assumptions)
XI = 100e-15                                      # VCMA coefficient, J/(V*m)
T_FILM = 0.4e-9                                   # magnetic film thickness, m
D_OX = 1e-9                                       # MgO barrier thickness, m
EPS_R = 9.8                                       # MgO relative permittivity
EPS0 = 8.854e-12
R_CONFIGS = [(3.6e-3, 45e-9, "d30"), (3.6e-3, 30e-9, "d30"), (3.6e-3, 45e-9, "d39")]
R_MEASURE = 3e-9


def head(D, temp=0, seed=1, skyrmion=True, gate=False, sensor=False):
    L = [f"SetGridSize({v2.N}, {v2.N}, 1)", f"SetCellSize({v2.DX}, {v2.DX}, {v2.DZ})",
         f"Msat  = {v2.FILM['Msat']}", f"Aex   = {v2.FILM['Aex']}", f"Ku1   = {v2.FILM['Ku']}",
         "AnisU = vector(0, 0, 1)", f"Dind  = {D}", f"alpha = {v2.FILM['alpha']}"]
    if sensor:
        L.append(f"DefRegion(1, Circle({2 * v2.SENSOR_R}))")
    if gate:
        L.append(f"DefRegion(2, Circle({GATE_D}))")
        L.append(f"Ku1.SetRegion(2, {v2.FILM['Ku']})")
    if temp > 0:
        L += ["SetSolver(2)", f"FixDt = {v2.T_DT}", f"Temp = {temp}", f"ThermSeed({seed})"]
    L.append("m = Uniform(0, 0, 1)")
    if skyrmion:
        L.append("m.SetInShape(Circle(30e-9), Uniform(0, 0, -1))")
    L += ["TableAdd(ext_topologicalcharge)", "TableAdd(B_ext)"]
    if sensor:
        L.append("TableAdd(m.Region(1))")
    return "\n".join(L) + "\n"


def gate_cost(frac):
    dk_v = frac * v2.FILM["Ku"]                   # J/m^3
    volt = dk_v * T_FILM * D_OX / XI              # V
    cap = EPS0 * EPS_R * np.pi * (GATE_D / 2) ** 2 / D_OX
    return round(float(volt), 2), round(float(cap * volt ** 2 * 1e15), 4)   # V, fJ


def vcma_run(name, op, frac, tau, ah, temp=0, seed=1):
    text = head(D_GATE, temp=temp, seed=seed, skyrmion=(op == "erase"), gate=True)
    text += "Run(1e-9)\n"
    if ah:
        path, _ = v2.make_map(f"V_{op}_h{int(round(ah * 1e9))}", "d30", ah, -1 if op == "write" else 1)
        text += f'B_ext.Add(LoadFile("{path}"), 1)\nRun(3e-10)\n'
    k_pulse = v2.FILM["Ku"] * (1 - frac if op == "write" else 1 + frac)
    text += (f"Ku1.SetRegion(2, {k_pulse:.6e})\nRun({tau})\nKu1.SetRegion(2, {v2.FILM['Ku']})\n"
             f"Run(1.5e-9)\nTableSave()\nSave(m)\n")
    od = v2.run(name, text)
    if not od:
        return None
    cols, tab = v2.read_table(od)
    return float(tab[-1, v2.col(cols, "topologicalcharge")])


def success(op, q):
    if q is None:
        return None
    return abs(q + 1) < 0.4 if op == "write" else abs(q) < 0.3


def part_V(res):
    v2.log(f"--- Part V: voltage-gate write/erase at {V_TEMP} K ---")
    rows = []
    for op in ("write", "erase"):
        for an, ah in ASSIST.items():
            for tau in TAUS:
                for f in FRACS[op]:
                    name = f"V{V_TEMP}_{op}_{an}_t{tau * 1e9:.1f}_f{int(f * 100)}".replace(".", "p")
                    q = vcma_run(name, op, f, tau, ah, temp=V_TEMP, seed=1)
                    volt, en = gate_cost(f)
                    rows.append(dict(op=op, assist=an, pulse_ns=tau * 1e9, frac=f, charge_end=q, success=success(op, q),
                                     gate_V=volt, energy_fJ=en))
    res["V"] = rows
    best = {}
    for r in rows:
        if r["success"]:
            k = f"{r['op']}_{r['assist']}"
            if k not in best or (r["frac"], r["pulse_ns"]) < (best[k]["frac"], best[k]["pulse_ns"]):
                best[k] = r
    res["V_best"] = best


def part_C(res):
    v2.log("--- Part C: best gate settings confirmed at 300 K ---")
    out = {}
    for k, b in res.get("V_best", {}).items():
        ok = []
        for s in C_SEEDS:
            name = f"C_{k}_t{b['pulse_ns']:.1f}_f{int(b['frac'] * 100)}_s{s}".replace(".", "p")
            q = vcma_run(name, b["op"], b["frac"], b["pulse_ns"] * 1e-9, ASSIST[b["assist"]], temp=V_TEMP, seed=s)
            ok.append(success(b["op"], q))
        out[k] = dict(settings=b, successes=sum(1 for x in ok if x), runs=len(ok))
    res["C"] = out


def part_R(res):
    v2.log("--- Part R: sensor at 300 K, three seeds, 3 ns read-out ---")
    out = {}
    for D, h, dot in R_CONFIGS:
        key = f"D{D * 1e3:.1f}_{dot}_h{int(round(h * 1e9))}"
        per_seed = []
        for s in SEEDS:
            means, stds, blocks = {}, {}, {}
            for sg, st in ((1, "up"), (-1, "down")):
                name = f"R_{key}_s{s}_{st}".replace(".", "p")
                path, _ = v2.make_map(name, dot, h, sg)
                text = (head(D, temp=300, seed=s, sensor=True) + f'B_ext.Add(LoadFile("{path}"), 1)\n'
                        + f"Run({v2.T_SETTLE})\nTableAutosave(1e-11)\nRun({R_MEASURE})\nSave(m)\n")
                od = v2.run(name, text)
                if not od:
                    continue
                cols, tab = v2.read_table(od)
                iz = v2.col(cols, "region", "z")
                if iz is None:
                    continue
                z = tab[:, iz]
                means[st], stds[st] = float(z.mean()), float(z.std())
                b = z[: len(z) // 25 * 25].reshape(-1, 25).mean(axis=1)
                blocks[st] = b
            if "up" in means and "down" in means:
                per_seed.append(dict(seed=s, up=means["up"], down=means["down"],
                                     contrast=abs(means["up"] - means["down"]),
                                     noise=float(np.sqrt((stds["up"] ** 2 + stds["down"] ** 2) / 2)),
                                     block_noise=float(np.sqrt((blocks["up"].std() ** 2 + blocks["down"].std() ** 2) / 2))))
        if per_seed:
            c = np.array([p["contrast"] for p in per_seed])
            bn = np.array([p["block_noise"] for p in per_seed])
            out[key] = dict(seeds=per_seed, contrast_mean=round(float(c.mean()), 4), contrast_min=round(float(c.min()), 4),
                            snr_0p25ns_mean=round(float((c / bn).mean()), 2), snr_0p25ns_min=round(float((c / bn).min()), 2))
    res["R"] = out


def summary(res):
    L = ["Skyrmion study, part 3", ""]
    if "V" in res:
        L.append(f"V. Voltage gate {GATE_D * 1e9:.0f} nm, DMI {D_GATE * 1e3:.1f}, at {V_TEMP} K. Gate model: VCMA {XI * 1e15:.0f} fJ/(V m), "
                 f"MgO {D_OX * 1e9:.0f} nm, film {T_FILM * 1e9:.1f} nm")
        for k, b in res.get("V_best", {}).items():
            L.append(f"  best {k}: anisotropy change {b['frac'] * 100:.0f}%, pulse {b['pulse_ns']} ns, gate {b['gate_V']} V, energy {b['energy_fJ']} fJ")
        for op in ("write", "erase"):
            for an in ASSIST:
                if f"{op}_{an}" not in res.get("V_best", {}):
                    L.append(f"  {op} with assist '{an}': no success in the tested range")
        L.append("  All runs: " + "; ".join(f"{r['op']}/{r['assist']}/{r['pulse_ns']}ns/{int(r['frac'] * 100)}%: "
                                           f"Q={None if r['charge_end'] is None else round(r['charge_end'], 2)}" for r in res["V"]))
        L.append("")
    if "C" in res:
        L.append(f"C. Best gate settings repeated at {V_TEMP} K with three new thermal seeds")
        for k, v in res["C"].items():
            L.append(f"  {k}: {v['successes']} of {v['runs']} succeeded")
        L.append("")
    if "R" in res:
        L.append(f"R. Sensor at 300 K, three seeds, {R_MEASURE * 1e9:.0f} ns read-out each")
        for k, v in res["R"].items():
            L.append(f"  {k}: contrast mean {v['contrast_mean']} (min {v['contrast_min']}), "
                     f"SNR over 0.25 ns mean {v['snr_0p25ns_mean']} (min {v['snr_0p25ns_min']})")
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
    only = sys.argv[sys.argv.index("--only") + 1].upper() if "--only" in sys.argv else "VCR"
    v2.log(f"=== skyrmion study part 3 started {time.strftime('%Y-%m-%d %H:%M:%S')}, parts {only} ===")
    rpath = os.path.join(OUT, "results.json")
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}
    if "V" in only:
        part_V(res)
        summary(res)
    if "C" in only:
        part_C(res)
        summary(res)
    if "R" in only:
        part_R(res)
        summary(res)
    v2.log("=== finished. Send skyrmion3_results/summary.txt and skyrmion3_results/results.json ===")


if __name__ == "__main__":
    main()
