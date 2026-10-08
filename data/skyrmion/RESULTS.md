# Skyrmion film as an optional alternative sensor: results

These are the MuMax3 results behind Section 6 of the manuscript. The main architecture uses direct field sensors (tunnel junction or Hall element); the skyrmion film is an optional alternative and the results of Sections 4 and 5 do not depend on it.

Scripts: `mumax/skyrmion/`. Raw summaries of part 5: `part5_summary.txt`, `part5_results.json`. Parts 1 to 4 can be added from the local output folders (`skyrmion_results`, `skyrmion2_results`, `skyrmion3_results`, `skyrmion4_results`) as `partN_summary.txt` and `partN_results.json`.

Film: Pt/Co-like, Msat 580 kA/m, A 15 pJ/m, Ku 0.8 MJ/m³, damping 0.3, interfacial DMI 3.0 to 3.8 mJ/m², 256 × 256 nm on 2 nm cells. Generator fields: MuMax3 vector maps of 30 and 39 nm perpendicular generators. Sensor signal: mean out-of-plane magnetisation in a disk of 20 nm radius.

## 0 K (clean reference)

| Finding | Value |
|---|---|
| Summed field of three generators vs sensor signal | correlation 0.998 to 0.999 |
| Response to +20 / −20 mT (skyrmion radius 15 nm) | +0.28 / −0.71 (asymmetric, built-in nonlinearity) |
| Best sensitivity | skyrmion radius ≈ sensor radius (18.8 nm vs 20 nm), DMI 3.6 |
| Contrast up vs down at the optimum | 0.47 (30 nm generator, 30 nm away), 0.38 (39 nm, 45 nm away), 0.24 (30 nm, 45 nm away) |
| Field needed to collapse the skyrmion | 200 to 480 mT, it does not return; generators produce at most about 25 mT |

## 300 K (39 nm generator 45 nm above, three thermal seeds × 3 ns)

| Geometry | Layer (nm) | DMI | Accuracy 0.25 ns | Accuracy 1 ns | Contrast mean (min) | Survival |
|---|---|---|---|---|---|---|
| open film | 0.4 | 3.6 | 0.69 | – | 0.39 (−0.12) | 85% |
| 60 nm disk | 0.4 | 3.6 | 0.65 | – | 0.22 (0.09) | 28% |
| 80 nm disk | 0.4 | 3.6 | 0.81 | – | 0.46 (0.15) | 38% |
| 80 nm disk | 1.2 | 3.6 | 0.76 | – | 0.09 (0.08) | 99% |
| 80 nm disk | 1.2 | 3.8 | 0.67 | 0.94 | 0.07 (0.06) | 99% |
| 80 nm disk | 2.0 | 3.6 | 0.81 | 0.89 | 0.10 (0.10) | 100% |
| **80 nm disk** | **2.0** | **3.4** | **0.875** | **1.00 (18/18)** | **0.10 (0.09)** | **100%** |

18 of 18 correct read-outs only bound the error rate below about 17% at 95% confidence.

## Writing skyrmions (not needed for sensing)

- Current pulse through a 30 nm contact: works only at 3 × 10¹² A/m² (about 2.1 mA, about 190 fJ per write at an assumed 0.3 V). A generator 21 nm above did not lower the threshold on the grid tested.
- Voltage gate (VCMA): nothing nucleates at 0 K. At 300 K in the stable 2.0 nm geometry, +100% anisotropy erased the skyrmion in 2 of 3 seeds (0 of 3 in the controls); lowering it by up to 95% wrote nothing. This needs about 16 V at a VCMA coefficient of 100 fJ/(V m).
- In the gate runs the sensor region overlapped the gate region, so the read-out was the annulus between 15 and 20 nm radius (reference levels 0.60 to 0.72 with a skyrmion, 0.97 without).

## In the interactive simulation

*Construction → Sensor type* switches between the field sensor (default), the skyrmion film at 0 K (measured transfer curve, DMI 3.5) and the skyrmion film at 300 K (same curve scaled to the contrast of the stable 2 nm disk, plus Gaussian noise of 0.06 in normalised units, from the 300 K runs).

Quick check in the network model (best configuration of the paper, one seed, not tuned for the skyrmion sensor): with the 0 K skyrmion curve the memory capacity stays similar (MC 8.7 to 9.0) but the 5-step Mackey–Glass error rises from 0.10 to 0.28–0.33 and three-bit parity falls to chance; with the 300 K sensor MC drops to 2 to 3. Associative recall of 24 patterns stayed at 1.00. The skyrmion response is weak at the few-millitesla fields inside a column, so the network would need its own tuning for this sensor.
