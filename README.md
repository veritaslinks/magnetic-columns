# Magnetic Columns

Interactive simulation, micromagnetic scripts and data for the preprint

**Stacked Hexagonal Magnetic Columns with Field-Coupled Generator–Sensor Loops: A Simulation and Micromagnetic Study of In-Memory Reservoir Computing and Associative Memory**
Andrew Pomazkov, CEO, VeritasLinks, Inc.
Preprint: the arXiv link will be added here after posting.

The architecture is a hexagonal lattice of vertical columns. Perpendicular nanomagnet *generators* and magnetic field *sensors* sit on the column walls and in a central pillar. Each generator is driven by a weighted combination of sensors that read the superposed stray field of nearby generators, so the field superposition performs the weighted summation of a recurrent network while the magnetization holds its state: memory and computation are the same physical state. Channels at the junctions of three cells carry simple relay logic between cells and act as heat sinks.

The work grew out of developing a neural model for assessing companies for the VeritasLinks platform: the goal was a network that keeps and remembers its own state between observations instead of recomputing every assessment from scratch. Followed down to the hardware level, that requirement leads to this substrate. The repository does not contain any company data or company assessments.

## The simulation runs on micromagnetic physics, not on a toy field

The field that every sensor reads in the simulation is not the point-dipole approximation. It is interpolated from stray-field maps computed in [MuMax3](https://mumax.github.io/) for perpendicular CoFeB/Co-like generators 21, 30 and 39 nm wide and 3 nm thick, and the simulation works in physical units: 1 model unit = 150 nm, fields in millitesla. The maps are embedded in `index.html` and provided separately in `data/field_tables.json`.

How the MuMax3 data were validated:

- A direct MuMax3 simulation of one variant-C cell over three layers (30 generators, each flipped in turn) showed that every generator stays saturated under its neighbours' field (`data/direct_30_generator_run/`).
- Rebuilding that cell from single-generator maps reproduced the direct result with a median error of 3.1%, 90% of pairs within 7.7%, correlation 0.99996. This validates the superposition used for larger networks.
- A sweep of 144 configurations (up to 19 cells and 10 layers, three generator sizes, three layer spacings, two gaps) is in `data/mumax_sweep_144_configurations.csv`.

The real near field is about half of the point-dipole value (25 mT instead of 46 mT at 21 nm above a 30 nm magnet), which is why the calibration matters.

What is **not** taken from MuMax3, so you know where the physics ends and the model begins: generator switching dynamics, the heat model and the corner-channel logic are phenomenological (Section 4.1 of the manuscript), and the lateral field components (used only by the optional multi-axis sensors and the base-architecture memory) follow the dipole law scaled to the same units. The toggle **Physics from MuMax3** in *Construction* switches between the calibrated field and the point dipole so you can compare them directly.

## Run it

**Online:** https://veritaslinks.github.io/magnetic-columns/ (no installation needed).

**Locally:** open `index.html` in a recent desktop browser (Chrome, Edge, Firefox, Safari). The page loads three.js from a CDN, so it needs an internet connection the first time.

**Best configuration from the paper:** click **Import settings**, choose `settings/best-settings.json` (or paste its content) and press **Apply**. This loads architecture C, 64 long-range links per generator, a 75 nm gap, the MuMax3 field of a 30 nm generator and a 2 mT sensor scale, with separate tuned dynamics for each mode.

## Interface

**Top bar.** Three modes: *Free dynamics*, *Reservoir computing*, *Associative memory*. Impulse buttons, *Pause*, *Export settings* and *Import settings* (JSON).

**3D view.** Drag to rotate, wheel or pinch to zoom. Clicking an element either *inspects* it or fires a *flash* at that point (choose in *Display*). Inspect draws amber lines from everything that influences the element right now and purple lines to everything that listens to it, and shows its state, temperature and strongest inputs.

**Every control explains itself.** Moving any slider or pressing any button shows a short note at the top of the 3D view and highlights the part of the network it affects (generators, sensors, corner channels, pillars, long-range links, input region, readout generators).

**Panels.**

- *Mode panel* (top of the sidebar): task, training or memory controls and live metrics for the current mode.
- *Construction*: number of cells (1, 7, 19), layers, layer spacing, gap, pillars and pillar radius, generator strength and orientation, architecture (base, A, B, C), MuMax3 physics, generator size, sensor saturation scale, sensor layer range, frequency bands, long-range links, multi-axis sensors.
- *Dynamics inside cells*: loop gain, noise, loop delay, weakening from heat, adaptation, generator heating, thermal fluctuations.
- *Links through the corner channels*: on/off, link strength, logic sensitivity, delay, what the logic passes, link type.
- *Cooling*: cooling at corners, pillars and edge, gap conductance, thermal throttle.
- *Signal and flashes*: flash strength, radius, height and duration, continuous input.
- *Display*: generator colouring (activity, temperature, readout weight, where learning happens, memory agreement, memory writing), heat cloud, top/side maps, field arrows, what clicking does.

## Experiments to try

Each recipe starts from a fresh page or from `settings/best-settings.json`.

**1. Watch the field do the summation.** In *Free dynamics*, set *Display → How to show the field* to the cell field, press *Impulse to the centre* and watch the arrows and the activity trail. Switch *Physics from MuMax3* off and on: the dipole field overweights the nearest neighbours.

**2. Reservoir computing.** Switch to *Reservoir computing*, choose *Predict a chaotic signal* and press *Train readout*. During training the colouring shows *where learning happens*; afterwards it shows which generators the answer relies on. Compare the network error with the dashed *error without the network*. Then change the layer spacing (0.2, 0.3, 0.4 = 30, 45, 60 nm): the MuMax3 sweep predicts a trade-off between mixing and signal strength, and you can see it in the error.

**3. Associative memory.** Switch to *Associative memory*, tick several random patterns and press *Store patterns*. Writing sweeps through the network: red generators are still being rewritten, green ones are fixed, and every weight change releases heat. Then press *Damage and recall* and watch the agreement spread. Repeat with *Long-range links per generator* at 0, 16 and 64, and with *Frequency bands* at 3, to see where the capacity comes from.

**4. Compare architectures.** In *Construction*, switch between base, A, B and C and store the same patterns. Variant C (generators alternating between walls and pillar from layer to layer) degrades most gracefully.

**5. No middle ground between silence and a storm.** In *Free dynamics*, raise *Logic sensitivity* in small steps from 0.6 to 0.8 and fire impulses. Below about 0.7 the signal stays in its cell; above about 0.75 activity becomes self-sustained and the heat cloud lights up. This is one of the negative results of the paper.

**6. Heat.** Raise *Generator heating*, turn *Thermal throttle* on and off, change cooling at corners and pillars, and watch the temperature colouring and the maps.

## Reproducing the paper numbers

The benchmarks run the same simulation code headlessly (Node.js 18 or newer):

```
cd benchmarks
npm install
node table3.js --quick     # seeds 3-5, a few minutes
node table3.js             # seeds 3-12, as in the paper
```

`table3.js` prints mean ± 95% confidence interval for every row of Table 3: memory capacity, Mackey–Glass prediction error, three-bit parity, temporal XOR and recall of 24 random patterns, plus the tuned echo state network baselines. Single runs: `node opt.js '<config json>' both <seed>` and `node esn2.js '<config json>' <seed>`.

## Micromagnetics (MuMax3)

Scripts are in `mumax/`. They were run with MuMax3 3.12 on Windows with an NVIDIA RTX 4070. Put them next to `mumax3.exe` and `mumax3-convert.exe`.

- `test_mumax.mx3`: quick check that MuMax3 works.
- `hexcell_C_coupling.mx3` + `analyze_coupling.py`: direct simulation of one variant-C cell over three layers and its analysis (coupling in mT, effective number of sources, fall-off exponent, sensor contrast).
- `run_mumax_study.py`: one command that runs MuMax3 for the three generator sizes, validates superposition against the direct run, evaluates the 144-configuration sweep and writes `study_results/`.
- `export_field_table.py`: exports the axially symmetric field tables used by the simulation.

Windows PowerShell notes: run executables as `.\mumax3.exe`, pass flags as `'-http=' '-cache=<folder>'` (PowerShell drops empty quotes otherwise), and convert files with
`Get-ChildItem <dir>\*.ovf | ForEach-Object { .\mumax3-convert.exe -numpy $_.FullName }`.

## Data

- `data/field_tables.json`: B_z(ρ, z) in mT around one generator for 21, 30 and 39 nm diameters, 3 nm grid, ρ and z up to 201 nm.
- `data/mumax_sweep_144_configurations.csv`: the sweep, one row per configuration.
- `data/direct_30_generator_run/`: MuMax3 log of the direct run and its analysis.
- `settings/best-settings.json`: best configuration from the paper, loadable with *Import settings*.

## Limitations

All results are simulations. The dynamical model is phenomenological with parameters found by search. The best associative capacity relies mainly on 64 addressed links per generator rather than on the field, field coupling alone does not carry signals between cells, and no claims about energy or speed relative to CMOS are made. Details are in Section 6 of the manuscript.

## Citation

The citation entry will be added after the preprint is posted on arXiv.

## License

MIT, see `LICENSE`.
