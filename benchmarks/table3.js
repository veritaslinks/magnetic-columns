// Reproduces Table 3 of the manuscript: runs each system on held-out seeds and prints mean ± 95% CI.
// Usage: node table3.js            (all rows, seeds 3-12; takes a while)
//        node table3.js --quick    (seeds 3-5 only)
const { execFileSync } = require('child_process');
const path = require('path');
const quick = process.argv.includes('--quick');
const seeds = quick ? [3, 4, 5] : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const T = { 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262 };
const rows = [
  ['C, as designed (dipole)', 'opt', { con: { arch: 'C', calib: false }, res: { gain: 0.3 }, memP: 24 }],
  ['C + 3 frequency bands (dipole)', 'opt', { con: { arch: 'C', bands: 3, gap: 0.5, calib: false }, res: { gain: 0.3 }, memP: 24 }],
  ['C + 64 links (dipole)', 'opt', { con: { arch: 'C', longR: 64, gap: 0.5, calib: false }, res: { gain: 0.6 }, memP: 24 }],
  ['C + 64 links (MuMax3-calibrated)', 'opt', { con: { arch: 'C', longR: 64, gap: 0.5, calib: true, dotType: 'd30', Bsens: 2 }, res: { gain: 0.7 }, memP: 24 }],
  ['ESN, input to 24 nodes', 'esn', { rho: 0.95, leak: 1.0, ins: 1.5, nin: 24 }],
  ['ESN, input to all nodes', 'esn', { rho: 0.95, leak: 1.0, ins: 1.5 }],
];
const ci = (v) => { const n = v.length, m = v.reduce((a, b) => a + b, 0) / n; if (n < 2) return m.toFixed(3); const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1)); return m.toFixed(3) + ' ± ' + (T[n] * sd / Math.sqrt(n)).toFixed(3); };
for (const [name, kind, cfg] of rows) {
  const out = [];
  for (const s of seeds) {
    const args = kind === 'opt' ? [path.join(__dirname, 'opt.js'), JSON.stringify(cfg), 'both', String(s)] : [path.join(__dirname, 'esn2.js'), JSON.stringify(cfg), String(s)];
    const line = execFileSync(process.execPath, args, { encoding: 'utf8', maxBuffer: 1 << 24 }).trim().split('\n').pop();
    out.push(JSON.parse(line));
    process.stderr.write(`${name}: seed ${s} done\n`);
  }
  const f = (k) => out[0][k] === undefined ? '–' : ci(out.map((r) => r[k]));
  console.log(`${name.padEnd(34)} MC ${f('MC')}  MG5 ${f('MG5')}  parity3 ${f('par3')}  tXOR ${f('tXOR')}  memory24 ${f('mem20')} / ${f('mem30')}`);
}
