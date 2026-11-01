// Replay an eval_app.py run (runs/eval/result.json) through the app's Identity logic with different
// thresholds, offline, in seconds. Picks CFG from data instead of guesses.
//   node train/replay_id.mjs [runs/eval/result.json]
// Metrics: fish created after clip A / after A+B (fewer = fewer duplicates; the pond has a fixed number),
// named share of live fish, identity switches per track, share of clip-B tracks re-using a clip-A fish.
import { readFileSync } from 'node:fs';
import { Identity } from '../app/reid.js';

const res = JSON.parse(readFileSync(process.argv[2] || new URL('runs/eval/result.json', import.meta.url), 'utf8'));
const clips = Object.keys(res);
const H = 960;

function run(cfg) {
  const id = new Identity(cfg);
  const out = { fish: [], named: [], switches: [], reuse: null, dup: [] };
  for (const clip of clips) {
    const d = res[clip], tracks = new Map(), emb = new Map();
    for (const x of d.embs) emb.set(`${x.track}:${x.f}`, x);
    const fishBefore = new Set(id.fish.keys());
    let named = 0, live = 0, dup = 0, withId = 0;
    d.frames.forEach((fr, i) => {
      const lv = [];
      for (const r of fr.tracks) {
        let t = tracks.get(r.id);
        if (!t) { t = { id: r.id, hits: 0, hist: null, bornAt: fr.t }; tracks.set(r.id, t); }
        t.hits++; t.seenAt = fr.t; // track age in video seconds (works for strided runs too)
        const e = emb.get(`${r.id}:${i + 1}`);
        t.obs = { geo: { cy: (r.box[1] + r.box[3]) / 2, len: r.len || 0, straight: 1, edge: false, theta: 0 },
          ...(e ? { emb: Float32Array.from(e.e), hist: e.hist, q: e.q } : {}) };
        lv.push(t);
      }
      id.step(lv, H);
      for (const t of lv) { t.ids ||= []; if (t.fishId && t.ids.at(-1) !== t.fishId) t.ids.push(t.fishId); }
      named += lv.filter(t => id.named(t)).length; live += lv.length;
      // two fish on screen at once with the same identity = distinct fish merged (under-splitting)
      const ids = lv.filter(t => t.fishId).map(t => id.resolve(t.fishId)); withId += ids.length; dup += ids.length - new Set(ids).size;
    });
    id.consolidate();
    out.fish.push(id.fish.size);
    out.named.push(live ? named / live : 0); out.dup.push(withId ? dup / withId : 0);
    const long = [...tracks.values()].filter(t => t.hits >= 30);
    out.switches.push(long.length ? long.reduce((a, t) => a + Math.max(0, (t.ids?.length || 0) - 1), 0) / long.length : 0);
    if (fishBefore.size) {
      const sure = [...tracks.values()].filter(t => t.sure);
      out.reuse = sure.length ? sure.filter(t => fishBefore.has(id.resolve(t.fishId))).length / sure.length : 0;
    }
  }
  return out;
}

const rows = [];
const grid = process.env.GRID ? JSON.parse(process.env.GRID) : { newSim: [0.3, 0.4, 0.45, 0.5, 0.55, 0.6], matchSim: [0.6, 0.65, 0.7, 0.75], showQ: [0.06, 0.1] };
for (const newSim of grid.newSim)
  for (const matchSim of grid.matchSim)
    for (const showQ of grid.showQ) {
      if (newSim >= matchSim) continue;
      const r = run({ newSim, matchSim, showQ, mergeSim: Math.max(matchSim, 0.6) });
      rows.push({ newSim, matchSim, showQ, fishA: r.fish[0], fishAB: r.fish.at(-1), namedA: +r.named[0].toFixed(2), namedB: +(r.named[1] ?? 0).toFixed(2), sw: +r.switches[0].toFixed(2), dupA: +r.dup[0].toFixed(2), dupB: +(r.dup[1] ?? 0).toFixed(2), reuseB: r.reuse == null ? null : +r.reuse.toFixed(2) });
    }
rows.sort((a, b) => (a.dupA + a.dupB) - (b.dupA + b.dupB) || a.fishAB - b.fishAB);
console.table(rows.slice(0, 48));
