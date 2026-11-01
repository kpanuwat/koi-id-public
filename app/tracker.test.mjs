// node app/tracker.test.mjs  — assert-based checks for tracker + identity logic
import assert from 'node:assert/strict';
import { hungarian, Tracker, COUNT, CAND } from './tracker.js';
import { Identity, SizeModel, maskGeometry, autoName, variety, mergeGallery, sharedFish } from './reid.js';
import { isNovel, evictIndex } from './samples.js';

let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

// hungarian == brute force on small random matrices (incl. rectangular)
function brute(c) {
  const n = c.length, m = c[0].length, k = Math.min(n, m); let best = Infinity;
  const go = (i, used, s) => { if (i === n) { if (used.size === k) best = Math.min(best, s); return; }
    for (let j = 0; j < m; j++) if (!used.has(j)) { used.add(j); go(i + 1, used, s + c[i][j]); used.delete(j); }
    go(i + 1, used, s); };
  go(0, new Set(), 0); return best;
}
for (let k = 0; k < 200; k++) {
  const n = 1 + Math.floor(rnd() * 4), m = 1 + Math.floor(rnd() * 4);
  const c = Array.from({ length: n }, () => Array.from({ length: m }, () => Math.round(rnd() * 10)));
  const a = hungarian(c), cost = a.reduce((s, j, i) => s + (j >= 0 ? c[i][j] : 0), 0);
  assert.equal(a.filter(j => j >= 0).length, Math.min(n, m));
  assert.equal(cost, brute(c), JSON.stringify(c));
}

// tracker keeps ids for two fish swimming past each other with a missed frame
{
  const tr = new Tracker(); let ids;
  for (let f = 0; f < 30; f++) {
    const d = [{ box: [10 + 3 * f, 50, 40 + 3 * f, 62], score: 0.9 }, { box: [200 - 3 * f, 100, 230 - 3 * f, 112], score: 0.8 }];
    const out = tr.update(f === 12 ? [d[0]] : d);
    if (f === 5) ids = out.map(t => t.id).sort();
  }
  assert.deepEqual(tr.tracks.map(t => t.id).sort(), ids);
}

// counting evidence under a panning camera: rock (static, middling score) drops out, resting koi
// (static, high score) and swimming koi stay; a flickering glint (area jumps) never counts
{
  const tr = new Tracker(); let out = [];
  for (let f = 0; f < 240; f++) { // 8 s at 30 fps, camera pans 1.5 px/frame
    const p = 1.5 * f, B = (x, y, w = 30, h = 12) => [x + p, y, x + p + w, y + h];
    out = tr.update([
      { box: B(20, 40), score: 0.65 },                       // rock (above spawn 0.6, below staticScore 0.7)
      { box: B(100, 40), score: 0.92 },                      // resting koi
      { box: B(20 + 2 * f, 120), score: 0.85 },              // swimming koi
      { box: B(200, 200), score: 0.62, area: f % 2 ? 40 : 360 }, // glint: area x9 every other frame
    ], f / 30);
  }
  const rock = out.find(t => t.box[1] === 40 && t.det.score < 0.7), rest = out.find(t => t.box[1] === 40 && t.det.score > 0.9);
  const swim = out.find(t => t.box[1] === 120), glint = out.find(t => t.box[1] === 200);
  assert.ok(rock.static && !rock.counted, 'rock not counted');
  assert.ok(!rest.static && rest.counted, 'resting koi counted');
  assert.ok(swim.counted, 'swimming koi counted');
  assert.ok(glint.jitter > COUNT.JITTER && !glint.counted, 'glint not counted');
}

// slow device (3 passes/s): a turning koi whose area varies moderately still counts; at 30 passes/s the same variation does not
{
  const run = dt => { const tr = new Tracker(); let o = [];
    for (let f = 0; f < 30; f++) o = tr.update([{ box: [100, 100, 160, 124], score: 0.8, area: f % 2 ? 600 : 1650 }], f * dt);
    return o[0]; };
  const slow = run(1 / 3), fast = run(1 / 30);
  assert.ok(slow.jitter > COUNT.JITTER && slow.jitter < COUNT.JITTER_SLOW, `jitter ${slow.jitter}`);
  assert.ok(slow.counted && !fast.counted, 'jitter threshold follows the pass rate');
}

// weak detections: a faint koi that keeps reappearing and swims becomes a counted track; a faint still blob does not
{
  const tr = new Tracker();
  for (let f = 0; f < 40; f++) tr.update([
    { box: [10 + 3 * f, 300, 40 + 3 * f, 312], score: 0.4 },   // faint swimming koi
    { box: [300, 50, 330, 62], score: 0.4 },                   // faint still blob
    { box: [100, 500, 130, 512], score: 0.9 }, { box: [200, 600, 230, 612], score: 0.9 }, { box: [250, 700, 280, 712], score: 0.9 }, // anchors for camera estimate
  ], f / 30);
  const swim = tr.tracks.find(t => t.box[1] === 300), blob = tr.tracks.find(t => t.box[1] === 50);
  assert.ok(swim && swim.promoted && swim.counted, 'faint swimming koi promoted and counted');
  assert.ok(blob && blob.promoted && !blob.counted, 'faint still blob promoted but not counted');
}

// stitching: a named fish dives and resurfaces a little further on => its own track continues (name kept)
{
  const tr = new Tracker(), anchors = [[100, 500], [200, 600], [250, 700]].map(([x, y]) => ({ box: [x, y, x + 30, y + 12], score: 0.9 }));
  let fishTrack;
  for (let f = 0; f < 10; f++) { const out = tr.update([{ box: [10 + 3 * f, 300, 40 + 3 * f, 312], score: 0.9 }, ...anchors], f / 30); fishTrack = out.find(t => t.box[1] === 300); }
  fishTrack.fishId = 7;
  for (let f = 10; f < 25; f++) tr.update(anchors, f / 30);                   // under water for 0.5 s
  const out = tr.update([{ box: [10 + 3 * 25 + 40, 300, 40 + 3 * 25 + 40, 312], score: 0.9 }, ...anchors], 25 / 30);
  assert.ok(fishTrack.lost === 0 && fishTrack.fishId === 7 && fishTrack.box[0] === 125, 'resurfaced fish continues its own track');
  assert.equal(tr.tracks.filter(t => t.box[1] === 300).length, 1, 'no duplicate track');
  // long dive: the track was dropped, only the name survives => new track gets fish 7 as a prior
  const tr2 = new Tracker({ maxLostS: 5 / 30 });
  let ft;
  for (let f = 0; f < 10; f++) { const o = tr2.update([{ box: [10 + 3 * f, 300, 40 + 3 * f, 312], score: 0.9 }, ...anchors], f / 30); ft = o.find(t => t.box[1] === 300) || ft; }
  ft.fishId = 7;
  for (let f = 10; f < 30; f++) tr2.update(anchors, f / 30);              // 0.7 s under water, track removed after 5 frames
  tr2.update([{ box: [120, 300, 150, 312], score: 0.9 }, ...anchors], 30 / 30);
  const re = tr2.tracks.find(t => t.box[1] === 300);
  assert.ok(re && re !== ft && re.prior === 7, 'new track inherits fish 7 as prior');
}

// a weak detection sitting on a tracked fish (its head) is not a second fish
{
  const tr = new Tracker();
  for (let f = 0; f < 20; f++) tr.update([{ box: [100, 100, 160, 124], score: 0.9 }, { box: [140, 104, 158, 120], score: 0.45 }], f / 30);
  assert.equal(tr.tracks.length, 1, 'head box never becomes its own track');
}

// geometry: bent fish (arc) length ~ arc length, not bbox
{
  const w = 120, h = 80, m = new Uint8Array(w * h), R = 60, cx = 60, cy = 75;
  for (let a = -0.6; a <= 0.6; a += 0.002) for (let r = -5; r <= 5; r += 0.5) {
    const x = Math.round(cx + (R + r) * Math.sin(a)), y = Math.round(cy - (R + r) * Math.cos(a)); m[y * w + x] = 1;
  }
  const g = maskGeometry(m, w, h);
  assert.ok(Math.abs(g.len - R * 1.2) / (R * 1.2) < 0.08, `len ${g.len}`);
}

// size model recovers relative sizes under perspective (a + b*y) from noisy obs
{
  const sm = new SizeModel(), truth = Array.from({ length: 10 }, () => 0.3 * gauss());
  for (let fr = 0; fr < 40; fr++) sm.fit(truth.map((s, i) => ({ y: (i + rnd()) / 10, logLen: 3.5 + 1.2 * ((i + 0.5) / 10) + s + 0.05 * gauss(), s: fr > 5 ? s : null })));
  assert.ok(Math.abs(sm.b - 1.2) < 0.15, `b ${sm.b}`);
  const est = truth.map((s, i) => sm.size(3.5 + 1.2 * ((i + 0.5) / 10) + s, (i + 0.5) / 10));
  truth.forEach((s, i) => assert.ok(Math.abs(est[i] - s) < 0.1, `fish ${i}: ${est[i]} vs ${s}`));
}

// identity: 6 fish, noisy multi-view embeddings; sessions re-identify, co-visible fish never merge
{
  const D = 32, unit = v => { const n = Math.hypot(...v); return Float32Array.from(v, x => x / n); };
  const proto = Array.from({ length: 6 }, () => [0, 1].map(() => unit(Array.from({ length: D }, gauss)))); // 2 views each
  const emb = (f, view) => unit(Array.from(proto[f][view], x => x + 0.08 * gauss()));
  const id = new Identity({ matchSim: 0.75, newSim: 0.6, mergeSim: 0.8 });
  let tid = 1;
  const session = (fishIdx, frames = 40) => {
    const tracks = fishIdx.map(f => ({ id: tid++, hits: 0, truth: f }));
    for (let fr = 0; fr < frames; fr++) {
      for (const t of tracks) { t.hits++; t.obs = { geo: { cx: 0, cy: 100, len: 50, straight: 1, edge: false, theta: 0 }, q: 0.8,
        emb: fr % 4 === 0 ? emb(t.truth, fr < frames / 2 ? 0 : 1) : null, hist: [0.2, 0, 0, 0.7, 0, 0, 0.1] }; }
      id.step(tracks, 960);
    }
    id.retire(tracks); return tracks;
  };
  const s1 = session([0, 1, 2, 3, 4, 5]);
  assert.equal(id.fish.size, 6);
  assert.equal(new Set(s1.map(t => t.fishId)).size, 6);
  const map = new Map(s1.map(t => [t.truth, t.fishId]));
  const s2 = session([5, 3, 1]);
  s2.forEach(t => assert.equal(t.fishId, map.get(t.truth), `re-id fish ${t.truth}`));
  assert.equal(id.fish.size, 6);
  assert.equal(id.consolidate(), 0); // all pairs co-visible => nothing merges
  for (const f of id.fish.values()) assert.ok(f.ex.length >= 2, 'learned both views');
  const back = new Identity().load(JSON.parse(JSON.stringify(id.toJSON())));
  assert.equal(back.fish.size, 6);
}

// population cap stops new fish
{
  const id = new Identity({ population: 1 });
  const e = Float32Array.from([1, 0]), e2 = Float32Array.from([0, 1]);
  const mk = (em) => ({ id: Math.random(), hits: 0 });
  const a = mk(), b = mk();
  for (let fr = 0; fr < 20; fr++) { a.hits++; a.obs = { emb: e }; id.step([a], 960); }
  for (let fr = 0; fr < 20; fr++) { b.hits++; b.obs = { emb: e2 }; id.step([b], 960); }
  assert.equal(id.fish.size, 1);
}

// naming
assert.equal(variety([0.25, 0, 0, 0.65, 0.02, 0, 0.08]), 'Kohaku');
assert.equal(variety([0, 0.05, 0.8, 0.1, 0, 0, 0.05]), 'Ogon');
assert.equal(autoName('Ogon', new Set(['ทองหยิบ'])), 'ทองหยอด');
{ // 300 fish of one colour: all names unique, none carries a number
  const taken = new Set(); for (let i = 0; i < 300; i++) taken.add(autoName('Ogon', taken));
  assert.equal(taken.size, 300); assert.ok(![...taken].some(n => /\d/.test(n)), 'no numbered names');
  assert.ok([...taken].every(n => [...n].length <= 10), `names <= 10 chars: ${[...taken].filter(n => [...n].length > 10)}`);
  const old = new Identity().load({ v: 1, nextFish: 3, covis: [], fish: [{ id: 1, name: 'ทองหยอด', variety: 'Ogon', ex: [] }, { id: 2, name: 'ทองหยอด 2', variety: 'Ogon', ex: [] }] });
  assert.ok(!/\d/.test(old.fish.get(2).name) && old.fish.get(2).name !== 'ทองหยอด', 'old numbered name replaced');
}
// partial views (head/tail only) never found a new fish
{
  const id = new Identity(); const t = { id: 1, hits: 0 };
  for (let k = 0; k < 40; k++) { t.hits++; t.obs = { emb: Float32Array.from([1, 0]), q: 0.8, geo: { partial: true, len: 20, cy: 100 } }; id.step([t], 960); }
  assert.equal(id.fish.size, 0);
}

// name gating: a fish with a near-twin in the gallery stays unnamed; a distinct one is named
{
  const D = 16, unit = v => { const n = Math.hypot(...v); return Float32Array.from(v, x => x / n); };
  const base = unit(Array.from({ length: D }, gauss)), twin = unit(Array.from(base, x => x + 0.02 * gauss())), other = unit(Array.from({ length: D }, gauss));
  const id = new Identity({ matchSim: 0.7, newSim: 0.5 });
  const run = (e, frames = 30) => { const t = { id: Math.random(), hits: 0 };
    for (let k = 0; k < frames; k++) { t.hits++; t.obs = { emb: k % 3 === 0 ? e : null, q: 0.8, hist: [0, 0, 1, 0, 0, 0, 0] }; id.step([t], 960); }
    return t; };
  run(base); run(other);                       // two fish founded
  id.fish.set(99, { ...id.fish.get(1), id: 99, name: 'twin', ex: [{ e: twin, view: 0, q: 1 }] }); // look-alike of fish 1
  const a = run(base), b = run(other);
  assert.equal(id.named(a), false, 'ambiguous with twin => no name');
  assert.equal(id.named(b), true, 'distinct fish => named');
}

// auto name follows the colour type once it settles; an owner-typed name stays
{
  const id = new Identity({ minTrackS: 3 / 30 }), e = Float32Array.from([1, 0]), t = { id: 1, hits: 0 };
  const go = (n, hist) => { for (let k = 0; k < n; k++) { t.hits++; t.obs = { emb: e, q: 0.8, hist, geo: { len: 40, cy: 100 } }; id.step([t], 960); } };
  go(5, [0, 0, 0, 0, 0.1, 0.9, 0]);                     // murky start: brown
  const f = id.fish.get(t.fishId);
  assert.equal(f.variety, 'Chagoi');
  go(200, [0.45, 0.05, 0, 0.45, 0, 0.05, 0]);           // clear view: red + white
  assert.equal(f.variety, 'Kohaku');
  assert.ok(['ซากุระ', 'ทับทิม', 'ธงชาติ', 'มะลิแดง', 'สตรอว์เบอร์รี่', 'ลิปสติก', 'แตงโม', 'กุหลาบ'].includes(f.name), `Kohaku name, got ${f.name}`);
  f.name = 'เจ้าของตั้ง'; f.autoName = false;
  go(200, [0, 0, 0, 0, 0.1, 0.9, 0]);
  assert.equal(f.name, 'เจ้าของตั้ง');
}

// merging keeps a name the owner typed, whichever fish survives
{
  const id = new Identity(), e = Float32Array.from([1, 0]);
  const a = id.newFish({ id: 1, embs: [e], sizes: [] }, 0), b = id.newFish({ id: 2, embs: [e], sizes: [] }, 0);
  b.name = 'บ็อบ'; b.autoName = false;
  id.merge(a.id, b.id);
  assert.ok(id.fish.get(a.id).name === 'บ็อบ' && id.fish.get(a.id).autoName === false, 'owner name kept after merge');
}

// red flag: two tracks claiming the same fish at once -> one keeps it, the other is told it is not that fish
{
  const id = new Identity({ minTrackS: 3 / 30 }), e = Float32Array.from([1, 0]);
  const a = { id: 1, hits: 0 }, b = { id: 2, hits: 0 };
  for (let k = 0; k < 6; k++) { a.hits++; a.obs = { emb: e, q: 0.8, geo: { len: 40, cy: 100 } }; id.step([a], 960); }
  const fid = a.fishId;
  b.votes = new Map([[fid, 5]]); b.fishId = fid; b.sure = true; b.embs = [Float32Array.from([0.6, 0.8])]; b.hits = 10; // a weaker look-alike claim
  a.obs = { emb: e, q: 0.8, geo: { len: 40, cy: 100 } }; b.obs = { emb: Float32Array.from([0.6, 0.8]), q: 0.8, geo: { len: 40, cy: 100 } };
  id.step([a, b], 960);
  assert.equal(a.fishId, fid, 'best fit keeps the fish');
  assert.ok(b.fishId !== fid && b.notFish.has(fid) && b.red === 1, 'the other is not that fish');
  assert.equal(id.grade(b), null, 'red-flagged track gives no training data');
  assert.ok(['silver', 'bronze'].includes(id.grade(a)), 'clean confirmed track is graded');
}

// a stored crop is placed on the right fish without a track (auto-labelling of the review queue)
{
  const id = new Identity(), e1 = Float32Array.from([1, 0]), e2 = Float32Array.from([0, 1]);
  id.newFish({ id: 1, embs: [e1], sizes: [] }, 0); id.newFish({ id: 2, embs: [e2], sizes: [] }, 0);
  const [fid, sc, second] = id.matchOne(Float32Array.from([0.98, 0.2]), null);
  assert.ok(fid === 1 && sc > second, 'nearest fish wins');
}

// central gallery = single source of truth: proposals from two devices never share a number, the same fish joins up,
// owner renames, deletions and merges reach every device
{
  const v = (...x) => Float32Array.from(x), E1 = v(1, 0, 0), E2 = v(0, 1, 0), E3 = v(0, 0, 1), E2b = v(0.1, 0.995, 0);
  const device = central => { const d = new Identity().load(JSON.parse(JSON.stringify(central))); d.central = true; return d; };
  const seed = new Identity(); seed.newFish({ id: 1, embs: [E1], sizes: [] }, 0);
  let central = mergeGallery(null, seed).gallery;
  assert.equal(central.fish.length, 1); assert.equal(central.fish[0].c, true);
  const A = device(central), B = device(central);
  const a2 = A.newFish({ id: 1, embs: [E2], sizes: [] }, 0), b2 = B.newFish({ id: 1, embs: [E3], sizes: [] }, 0);
  assert.ok(a2.c === false && a2.id === b2.id, 'both devices propose the same number');
  assert.ok(!A.named({ sure: true, fishId: a2.id, qEMA: 1, founder: true }), 'a proposal shows no name yet');
  const ra = mergeGallery(central, A); central = ra.gallery;
  assert.deepEqual([...ra.added], [2]);
  const rb = mergeGallery(central, B);
  assert.equal(rb.map.get(2), 3, 'the second device gets the next central number');
  assert.equal(rb.gallery.fish.length, 3);
  const names = rb.gallery.fish.map(f => f.name); assert.equal(new Set(names).size, 3, 'names stay unique');
  central = rb.gallery;
  // same fish seen on another device joins the central fish instead of becoming a new one
  const C = device(central); const c2 = C.newFish({ id: 1, embs: [E2b], sizes: [] }, 0);
  const rc = mergeGallery(central, C);
  assert.equal(rc.map.get(c2.id), 2); assert.equal(rc.added.size, 0);
  // ...but never a fish it was seen on screen together with
  const D = device(central); const d9 = D.newFish({ id: 1, embs: [E2b], sizes: [] }, 0); D.covis.add(`2|${d9.id}`);
  assert.notEqual(mergeGallery(central, D).map.get(d9.id), 2);
  // owner rename on one device, delete on another, merge on a third
  const R = device(central); Object.assign(R.fish.get(1), { name: 'โชคดี', autoName: false, namedAt: 5 });
  central = mergeGallery(central, R).gallery;
  assert.equal(central.fish.find(f => f.id === 1).name, 'โชคดี');
  const X = device(central); X.remove(3);
  central = mergeGallery(central, X).gallery;
  const Y = device(central); assert.ok(!Y.fish.has(3), 'deleted on every device');
  const old = device(rb.gallery); assert.equal(mergeGallery(central, old).map.get(3), null, 'a stale device drops it too');
  const M = device(central); M.merge(1, 2); central = mergeGallery(central, M).gallery;
  assert.deepEqual(central.fish.map(f => f.id), [1]); assert.equal(central.fish[0].name, 'โชคดี');
  assert.equal(mergeGallery(central, device(rb.gallery)).map.get(2), 1, 'a device still holding fish 2 follows the merge');
}

// relative sizes heal: a gallery whose shared offset drifted (every fish x0.00) is pinned back to mean x1.00 on load
{
  const g = { v: 1, nextFish: 5, covis: [], size: { a: 9.9, b: 0 },
    fish: [-5.8, -5.5, -6.1, -5.6].map((m, i) => ({ id: i + 1, name: `n${i}`, variety: 'Koi', ex: [], hist: new Array(7).fill(0), size: { m, v: 0.1, n: 50 }, seen: 1 })) };
  const id = new Identity().load(g);
  const ms = [...id.fish.values()].map(f => f.size.m);
  assert.ok(ms.some(m => Math.abs(m) < 1e-9) && ms.every(m => Math.abs(m) < 0.6), 'typical fish back to x1.00');
  assert.ok(Math.abs((ms[2] - ms[1]) - (-6.1 + 5.5)) < 1e-9, 'differences between fish kept');
  assert.ok(Math.abs(id.size.a - (9.9 - 5.6)) < 1e-9, 'image-to-size offset moves with them');
  const g2 = { ...g, fish: [...g.fish, { ...g.fish[0], id: 9, name: 'x', size: { m: -2.0, v: 0.1, n: 50 } }] };
  const id2 = new Identity().load(JSON.parse(JSON.stringify(g2)));
  assert.equal(id2.fish.get(9).size.n, 0, 'a fish drifted far from the pond (x45) relearns its size');
  assert.ok(Math.abs(id2.fish.get(1).size.m) < 0.3, 'the outlier does not drag the others');
}

// two pond records that share fish are one body of water (footbridge); unrelated ponds share none
{
  const v = (...x) => { const n = Math.hypot(...x); return Float32Array.from(x.map(y => y / n)); };
  const pond = es => { const id = new Identity(); id.embedTag = 't'; es.forEach(e => id.newFish({ id: 1, embs: [e], sizes: [] }, 0)); return id; };
  const A = pond([v(1, 0, 0, 0), v(0, 1, 0, 0), v(0, 0, 1, 0), v(0, 0, 0, 1)]);
  const B = pond([v(1, 0.05, 0, 0), v(0, 1, 0.05, 0), v(0.05, 0, 1, 0), v(1, 1, 1, 1)]);
  assert.equal(sharedFish(A, B), 3);
  assert.equal(sharedFish(A, pond([v(1, 1, 1, 1), v(1, -1, 1, -1)])), 0);
}

// fish deleted (or library cleared) while a confirmed track shows it: the track forgets it and may found a new fish
{
  const id = new Identity({ minTrackS: 3 / 30 }), e = Float32Array.from([1, 0]), t = { id: 1, hits: 0 };
  const go = n => { for (let k = 0; k < n; k++) { t.hits++; t.obs = { emb: e, q: 0.8, geo: { len: 40, cy: 100 } }; id.step([t], 960); } };
  go(5);
  assert.ok(t.sure && t.fishId === 1);
  id.fish.delete(1); go(1);
  assert.ok(t.fishId !== 1 && ![...t.votes.keys()].includes(1), 'deleted fish forgotten');
  go(3);
  assert.ok(t.sure && id.fish.has(t.fishId), 'track founds a live fish again');
}

// harvest: keep only new views; evict the most redundant, never an owner label
{
  const a = Float32Array.from([1, 0, 0]), a2 = Float32Array.from([0.99, 0.141, 0]), b = Float32Array.from([0, 1, 0]);
  assert.equal(isNovel(a2, [a]), false);
  assert.equal(isNovel(b, [a]), true);
  const items = [{ emb: a, q: 0.9, at: 1, label: 'owner' }, { emb: a2, q: 0.5, at: 2, label: 'track' }, { emb: b, q: 0.5, at: 3, label: 'track' }];
  assert.equal(evictIndex(items), 1);
}
// glint detector on synthetic canonical crops (224x112 RGBA, background = ImageNet-mean grey)
{
  const { blobShape, isGlint } = await import('./reid.js');
  {
    const img = (fn) => { const p = new Uint8ClampedArray(224 * 112 * 4); for (let y = 0; y < 112; y++) for (let x = 0; x < 224; x++) { const i = 4 * (y * 224 + x), c = fn(x, y) || [124, 116, 104]; p.set([...c, 255], i); } return p; };
    const glint = img((x, y) => ((x - 112) ** 2 / 90 ** 2 + (y - 56) ** 2 / 55 ** 2 < 1 ? (x % 9 < 2 ? [60, 60, 60] : [235, 235, 230]) : null));
    const koi = img((x, y) => ((x - 112) ** 2 / 100 ** 2 + (y - 56) ** 2 / 22 ** 2 < 1 ? [230, 90, 40] : null));
    const whiteKoi = img((x, y) => ((x - 112) ** 2 / 100 ** 2 + (y - 56) ** 2 / 22 ** 2 < 1 ? [235, 235, 225] : null));
    assert.ok(isGlint(blobShape(glint)), 'round white streaky blob = glint');
    assert.ok(!isGlint(blobShape(koi)), 'orange koi kept');
    assert.ok(!isGlint(blobShape(whiteKoi)), 'elongated white koi kept');
  }
}
assert.equal(variety([0.2, 0, 0, 0.3, 0, 0, 0.5]), 'Kohaku', 'variety ignores unreadable pixels');
console.log('ok');
