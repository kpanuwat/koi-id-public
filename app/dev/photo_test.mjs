// photo mode (v63.13): still geometry, matchStill (read only, unique assignment) and the session tally
import assert from 'node:assert/strict';
import { Identity, StillSession, stillGeo, LEARN } from '../reid.js';

const unit = (...v) => { const a = Float32Array.from(v), n = Math.hypot(...a); return a.map(x => x / n); };
const H1 = [1, 0, 0, 0, 0, 0, 0], H2 = [0, 1, 0, 0, 0, 0, 0];

// gallery: fish 1 along x, fish 2 along y
const id = new Identity();
const f1 = id.newFish({ embs: [unit(1, 0, 0)], hist: H1, sizes: [] }, 1);
const f2 = id.newFish({ embs: [unit(0, 1, 0)], hist: H2, sizes: [] }, 1);
const snapshot = JSON.stringify(id.toJSON());

// clear matches
let r = id.matchStill([{ emb: unit(0.05, 1, 0), hist: H2 }, { emb: unit(1, 0.05, 0), hist: H1 }]);
assert.deepEqual(r.map(x => x.fish), [f2.id, f1.id]);
assert.ok(r[0].score >= id.cfg.matchSim && r[0].score - r[0].second >= id.cfg.showMargin);

// two dets that both look like fish 1: never the same fish twice
r = id.matchStill([{ emb: unit(1, 0, 0), hist: H1 }, { emb: unit(1, 0.02, 0), hist: H1 }]);
assert.equal(r.filter(x => x.fish === f1.id).length, 1);
assert.notEqual(r[0].best, r[1].best);

// halfway between the two fish: no clear lead -> no match
r = id.matchStill([{ emb: unit(1, 1, 0), hist: [0.5, 0.5, 0, 0, 0, 0, 0] }]);
assert.equal(r[0].fish, null);

// empty input / empty gallery
assert.deepEqual(id.matchStill([]), []);
assert.equal(new Identity().matchStill([{ emb: unit(1, 0, 0), hist: H1 }])[0].fish, null);

// read only: the gallery is unchanged
assert.equal(JSON.stringify(id.toJSON()), snapshot);

// session: unknown fish get a hex ID that holds across photos; gallery fish count once
const s = new StillSession();
let out = s.label(id, [{ emb: unit(1, 0, 0), hist: H1 }, { emb: unit(0, 0, 1), hist: [0, 0, 1, 0, 0, 0, 0] }]);
assert.equal(out[0].fish, f1.id);
assert.match(out[1].hex, /^[0-9A-F]{2}$/);
const hx = out[1].hex;
assert.equal(s.tally, 2);
out = s.label(id, [{ emb: unit(0, 0.02, 1), hist: [0, 0, 1, 0, 0, 0, 0] }, { emb: unit(1, 0.01, 0), hist: H1 }, { emb: unit(0, 1, 0), hist: H2 }]);
assert.equal(out[0].hex, hx);
assert.equal(out[1].fish, f1.id);
assert.equal(out[2].fish, f2.id);
assert.equal(s.tally, 3);
assert.equal(s.shots, 2);
assert.equal(JSON.stringify(id.toJSON()), snapshot);

// between two fish: an ID for this photo only (never '?'), not kept as an unknown, not counted; every hex is unique
out = s.label(id, [{ emb: unit(1, 1, 0), hist: [0.5, 0.5, 0, 0, 0, 0, 0] }]);
assert.equal(out[0].fish, null);
assert.match(out[0].hex, /^[0-9A-F]{2}$/);
assert.equal(out[0].temp, true);
assert.notEqual(out[0].hex, hx);
assert.equal(s.unknown.length, 1);
assert.equal(s.tally, 3);
const ids = new StillSession(), seen = new Set();
for (let i = 0; i < 256; i++) seen.add(ids.hex());
assert.equal(seen.size, 256);
assert.equal(ids.hex(), null);

// still geometry: a horizontal bar, wide on the right; mask px scaled back by msc. Heading follows app.js process()
// exactly (headSign = widthSign, here -1 -> theta + PI), so photo crops are oriented like the gallery's video crops
const mw = 40, mh = 10, mask = new Uint8Array(mw * mh);
for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) if (Math.abs(y - 4.5) <= 1 + 4 * x / mw) mask[y * mw + x] = 1;
const g = stillGeo({ box: [100, 50, 180, 70], mask, mw, mh, msc: 0.5 }, 1920, 1440);
assert.ok(Math.abs(g.cx - (100 + 2 * 22)) < 6 && Math.abs(g.cy - 59) < 3, `centre ${g.cx},${g.cy}`);
assert.ok(g.len > 60 && g.len < 90, `len ${g.len}`);
assert.ok(Math.abs(Math.cos(g.heading) + 1) < 0.05, `heading ${g.heading}`);
assert.equal(g.edge, false);
assert.equal(stillGeo({ box: [0, 50, 80, 70], mask, mw, mh, msc: 0.5 }, 1920, 1440).edge, true);
assert.equal(stillGeo({ box: [0, 0, 4, 4], mask: new Uint8Array(4), mw: 2, mh: 2, msc: 1 }, 100, 100), null);

// ---- learning (v63.17): photos teach the gallery; LEARN gates; unknowns seen in 2 photos become fish ----
assert.deepEqual(LEARN, { sim: 0.75, lead: 0.10, q: 0.12, shots: 2 });
const item = (emb, q = 0.6, hist = H1, thumb = null) => ({ emb, hist, q, thumb });
const gal = () => { // fish A along x (H1), fish B along y (H2)
  const g = new Identity(), A = g.newFish({ embs: [unit(1, 0, 0)], hist: H1, sizes: [] }, 1), B = g.newFish({ embs: [unit(0, 1, 0)], hist: H2, sizes: [] }, 1);
  return { g, A, B };
};

// (a) clear match + q ok -> exemplar added and last seen updated; each gate on its own -> not
{
  const { g, A } = gal(), s = new StillSession();
  const it = item(unit(0.8, 0.6, 0));
  const out = s.label(g, [it]);
  assert.equal(out[0].fish, A.id);
  assert.ok(out[0].second >= 0 && out[0].second < 0.6, `second ${out[0].second}`);
  assert.deepEqual(s.learn(g, [it], out, 5), { added: 1, promoted: [] });
  assert.equal(A.ex.length, 2);
  assert.equal(A.last, 5);
  // the same view again: novelty rule refuses it (no duplicate exemplar), but the fish was seen
  assert.equal(s.learn(g, [it], s.label(g, [it]), 6).added, 0);
  assert.equal(A.ex.length, 2);
  assert.equal(A.last, 6);
}
{ // score below LEARN.sim (0.72 < 0.75) but a display match: shown, not learned
  const { g, A } = gal(), s = new StillSession();
  const it = item(unit(0.65, Math.sqrt(1 - 0.4225), 0));
  const out = s.label(g, [it]);
  assert.equal(out[0].fish, A.id);
  assert.ok(out[0].score < LEARN.sim && out[0].score >= g.cfg.matchSim, `score ${out[0].score}`);
  assert.equal(s.learn(g, [it], out).added, 0);
  assert.equal(A.ex.length, 1);
}
{ // lead over the runner-up 0.064 < LEARN.lead (0.10): shown (showMargin 0.05), not learned
  const g = new Identity(), A = g.newFish({ embs: [unit(1, 0, 0)], hist: H1, sizes: [] }, 1);
  g.newFish({ embs: [unit(0.16, 0.987, 0)], hist: H1, sizes: [] }, 1);
  const s = new StillSession(), it = item(unit(0.8, 0.6, 0));
  const out = s.label(g, [it]);
  assert.equal(out[0].fish, A.id);
  assert.ok(out[0].score - out[0].second < LEARN.lead && out[0].score - out[0].second >= g.cfg.showMargin);
  assert.equal(s.learn(g, [it], out).added, 0);
  assert.equal(A.ex.length, 1);
}
{ // q below LEARN.q (0.11): not learned; exactly 0.12 learned
  const { g, A } = gal(), s = new StillSession();
  const lo = item(unit(0.8, 0.6, 0), 0.11), hi = item(unit(0.8, 0.6, 0), 0.12);
  assert.equal(s.learn(g, [lo], s.label(g, [lo])).added, 0);
  assert.equal(A.ex.length, 1);
  assert.equal(s.learn(g, [hi], s.label(g, [hi])).added, 1);
}
{ // temp (an ID for this photo only) is never learned from, whatever the score says
  const { g, A } = gal(), s = new StillSession();
  const it = item(unit(0.8, 0.6, 0));
  assert.equal(s.learn(g, [it], [{ fish: A.id, hex: null, score: 0.99, second: -1, temp: true }]).added, 0);
  assert.equal(A.ex.length, 1);
}

// (b) an unknown seen in 1 photo -> no fish; in 2 photos -> exactly one new fish: auto name, out of the session's unknowns,
// tally unchanged, its hex remembered, the best thumbnail kept
{
  const g = new Identity(); g.central = true;
  const H3 = [0, 0, 1, 0, 0, 0, 0], s = new StillSession(), U = unit(0, 0, 1);
  const out1 = s.label(g, [item(U, 0.6, H3, 'T1')]);
  assert.match(out1[0].hex, /^[0-9A-F]{2}$/);
  assert.equal(s.tally, 1);
  assert.deepEqual(s.learn(g, [item(U, 0.6, H3, 'T1')], out1), { added: 0, promoted: [] });
  assert.equal(g.fish.size, 0);
  assert.equal(s.unknown.length, 1);
  // second photo: the same fish, clear enough to match its unknown
  const it2 = item(unit(0.02, 0.01, 1), 0.9, H3, 'T2');
  const out2 = s.label(g, [it2]);
  assert.equal(out2[0].hex, out1[0].hex);
  assert.equal(s.tally, 1);
  const r2 = s.learn(g, [it2], out2);
  assert.equal(r2.promoted.length, 1);
  const { fish, hex } = r2.promoted[0];
  assert.equal(hex, out1[0].hex);
  assert.equal(g.fish.size, 1);
  const f = g.fish.get(fish);
  assert.ok(f && f.autoName === true && f.name.length > 0 && f.c === false, JSON.stringify(f));
  assert.equal(f.thumb, 'T2'); // q 0.9 beats the first photo's 0.6
  assert.equal(s.unknown.length, 0);
  assert.equal(s.tally, 1, 'no double count');
  assert.ok(s.seen.has(`f:${fish}`) && !s.seen.has(`u:${hex}`));
  assert.equal(s.fishHex(fish), hex);
  // third photo: the new fish comes back as a gallery match, not a new unknown; tally still 1, nothing promoted
  const out3 = s.label(g, [item(U, 0.8, H3)]);
  assert.equal(out3[0].fish, fish);
  assert.equal(out3[0].hex, null);
  assert.equal(s.tally, 1);
  assert.deepEqual(s.learn(g, [item(U, 0.8, H3)], out3).promoted, []);
  assert.equal(s.unknown.length, 0);
  assert.equal(s.fishHex(fish), hex);
}

// (c) an unknown close to a gallery fish not seen with it (>= matchSim) is never promoted, however often it is seen
{
  const g = new Identity(), s = new StillSession(), H3 = [0, 0, 1, 0, 0, 0, 0], U = unit(0.8, 0, 0.6);
  s.label(g, [item(U, 0.6, H3)]);
  g.newFish({ embs: [unit(0.8, 0.6, 0)], hist: H3, sizes: [] }, 1); // a fish that grows near the unknown after the first photo
  assert.ok(g.score({ embs: [U], hist: H3, sizes: [] }, [...g.fish.values()][0]) >= g.cfg.matchSim);
  for (let k = 0; k < 3; k++) {
    const out = s.label(g, [item(U, 0.6, H3)]);
    assert.equal(s.learn(g, [item(U, 0.6, H3)], out).promoted.length, 0);
  }
  assert.equal(g.fish.size, 1);
  assert.equal(s.unknown.length, 1);
}

// (e) look-alike koi seen together in two photos all become fish: covisible fish do not block each other's promotion,
// and a view between newSim and matchSim of an earlier fish does not block either (field: only 2 of 15 got in with newSim)
{
  const g = new Identity(), s = new StillSession(), H3 = [0, 0, 1, 0, 0, 0, 0];
  const P = unit(1, 0, 0), Q = unit(0.6, 0.8, 0), R = unit(0, 0.6, 0.8); // P.Q 0.6, Q.R 0.48: look-alikes, all < matchSim
  const shot = () => [item(P, 0.2, H3), item(Q, 0.2, H3), item(R, 0.2, H3)];
  let its = shot(); s.learn(g, its, s.label(g, its));
  assert.equal(s.unknown.length, 3);
  its = shot(); const r = s.learn(g, its, s.label(g, its));
  assert.equal(r.promoted.length, 3, JSON.stringify(r));
  assert.equal(g.fish.size, 3);
  assert.equal(s.tally, 3);
  assert.equal(new Set([...g.fish.values()].map(f => f.name)).size, 3);
}

// (d) renumbering (adopt after a central sync): fish keys, hex memory and the tally follow the map; a deleted fish keeps its slot
{
  const { g, A, B } = gal(), s = new StillSession();
  const out = s.label(g, [item(unit(1, 0, 0)), item(unit(0, 1, 0), 0.6, H2), item(unit(0, 0, 1), 0.6, [0, 0, 1, 0, 0, 0, 0])]);
  assert.equal(out[0].fish, A.id);
  assert.equal(out[1].fish, B.id);
  const hu = out[2].hex;
  assert.equal(s.tally, 3);
  const hA = s.fishHex(A.id);
  s.remap(new Map([[A.id, 7], [B.id, null]]));
  assert.equal(s.tally, 3, 'tally kept through a renumber and a deletion');
  assert.ok(s.seen.has('f:7') && s.seen.has(`x:${B.id}`) && s.seen.has(`u:${hu}`), [...s.seen].join(','));
  assert.equal(s.fishHex(7), hA);
  assert.equal(s.hexOf.has(A.id), false);
  assert.equal(s.hexOf.has(B.id), false);
  assert.equal(s.unknown.length, 1);
  // an empty map changes nothing
  const before = [...s.seen].sort().join(',');
  s.remap(new Map());
  assert.equal([...s.seen].sort().join(','), before);
}
console.log('photo_test ok');
