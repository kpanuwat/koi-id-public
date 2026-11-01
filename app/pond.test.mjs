// node app/pond.test.mjs  — pond chosen by the fish (pondByFish) and photo co-visibility (StillSession.learn -> covis)
import assert from 'node:assert/strict';
import { Identity, StillSession, pondByFish, mergeGallery } from './reid.js';

const D = 16;
const unit = i => { const e = new Float32Array(D); e[i] = 1; return e; };
const hist = [0, 0, 0, 1, 0, 0, 0];
// a pond whose fish k has the one-hot look `looks[k]`
const pondOf = looks => {
  const id = new Identity(); id.embedTag = 't';
  for (const i of looks) id.newFish({ embs: [unit(i)], hist, sizes: [] }, 0);
  return id;
};
const items = looks => looks.map(i => ({ emb: unit(i), hist }));

// pondByFish: another pond wins only with >= 2 named, >= 2 more than here and >= twice as many
{
  const here = pondOf([0, 1]), other = pondOf([2, 3, 4, 5]), third = pondOf([2, 3, 4]);
  assert.equal(pondByFish(items([2, 3, 4]), here, [['o', other]]).key, 'o', '0 here, 3 there: switch');
  assert.deepEqual(pondByFish(items([2, 3, 4]), here, [['o', other]]), { key: 'o', matchedFrom: 0, matchedTo: 3 });
  assert.equal(pondByFish(items([2]), here, [['o', other]]), null, 'one fish is never enough');
  assert.equal(pondByFish(items([0, 2, 3]), here, [['o', other]]), null, '1 here, 2 there: not 2 more');
  assert.equal(pondByFish(items([0, 1, 2, 3, 4]), here, [['o', other]]), null, '2 here, 3 there: not twice');
  assert.equal(pondByFish(items([0, 2, 3, 4, 5]), here, [['o', other]]).matchedTo, 4, '1 here, 4 there: switch');
  assert.equal(pondByFish(items([2, 3, 4, 5]), here, [['3', third], ['o', other]]).key, 'o', 'the pond naming most wins');
  assert.equal(pondByFish(items([2, 3]), here, []), null, 'no other pond');
  assert.equal(pondByFish([], here, [['o', other]]), null);
}

// photo covisibility: gallery fish seen in one photo go into identity.covis, so consolidate() and mergeGallery keep them apart
{
  const id = pondOf([0, 1, 2]);
  const ss = new StillSession(), it = items([0, 1]);
  const lab = ss.label(id, it);
  assert.deepEqual(lab.map(l => l.fish), [1, 2]);
  const r = ss.learn(id, it, lab, 1);
  assert.equal(r.covis, 1);
  assert.ok(id.covis.has('1|2'), 'pair stored as "a|b", smaller first');
  assert.equal(ss.learn(id, it, lab, 1).covis, 0, 'not counted twice');
  // a later photo with all three: the two new pairs
  const it3 = items([2, 1, 0]), lab3 = ss.label(id, it3);
  assert.equal(ss.learn(id, it3, lab3, 2).covis, 2);
  assert.deepEqual([...id.covis].sort(), ['1|2', '1|3', '2|3']);
  // two look-alikes (cos 0.78, above mergeSim) that sat in one photo are never merged; without the photo they would be
  const n = Math.hypot(1, 0.8), b = Float32Array.from(unit(0), (v, i) => (i === 0 ? 1 / n : i === 1 ? 0.8 / n : 0));
  const twins = () => { const t = pondOf([0]); t.newFish({ embs: [b], hist, sizes: [] }, 0); return t; };
  const lone = twins(); lone.consolidate();
  assert.equal(lone.fish.size, 1, 'control: look-alikes merge without co-visibility');
  const tw = twins(), ts = new StillSession(), ti = [{ emb: unit(0), hist }, { emb: b, hist }], tl = ts.label(tw, ti);
  assert.deepEqual(tl.map(l => l.fish), [1, 2]);
  ts.learn(tw, ti, tl, 1); tw.consolidate();
  assert.equal(tw.fish.size, 2, 'consolidate keeps co-visible fish apart');
  // survives a central sync: covis is in the merged gallery
  const merged = mergeGallery(id.toJSON(), id).gallery;
  assert.ok(merged.covis.includes('1|2') && merged.covis.includes('2|3'));
}
console.log('ok');
