import assert from 'node:assert/strict';

// models.js reads the worker's URL and storage at import; a fresh import per query string
globalThis.self = globalThis;
async function load(search) {
  globalThis.location = { search };
  return (await import(`../models.js${search}`)).Models;
}

// fake embedder: embedding = [batch size, first pixel of the crop]; records each run's shape
function fake(Models) {
  const m = new Models(), shapes = [];
  m.emb = { s: { run: async ({ pixel_values: t }) => {
    shapes.push(t.dims[0]);
    const N = t.dims[1] * t.dims[2] * t.dims[3], data = new Float32Array(t.dims[0] * 2);
    for (let i = 0; i < t.dims[0]; i++) { data[2 * i] = t.dims[0]; data[2 * i + 1] = t.data[i * N]; }
    return { embedding: { dims: [t.dims[0], 2], data } };
  } } };
  return { m, shapes };
}
const crop = v => { const x = new Float32Array(3 * 224 * 112); x[0] = v; return x; };

// lean (iPhone, or safe >= 1): always batch 2, zero padded, longer lists in chunks
const lean = fake(await load('?safe=1'));
assert.deepEqual(await lean.m.embed([]), []);
assert.deepEqual((await lean.m.embed([crop(7)])).map(e => Array.from(e)), [[2, 7]]);
assert.deepEqual((await lean.m.embed([crop(1), crop(2), crop(3)])).map(e => Array.from(e)), [[2, 1], [2, 2], [2, 3]]);
assert.deepEqual(lean.shapes, [2, 2, 2]);

// desktop: one run in the list's own size, as before
const desk = fake(await load('?safe=0'));
assert.deepEqual((await desk.m.embed([crop(1), crop(2), crop(3)])).map(e => Array.from(e)), [[3, 1], [3, 2], [3, 3]]);
assert.deepEqual(desk.shapes, [3]);

// ?embB= overrides (eval of the padded path on a desktop browser)
const forced = fake(await load('?safe=0&embB=4'));
await forced.m.embed([crop(1)]);
assert.deepEqual(forced.shapes, [4]);

console.log('embed_batch_test OK');
