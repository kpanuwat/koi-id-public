import assert from 'node:assert/strict';
import { Identity, sameEmbedder, sharedFish } from '../reid.js';

const cur = { embedTag: 'sha:0123456789abcdef', probeTag: '256:0.044,0.067,-0.007,-0.062' };
const close = '256:0.033,0.067,-0.007,-0.062';
const rounded = '256:0.043,0.066,-0.008,-0.061';
const far = '256:0.064,0.067,-0.007,-0.062';

assert.equal(sameEmbedder(cur.embedTag, cur), true);
assert.equal(sameEmbedder('sha:fedcba9876543210', cur), false);
assert.equal(sameEmbedder(close, cur), true);
assert.equal(sameEmbedder(rounded, cur), true);
assert.equal(sameEmbedder(far, cur), false);
assert.equal(sameEmbedder('512:0.044,0.067,-0.007,-0.062', cur), false);
assert.equal(sameEmbedder(null, cur), true);
assert.equal(sameEmbedder(undefined, cur), true);

const vector = Float32Array.from([1, 0]);
function gallery(tag) {
  const id = new Identity();
  id.embedTag = tag;
  id.newFish({ embs: [vector], hist: [1, 0, 0, 0, 0, 0, 0], sizes: [] }, 1);
  const saved = id.toJSON();
  delete saved.probeTag;
  return saved;
}

const accepted = new Identity().load(gallery(rounded), cur);
assert.equal(accepted.fish.get(1).ex.length, 1);
assert.equal(accepted.embedTag, cur.embedTag);
assert.equal(accepted.probeTag, cur.probeTag);
assert.equal(new Identity().load(gallery(rounded), rounded).fish.get(1).ex.length, 1);

const rejectedLegacy = new Identity().load(gallery(far), cur);
assert.equal(rejectedLegacy.fish.get(1).ex.length, 0);
assert.equal(rejectedLegacy.needsReembed, true);

const rejectedSha = new Identity().load(gallery('sha:fedcba9876543210'), cur);
assert.equal(rejectedSha.fish.get(1).ex.length, 0);
assert.equal(rejectedSha.needsReembed, true);

const left = new Identity(), right = new Identity();
left.embedTag = close;
right.embedTag = rounded;
left.newFish({ embs: [vector], hist: [1, 0, 0, 0, 0, 0, 0], sizes: [] }, 1);
right.newFish({ embs: [vector], hist: [1, 0, 0, 0, 0, 0, 0], sizes: [] }, 1);
assert.equal(sharedFish(left, right), 1);

console.log('embedtag_test: OK');
