// Local data harvest (IndexedDB only, nothing leaves the device): fish crops with labels
// and sparse full frames with detector pseudo-labels. Ids are UUIDs + timestamps so a later
// sync (e.g. Cloudflare) can merge without collisions.
//
// sample: {id, at, session, pond, synced, imgSynced, tag (embedder), fish|null, label: 'track'|'owner'|'pending', cand:[fishId], q, len, s, view, emb, hist, img:Blob}
// frame:  {id, at, session, pond, synced, w, h, img:Blob, dets:[{box, score, poly}]}
// Records from before ponds existed have no pond: they belong to the first pond (legacyPond).
import { sameEmbedder } from './reid.js';

export const LIMITS = { perFish: 40, samples: 3000, frames: 300, novelty: 0.9, frameEveryS: 10 };

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// Keep a confident crop only if it shows the fish in a way we have not stored yet.
export function isNovel(emb, stored, novelty = LIMITS.novelty) {
  for (const e of stored) if (dot(emb, e) >= novelty) return false;
  return true;
}

// Which sample to drop when a fish is over budget: the most redundant, never owner-verified.
export function evictIndex(items) {
  let worst = -1, ws = -Infinity;
  items.forEach((x, i) => {
    if (x.label === 'owner') return;
    let m = -1; items.forEach((y, j) => { if (i !== j) m = Math.max(m, dot(x.emb, y.emb)); });
    const r = m - 0.2 * x.q - 1e-12 * x.at; // redundant, low quality, then oldest
    if (r > ws) { ws = r; worst = i; }
  });
  return worst;
}

let dbp;
function db() {
  return (dbp ||= new Promise((res, rej) => {
    const r = indexedDB.open('koi-data', 1);
    r.onupgradeneeded = () => {
      const s = r.result.createObjectStore('samples', { keyPath: 'id' });
      s.createIndex('fish', 'fish'); s.createIndex('label', 'label');
      r.result.createObjectStore('frames', { keyPath: 'id' }).createIndex('at', 'at');
    };
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  }));
}
async function tx(store, mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(store, mode), out = fn(t.objectStore(store));
    t.oncomplete = () => res(out?.result ?? out); t.onerror = () => rej(t.error);
  });
}
const all = (store, index, key) => tx(store, 'readonly', s => (index ? s.index(index).getAll(key) : s.getAll()));

export class Harvest {
  constructor() { this.session = crypto.randomUUID(); this.cache = new Map(); this.lastFrameAt = -Infinity; this.counts = { samples: 0, frames: 0, owner: 0, pending: 0 }; }

  async init(cur, pond = null, legacyPond = pond) {
    navigator.storage?.persist?.().catch(() => {}); // ask the browser not to evict our data
    const current = typeof cur === 'string' ? { embedTag: cur, probeTag: cur } : cur;
    this.tag = current?.embedTag; this.pond = pond; this.legacyPond = legacyPond; this.cache = new Map();
    for (const x of await all('samples')) {
      // embeddings from another embedder (a retrained model keeps the same length) are ignored; their crops stay
      if (x.fish != null && sameEmbedder(x.tag, current) && this.mine(x)) { if (!this.cache.has(x.fish)) this.cache.set(x.fish, []); this.cache.get(x.fish).push(x.emb); }
    }
    await this.recount();
    return this;
  }
  async recount() {
    const s = await all('samples');
    this.counts = { samples: s.length, frames: (await all('frames')).length, owner: s.filter(x => x.label === 'owner').length, pending: s.filter(x => x.label === 'pending').length };
  }

  // confident track view of a known fish
  async addConfident(fishId, o, img) {
    const c = this.cache.get(fishId) || [];
    if (!isNovel(o.emb, c)) return false;
    await this.put({ fish: fishId, label: 'track', cand: [], ...o, img });
    this.cache.set(fishId, [...c, o.emb]);
    const mine = await this.ofFish(fishId);
    if (mine.length > LIMITS.perFish) {
      const i = evictIndex(mine);
      if (i >= 0) {
        await tx('samples', 'readwrite', s => s.delete(mine[i].id)); this.counts.samples--;
        this.cache.set(fishId, mine.filter((_, k) => k !== i).map(x => x.emb));
      }
    }
    return true;
  }

  // a track the system could not decide on: goes to the owner's review queue
  async addPending(cand, o, img) {
    if (this.counts.pending >= 50) return false;
    await this.put({ fish: null, label: 'pending', cand, ...o, img });
    this.counts.pending++;
    return true;
  }

  async put(x) {
    const s = { id: crypto.randomUUID(), at: Date.now(), session: this.session, pond: this.pond, synced: 0, imgSynced: 0, tag: this.tag, ...x };
    s.emb = Float32Array.from(s.emb);
    await tx('samples', 'readwrite', st => st.put(s)); this.counts.samples++;
    if (this.counts.samples > LIMITS.samples) { // global cap: drop oldest confident sample
      const old = (await all('samples', 'label', 'track')).sort((a, b) => a.at - b.at)[0];
      if (old) { await tx('samples', 'readwrite', st => st.delete(old.id)); this.counts.samples--; }
    }
  }

  mine(x) { return (x.pond ?? this.legacyPond) === this.pond; }
  // a pond turned out to be part of another one: its crops and frames move over, fish renumbered (map: old -> new, null = gone)
  async movePond(from, to, map) {
    const own = x => (x.pond ?? this.legacyPond) === from;
    const xs = (await all('samples')).filter(own), fs = (await all('frames')).filter(own);
    await tx('samples', 'readwrite', st => xs.forEach(x => {
      const fish = x.fish == null ? null : map.has(x.fish) ? map.get(x.fish) : x.fish;
      if (x.fish != null && fish == null) st.delete(x.id); else st.put({ ...x, pond: to, fish, cand: [], synced: 0 });
    }));
    await tx('frames', 'readwrite', st => fs.forEach(x => st.put({ ...x, pond: to, synced: 0 })));
  }
  async pending() { return (await all('samples', 'label', 'pending')).filter(x => this.mine(x)); }
  async ofFish(fishId) { return (await all('samples', 'fish', fishId)).filter(x => this.mine(x)); }

  // ---- central sync: what has not reached the server yet (labelled samples only; pending ones wait for an answer)
  async unsynced(limit = 40) {
    const own = x => ({ ...x, pond: x.pond ?? this.legacyPond });
    return { samples: (await all('samples')).filter(x => !x.synced && x.label !== 'pending').slice(0, limit).map(own),
      frames: (await all('frames')).filter(x => !x.synced).slice(0, Math.ceil(limit / 4)).map(own) };
  }
  async markSynced(store, x) { // only if nothing changed since it was read (a relabel in between stays unsynced)
    const d = await db();
    await new Promise((res, rej) => {
      const t = d.transaction(store, 'readwrite'), st = t.objectStore(store), r = st.get(x.id);
      r.onsuccess = () => { const cur = r.result; if (cur && cur.label === x.label && cur.fish === x.fish) st.put({ ...cur, pond: x.pond, synced: 1, imgSynced: 1 }); };
      t.oncomplete = res; t.onerror = () => rej(t.error);
    });
  }
  async setEmb(sample, emb) { await tx('samples', 'readwrite', s => s.put({ ...sample, emb: Float32Array.from(emb), tag: this.tag })); }

  // answer for a pending sample: fishId, or null = discard. label 'owner' = the owner said so (gold),
  // 'auto' = the gallery later placed it clearly by itself (bronze)
  async resolve(sample, fishId, label = 'owner') {
    if (fishId == null) { await tx('samples', 'readwrite', s => s.delete(sample.id)); this.counts.pending--; this.counts.samples--; return; }
    await tx('samples', 'readwrite', s => s.put({ ...sample, tag: this.tag, fish: fishId, label, grade: label === 'owner' ? 'gold' : 'bronze', labeledAt: Date.now(), synced: 0 }));
    this.cache.set(fishId, [...(this.cache.get(fishId) || []), sample.emb]);
    this.counts.pending--; if (label === 'owner') this.counts.owner++;
  }

  // keep labels consistent with gallery edits
  async relabel(goneId, keepId) {
    const xs = await this.ofFish(goneId);
    await tx('samples', 'readwrite', s => xs.forEach(x => s.put({ ...x, fish: keepId, label: x.label === 'owner' ? 'owner' : 'track', mergedFrom: goneId, synced: 0 })));
    this.cache.set(keepId, [...(this.cache.get(keepId) || []), ...(this.cache.get(goneId) || [])]); this.cache.delete(goneId);
  }
  async dropFish(fishId) {
    const xs = await this.ofFish(fishId);
    await tx('samples', 'readwrite', s => xs.forEach(x => s.delete(x.id)));
    this.cache.delete(fishId); await this.recount();
  }

  // sparse full frames with the detector's masks as pseudo-labels (future detector retraining).
  // frameDue() is checked synchronously so the caller snapshots the frame that matches dets.
  frameDue(t) {
    if (t - this.lastFrameAt < LIMITS.frameEveryS) return false;
    this.lastFrameAt = t; return true;
  }
  async addFrame(img, w, h, dets, quality = null) {
    await tx('frames', 'readwrite', s => s.put({ id: crypto.randomUUID(), at: Date.now(), session: this.session, pond: this.pond, synced: 0, w, h, img, dets, quality }));
    this.counts.frames++;
    if (this.counts.frames > LIMITS.frames) {
      const old = (await all('frames')).sort((a, b) => a.at - b.at)[0];
      await tx('frames', 'readwrite', s => s.delete(old.id)); this.counts.frames--;
    }
  }
}
