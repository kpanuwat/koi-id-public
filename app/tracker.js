// ByteTrack-lite: two-stage IoU association (high then low score dets) with a
// constant-velocity box predictor. Boxes are [x0,y0,x1,y1] in video pixels.
// Camera motion: a handheld pan moves every fish together, so the median shift of confident matches
// is applied to all predictions before association (translation only; ponytail: add scale/rotation if zooming hurts).

export function hungarian(cost) {
  // Min-cost assignment for an n x m matrix (rectangular ok). Returns row -> col (or -1).
  const n = cost.length, m = n ? cost[0].length : 0;
  if (!n || !m) return new Array(n).fill(-1);
  const N = Math.max(n, m);
  const a = Array.from({ length: N + 1 }, (_, i) => Array.from({ length: N + 1 }, (_, j) =>
    i && j && i <= n && j <= m ? cost[i - 1][j - 1] : 0));
  const u = new Float64Array(N + 1), v = new Float64Array(N + 1), p = new Int32Array(N + 1), way = new Int32Array(N + 1);
  for (let i = 1; i <= N; i++) {
    p[0] = i; let j0 = 0;
    const minv = new Float64Array(N + 1).fill(Infinity), used = new Uint8Array(N + 1);
    do {
      used[j0] = 1; const i0 = p[j0]; let d = Infinity, j1 = 0;
      for (let j = 1; j <= N; j++) if (!used[j]) {
        const c = a[i0][j] - u[i0] - v[j];
        if (c < minv[j]) { minv[j] = c; way[j] = j0; }
        if (minv[j] < d) { d = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= N; j++) if (used[j]) { u[p[j]] += d; v[j] -= d; } else minv[j] -= d;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const res = new Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j] && p[j] <= n) res[p[j] - 1] = j - 1;
  return res;
}

// share of the smaller box covered by the other (catches a head box inside a body box)
export function overlapSmall(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
}

export function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const i = w * h;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

// buffered IoU: small fish + handheld shake rarely overlap frame to frame; grow boxes before comparing
const grow = (b, k) => { const w = (b[2] - b[0]) * (k - 1) / 2, h = (b[3] - b[1]) * (k - 1) / 2; return [b[0] - w, b[1] - h, b[2] + w, b[3] + h]; };
export const BUFFER = 1.6;

// SLOW: a device that manages a detector pass only every SLOW.S seconds or more (iPhone ~1 s, field logs 25 Oct 2026).
// Pass-count rules are then eased so they mean roughly the same time: count after 2 passes, wider matching boxes
// (a fish swims further between passes), quicker candidate promotion. Fast devices and stride 5/10 runs are unchanged.
// hits 1 (v60): at ~1 pass/s a confident fish counts on its first pass (eval s20/s30: +1 to +2 fish, no sure-dups)
export const SLOW = { S: 0.4, hits: 1, lowHits: 3, buffer: 2.2, candHits: 2, candWindow: 4 };
let buffer = BUFFER;
function match(tracks, dets, minIou) {
  const cost = tracks.map(t => dets.map(d => 1 - iou(grow(t.pred, buffer), grow(d.box, buffer))));
  const asg = hungarian(cost), pairs = [], ut = [], used = new Set();
  asg.forEach((j, i) => (j >= 0 && cost[i][j] <= 1 - minIou ? (pairs.push([tracks[i], dets[j]]), used.add(j)) : ut.push(tracks[i])));
  return { pairs, ut, ud: dets.filter((_, j) => !used.has(j)) };
}

// Multi-frame evidence for counting (t.counted): a track counts only when it was seen >= 3 detector passes,
// its average score is >= 0.5, it is not a static object (rock, drain, stuck leaf: no motion relative to
// the pond for STILL_S seconds AND a middling score; a resting koi keeps a high score so it still counts),
// and its mask area does not flicker like a sun glint (log-area std > JITTER). Calibration knobs.
// Low-score but persistent objects that swim on their own (deep or water-coloured koi) also count:
// hits >= lowHits with average score >= lowScore, but only after they moved relative to the pond.
// JITTER_SLOW: the last 10 areas span ~3 s on a phone (~3 passes/s) instead of 0.3 s, so a turning fish varies more;
// 0.55 counted real koi that 0.45 dropped (24 crops checked by eye, 25 Oct 2026) with no rise in name clashes at stride 5/10.
export const COUNT = { hits: 3, score: 0.5, STILL_S: 5, staticScore: 0.7, moveFrac: 0.25, JITTER: 0.45, JITTER_SLOW: 0.55, SLOW_PASS_S: 0.1, lowHits: 6, lowScore: 0.3 };
// Candidates: weak detections (0.3..spawn) that keep reappearing in the same place become tracks
// (after 3 hits if they swam, 5 if still, within 8 detector passes). Stitching: a new track that appears
// where a named fish vanished (<= STITCH_S ago, plausible distance and size) inherits that fish as a prior.
export const CAND = { min: 0.3, hitsMoving: 3, hitsStill: 5, window: 8 };
export const STITCH = { S: 3, sizeRatio: 1.7 };

export class Tracker {
  // a lost track is dropped after maxLostS seconds unseen (a dive), or after maxLost detector passes without a match,
  // whichever comes first: every unmatched pass is another chance to latch onto the wrong fish (fast devices, eval)
  constructor({ high = 0.5, low = 0.2, spawn = 0.6, minIou = 0.2, maxLostS = 2.5, maxLost = 45 } = {}) {
    Object.assign(this, { high, low, spawn, minIou, maxLostS, maxLost });
    this.tracks = []; this.nextId = 1; this.frame = 0;
    this.cam = [0, 0]; // accumulated camera shift: image coords minus this = pond ("world") coords
    this.cands = []; this.gone = []; // weak-detection candidates; recently lost named tracks (for stitching)
  }

  // dets: [{box, score, area?, ...extra}], now: seconds -> live tracks (each carries the latest det as .det)
  update(dets, now = this.frame / 30) {
    this.frame++;
    if (this.lastNow != null && now > this.lastNow) this.passS = this.passS == null ? now - this.lastNow : 0.8 * this.passS + 0.2 * (now - this.lastNow);
    // see app.js (?slowAdapt=0 turns it off); a trailing comment had swallowed the buffer line until v60
    this.slow = this.adaptSlow !== false && (this.passS ?? 0) > SLOW.S; buffer = this.slow ? SLOW.buffer : BUFFER;
    this.lastNow = now;
    for (const t of this.tracks) {
      t.pred = t.box.map((x, k) => x + t.vel[k]);
    }
    const hi = dets.filter(d => d.score >= this.high), lo = dets.filter(d => d.score >= this.low && d.score < this.high);
    const pre = match(this.tracks.filter(t => t.lost === 0), hi, 0.5).pairs;
    if (pre.length >= 3) {
      const med = xs => xs.sort((a, b) => a - b)[xs.length >> 1];
      const cx = b => (b[0] + b[2]) / 2, cy = b => (b[1] + b[3]) / 2;
      const dx = med(pre.map(([t, d]) => cx(d.box) - cx(t.pred))), dy = med(pre.map(([t, d]) => cy(d.box) - cy(t.pred)));
      for (const t of this.tracks) t.pred = [t.pred[0] + dx, t.pred[1] + dy, t.pred[2] + dx, t.pred[3] + dy];
    }
    const m1 = match(this.tracks, hi, this.minIou);
    const m2 = match(m1.ut.filter(t => t.lost === 0), lo, this.minIou + 0.2);
    // camera step = median raw displacement of everything matched (fish swim every which way, the pan moves all)
    const pairs = [...m1.pairs, ...m2.pairs], cur = pairs.filter(([t]) => t.lost === 0);
    this.camStep = [0, 0];
    if (cur.length >= 3) {
      const med = xs => xs.sort((a, b) => a - b)[xs.length >> 1], c = (b, i) => (b[i] + b[i + 2]) / 2;
      this.camStep = [med(cur.map(([t, d]) => c(d.box, 0) - c(t.box, 0))), med(cur.map(([t, d]) => c(d.box, 1) - c(t.box, 1)))];
      this.cam[0] += this.camStep[0]; this.cam[1] += this.camStep[1];
    }
    for (const [t, d] of pairs) {
      t.vel = t.vel.map((v, k) => 0.7 * v + 0.3 * (d.box[k] - t.box[k]) / (t.lost + 1));
      t.box = d.box; t.det = d; t.lost = 0; t.hits++; t.age++;
      this.evidence(t, d, now);
    }
    const matched = new Set([...m1.pairs, ...m2.pairs].map(([t]) => t));
    for (const t of this.tracks) if (!matched.has(t)) { t.lost++; t.age++; t.det = null; t.box = t.pred; t.vel = t.vel.map(v => v * 0.8); }
    for (const d of m1.ud.filter(d => d.score >= this.spawn)) this.spawnTrack(d, now, 1); // weak dets never start tracks directly
    // weak detections: keep as candidates, promote the ones that keep coming back
    // weak detections inside/over a fish already tracked are parts of that fish (head, fin), not new fish
    const liveBoxes = this.tracks.filter(t => t.lost === 0).map(t => t.box);
    const covered = b => liveBoxes.some(L => overlapSmall(L, b) > 0.4);
    const weak = [...m1.ud.filter(d => d.score < this.spawn), ...m2.ud].filter(d => d.score >= CAND.min && !covered(d.box));
    for (const c of this.cands) c.pred = c.box;
    const mc = match(this.cands, weak, this.minIou);
    for (const [c, d] of mc.pairs) {
      const b = d.box, w = [(b[0] + b[2]) / 2 - this.cam[0], (b[1] + b[3]) / 2 - this.cam[1]];
      if (Math.hypot(w[0] - c.w0[0], w[1] - c.w0[1]) > 0.5 * Math.hypot(b[2] - b[0], b[3] - b[1])) c.moved = true;
      c.box = b; c.det = d; c.hits++; c.sum += d.score; c.last = this.frame;
    }
    for (const d of mc.ud) {
      const b = d.box;
      this.cands.push({ box: b, det: d, hits: 1, sum: d.score, born: this.frame, last: this.frame, moved: false, w0: [(b[0] + b[2]) / 2 - this.cam[0], (b[1] + b[3]) / 2 - this.cam[1]] });
    }
    this.cands = this.cands.filter(c => {
      if (c.hits >= (this.slow ? SLOW.candHits : c.moved ? CAND.hitsMoving : CAND.hitsStill)) { const t = this.spawnTrack(c.det, now, c.hits); t.scoreEMA = c.sum / c.hits; t.selfMoved = c.moved; t.promoted = true; return false; }
      return this.frame - c.last <= 2 && this.frame - c.born <= (this.slow ? SLOW.candWindow : CAND.window);
    });
    const expired = t => now - t.seenAt > this.maxLostS || t.lost > this.maxLost, out = this.tracks.filter(expired);
    for (const t of out) if (t.fishId != null && !t.stitched) this.gone.push({ fishId: t.fishId, w: t.lastW, at: t.seenAt, frame: t.seenFrame, diag: t.diag, speed: t.seenSpeed });
    this.gone = this.gone.filter(g => now - g.at <= STITCH.S);
    this.removed = out;
    this.tracks = this.tracks.filter(t => !expired(t));
    return this.tracks.filter(t => t.lost === 0 && t.hits >= 2);
  }

  spawnTrack(d, now, hits) {
    const t = { id: this.nextId++, box: d.box, pred: d.box, vel: [0, 0, 0, 0], det: d, lost: 0, hits, age: hits, born: this.frame, bornAt: now };
    this.evidence(t, d, now);
    // stitching: inherit the closest named fish that vanished nearby a moment ago (prior only; embeddings confirm)
    let best = null, bd = Infinity;
    const lost = this.tracks.filter(o => o.lost >= 1 && o !== t && !o.stitched)
      .map(o => ({ fishId: o.fishId, w: o.lastW, at: o.seenAt, frame: o.seenFrame, diag: o.diag, speed: o.seenSpeed, track: o }));
    for (const g of [...this.gone, ...lost]) {
      const dist = Math.hypot(t.lastW[0] - g.w[0], t.lastW[1] - g.w[1]), reach = 1.5 * g.diag + g.speed * (this.frame - g.frame) + 20;
      const ratio = Math.max(t.diag / g.diag, g.diag / t.diag);
      if (dist <= reach && ratio <= STITCH.sizeRatio && dist < bd) { bd = dist; best = g; }
    }
    if (best?.track) { // the fish surfaced again further on: continue the old track (keeps its votes, name, history)
      const o = best.track;
      o.box = d.box; o.pred = d.box; o.det = d; o.lost = 0; o.hits += hits; o.age += hits; o.vel = [0, 0, 0, 0];
      this.evidence(o, d, now);
      return o;
    }
    if (best) { t.prior = best.fishId; this.gone.splice(this.gone.indexOf(best), 1); }
    this.tracks.push(t);
    return t;
  }

  evidence(t, d, now) {
    const C = COUNT, b = d.box, diag = Math.hypot(b[2] - b[0], b[3] - b[1]);
    t.scoreEMA = t.scoreEMA == null ? d.score : 0.8 * t.scoreEMA + 0.2 * d.score;
    const w = [(b[0] + b[2]) / 2 - this.cam[0], (b[1] + b[3]) / 2 - this.cam[1]];
    t.lastW = w; t.diag = diag; t.seenAt = now; t.seenFrame = this.frame;
    t.seenSpeed = Math.hypot(t.vel[0] + t.vel[2], t.vel[1] + t.vel[3]) / 2; // px per detector pass, before it decays while lost
    if (!t.anchor || Math.hypot(w[0] - t.anchor[0], w[1] - t.anchor[1]) > C.moveFrac * diag) { if (t.anchor) t.selfMoved = true; t.anchor = w; t.movedAt = now; }
    // display box: light smoothing in pond coordinates (follows camera pans without lag)
    const st = this.camStep || [0, 0];
    t.sbox = t.sbox ? t.sbox.map((v, k) => 0.4 * (v + st[k % 2]) + 0.6 * b[k]) : b.slice();
    t.static = now - t.movedAt > C.STILL_S && t.scoreEMA < C.staticScore;
    (t.logA ||= []).push(Math.log(Math.max(1, d.area ?? (b[2] - b[0]) * (b[3] - b[1]))));
    if (t.logA.length > 10) t.logA.shift();
    if (t.logA.length >= 5) { const m = t.logA.reduce((a, v) => a + v, 0) / t.logA.length; t.jitter = Math.sqrt(t.logA.reduce((a, v) => a + (v - m) ** 2, 0) / t.logA.length); }
    const jit = (this.passS ?? 0) > C.SLOW_PASS_S ? C.JITTER_SLOW : C.JITTER;
    t.counted = t.hits >= (this.slow ? SLOW.hits : C.hits) && !t.static && !(t.jitter > jit) &&
      (t.scoreEMA >= C.score || (t.hits >= (this.slow ? SLOW.lowHits : C.lowHits) && t.scoreEMA >= C.lowScore && !!t.selfMoved));
  }
}
