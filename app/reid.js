// Koi identity: mask geometry, relative size, multi-view exemplar gallery,
// identity graph (cannot-link + consolidation), auto naming, persistence.
// Pure logic: crops/embeddings come in from models.js, so this runs in node for tests.
import { hungarian } from './tracker.js';

// Calibration knobs. score = 0.8*cos(fine-tuned embedder) + 0.2*colour-hist similarity - size penalty.
// Picked by replaying real runs offline (train/replay_id.mjs, 24 Oct 2026): newSim 0.45 / matchSim 0.70 gave
// 23 fish for clip A (max 24 seen at once), 35 for A+B, <=2% frames with two fish sharing a name.
export const CFG = {
  matchSim: 0.70,     // track->fish accept
  newSim: 0.45,       // below this vs every fish => new fish
  margin: 0.02,       // best must beat 2nd best by this
  minTrackS: 0.5,     // track must be followed this long (seconds) before it may found a new fish
  minEmbeds: 3,
  maxExemplars: 8,
  novelty: 0.88,      // confirmed track embedding below this vs fish exemplars => new view, store it
  sizeWeight: 0.15,   // max score penalty for size mismatch
  sizeSigma: 0.10,    // floor on log-size std
  mergeSim: 0.72,     // consolidation: auto-merge two fish never seen together
  suggestBand: 0.06,  // [mergeSim-band, mergeSim) => ask the owner
  population: 0,      // fish in pond (0 = unknown). When reached, no new fish are created
  showMargin: 0.05,   // show a name only when the fish beats its closest look-alike (look+colour+size) by this
  showQ: 0.06,        // ...and recent views are sharp/large enough to tell fish apart (0.25 hid every name on 540p phone video)
};

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const median = xs => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : NaN; };

// ---------- geometry from a binary mask ----------
// mask: Uint8Array w*h (bbox-local), ox/oy: bbox origin in frame px.
export function maskGeometry(mask, w, h, ox = 0, oy = 0, frameW = Infinity, frameH = Infinity) {
  let n = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) { n++; sx += x; sy += y; }
  if (n < 12) return null;
  const cx = sx / n, cy = sy / n;
  let xx = 0, yy = 0, xy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    const dx = x - cx, dy = y - cy; xx += dx * dx; yy += dy * dy; xy += dx * dy;
  }
  const theta = 0.5 * Math.atan2(2 * xy, xx - yy), ux = Math.cos(theta), uy = Math.sin(theta);
  // centerline: slice along the major axis, chain slice centroids -> length works for bent fish
  const K = 12; let umin = Infinity, umax = -Infinity;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    const u = (x - cx) * ux + (y - cy) * uy; if (u < umin) umin = u; if (u > umax) umax = u;
  }
  const bins = Array.from({ length: K }, () => ({ n: 0, u: 0, v: 0, vmin: Infinity, vmax: -Infinity }));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    const dx = x - cx, dy = y - cy, u = dx * ux + dy * uy, v = -dx * uy + dy * ux;
    const b = bins[Math.min(K - 1, Math.floor((u - umin) / (umax - umin + 1e-6) * K))];
    b.n++; b.u += u; b.v += v; if (v < b.vmin) b.vmin = v; if (v > b.vmax) b.vmax = v;
  }
  const pts = bins.filter(b => b.n).map(b => [b.u / b.n, b.v / b.n]);
  let len = 0; for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  len += (umax - umin) / K; // half a bin at each end
  const width = bins.map(b => (b.n ? b.vmax - b.vmin : 0));
  const front = (width[1] + width[2] + width[3]) / 3, back = (width[K - 4] + width[K - 3] + width[K - 2]) / 3;
  const edge = ox <= 2 || oy <= 2 || ox + w >= frameW - 2 || oy + h >= frameH - 2;
  return {
    cx: cx + ox, cy: cy + oy, theta, len, area: n,
    straight: (umax - umin) / Math.max(len, 1e-6),
    widthSign: Math.sign(front - back), // + => wide end (head) at +u
    edge,
  };
}

// ---------- relative size: log(len_px) = a + b*y + s_fish ----------
export class SizeModel {
  constructor() { this.a = null; this.b = 0; }
  // obs: [{y (0..1), logLen, s (known fish size or null), w}]
  fit(obs) {
    const anchors = obs.filter(o => o.s != null);
    let use = anchors;
    if (anchors.length < 3 || spread(anchors) < 0.15) {
      if (this.a != null && anchors.length) { // shift-only update from the anchors we have
        const da = median(anchors.map(o => o.logLen - o.s - this.a - this.b * o.y));
        this.a += 0.3 * da; return this;
      }
      if (this.a != null || obs.length < 4) return this;
      use = obs.map(o => ({ ...o, s: 0 })); // bootstrap: sizes relative to the population mean
    }
    let [a, b] = wls(use, use.map(o => o.w ?? 1));
    const r = use.map(o => o.logLen - o.s - a - b * o.y), mad = median(r.map(Math.abs)) + 1e-3;
    [a, b] = wls(use, use.map((o, i) => (o.w ?? 1) / Math.max(1, Math.abs(r[i]) / (1.5 * mad)))); // Huber reweight
    if (this.a == null) { this.a = a; this.b = b; } else { this.a = 0.7 * this.a + 0.3 * a; this.b = 0.7 * this.b + 0.3 * b; }
    return this;
  }
  size(logLen, y) { return this.a == null ? null : logLen - this.a - this.b * y; }
}
function spread(o) { const ys = o.map(q => q.y); return Math.max(...ys) - Math.min(...ys); }
function wls(o, w) {
  let S = 0, Sy = 0, Syy = 0, St = 0, Syt = 0;
  o.forEach((q, i) => { const t = q.logLen - q.s; S += w[i]; Sy += w[i] * q.y; Syy += w[i] * q.y * q.y; St += w[i] * t; Syt += w[i] * q.y * t; });
  const det = S * Syy - Sy * Sy;
  if (Math.abs(det) < 1e-9) return [St / S, 0];
  return [(St * Syy - Sy * Syt) / det, (S * Syt - Sy * St) / det];
}

// Shape + colour of the object in the canonical crop: sky glints on the water come out as stubby,
// washed-out white blobs with ripple streaks; koi are elongated. Measured on real crops (24 Oct 2026):
// glints white >= 0.6 and aspect < 2.6 (11 of 12 caught, 0 of 36 fish rejected). See train/runs/variety/.
export function blobShape(p) {
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, white = 0;
  for (let i = 0, N = 224 * 112; i < N; i++) {
    const r = p[4 * i], g = p[4 * i + 1], b = p[4 * i + 2];
    if (Math.abs(r - 124) + Math.abs(g - 116) + Math.abs(b - 104) <= 30) continue; // background fill
    const x = i % 224, y = (i / 224) | 0, mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    if (mx > 0.45 * 255 && (mx - mn) / Math.max(mx, 1) < 0.28) white++;
  }
  if (n < 30) return { aspect: 0, white: 0 };
  const cxx = sxx / n - (sx / n) ** 2, cyy = syy / n - (sy / n) ** 2, cxy = sxy / n - (sx / n) * (sy / n);
  const tr = cxx + cyy, det = cxx * cyy - cxy * cxy, d = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  return { aspect: Math.sqrt((tr / 2 + d) / Math.max(1e-6, tr / 2 - d)), white: white / n };
}
export const isGlint = c => c.white >= 0.6 && c.aspect > 0 && c.aspect < 2.6;

// ---------- naming ----------
// hist: fractions [red, orange, yellow, white, black, brown, other]
export const COLORS = ['red', 'orange', 'yellow', 'white', 'black', 'brown', 'other'];
export function colorClass(h, s, v) { // h 0..360, s/v 0..1; white/black loosened for koi seen through water
  if (v < 0.18) return 4;
  if (s < 0.28 && v > 0.45) return 3;
  if (s > 0.2 && s < 0.5 && v < 0.55 && h >= 15 && h < 45) return 5;
  if (s > 0.45 && (h < 15 || h >= 340)) return 0;
  if (s > 0.45 && h >= 15 && h < 35) return 1;
  if (s > 0.3 && h >= 35 && h < 70) return 2;
  return 6;
}
// rules work on the fish colours only: "other" (unreadable, mixed-with-water pixels) is left out
export function variety(f) {
  const t = f.slice(0, 6).reduce((a, v) => a + v, 0) || 1, [r, o, y, w, k, br] = f.slice(0, 6).map(v => v / t);
  if (k > 0.5) return 'Karasu';
  if (br > 0.4) return 'Chagoi';
  if (r + o > 0.12 && w > 0.2 && k > 0.08) return 'Sanke';
  if (r + o > 0.12 && w > 0.2) return 'Kohaku';
  if (y > 0.12 && w > 0.25) return 'Hariwake';
  if (w > 0.6) return 'Platinum';
  if (y > 0.4 && y >= o) return 'Ogon';
  if (o > 0.3 && o >= r) return 'Orenji';
  if (r + o > 0.4) return 'Benigoi';
  return 'Koi';
}
const NAMES = {
  Kohaku: ['ซากุระ', 'ทับทิม', 'ธงชาติ', 'มะลิแดง', 'สตรอเบอรี่', 'ลิปสติก', 'แตงโม', 'กุหลาบ', 'ลูกกวาด', 'ชมพู่', 'ซูชิ', 'บีทรูท'],
  Sanke: ['สามสี', 'ปิกัสโซ่', 'ลายเสือ', 'ตุ๊กตา', 'ม้าลาย', 'แพนด้า', 'โมเสก', 'ลายจุด', 'ดัลเมเชียน', 'หมากรุก'],
  Ogon: ['ทองหยิบ', 'ทองหยอด', 'ฝอยทอง', 'กล้วยหอม', 'มะม่วง', 'ขมิ้น', 'ไข่เจียว', 'เลมอน', 'ซันนี่', 'ทองคำ', 'ข้าวโพด', 'ทานตะวัน'],
  Orenji: ['ส้มจี๊ด', 'ส้มโอ', 'ฟักทอง', 'แครอท', 'ส้มซ่า', 'มะละกอ', 'แอปริคอต', 'ส้มตำ', 'ทุเรียน', 'พีช'],
  Benigoi: ['พริกขี้หนู', 'มะเขือเทศ', 'ลิ้นจี่', 'เชอร์รี่', 'แดงแจ๋', 'ทับทิมกรอบ', 'แอปเปิ้ล', 'ซอสมะเขือ', 'กระเจี๊ยบ', 'แดงชาด'],
  Platinum: ['สำลี', 'ปุยฝ้าย', 'หิมะ', 'ไข่มุก', 'นมสด', 'มะพร้าว', 'โมจิ', 'ข้าวเหนียว', 'น้ำแข็ง', 'เกล็ดหิมะ'],
  Hariwake: ['ครีม', 'ขนมชั้น', 'คัสตาร์ด', 'บานาน่า', 'ไข่ดาว', 'ฟองนม', 'ชีส', 'พุดดิ้ง'],
  Karasu: ['นิล', 'ถ่าน', 'กาแฟ', 'โอรีโอ้', 'งาดำ', 'ราตรี', 'เฉาก๊วย', 'อีกา'],
  Chagoi: ['โกโก้', 'ชาเย็น', 'คาราเมล', 'มอคค่า', 'ช็อกโกแลต', 'อัลมอนด์', 'ขนมปังปิ้ง', 'เกาลัด'],
  Koi: ['น้องปลา', 'ลูกชิ้น', 'ขนมปัง', 'บัวลอย', 'เต้าฮวย', 'ทองม้วน', 'ลูกชุบ', 'สายไหม'],
};
// second word for compound names once a colour's list is used up: every auto name stays unique, at most
// NAME_MAX characters (Thai vowel and tone marks count, owner 25 Oct 2026) and never gets a number
const MORE = ['ดาว', 'เมฆ', 'ฝน', 'ลม', 'ฟ้า', 'รุ้ง', 'หมอก', 'จันทร์', 'ตะวัน', 'ธารา', 'ภูผา', 'ใบเตย', 'ใบบัว', 'ใบไผ่', 'มะลิ', 'ดอกแก้ว',
  'ลำดวน', 'พุดซ้อน', 'ขนนก', 'ระฆัง', 'โบว์', 'กระดิ่ง', 'ทราย', 'กรวด', 'มรกต', 'ไพลิน', 'บุษบา', 'พลอย', 'น้ำผึ้ง', 'ข้าวหอม', 'คลื่น', 'ชบา'];
export const NAME_MAX = 10;
const ALL = Object.values(NAMES).flat();
export function autoName(v, taken) {
  const pool = NAMES[v] || NAMES.Koi, ok = n => [...n].length <= NAME_MAX && !taken.has(n);
  for (const n of pool) if (ok(n)) return n;
  for (const m of MORE) for (const n of pool) if (ok(n + m)) return n + m;       // same colour first
  for (const m of MORE) for (const n of ALL) if (ok(n + m)) return n + m;
  for (const m of MORE) for (const p of MORE) if (ok(p + m)) return p + m;       // ~2,000 short names in all
  return pool[0]; // ponytail: past ~2,000 fish names start repeating; add words to MORE if a library ever gets there
}

// ---------- identity ----------
const viewOf = theta => ((Math.round(theta / (Math.PI / 2)) % 4) + 4) % 4; // heading quadrant

export function sameEmbedder(tag, cur) {
  if (tag == null) return true;
  if (typeof tag !== 'string') return false;
  const current = typeof cur === 'string' ? { embedTag: cur, probeTag: cur } : cur || {};
  if (tag.startsWith('sha:')) return tag === current.embedTag;
  const parse = value => {
    if (typeof value !== 'string') return null;
    const i = value.indexOf(':');
    if (i < 1 || !/^\d+$/.test(value.slice(0, i))) return null;
    const parts = value.slice(i + 1).split(',').map(Number);
    return parts.length === 4 && parts.every(Number.isFinite) ? [value.slice(0, i), parts] : null;
  };
  const a = parse(tag), b = parse(current.probeTag);
  return !!a && !!b && a[0] === b[0] && a[1].every((v, i) => Math.abs(v - b[1][i]) <= 0.011 + Number.EPSILON);
}

export class Identity {
  constructor(cfg = {}) {
    this.cfg = { ...CFG, ...cfg };
    this.embedTag = null; this.probeTag = null;
    this.fish = new Map(); this.nextFish = 1;
    this.covis = new Set();       // "a|b" fish pairs seen at the same time => never the same fish
    this.suggestions = [];        // [{a, b, sim}] for the owner to confirm
    this.size = new SizeModel();
    this.cmK = null;              // cm = cmK * exp(s), set by calibrate()
    this.log = [];                // events for UI / eval
    this.alias = new Map();       // merged-away fish id -> surviving id
    this.onMerge = null;          // (keepId, goneId) => void, keeps harvested labels in step
    this.central = false;         // synced pond: the central gallery numbers and names new fish (see mergeGallery)
    this.gone = new Set();        // centrally numbered fish the owner deleted: stay deleted on every device
    this.joins = [];              // [keep, gone] merges of centrally numbered fish, applied centrally at the next sync
  }
  resolve(id) { while (this.alias.has(id)) id = this.alias.get(id); return id; }

  // ---- scoring ----
  appSim(t, f) { // mean over the track's recent views of best exemplar match
    if (!t.embs.length || !f.ex.length) return -1;
    let s = 0;
    for (const e of t.embs) { let best = -1; for (const x of f.ex) best = Math.max(best, dot(e, x.e)); s += best; }
    return s / t.embs.length;
  }
  score(t, f) {
    let sc = 0.8 * this.appSim(t, f) + 0.2 * histSim(t.hist, f.hist);
    const ts = trackSize(t);
    if (ts && f.size.n >= 3) {
      const sd = Math.hypot(Math.sqrt(f.size.v), ts.sd, this.cfg.sizeSigma), z = (ts.m - f.size.m) / sd;
      sc -= this.cfg.sizeWeight * (1 - Math.exp(-0.5 * z * z));
    }
    return sc;
  }

  // ---- per frame ----
  // tracks: live tracker tracks carrying t.obs = {geo, emb?, hist?, q, thumb?}; frameH for size y-normalisation
  step(tracks, frameH, now = Date.now()) {
    const C = this.cfg;
    for (const t of tracks) {
      t.embs ||= []; t.votes ||= new Map(); t.sizes ||= [];
      // stitched track: the fish that vanished here a moment ago is a prior vote, not yet "sure"
      if (t.prior != null && !t.votes.size && this.fish.has(this.resolve(t.prior))) t.votes.set(this.resolve(t.prior), 0.25);
      // fish merged away: follow it to the survivor; fish deleted or library cleared: forget it
      for (const id of t.votes.keys()) if (!this.fish.has(id)) {
        t.votes = [...t.votes].reduce((m, [k, v]) => { k = this.resolve(k); return this.fish.has(k) ? m.set(k, (m.get(k) || 0) + v) : m; }, new Map());
        break;
      }
      if (t.fishId && !this.fish.has(t.fishId)) {
        const k = this.resolve(t.fishId);
        if (this.fish.has(k)) t.fishId = k; else { t.fishId = null; t.sure = false; t.founder = false; }
      }
      const o = t.obs; if (!o) continue;
      if (o.emb) { t.embs.push(o.emb); if (t.embs.length > 6) t.embs.shift(); t.embN = (t.embN || 0) + 1; if (!o.geo?.partial) t.fullEmbN = (t.fullEmbN || 0) + 1; }
      if (o.hist) t.hist = t.hist ? t.hist.map((x, i) => 0.8 * x + 0.2 * o.hist[i]) : o.hist.slice();
      if (o.thumb && (!t.thumb || o.q > t.thumbQ)) { t.thumb = o.thumb; t.thumbQ = o.q; }
      if (o.q != null) t.qEMA = t.qEMA == null ? o.q : 0.7 * t.qEMA + 0.3 * o.q;
    }
    // relative size
    const sobs = [];
    for (const t of tracks) {
      const g = t.obs?.geo;
      if (!g || g.edge || g.partial || g.straight < 0.85 || g.len < 12) continue; // partial views would shrink the size
      const f = this.fish.get(t.fishId);
      sobs.push({ t, y: g.cy / frameH, logLen: Math.log(g.len), s: f && f.size.n >= 5 && t.sure ? f.size.m : null, w: Math.min(1, g.len / 60) });
    }
    this.size.fit(sobs);
    for (const o of sobs) {
      const s = this.size.size(o.logLen, o.y);
      if (s != null) { o.t.sizes.push(s); if (o.t.sizes.length > 30) o.t.sizes.shift(); }
    }
    // joint assignment: confirmed tracks block their fish for everyone else (cannot-link)
    const cand = tracks.filter(t => t.embs.length && t.obs?.emb);
    const taken = new Map(tracks.filter(t => t.sure).map(t => [t.fishId, t]));
    const fl = [...this.fish.values()];
    if (cand.length) {
      const S = cand.map(t => fl.map(f => ((taken.has(f.id) && taken.get(f.id) !== t) || t.notFish?.has(f.id) ? -9 : this.score(t, f))));
      const asg = fl.length ? hungarian(S.map(r => r.map(x => 1 - x))) : cand.map(() => -1);
      cand.forEach((t, i) => {
        const j = asg[i], sc = j >= 0 ? S[i][j] : -1;
        const second = Math.max(-1, ...S[i].filter((_, k) => k !== j));
        const full = C.population > 0 && this.fish.size >= C.population;
        if (j >= 0 && sc > -9 && ((sc >= C.matchSim && sc - second >= C.margin) || (full && sc > 0))) {
          t.votes.set(fl[j].id, (t.votes.get(fl[j].id) || 0) + (sc - C.newSim) + 0.05);
          t.lastScore = sc; t.lastBest = fl[j].id; t.lastMargin = sc - second;
        } else if (!t.fishId && (j < 0 || sc < C.newSim) && !full &&
                   trackAge(t) >= C.minTrackS && (t.fullEmbN || 0) >= C.minEmbeds) { // never found a fish from head/tail-only views
          const f = this.newFish(t, now);
          t.votes.set(f.id, 1); t.lastScore = 1; t.founder = true;
        }
        // identity = vote leader; "sure" once it leads clearly
        let best = null, bv = 0, sv = 0;
        for (const [id, v] of t.votes) if (v > bv) { sv = bv; bv = v; best = id; } else if (v > sv) sv = v;
        if (best != null && this.fish.has(best)) {
          if (t.fishId != null && t.fishId !== best) t.switches = (t.switches || 0) + 1; // identity flip: a red flag for learning
          t.fishId = best; t.sure = bv >= 0.3 && bv >= 2 * sv;
        }
        // distinguishability: how clearly this track beats its closest look-alike, smoothed
        const m = t.lastBest === t.fishId ? (t.lastMargin ?? 0) : 0;
        t.mEMA = t.mEMA == null ? m : 0.7 * t.mEMA + 0.3 * m;
      });
    }
    // RED FLAG, one fish in two places: impossible. The track that fits the fish best (appearance, colour, size) keeps it;
    // the other learns it is NOT that fish. If it keeps showing evidence it founds a look-alike fish of its own, and once
    // both are confirmed on screen together they are never merged.
    const claims = new Map();
    for (const t of tracks) if (t.fishId != null && this.fish.has(t.fishId)) { if (!claims.has(t.fishId)) claims.set(t.fishId, []); claims.get(t.fishId).push(t); }
    for (const [fid, ts] of claims) if (ts.length > 1) {
      const f = this.fish.get(fid), fit = t => (t.embs.length ? this.score(t, f) : -1) + (t.sure ? 0.05 : 0) + (t.founder ? 0.1 : 0);
      ts.sort((a, b) => fit(b) - fit(a));
      for (const t of ts.slice(1)) {
        (t.notFish ||= new Set()).add(fid); t.votes.delete(fid);
        t.fishId = null; t.sure = false; t.founder = false; t.red = (t.red || 0) + 1; this.redFlags = (this.redFlags || 0) + 1;
      }
    }
    // learn from confirmed tracks: new views, colour, size. Red-flagged ones (size far off the fish, identity flipping) do not teach
    for (const t of tracks) {
      const f = this.fish.get(t.fishId); if (!f || !t.sure) continue;
      const o = t.obs; f.last = now;
      const ts = trackSize(t);
      t.sizeOff = !!ts && f.size.n >= 5 && Math.abs(ts.m - f.size.m) > 3 * Math.hypot(Math.sqrt(f.size.v), ts.sd, C.sizeSigma);
      if (t.sizeOff || (t.switches || 0) >= 2) continue;
      if (o?.emb && o.q > 0.3) this.addExemplar(f, o.emb, viewOf(o.geo?.heading ?? o.geo?.theta ?? 0), o.q);
      if (t.hist) { f.hist = f.hist.map((x, i) => 0.95 * x + 0.05 * t.hist[i]); f.variety = variety(f.hist); this.renameAuto(f); }
      const s = t.sizes.at(-1);
      if (s != null && t.obs?.geo) addSize(f.size, s);
      if (o?.thumb && o.q > (f.thumbQ || 0)) { f.thumb = o.thumb; f.thumbQ = o.q; }
    }
    const ids = [...new Set(tracks.filter(t => t.sure).map(t => t.fishId))];
    for (let i = 0; i < ids.length; i++) for (let k = i + 1; k < ids.length; k++) this.covis.add(pair(ids[i], ids[k]));
  }

  // best gallery match for one stored crop (no track): [fishId, score, runner-up score]
  matchOne(e, hist) {
    let best = null, bs = -1, ss = -1;
    for (const f of this.fish.values()) {
      if (!f.ex.length) continue;
      let m = -1; for (const x of f.ex) m = Math.max(m, dot(e, x.e));
      const sc = 0.8 * m + 0.2 * histSim(hist, f.hist);
      if (sc > bs) { ss = bs; bs = sc; best = f.id; } else if (sc > ss) ss = sc;
    }
    return [best, bs, ss];
  }

  // photo mode: one still, no tracks. items [{emb, hist}] against the gallery plus optional fish-like records `extra`
  // ({id, ex: [{e}], hist, size: {n: 0}}, e.g. unknowns from earlier photos of the session). Scored like step() (one view,
  // no size: a still's pixel scale differs from the video the size model came from), assigned jointly so two items never
  // get the same fish; a match needs matchSim and a clear lead over the runner-up (showMargin, as for a name on screen).
  // Read only: no votes, exemplars or new fish.
  matchStill(items, extra = []) {
    const C = this.cfg, fl = [...this.fish.values()].filter(f => f.ex.length).concat(extra);
    if (!items.length || !fl.length) return items.map(() => ({ fish: null, best: null, score: -1, second: -1 }));
    const S = items.map(it => { const t = { embs: [it.emb], hist: it.hist, sizes: [] }; return fl.map(f => this.score(t, f)); });
    const asg = hungarian(S.map(r => r.map(x => 1 - x)));
    return items.map((_, i) => {
      const j = asg[i], sc = j >= 0 ? S[i][j] : -1, second = Math.max(-1, ...S[i].filter((_, k) => k !== j));
      const ok = j >= 0 && sc >= C.matchSim && sc - second >= C.showMargin;
      return { fish: ok ? fl[j].id : null, best: j >= 0 ? fl[j].id : null, score: sc, second };
    });
  }

  // GREEN FLAGS, how much a confirmed crop is worth as training data: silver = followed >= 2 s, >= 3 full views and told
  // apart from its look-alikes, with no red flag; bronze = confirmed only. Red-flagged tracks give no training data (null).
  grade(t) {
    if (!t.sure || !this.fish.has(t.fishId) || t.red || t.sizeOff || (t.switches || 0) >= 2) return null;
    return trackAge(t) >= 2 && (t.fullEmbN || 0) >= 3 && this.named(t) ? 'silver' : 'bronze';
  }

  // name on screen only when the evidence separates this fish from the others; otherwise stay silent
  named(t) { // a fish the centre has not accepted yet (c false) keeps its hex ID until the next sync gives it a number and name
    return !!(t.sure && this.fish.has(t.fishId) && this.fish.get(t.fishId).c !== false && (t.qEMA ?? 0) >= this.cfg.showQ && (t.founder || (t.mEMA ?? 0) >= this.cfg.showMargin));
  }

  // the first colours come from a few murky crops: an auto name follows the type once the new type has held for a while
  // (about 5 s of confirmed views); a name the owner typed never changes
  renameAuto(f) {
    if (!f.autoName || f.variety === f.nameVariety) { f.vStreak = 0; return; }
    if (++f.vStreak < 30) return;
    f.name = autoName(f.variety, new Set([...this.fish.values()].filter(x => x !== f).map(x => x.name)));
    f.nameVariety = f.variety; f.vStreak = 0;
  }

  // called with tracks the tracker dropped
  retire(tracks) {
    for (const t of tracks) { const f = this.fish.get(t.fishId); if (f && t.sure) f.seen++; }
  }

  newFish(t, now) {
    const hist = t.hist ? t.hist.slice() : new Array(7).fill(0);
    const v = variety(hist), names = new Set([...this.fish.values()].map(f => f.name));
    const f = { id: this.nextFish++, name: autoName(v, names), variety: v, nameVariety: v, autoName: true, vStreak: 0, ex: [], hist, size: { m: 0, v: 0, n: 0 },
      thumb: t.thumb || null, thumbQ: t.thumbQ || 0, seen: 0, first: now, last: now, ...(this.central ? { c: false } : {}) };
    for (const e of t.embs) this.addExemplar(f, e, 0, 0.5);
    for (const s of t.sizes) addSize(f.size, s);
    this.fish.set(f.id, f);
    this.log.push({ ev: 'new', id: f.id, name: f.name, track: t.id, at: now });
    return f;
  }

  addExemplar(f, e, view, q) {
    let best = -1; for (const x of f.ex) best = Math.max(best, dot(e, x.e));
    if (best >= this.cfg.novelty) return false;
    f.ex.push({ e, view, q });
    if (f.ex.length > this.cfg.maxExemplars) { // drop the most redundant (closest to another), keeps view diversity
      let worst = 0, ws = -2;
      f.ex.forEach((x, i) => { let m = -1; f.ex.forEach((y, j) => { if (i !== j) m = Math.max(m, dot(x.e, y.e)); }); const r = m - 0.1 * x.q; if (r > ws) { ws = r; worst = i; } });
      f.ex.splice(worst, 1);
    }
    return true;
  }

  fishSim(a, b) { // mean of each exemplar's best match in the other fish
    if (!a.ex.length || !b.ex.length) return -1;
    let s = 0; for (const x of a.ex) { let m = -1; for (const y of b.ex) m = Math.max(m, dot(x.e, y.e)); s += m; }
    return s / a.ex.length;
  }

  pairSim(a, b) { // same fish? both directions must agree; clearly different sizes count against
    let sim = Math.min(this.fishSim(a, b), this.fishSim(b, a));
    if (a.size.n >= 5 && b.size.n >= 5 && Math.abs(a.size.m - b.size.m) > 3 * Math.hypot(Math.sqrt(a.size.v), Math.sqrt(b.size.v), this.cfg.sizeSigma)) sim -= 0.1;
    return sim;
  }

  // sizes are only known relative to each other (log size = image length - a - b*y): nothing pins the shared offset, so
  // over hours of use it drifted until every fish read x0.00 (owner's gallery, 25 Oct 2026: mean log size -5.8).
  // Pin the pond's average fish to x1.00; a drifted gallery heals itself on the next load or clean-up.
  // Median, not mean: fish whose size drifted at another rate must not drag the pond. A fish more than 4x longer or shorter
  // than the pond's typical koi is a drift artefact (x45 seen after the fix above): its size is forgotten and relearned.
  recenter() {
    const fl = [...this.fish.values()].filter(f => f.size?.n >= 5);
    if (fl.length < 3) return;
    const d = median(fl.map(f => f.size.m));
    if (Math.abs(d) >= 0.01) {
      for (const f of this.fish.values()) if (f.size?.n) f.size.m -= d;
      if (this.size.a != null) this.size.a += d;
      if (this.cmK) this.cmK *= Math.exp(d);
    }
    for (const f of fl) if (Math.abs(f.size.m) > Math.log(4)) f.size = { m: 0, v: 0, n: 0 };
  }

  // periodic graph clean-up: merge duplicates that were never seen together
  consolidate() {
    this.recenter();
    const C = this.cfg, fl = [...this.fish.values()], sugg = [];
    let merged = 0;
    for (let i = 0; i < fl.length; i++) for (let k = i + 1; k < fl.length; k++) {
      const a = fl[i], b = fl[k];
      if (!this.fish.has(a.id) || !this.fish.has(b.id) || this.covis.has(pair(a.id, b.id))) continue;
      const sim = this.pairSim(a, b);
      if (sim >= C.mergeSim) { this.merge(a.id, b.id); merged++; }
      else if (sim >= C.mergeSim - C.suggestBand) sugg.push({ a: a.id, b: b.id, sim });
    }
    this.suggestions = sugg.sort((x, y) => y.sim - x.sim).slice(0, 10);
    return merged;
  }

  merge(keepId, goneId) {
    const k = this.fish.get(keepId), g = this.fish.get(goneId);
    if (!k || !g || keepId === goneId) return;
    for (const x of g.ex) this.addExemplar(k, x.e, x.view, x.q);
    k.hist = k.hist.map((x, i) => (x * (k.seen + 1) + g.hist[i] * (g.seen + 1)) / (k.seen + g.seen + 2));
    if (g.size.n) { const n = k.size.n + g.size.n; k.size.m = (k.size.m * k.size.n + g.size.m * g.size.n) / n; k.size.v = Math.max(k.size.v, g.size.v); k.size.n = n; }
    k.seen += g.seen; k.first = Math.min(k.first, g.first);
    if (k.autoName && g.autoName === false) { k.name = g.name; k.autoName = false; } // a name the owner typed survives the merge
    if ((g.thumbQ || 0) > (k.thumbQ || 0)) { k.thumb = g.thumb; k.thumbQ = g.thumbQ; }
    if (g.c === true && k.c === true) this.joins.push([keepId, goneId]);
    this.fish.delete(goneId); this.alias.set(goneId, keepId);
    this.onMerge?.(keepId, goneId);
    this.covis = new Set([...this.covis].map(p => p.split('|').map(Number).map(x => (x === goneId ? keepId : x))).filter(([x, y]) => x !== y).map(([x, y]) => pair(x, y)));
    this.log.push({ ev: 'merge', keep: keepId, gone: goneId, at: Date.now() });
  }

  remove(fid) { // owner deleted a fish; a centrally numbered one is deleted on every device
    if (this.fish.get(fid)?.c === true) this.gone.add(fid);
    this.fish.delete(fid);
  }

  // two taps on a known length (px at frame y) -> cm for every fish via the shared size scale
  calibrate(lenPx, lenCm, yNorm) {
    if (this.size.a == null) return false;
    this.cmK = Math.exp(this.size.a + this.size.b * yNorm) * lenCm / lenPx;
    return true;
  }
  cm(f) { return this.cmK && f.size.n >= 3 ? this.cmK * Math.exp(f.size.m) : null; }

  toJSON() {
    return { v: 1, embedTag: this.embedTag, probeTag: this.probeTag, nextFish: this.nextFish, cmK: this.cmK, population: this.cfg.population, covis: [...this.covis], size: { a: this.size.a, b: this.size.b },
      alias: [...this.alias], gone: [...this.gone], joins: this.joins,
      fish: [...this.fish.values()].map(f => ({ ...f, ex: f.ex.map(x => ({ ...x, e: Array.from(x.e) })) })) };
  }
  load(j, cur) {
    if (!j || j.v !== 1) return this;
    // gallery built with another embedder: keep names, sizes, colours, thumbnails; drop the incomparable
    // embeddings and let the app rebuild them from the crops kept on this device (migrate())
    const current = typeof cur === 'string' ? { embedTag: cur, probeTag: cur } : cur;
    const other = current && !sameEmbedder(j.embedTag, current);
    this.embedTag = current?.embedTag ?? this.embedTag ?? j.embedTag ?? null;
    this.probeTag = current?.probeTag ?? this.probeTag ?? j.probeTag ?? null;
    this.nextFish = j.nextFish; this.cmK = j.cmK; this.covis = new Set(j.covis); this.cfg.population = j.population || 0;
    this.alias = new Map(j.alias || []); this.gone = new Set(j.gone || []); this.joins = j.joins || [];
    if (j.size) { this.size.a = j.size.a; this.size.b = j.size.b ?? 0; }
    this.fish = new Map(j.fish.map(f => [f.id, { ...f, ex: other ? [] : f.ex.map(x => ({ ...x, e: Float32Array.from(x.e) })) }]));
    if (other) this.needsReembed = true;
    // older galleries numbered repeat names ("ทองหยอด 2") or made long ones: give those auto names a short unique name
    for (const f of this.fish.values()) if (f.autoName !== false && (/\s\d+$/.test(f.name) || [...f.name].length > NAME_MAX)) {
      f.name = autoName(f.variety, new Set([...this.fish.values()].map(x => x.name))); f.autoName = true; f.nameVariety = f.variety;
    }
    this.recenter();
    return this;
  }
}

// ---------- central gallery: the single source of truth for fish numbers and names ----------
// A device numbers a new fish itself (it must count and track offline), but that number is only a proposal (c false):
// another device may have used it meanwhile. At sync the device merges its gallery into the central one:
//   fish the centre accepted (c true) merge by number (renamed, deleted or merged centrally: follow the centre);
//   proposals join the central fish they clearly are (never one seen on screen together with them),
//   otherwise they get the next central number and a unique name.
// Returns the new central gallery, the local -> central number map (null = deleted) and the newly numbered fish.
// The server accepts the result only on top of the version it was built from, so merges never overwrite each other.
export function mergeGallery(remote, local) {
  const C = new Identity(local.cfg); C.embedTag = local.embedTag; C.probeTag = local.probeTag;
  if (remote) C.load(remote, { embedTag: local.embedTag, probeTag: local.probeTag });
  C.joins = [];
  const map = new Map(), fresh = [], added = new Set();
  const absorb = (k, f) => {
    for (const x of f.ex) C.addExemplar(k, x.e, x.view, x.q);
    k.hist = k.hist.map((x, i) => (x * (k.seen + 1) + f.hist[i] * (f.seen + 1)) / (k.seen + f.seen + 2)); k.variety = variety(k.hist);
    if (f.size.n > k.size.n) k.size = { ...f.size };
    k.seen = Math.max(k.seen, f.seen); k.first = Math.min(k.first, f.first); k.last = Math.max(k.last, f.last);
    if (f.autoName === false && (k.autoName !== false || (f.namedAt || 0) > (k.namedAt || 0))) { k.name = f.name; k.autoName = false; k.namedAt = f.namedAt; }
    if ((f.thumbQ || 0) > (k.thumbQ || 0)) { k.thumb = f.thumb; k.thumbQ = f.thumbQ; }
  };
  for (const g of local.gone) { C.gone.add(g); C.fish.delete(g); }
  for (const f of local.fish.values()) {
    if (f.c !== true) { fresh.push(f); continue; }
    if (C.gone.has(f.id)) { map.set(f.id, null); continue; }
    const r = C.resolve(f.id);
    if (!C.fish.has(r)) { fresh.push(f); continue; } // the centre lost it (store reset): rejoin as a proposal
    absorb(C.fish.get(r), f); map.set(f.id, r);
  }
  for (const [keep, gone] of local.joins) { // merges done on this device
    const k = map.get(local.resolve(keep)) ?? C.resolve(keep), g = C.resolve(gone);
    if (C.fish.has(k) && C.fish.has(g) && k !== g && !C.covis.has(pair(k, g))) C.merge(k, g);
  }
  for (const [l, c] of map) if (c != null && !C.fish.has(c)) map.set(l, C.resolve(c));
  // ponytail: greedy best-first matching; Hungarian if a pond ever has many near-identical proposals at once
  const cands = [];
  for (const f of fresh) for (const r of C.fish.values()) { const s = C.pairSim(f, r); if (s >= C.cfg.mergeSim) cands.push([s, f, r.id]); }
  cands.sort((a, b) => b[0] - a[0]);
  const apart = (f, r) => [...map].some(([l, c]) => c === r && (l === f.id || local.covis.has(pair(l, f.id))));
  for (const [, f, r] of cands) if (!map.has(f.id) && !apart(f, r)) { absorb(C.fish.get(r), f); map.set(f.id, r); }
  for (const f of fresh) if (!map.has(f.id)) {
    const n = C.nextFish++, taken = new Set([...C.fish.values()].map(x => x.name));
    const name = f.autoName === false || !taken.has(f.name) ? f.name : autoName(f.variety, taken);
    C.fish.set(n, { ...f, id: n, name, nameVariety: name === f.name ? f.nameVariety : f.variety, ex: f.ex.slice(), c: true });
    map.set(f.id, n); added.add(n);
  }
  for (const f of C.fish.values()) f.c = true;
  C.joins = []; // applied here; the central copy carries no pending work
  for (const p of local.covis) {
    const [a, b] = p.split('|').map(Number).map(x => map.get(x));
    if (a != null && b != null && a !== b) C.covis.add(pair(a, b));
  }
  if (local.cmK) C.cmK = local.cmK;
  if (local.cfg.population) C.cfg.population = local.cfg.population;
  if (C.size.a == null && local.size.a != null) { C.size.a = local.size.a; C.size.b = local.size.b; }
  return { gallery: C.toJSON(), map, added };
}

// How many fish two pond galleries have in common (one-to-one, clearly the same look). Ponds that share several fish are
// one body of water, e.g. both sides of a footbridge: from above they look like two ponds, but the fish swim across.
export function sharedFish(a, b) {
  const current = x => ({ embedTag: x.embedTag, probeTag: x.probeTag ?? x.embedTag });
  if (!sameEmbedder(a.embedTag, current(b)) && !sameEmbedder(b.embedTag, current(a))) return 0;
  const pairs = [];
  for (const f of a.fish.values()) for (const g of b.fish.values()) { const s = a.pairSim(f, g); if (s >= a.cfg.mergeSim) pairs.push([s, f.id, g.id]); }
  pairs.sort((x, y) => y[0] - x[0]);
  const ua = new Set(), ub = new Set();
  for (const [, i, j] of pairs) if (!ua.has(i) && !ub.has(j)) { ua.add(i); ub.add(j); }
  return ua.size;
}

const pair = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
function histSim(a, b) { if (!a || !b) return 0.5; let d = 0; for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]); return 1 - d / 2; }
// seconds this track has been followed (tracker time); bare test/replay tracks count hits at 30 fps
const trackAge = t => (t.bornAt != null ? t.seenAt - t.bornAt : t.hits / 30);
function trackSize(t) {
  if (!t.sizes || t.sizes.length < 3) return null;
  const m = median(t.sizes), sd = median(t.sizes.map(s => Math.abs(s - m))) * 1.48 + 0.02;
  return { m, sd };
}
function addSize(sz, s) { // EMA mean/var (fish grow: old evidence fades)
  sz.n++; const a = Math.max(0.02, 1 / sz.n), d = s - sz.m;
  sz.m += a * d; sz.v = (1 - a) * (sz.v + a * d * d);
}

// Which pond are these fish from? items [{emb, hist}] (one photo or video pass) are scored with matchStill against the
// current pond's gallery `cur` and each of `others` ([[key, Identity]]). Another pond wins only when it names at least
// POND_SWITCH.min fish, POND_SWITCH.lead more than `cur` and POND_SWITCH.ratio times as many. -> {key, matchedFrom,
// matchedTo} or null (stay).
export const POND_SWITCH = { min: 2, lead: 2, ratio: 2 };
export function pondByFish(items, cur, others) {
  const named = g => g.matchStill(items).filter(r => r.fish != null).length;
  const here = named(cur);
  let best = null;
  for (const [key, g] of others) {
    const k = named(g);
    if (k >= POND_SWITCH.min && k >= here + POND_SWITCH.lead && k >= POND_SWITCH.ratio * here && (!best || k > best.matchedTo)) best = { key, matchedFrom: here, matchedTo: k };
  }
  return best;
}

// ---------- persistence (browser only) ----------
const DB = 'koi-id';
let dbp; // one connection for the page's lifetime
function idb() {
  return (dbp ||= new Promise((res, rej) => { const r = indexedDB.open(DB, 1); r.onupgradeneeded = () => r.result.createObjectStore('kv'); r.onsuccess = () => res(r.result); r.onerror = () => { dbp = null; rej(r.error); }; }));
}
// one gallery per pond (key 'gallery:<pond id>'); 'gallery' is the single-pond key from before ponds existed
export async function saveGallery(id, key = 'gallery') {
  const db = await idb();
  await new Promise((res, rej) => { const tx = db.transaction('kv', 'readwrite'); tx.objectStore('kv').put(id.toJSON(), key); tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
}
export async function loadGallery(key = 'gallery') {
  try {
    const db = await idb();
    return await new Promise(res => { const r = db.transaction('kv').objectStore('kv').get(key); r.onsuccess = () => res(r.result); r.onerror = () => res(null); });
  } catch { return null; }
}

// ---------- photo mode (one still, no tracks) ----------
// detection in processing px -> geometry as in app.js process(): mask px -> frame px; a still has no motion, so the head
// is at the wide end of the body
export function stillGeo(d, W, H) {
  const g = maskGeometry(d.mask, d.mw, d.mh);
  if (!g) return null;
  return { ...g, cx: d.box[0] + g.cx / d.msc, cy: d.box[1] + g.cy / d.msc, len: g.len / d.msc,
    edge: d.box[0] <= 2 || d.box[1] <= 2 || d.box[2] >= W - 2 || d.box[3] >= H - 2,
    heading: g.theta + ((g.widthSign || 1) < 0 ? Math.PI : 0) };
}

// The photos of one session: gallery fish seen, plus fish the gallery does not know, kept in memory only until learn() puts
// the clear ones into the gallery. An unknown gets a 2-digit hex ID that stays the same when a later photo shows it again. A
// fish between two candidates gets an ID for that photo only (owner, 27 Oct 2026: an ID, never '?'); it is not kept as an
// unknown (it would compete with the gallery fish it may be in later photos) nor counted. Every hex of the session is unique.
// Photos teach the gallery (27 Oct 2026, owner): LEARN sets how clear a match must be before it is learned from.
// q: a photo has one view per fish, the video path picks sharp frames out of hundreds (its gate is q > 0.3); measured on a
// pond photo, crop q ran 0.08-0.31 (median 0.14), so 0.3 learned from 1 fish in 15. The match gate already proves the fish.
export const LEARN = { sim: 0.75, lead: 0.10, q: 0.12, shots: 2 };
export class StillSession {
  constructor() { this.unknown = []; this.seen = new Set(); this.shots = 0; this.used = new Set(); this.hexOf = new Map(); this.shotFish = new Map(); }
  hex() {
    if (this.used.size >= 256) return null;
    let h; do h = Math.floor(Math.random() * 256).toString(16).toUpperCase().padStart(2, '0'); while (this.used.has(h));
    this.used.add(h); return h;
  }
  // the hex a fish the centre has not accepted yet shows: the one it was promoted with, else a new one kept for the session
  fishHex(fid) {
    if (!this.hexOf.has(fid)) { const h = this.hex(); if (h) this.hexOf.set(fid, h); }
    return this.hexOf.get(fid) ?? null;
  }
  // items [{emb, hist, q?, thumb?}] of one photo -> per item {fish (gallery id) | hex (session unknown or this photo only),
  // score, second, temp?}. Clearly new (below newSim against every gallery fish and session unknown) -> a new session
  // unknown; in between -> an ID for this photo only. A matched unknown counts this photo and keeps its best thumbnail.
  label(identity, items) {
    this.shots++;
    const res = identity.matchStill(items, this.unknown), out = [];
    items.forEach((it, i) => {
      const r = res[i], u = r.fish != null && this.unknown.find(x => x.id === r.fish), at = { score: r.score, second: r.second };
      let h;
      if (u) {
        if (u.ex.length < identity.cfg.maxExemplars) u.ex.push({ e: it.emb });
        u.photos.add(this.shots);
        if (it.thumb && (it.q || 0) > (u.thumbQ || 0)) { u.thumb = it.thumb; u.thumbQ = it.q; }
        this.seen.add(u.id); out.push({ fish: null, hex: u.hex, ...at });
      } else if (r.fish != null) {
        this.seen.add(`f:${r.fish}`); out.push({ fish: r.fish, hex: null, ...at });
      } else if (Math.max(r.score, r.second) < identity.cfg.newSim && (h = this.hex())) {
        this.unknown.push({ id: `u:${h}`, hex: h, ex: [{ e: it.emb }], hist: it.hist, size: { n: 0 }, photos: new Set([this.shots]), thumb: it.thumb || null, thumbQ: it.q || 0 });
        this.seen.add(`u:${h}`); out.push({ fish: null, hex: h, ...at });
      } else out.push({ fish: null, hex: this.hex(), ...at, temp: true });
    });
    this.shotFish.set(this.shots, new Set(out.filter(o => o.fish != null).map(o => o.fish))); // seen together = different fish
    return out;
  }
  // after label() of the same photo (items and out in the same order). Clear matches teach their gallery fish: a fish
  // matched with score >= LEARN.sim, a lead over the runner-up >= LEARN.lead and a crop q >= LEARN.q gets the view as an
  // exemplar (addExemplar's novelty rule keeps the set diverse). An unknown seen in >= LEARN.shots photos, none of whose
  // exemplars scores matchSim against a gallery fish not seen with it, becomes a new fish (c false until the centre accepts it; the
  // session keeps its hex for it). Returns { added (exemplars), promoted: [{fish, hex}] }.
  learn(identity, items, out, now = Date.now()) {
    let added = 0;
    const promoted = [];
    items.forEach((it, i) => {
      const o = out[i];
      if (o?.fish == null || o.temp) return;
      if (!(o.score >= LEARN.sim && o.score - o.second >= LEARN.lead && it.q >= LEARN.q)) return;
      const f = identity.fish.get(o.fish);
      if (!f) return;
      f.last = now;
      if (identity.addExemplar(f, it.emb, 0, it.q)) added++;
    });
    // not promoted while any view of it would be shown as a gallery fish (matchSim, the name-on-screen rule); fish seen in
    // the same photo as it are different fish and do not count (with newSim here, 2 of 15 co-visible koi got in)
    for (const u of [...this.unknown]) {
      if (u.photos.size < LEARN.shots) continue;
      const covis = new Set([...u.photos].flatMap(p => [...(this.shotFish.get(p) || [])]));
      const gallery = [...identity.fish.values()].filter(f => f.ex.length && !covis.has(f.id));
      if (u.ex.some(x => gallery.some(f => identity.score({ embs: [x.e], hist: u.hist, sizes: [] }, f) >= identity.cfg.matchSim))) continue;
      const f = identity.newFish({ embs: u.ex.map(x => x.e), hist: u.hist, sizes: [], thumb: u.thumb, thumbQ: u.thumbQ }, now);
      this.unknown.splice(this.unknown.indexOf(u), 1);
      this.seen.delete(u.id); this.seen.add(`f:${f.id}`); this.hexOf.set(f.id, u.hex);
      for (const p of u.photos) this.shotFish.get(p)?.add(f.id);
      promoted.push({ fish: f.id, hex: u.hex });
    }
    // gallery fish in one photo are different fish: into identity.covis, so consolidate() and mergeGallery never join them
    let covis = 0;
    for (const fs of this.shotFish.values()) {
      const a = [...fs].filter(x => identity.fish.has(x));
      for (let i = 0; i < a.length; i++) for (let k = i + 1; k < a.length; k++) {
        const p = pair(a[i], a[k]);
        if (!identity.covis.has(p)) { identity.covis.add(p); covis++; }
      }
    }
    return { added, promoted, covis };
  }
  // gallery numbers moved (adopt after a central sync; map old -> new, null = deleted centrally): the session follows.
  // A deleted fish keeps its slot in the tally (x:<old id>), so the count does not move.
  remap(map) {
    const m = x => (map.has(x) ? map.get(x) : x);
    this.seen = new Set([...this.seen].map(k => {
      if (!k.startsWith('f:')) return k;
      const id = +k.slice(2), n = m(id);
      return n == null ? `x:${id}` : `f:${n}`;
    }));
    this.hexOf = new Map([...this.hexOf].filter(([id]) => m(id) != null).map(([id, h]) => [m(id), h]));
    for (const [p, fs] of this.shotFish) this.shotFish.set(p, new Set([...fs].map(m).filter(x => x != null)));
  }
  get tally() { return this.seen.size; }
}
