// Boxes that follow the fish between detector passes (an iPhone manages ~1.5 passes a second, each ~0.55 s late; moving
// a box along its per-pass velocity drifts off a fish that turns, stops or is shaken by the hand). Cheap CPU block
// matching on the main thread, detector passes stay the truth:
// - each video frame (at most HZ a second) becomes a small gray picture (shorter side ~135 px) kept in a ring of RING
//   reused buffers: the luma plane of a VideoFrame (copyTo is ~2 ms in WebKit; drawImage+getImageData of a video costs
//   ~7 ms whatever the canvas size), else a small canvas;
// - a detector pass describes the frame it started on (the "key"). When its boxes arrive, every counted track gets a
//   template from the key frame: gray levels under the fish's own mask pixels (box corners are water and would pin the
//   template to the background), <= MAXP points. The template is then followed through the ring frames since the key,
//   so the box is where the fish is NOW, not where it was when the pass began;
// - per frame each template is searched (sum of absolute differences) within +/-R px of its last place. A poor,
//   ambiguous or implausible match (glare, another fish over it) moves it with the camera shake instead (median shift of
//   the good matches); after MISS_MS of misses the app falls back to its velocity prediction.
// Nothing here touches tracks: it only says where to DRAW them.
const SHORT = 135, MAXP = 120, MINP = 12, COST = 40, RATIO = 0.85, MISS_MS = 350, RING = 32;
// a followed fish stays within REACH0 box diagonals of its detector box, plus REACH_S diagonals per second since the key
const REACH0 = 1, REACH_S = 1.5;

export class Follow {
  constructor({ hz = 30, R = 4, short = SHORT, mask = true, vf = true, dump = false } = {}) {
    this.R = R; this.short = short; this.mask = mask; this.useVF = vf && typeof VideoFrame === 'function';
    this.minMs = hz > 0 ? 1000 / hz - 6 : 0;
    this.W = 0; this.H = 0; this.F = 1; this.vw = 0; this.vh = 0; this.ring = []; this.ringAt = new Float64Array(RING); this.cam = new Float32Array(2 * RING);
    this.head = -1; this.keySlot = -1; this.keyN = 0; // keyN: frames written since the key (the key is lost after RING - 1)
    this.items = new Map(); this.pool = []; this.buf = null; this.acc = null; this.cv = null; this.ctx = null;
    this.surf = new Float32Array((2 * R + 1) ** 2); this.m = { x: 0, y: 0, cost: 0, ratio: 0, edge: false };
    this.disp = new Float32Array(64); this.lumaFull = false; this.busy = false; this.vfFail = 0; this.waitMs = 0;
    this.costs = dump ? [] : null; this.resetCosts = dump ? [] : null; // ms per frame / per pass reset (probe only)
  }

  size(vw, vh) {
    if (vw === this.vw && vh === this.vh) return;
    const F = Math.max(1, Math.round(Math.min(vw, vh) / this.short));
    this.vw = vw; this.vh = vh; this.F = F; this.W = Math.floor(vw / F); this.H = Math.floor(vh / F);
    this.ring = Array.from({ length: RING }, () => new Uint8Array(this.W * this.H));
    this.acc = new Float32Array(this.W); this.buf = null; this.cv = null; this.clear();
  }

  clear() { for (const st of this.items.values()) this.pool.push(st); this.items.clear(); this.keySlot = -1; this.head = -1; }

  // one video frame: into the ring, every template followed on it; isKey: a detector pass starts on this frame
  async frame(video, now, isKey) {
    if (isKey) this.keySlot = -1; // a new pass: the old key no longer matches its boxes, even if this grab fails
    const vw = video.videoWidth, vh = video.videoHeight;
    // the canvas fallback reads the video back (~7 ms in WebKit): at most 15 frames a second there
    const minMs = this.useVF ? this.minMs : Math.max(this.minMs, 1000 / 15 - 6);
    if (!vw || !vh || this.busy || (!isKey && this.head >= 0 && now - this.ringAt[this.head] < minMs)) return;
    this.size(vw, vh);
    this.busy = true;
    const t0 = performance.now();
    let ok = false;
    this.waitMs = 0;
    try {
      const slot = (this.head + 1) % RING, g = this.ring[slot];
      if (this.useVF) {
        const r = await this.grabVF(video, g);
        this.vfFail = r === null ? this.vfFail + 1 : 0; ok = !!r;
        if (this.vfFail >= 3) this.useVF = false; // not usable here (format, API): the canvas for good
      }
      else ok = this.grabCanvas(video, g);
      if (vw !== this.vw || vh !== this.vh) ok = false; // rotated meanwhile
      if (!ok) { this.clear(); return; }
      this.head = slot; this.ringAt[slot] = now;
      if (this.keySlot >= 0 && ++this.keyN >= RING) this.keySlot = -1; // the key frame was overwritten: too late to follow
      this.step(g, slot, now);
      if (isKey) { this.keySlot = slot; this.keyN = 0; }
    } finally { this.busy = false; }
    // main-thread time only: the wait for the copy (it may finish off-thread, other work runs meanwhile) is left out
    if (this.costs && this.costs.length < 50000) this.costs.push(performance.now() - t0 - this.waitMs);
  }

  // VideoFrame luma -> small gray (F x F block means). null: failed (3 in a row: the canvas takes over for good)
  async grabVF(video, g) {
    let f;
    try { f = new VideoFrame(video); } catch { return null; }
    try {
      const fmt = f.format || '', yuv = /^(NV12|I420|I420A|I422|I444)$/.test(fmt), rgb = /^(RGB|BGR)[AX]$/.test(fmt);
      if (!yuv && !rgb) return null;
      const n = f.allocationSize();
      if (!this.buf || this.buf.length < n) this.buf = new Uint8Array(n);
      const pending = f.copyTo(this.buf), tw = performance.now(), layout = await pending, vis = f.visibleRect;
      this.waitMs = performance.now() - tw;
      if (vis.width !== this.vw || vis.height !== this.vh) return false;
      const { offset, stride } = layout[0], F = this.F, W = this.W, H = this.H, b = this.buf, acc = this.acc, k = 1 / (F * F);
      const ri = fmt[0] === 'B' ? 2 : 0, bi = 2 - ri;
      for (let y = 0; y < H; y++) {
        acc.fill(0);
        for (let dy = 0; dy < F; dy++) {
          let p = offset + (y * F + dy) * stride;
          if (yuv) for (let x = 0; x < W; x++) { let s = 0; for (let dx = 0; dx < F; dx++) s += b[p++]; acc[x] += s; }
          else for (let x = 0; x < W; x++) { let s = 0; for (let dx = 0; dx < F; dx++, p += 4) s += (b[p + ri] * 54 + b[p + 1] * 183 + b[p + bi] * 19) >> 8; acc[x] += s; }
        }
        const o = y * W;
        for (let x = 0; x < W; x++) g[o + x] = acc[x] * k;
      }
      this.lumaFull = rgb || !!f.colorSpace?.fullRange;
      return true;
    } catch { return null; } finally { f.close(); }
  }

  grabCanvas(video, g) {
    const W = this.W, H = this.H, F = this.F;
    if (!this.cv) { this.cv = document.createElement('canvas'); this.cv.width = W; this.cv.height = H; this.ctx = this.cv.getContext('2d', { willReadFrequently: true, alpha: false }); }
    let px;
    try { this.ctx.drawImage(video, 0, 0, W * F, H * F, 0, 0, W, H); px = this.ctx.getImageData(0, 0, W, H).data; } catch { return false; }
    for (let i = 0, j = 0; i < W * H; i++, j += 4) g[i] = (px[j] * 54 + px[j + 1] * 183 + px[j + 2] * 19) >> 8;
    this.lumaFull = true;
    return true;
  }

  // follow every template on gray frame g (ring slot); misses move with the camera (median shift of the good matches)
  step(g, slot, now) {
    const m = this.m, D = this.disp;
    let n = 0;
    for (const st of this.items.values()) {
      st.good = this.match(st, g, st.x, st.y, this.R) && m.cost <= COST && m.ratio <= RATIO && !m.edge && this.sane(st, m.x, m.y, now);
      if (st.good) { st.mx = m.x; st.my = m.y; if (n < 32) { D[2 * n] = m.x - st.x; D[2 * n + 1] = m.y - st.y; n++; } }
    }
    const cx = n >= 3 ? median(D, n, 0) : 0, cy = n >= 3 ? median(D, n, 1) : 0;
    this.cam[2 * slot] = cx; this.cam[2 * slot + 1] = cy;
    for (const st of this.items.values()) this.advance(st, cx, cy, now);
  }

  advance(st, cx, cy, now) {
    if (st.good) { st.x = st.mx; st.y = st.my; st.miss = 0; st.ok = true; }
    else { st.x += cx; st.y += cy; st.miss += now - st.at; if (st.miss > MISS_MS) st.ok = false; }
    st.at = now;
  }

  // a followed box stays near its detector box (one box size, + 1.5 sizes per second since the key frame): else it
  // latched onto something else (water, another fish)
  sane(st, x, y, now) { return Math.hypot(x - st.dx, y - st.dy) <= st.reach * (REACH0 + REACH_S * (now - st.rAt) / 1000); }

  // detector pass result (boxes describe the key frame): templates from the key frame, followed through the ring frames
  // written since then (with the camera shifts measured on them) up to the newest one
  reset(tracks) {
    const t0 = performance.now();
    if (this.keySlot < 0) { this.clear(); return; }
    const old = this.items, next = new Map(), s = 1 / this.F, key = this.ring[this.keySlot], kAt = this.ringAt[this.keySlot];
    for (const t of tracks) {
      let st = old.get(t);
      if (st) old.delete(t); else st = this.pool.pop() || newState();
      if (!this.cut(st, t, key)) { this.pool.push(st); continue; }
      const b = t.box;
      st.x = st.dx = (b[0] + b[2]) / 2 * s; st.y = st.dy = (b[1] + b[3]) / 2 * s; st.rAt = st.at = kAt; st.miss = 0; st.ok = true;
      st.reach = Math.max(4, Math.hypot((b[2] - b[0]) * s, (b[3] - b[1]) * s));
      next.set(t, st);
    }
    for (const st of old.values()) this.pool.push(st);
    this.items = next;
    // catch up: the same steps as on live frames, on the frames stored since the key
    for (let i = 1; i <= this.keyN; i++) { const slot = (this.keySlot + i) % RING; this.step(this.ring[slot], slot, this.ringAt[slot]); }
    if (this.resetCosts && this.resetCosts.length < 10000) this.resetCosts.push(performance.now() - t0);
  }

  // template of track t from gray frame K: gray levels at mask pixels inside the box (sub-sampled to <= MAXP points);
  // a fish without a usable mask gets the middle 60 % of its box
  cut(st, t, K) {
    const b = t.box, d = t.det, W = this.W, H = this.H, s = 1 / this.F;
    const cx = (b[0] + b[2]) / 2 * s, cy = (b[1] + b[3]) / 2 * s, ix = Math.round(cx), iy = Math.round(cy);
    const gx0 = Math.max(0, Math.floor(b[0] * s)), gx1 = Math.min(W - 1, Math.ceil(b[2] * s) - 1);
    const gy0 = Math.max(0, Math.floor(b[1] * s)), gy1 = Math.min(H - 1, Math.ceil(b[3] * s) - 1);
    if (gx1 < gx0 || gy1 < gy0) return false;
    let n = 0;
    const add = (gx, gy) => { st.off[n] = (gy - iy) * W + (gx - ix); st.px[n] = gx - ix; st.py[n] = gy - iy; st.val[n] = K[gy * W + gx]; n++; };
    if (this.mask && d?.mask && d.mw && d.mh && d.msc) {
      const step = Math.max(1, Math.ceil(Math.sqrt(0.6 * (gx1 - gx0 + 1) * (gy1 - gy0 + 1) / MAXP)));
      for (let gy = gy0; gy <= gy1 && n < MAXP; gy += step) {
        const my = Math.floor(((gy + 0.5) / s - b[1]) * d.msc);
        if (my < 0 || my >= d.mh) continue;
        for (let gx = gx0; gx <= gx1 && n < MAXP; gx += step) {
          const mx = Math.floor(((gx + 0.5) / s - b[0]) * d.msc);
          if (mx >= 0 && mx < d.mw && d.mask[my * d.mw + mx]) add(gx, gy);
        }
      }
    }
    if (n < MINP) {
      n = 0;
      const ax = Math.round(gx0 + 0.2 * (gx1 - gx0)), bx = Math.round(gx1 - 0.2 * (gx1 - gx0));
      const ay = Math.round(gy0 + 0.2 * (gy1 - gy0)), by = Math.round(gy1 - 0.2 * (gy1 - gy0));
      const step = Math.max(1, Math.ceil(Math.sqrt((bx - ax + 1) * (by - ay + 1) / MAXP)));
      for (let gy = ay; gy <= by; gy += step) for (let gx = ax; gx <= bx && n < MAXP; gx += step) add(gx, gy);
      if (n < 4) return false;
    }
    st.n = n; st.fx = cx - ix; st.fy = cy - iy;
    let x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (let p = 0; p < n; p++) { x0 = Math.min(x0, st.px[p]); x1 = Math.max(x1, st.px[p]); y0 = Math.min(y0, st.py[p]); y1 = Math.max(y1, st.py[p]); }
    st.x0 = x0; st.x1 = x1; st.y0 = y0; st.y1 = y1;
    return true;
  }

  // SAD search of template st around centre (cx, cy) within +/-R in gray frame g -> this.m (sub-pixel centre, mean cost
  // per point, best/mean ratio over the window: near 1 = flat, ambiguous; edge: best on the window border)
  match(st, g, cx, cy, R) {
    const W = this.W, H = this.H, D = 2 * R + 1, S = this.surf, n = st.n, off = st.off, val = st.val;
    const bx = Math.round(cx - st.fx), by = Math.round(cy - st.fy);
    let best = Infinity, bi = 0, bj = 0, sum = 0, cnt = 0;
    for (let j = -R; j <= R; j++) {
      const y = by + j, rowOk = y + st.y0 >= 0 && y + st.y1 < H;
      for (let i = -R; i <= R; i++) {
        const x = bx + i, k = (j + R) * D + i + R;
        if (!rowOk || x + st.x0 < 0 || x + st.x1 >= W) { S[k] = -1; continue; }
        const base = y * W + x;
        let sad = 0;
        for (let p = 0; p < n; p++) { const v = g[base + off[p]] - val[p]; sad += v < 0 ? -v : v; }
        S[k] = sad; sum += sad; cnt++;
        if (sad < best) { best = sad; bi = i; bj = j; }
      }
    }
    if (cnt < D) return false;
    const k = (bj + R) * D + bi + R, m = this.m;
    m.x = bx + bi + st.fx + (Math.abs(bi) < R ? sub(S[k - 1], best, S[k + 1]) : 0);
    m.y = by + bj + st.fy + (Math.abs(bj) < R ? sub(S[k - D], best, S[k + D]) : 0);
    m.cost = best / n; m.ratio = best / Math.max(1e-6, sum / cnt); m.edge = Math.abs(bi) === R || Math.abs(bj) === R;
    return true;
  }

  // drawn centre of track t in video px (out[0], out[1]); false when it is not followed (the app predicts by velocity)
  pos(t, out) {
    const st = this.items.get(t);
    if (!st || !st.ok) return false;
    out[0] = st.x * this.F; out[1] = st.y * this.F; return true;
  }

  // mean luma (0-255, full range) of a video-px rectangle on the newest frame when it is fresh (label ink), else -1
  luma(x0, y0, x1, y1, now) {
    if (this.head < 0 || now - this.ringAt[this.head] > 250) return -1;
    const g = this.ring[this.head], F = this.F, W = this.W, H = this.H;
    const X0 = Math.max(0, Math.floor(x0 / F)), X1 = Math.min(W - 1, Math.ceil(x1 / F)), Y0 = Math.max(0, Math.floor(y0 / F)), Y1 = Math.min(H - 1, Math.ceil(y1 / F));
    let sum = 0, n = 0;
    for (let y = Y0; y <= Y1; y++) for (let x = X0; x <= X1; x++) { sum += g[y * W + x]; n++; }
    if (!n) return -1;
    return this.lumaFull ? sum / n : (sum / n - 16) * 255 / 219;
  }
}

const sub = (a, c, b) => (a < 0 || b < 0 ? 0 : Math.max(-0.5, Math.min(0.5, (a - b) / (2 * Math.max(1e-6, a - 2 * c + b)))));
const tmp = new Float32Array(32);
function median(D, n, o) { for (let i = 0; i < n; i++) tmp[i] = D[2 * i + o]; return tmp.subarray(0, n).sort()[n >> 1]; }

function newState() {
  return { off: new Int32Array(MAXP), px: new Int16Array(MAXP), py: new Int16Array(MAXP), val: new Uint8Array(MAXP), n: 0, fx: 0, fy: 0,
    x0: 0, x1: 0, y0: 0, y1: 0, x: 0, y: 0, mx: 0, my: 0, good: false, at: 0, miss: 0, ok: false, dx: 0, dy: 0, reach: 0, rAt: 0 };
}
