// ONNX models in the browser: RF-DETR-seg koi detector + DINOv2-S re-ID embedder,
// plus the canonical crop (body axis horizontal, head right, background removed).
import * as ort from './vendor/ort.webgpu.min.mjs';
import { colorClass, blobShape } from './reid.js';
import { iou } from './tracker.js';

ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
ort.env.logLevel = 'error'; // fp16 graph-optimiser warnings are noise, keep real errors visible
// iPhone/iPad (every iOS browser is WebKit): the tab is killed above roughly 1.5 GB, so run lean there (measured in WebKit
// 25 Oct 2026: 1.8 GB peak with all three models). Same test in app.js.
export const LITE = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 1);
// safe level from app.js (raised after a crash): 1+ = lean like an iPhone. Worker: from its URL; in page: from storage
const Q = new URLSearchParams(self.location?.search);
const SAFE = +(Q.get('safe') ?? (self.localStorage?.getItem('koi.safe') || 0)) || 0, LEAN = LITE || SAFE >= 1;
// WebGPU runs the models; wasm threads would only add memory (each thread its own stack on a shared heap)
ort.env.wasm.numThreads = self.crossOriginIsolated && !(LEAN && navigator.gpu && Q.get('ep') !== 'wasm') ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
export const CROP_W = 224, CROP_H = 112;
// lean devices run the embedder in ONE batch shape (zero crops pad it, longer lists go in chunks): WebGPU builds new
// pipelines for each new input shape, and the first batch of 2 mid-run spiked WebKit ~15 s after a pick to 1.1 GB page
// + 0.65 GB GPU (pending/v64/NOTES_B.md). The load's probe builds that shape. 0 = any batch size (desktop, as before).
// ?count=1: count only (owner, 31 Oct 2026): YOLOX-tiny boxes (Apache-2.0, 5 M params, ~20 MB), no embedder, no naming
const COUNT = Q.get('count') !== '0'; // the default since v0.064 (app.js COUNT_ONLY); ?count=0 = full app models
// ?count=1&id=1: the same detector plus the embedder, so photos get the pond gallery's permanent fish numbers
const IDS = COUNT && Q.get('id') === '1';
const EMB_B = +(Q.get('embB') ?? (LEAN ? 2 : 0)) || 0;

// models ship in <=20 MB parts (static hosts cap file size, train/split_models.py); join them back into one buffer
let partsIndex;
const PART = 20 * 1024 * 1024; // train/split_models.py
async function modelBytes(url, onBytes = () => {}) {
  partsIndex ||= fetch(new URL('./models/parts.json', import.meta.url)).then(r => r.json());
  const { parts: n, bytes } = (await partsIndex)[url.split('/').pop()] || {};
  if (!n) throw new Error(`no parts for ${url}`);
  // one buffer of the final size; parts download in parallel (a phone on 4G took 18 min one by one) and each is copied to
  // its place as it arrives, streamed so progress shows and the model is never held twice (memory peak on iPhones)
  const out = new Uint8Array(bytes);
  let got = 0;
  await Promise.all(Array.from({ length: n }, async (_, i) => {
    const r = await fetch(new URL(`${url}.part${i}`, import.meta.url));
    if (!r.ok) throw new Error(`${url}.part${i}: HTTP ${r.status}`);
    let o = i * PART;
    for (const rd = r.body.getReader(); ;) {
      const { done, value } = await rd.read(); if (done) break;
      out.set(value, o); o += value.byteLength; got += value.byteLength; onBytes(value.byteLength);
    }
  }));
  if (got !== bytes) throw new Error(`${url}: ${got} of ${bytes} bytes`);
  return out;
}

async function session(url, onBytes) { return sessionFrom(await modelBytes(url, onBytes), url); }
async function sessionFrom(bytes, url, eps = ['webgpu', 'wasm']) {
  for (const ep of Q.get('ep') ? [Q.get('ep')] : eps) { // ?ep=wasm: CPU only (iPhone crash test)
    if (ep === 'webgpu' && !navigator.gpu) continue;
    try { return { s: await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all', logSeverityLevel: 3 }), ep }; }
    catch (e) { console.warn(`[koi] ${ep} failed for ${url}`, e); }
  }
  throw new Error(`cannot load ${url}`);
}

export class Models {
  async load(onStatus = () => {}) {
    // one progress figure for both models (MB from parts.json), shown until ready
    partsIndex ||= fetch(new URL('./models/parts.json', import.meta.url)).then(r => r.json());
    const idx = await partsIndex, det = Q.get('det') || (COUNT ? 'koi_yx320' : LEAN ? 'koi_det_lite16' : 'koi_det'); // lite16: 312 px, fp16 attention Softmax (less memory traffic, the iPhone bottleneck)
    const total = idx[`${det}.onnx`].bytes + (COUNT && !IDS ? 0 : idx['embed.onnx'].bytes); let done = 0, shown = -1;
    const onBytes = k => { done += k; const pc = Math.floor(100 * done / total); if (pc !== shown) { shown = pc; onStatus(`กำลังโหลดโมเดล ${pc}% (${Math.round(done / 1e6)}/${Math.round(total / 1e6)} MB) ครั้งแรกเท่านั้น`); } };
    // iPhones (and devices after a crash) get the 312 px export of the same detector: about half the compute and memory
    // (full 432 px took 2.4 s per pass on an iPhone 16e and the tab was killed, field logs 25 Oct 2026)
    const [d, e] = await Promise.all([session(`./models/${det}.onnx`, onBytes), COUNT && !IDS ? null : modelBytes('./models/embed.onnx', onBytes)]);
    this.det = d; // fp16 weights, fp32 io
    this.R = this.det.s.inputMetadata?.[0]?.shape?.[2] || 432;
    this.yolox = det.startsWith('koi_yx'); // YOLOX export: one output [1, N, 6] = cx, cy, w, h, objectness, koi
    this.prep = new OffscreenCanvas(this.R, this.R);
    this.pctx = this.prep.getContext('2d', { willReadFrequently: true });
    if (COUNT && !IDS) { this.emb = null; this.embedTag = this.probeTag = 'count'; return this; }
    onStatus('กำลังเตรียมโมเดลจำหน้าปลา...');
    // file fingerprint from 16 evenly spaced 64 KB samples + the length (1 MB hashed, not 43 MB: this runs at the load's
    // memory peak, where iPhones get killed); any retrain changes weights all through the file
    const S = 65536, N = 16, sample = new Uint8Array(N * S + 8);
    for (let i = 0; i < N; i++) { const o = Math.floor(i * Math.max(0, e.length - S) / (N - 1)); sample.set(e.subarray(o, o + S), i * S); }
    new DataView(sample.buffer).setFloat64(N * S, e.length);
    const digest = await crypto.subtle.digest('SHA-256', sample);
    this.embedTag = `sha:${Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('').slice(0, 16)}`;
    // id=1 (owner A/B, 1 Nov 2026): the embedder on wasm so WebGPU holds one session only (v63.24 id mode died after 2-3
    // photos, count mode with the one detector session did not); ?embEp=webgpu puts it back on the GPU
    this.emb = await sessionFrom(e, './models/embed.onnx', IDS && Q.get('embEp') !== 'webgpu' ? ['wasm'] : undefined);
    // Keep the rounded device probe only to recognize galleries written before the file hash was added.
    // It is also the embedder's first run, so it builds the fixed EMB_B shape here, never during a run.
    const probe = await this.embed([new Float32Array(3 * CROP_W * CROP_H)]);
    this.probeTag = `${probe[0].length}:${Array.from(probe[0].slice(0, 4), v => v.toFixed(3)).join(',')}`;
    // no detector warm-up here: v55 tried one and iPhones died during load (it stacked on the load's own peak)
    this.out = new OffscreenCanvas(CROP_W, CROP_H);
    this.octx = this.out.getContext('2d', { willReadFrequently: true });
    return this;
  }
  // one zero-input detector pass, once: the first run builds the GPU pipelines (WebKit: +600 MB page, +550 MB GPU for
  // ~10 s, freed by the next idle GC). It must not stack on the load's peak (v55 warmed 2 s after load: died in load) nor
  // on live passes (no warm-up: iPhone tabs died 20-45 s into runs; v55 runs that got past it lasted). app.js spaces it.
  async warm() {
    if (this.warmed) return false; this.warmed = true;
    const R = this.R; await this.det.s.run({ [this.det.s.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * R * R), [1, 3, R, R]) });
    return true;
  }
  get backend() { return `${this.det.ep}/${this.emb?.ep || '-'}`; }

  // everyday objects around the pond (COCO-pretrained RF-DETR Nano, 80 classes): loaded after the fish models, optional
  async loadObjects() {
    this.coco = await session('./models/coco_det.onnx');
    this.cocoNames = await (await fetch(new URL('./models/coco_classes.json', import.meta.url))).json();
    const R = this.coco.s.inputMetadata?.[0]?.shape?.[2] || 384;
    this.cprep = new OffscreenCanvas(R, R); this.cctx = this.cprep.getContext('2d', { willReadFrequently: true });
    return true;
  }
  // src -> [{name (COCO English), score, cx, cy, w, h}] in src px, one per object (class-wise NMS)
  async objects(src, W, H, minScore = 0.5) {
    const R = this.cprep.width;
    this.cctx.drawImage(src, 0, 0, R, R);
    const px = this.cctx.getImageData(0, 0, R, R).data, n = R * R, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[c * n + i] = (px[4 * i + c] / 255 - MEAN[c]) / STD[c];
    const out = await this.coco.s.run({ [this.coco.s.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, R, R]) });
    const boxes = out.dets.data, logits = out.labels.data, [, Q, C] = out.labels.dims, found = [];
    for (let q = 0; q < Q; q++) {
      let bc = 0, bl = -Infinity;
      for (let c = 1; c < C; c++) if (logits[q * C + c] > bl) { bl = logits[q * C + c]; bc = c; }
      const score = 1 / (1 + Math.exp(-bl)), name = this.cocoNames[bc];
      if (score < minScore || !name) continue;
      const [cx, cy, bw, bh] = [0, 1, 2, 3].map(k => boxes[q * 4 + k]);
      found.push({ name, score, cx: cx * W, cy: cy * H, w: bw * W, h: bh * H, box: [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H] });
    }
    found.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const o of found) if (keep.every(k => k.name !== o.name || iou(k.box, o.box) < 0.5)) keep.push(o);
    return keep.map(({ box, ...o }) => o);
  }

  // src: canvas/video of size W x H -> [{box:[x0,y0,x1,y1], score, mask:Uint8Array, mw, mh, msc}] in src px.
  // Only fish inside the pond are returned (2-class model: koi + pond water). this.pond keeps the water mask.
  async detect(src, W, H, minScore = 0.2) {
    if (this.yolox) return this.detectYolox(src, W, H, minScore);
    const R = this.R;
    this.pctx.drawImage(src, 0, 0, R, R); // RF-DETR uses a plain square resize
    const px = this.pctx.getImageData(0, 0, R, R).data, x = new Float32Array(3 * R * R);
    let sr = 0, sg = 0, sb = 0;
    for (let i = 0, n = R * R; i < n; i++) {
      for (let c = 0; c < 3; c++) x[c * n + i] = (px[4 * i + c] / 255 - MEAN[c]) / STD[c];
      sr += px[4 * i]; sg += px[4 * i + 1]; sb += px[4 * i + 2];
    }
    // light/white-balance compensation for colour features only (never applied to the image itself):
    // gray-world gains + exposure gain, smoothed over frames
    const n = R * R, m = (sr + sg + sb) / (3 * n), cl = (v, a, b) => Math.min(b, Math.max(a, v));
    const exp = cl(110 / Math.max(m, 1), 0.7, 1.6), g = [m / Math.max(sr / n, 1), m / Math.max(sg / n, 1), m / Math.max(sb / n, 1)].map(v => cl(v, 0.75, 1.33) * exp);
    this.gains = this.gains ? this.gains.map((v, k) => 0.9 * v + 0.1 * g[k]) : g;
    const out = await this.det.s.run({ [this.det.s.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, R, R]) });
    const boxes = out.dets.data, logits = out.labels.data, masks = out.masks.data;
    const [, Q, C] = out.labels.dims, [, , Hm, Wm] = out.masks.dims;
    const POND = C >= 3 ? 1 : -1, KOI = 0;
    this.hasPond = POND >= 0; // rfdetr drops the root category: koi=0, pond=1, last = no-object
    const sig = v => 1 / (1 + Math.exp(-v));
    // pond: best pond query's mask on the model grid, dilated one cell so fish on the edge still count
    if (POND >= 0) {
      let bq = -1, bs = 0.4;
      for (let q = 0; q < Q; q++) { const s = sig(logits[q * C + POND]); if (s > bs) { bs = s; bq = q; } }
      if (bq >= 0) {
        const pm = new Uint8Array(Hm * Wm), base = bq * Hm * Wm;
        for (let j = 0; j < Hm; j++) for (let i = 0; i < Wm; i++) if (masks[base + j * Wm + i] > 0)
          for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) { const jj = j + dj, ii = i + di; if (jj >= 0 && jj < Hm && ii >= 0 && ii < Wm) pm[jj * Wm + ii] = 1; }
        this.pond = { m: pm, Wm, Hm, score: bs, age: 0 };
      } else if (this.pond && ++this.pond.age > 15) this.pond = null; // keep last pond briefly through a bad frame
    }
    // share of pond pixels that are blown-out sky reflection (near-white, all channels high)
    if (this.pond) {
      let pn = 0, pg = 0;
      for (let j = 0; j < R; j += 2) for (let i = 0; i < R; i += 2) {
        if (!this.pond.m[((j / R * Hm) | 0) * Wm + ((i / R * Wm) | 0)]) continue;
        const k = 4 * (j * R + i); pn++;
        if (px[k] > 225 && px[k + 1] > 225 && px[k + 2] > 225) pg++;
      }
      this.glare = pn ? pg / pn : 0;
    }
    const inPond = (fx, fy) => !this.pond || this.pond.m[Math.min(Hm - 1, (fy / H * Hm) | 0) * Wm + Math.min(Wm - 1, (fx / W * Wm) | 0)];
    const dets = [];
    for (let q = 0; q < Q; q++) {
      const score = sig(POND >= 0 ? logits[q * C + KOI] : Math.max(...Array.from({ length: C }, (_, c) => logits[q * C + c])));
      if (score < minScore) continue;
      const [cx, cy, bw, bh] = [0, 1, 2, 3].map(k => boxes[q * 4 + k]);
      const x0 = Math.max(0, (cx - bw / 2) * W), y0 = Math.max(0, (cy - bh / 2) * H);
      const x1 = Math.min(W, (cx + bw / 2) * W), y1 = Math.min(H, (cy + bh / 2) * H);
      if (x1 - x0 < 4 || y1 - y0 < 4) continue;
      if (!inPond((x0 + x1) / 2, (y0 + y1) / 2)) { this.outside = (this.outside || 0) + 1; continue; } // leaves, plants, rim
      // bilinear-sample this query's mask logits over its box (cap 96 px on the long side)
      const msc = Math.min(1, 96 / Math.max(x1 - x0, y1 - y0)), mw = Math.max(2, Math.round((x1 - x0) * msc)), mh = Math.max(2, Math.round((y1 - y0) * msc));
      const mask = new Uint8Array(mw * mh), base = q * Hm * Wm;
      for (let j = 0; j < mh; j++) for (let i = 0; i < mw; i++) {
        const u = ((x0 + (i + 0.5) / msc) / W) * Wm - 0.5, v = ((y0 + (j + 0.5) / msc) / H) * Hm - 0.5;
        const iu = Math.max(0, Math.min(Wm - 2, Math.floor(u))), iv = Math.max(0, Math.min(Hm - 2, Math.floor(v)));
        const fu = Math.min(1, Math.max(0, u - iu)), fv = Math.min(1, Math.max(0, v - iv)), k = base + iv * Wm + iu;
        const val = (masks[k] * (1 - fu) + masks[k + 1] * fu) * (1 - fv) + (masks[k + Wm] * (1 - fu) + masks[k + Wm + 1] * fu) * fv;
        mask[j * mw + i] = val > 0 ? 1 : 0;
      }
      let a = 0; for (let i = 0; i < mask.length; i++) a += mask[i];
      dets.push({ box: [x0, y0, x1, y1], score, mask, mw, mh, msc, area: a / (msc * msc) });
    }
    // DETR can emit 2 queries for one fish; duplicates spawn extra tracks. Class-agnostic NMS.
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (keep.every(k => iou(k.box, d.box) < 0.6)) keep.push(d);
    this.rawCount = dets.length;
    return keep;
  }

  // YOLOX (pending/yolox): letterbox into R x R at the top-left, pad grey 114, BGR 0-255, no normalisation (as trained).
  // Boxes only: the mask is the ellipse inside the box, so the tracker, areas and the overlay work unchanged. No pond class.
  async detectYolox(src, W, H, minScore) {
    const R = this.R, r = Math.min(R / W, R / H), n = R * R;
    this.pctx.fillStyle = 'rgb(114,114,114)'; this.pctx.fillRect(0, 0, R, R);
    this.pctx.drawImage(src, 0, 0, Math.round(W * r), Math.round(H * r));
    const px = this.pctx.getImageData(0, 0, R, R).data, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { x[i] = px[4 * i + 2]; x[n + i] = px[4 * i + 1]; x[2 * n + i] = px[4 * i]; }
    const out = await this.det.s.run({ [this.det.s.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, R, R]) });
    const o = out[this.det.s.outputNames[0]], v = o.data, N = o.dims[1], dets = [];
    this.hasPond = false; this.pond = null; this.glare = 0;
    for (let q = 0; q < N; q++) {
      const score = v[q * 6 + 4] * v[q * 6 + 5];
      if (score < minScore) continue;
      const [cx, cy, bw, bh] = [0, 1, 2, 3].map(k => v[q * 6 + k] / r);
      const x0 = Math.max(0, cx - bw / 2), y0 = Math.max(0, cy - bh / 2), x1 = Math.min(W, cx + bw / 2), y1 = Math.min(H, cy + bh / 2);
      if (x1 - x0 < 4 || y1 - y0 < 4) continue;
      const msc = Math.min(1, 96 / Math.max(x1 - x0, y1 - y0)), mw = Math.max(2, Math.round((x1 - x0) * msc)), mh = Math.max(2, Math.round((y1 - y0) * msc));
      // id mode: the crop's rotation needs the fish's own axis (pending/yxid: the ellipse takes the box's axis, > 15 deg off
      // for 64 % of views, and halved top-1); count only: the ellipse is enough
      let mask = IDS ? colourMask(src, x0, y0, x1, y1, mw, mh) : null;
      if (!mask) {
        mask = new Uint8Array(mw * mh);
        for (let j = 0; j < mh; j++) for (let i = 0; i < mw; i++) { const u = (i + 0.5) / mw - 0.5, w = (j + 0.5) / mh - 0.5; if (u * u + w * w <= 0.25) mask[j * mw + i] = 1; }
      }
      let a = 0; for (let i = 0; i < mask.length; i++) a += mask[i];
      dets.push({ box: [x0, y0, x1, y1], score, mask, mw, mh, msc, area: a / (msc * msc) });
    }
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (keep.every(k => iou(k.box, d.box) < 0.6)) keep.push(d);
    this.rawCount = dets.length;
    return keep;
  }

  // canonical crop for one detection -> {x: Float32Array 3*CROP_H*CROP_W, hist[7], glare, sharp}
  crop(frame, det, geo, heading) {
    const [x0, y0, x1, y1] = det.box, bw = Math.ceil(x1 - x0), bh = Math.ceil(y1 - y0);
    const mc = new OffscreenCanvas(det.mw, det.mh), mctx = mc.getContext('2d'), mi = mctx.createImageData(det.mw, det.mh);
    for (let i = 0; i < det.mask.length; i++) mi.data[4 * i + 3] = det.mask[i] * 255;
    mctx.putImageData(mi, 0, 0);
    const fc = new OffscreenCanvas(bw, bh), fctx = fc.getContext('2d', { willReadFrequently: true });
    fctx.drawImage(frame, x0, y0, bw, bh, 0, 0, bw, bh);
    fctx.globalCompositeOperation = 'destination-in';
    fctx.drawImage(mc, 0, 0, bw, bh);
    // colour histogram + glare on the fish pixels only
    const d = fctx.getImageData(0, 0, bw, bh).data, hist = new Array(7).fill(0);
    let n = 0, glare = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] < 128) continue;
      const G = this.gains || [1, 1, 1];
      const [h, s, v] = hsv(Math.min(255, d[i] * G[0]), Math.min(255, d[i + 1] * G[1]), Math.min(255, d[i + 2] * G[2]));
      n++;
      if (v > 0.97 && s < 0.06) { glare++; continue; }
      hist[colorClass(h, s, v)]++;
    }
    const hn = Math.max(1, n - glare);
    const o = this.octx;
    o.setTransform(1, 0, 0, 1, 0, 0);
    o.fillStyle = 'rgb(124,116,104)'; // ImageNet mean => zero after normalisation
    o.fillRect(0, 0, CROP_W, CROP_H);
    const sc = (0.88 * CROP_W) / Math.max(geo.len, 4);
    o.translate(CROP_W / 2, CROP_H / 2); o.rotate(-heading); o.scale(sc, sc);
    o.drawImage(fc, -(geo.cx - x0), -(geo.cy - y0));
    o.setTransform(1, 0, 0, 1, 0, 0);
    const p = o.getImageData(0, 0, CROP_W, CROP_H).data, N = CROP_W * CROP_H, x = new Float32Array(3 * N);
    let lap = 0, lapN = 0;
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) x[c * N + i] = (p[4 * i + c] / 255 - MEAN[c]) / STD[c];
      const r = (i / CROP_W) | 0, cc = i % CROP_W;
      if (r > 0 && r < CROP_H - 1 && cc > 0 && cc < CROP_W - 1 && (i & 3) === 0) {
        const g = k => p[4 * k] + p[4 * k + 1] + p[4 * k + 2];
        lap += (4 * g(i) - g(i - 1) - g(i + 1) - g(i - CROP_W) - g(i + CROP_W)) ** 2; lapN++;
      }
    }
    return { x, hist: hist.map(v => v / hn), glare: n ? glare / n : 1, sharp: Math.sqrt(lap / Math.max(1, lapN)) / 3, ...blobShape(p) };
  }

  // stored crop image (Blob or data URL, any size) -> model input, for re-embedding after a model update
  async tensorFrom(src) {
    const bmp = await createImageBitmap(typeof src === 'string' ? await (await fetch(src)).blob() : src);
    const o = this.octx; o.setTransform(1, 0, 0, 1, 0, 0); o.drawImage(bmp, 0, 0, CROP_W, CROP_H); bmp.close?.();
    const p = o.getImageData(0, 0, CROP_W, CROP_H).data, N = CROP_W * CROP_H, x = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) x[c * N + i] = (p[4 * i + c] / 255 - MEAN[c]) / STD[c];
    return x;
  }

  async embed(xs) {
    if (!xs.length) return [];
    const N = CROP_W * CROP_H * 3, B = EMB_B || xs.length, res = [];
    for (let k = 0; k < xs.length; k += B) {
      const part = xs.slice(k, k + B), buf = new Float32Array(B * N);
      part.forEach((x, i) => buf.set(x, i * N));
      const out = await this.emb.s.run({ pixel_values: new ort.Tensor('float32', buf, [B, 3, CROP_H, CROP_W]) });
      const D = out.embedding.dims[1];
      part.forEach((_, i) => res.push(out.embedding.data.slice(i * D, (i + 1) * D)));
    }
    return res;
  }
}

function hsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d) h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(h * 60 + 360) % 360, mx ? d / mx : 0, mx];
}

// Cheap fish mask inside a YOLOX box at the det-mask grid (mw x mh, <= 96 px), port of pending/yxid/run.py colour_mask:
// Lab distance to the water colour (median of a 2-cell ring along the box border, L weighted 0.5), Otsu threshold, 3x3
// close, largest 8-connected blob, holes filled. null when the blob is < 8 % or > 92 % of the box (no contrast).
// Exported for the JS/Python parity check (mask IoU 0.85-0.99 on the test stills, 31 Oct 2026).
let cmCanvas = null, cmCtx = null;
export function colourMask(src, x0, y0, x1, y1, mw, mh) {
  if (!cmCanvas || cmCanvas.width < mw || cmCanvas.height < mh) { cmCanvas = new OffscreenCanvas(96, 96); cmCtx = cmCanvas.getContext('2d', { willReadFrequently: true }); }
  cmCtx.drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, mw, mh);
  const px = cmCtx.getImageData(0, 0, mw, mh).data, n = mw * mh, L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  const lin = v => (v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4, f = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  for (let i = 0; i < n; i++) {
    const r = lin(px[4 * i]), g = lin(px[4 * i + 1]), b = lin(px[4 * i + 2]);
    const fx = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047), fy = f(0.2126 * r + 0.7152 * g + 0.0722 * b), fz = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
    L[i] = (116 * fy - 16) * 2.55; A[i] = 500 * (fx - fy); B[i] = 200 * (fy - fz); // OpenCV 8-bit Lab scale (offsets cancel)
  }
  const ring = [];
  for (let j = 0; j < mh; j++) for (let i = 0; i < mw; i++) if (j < 2 || j >= mh - 2 || i < 2 || i >= mw - 2) ring.push(j * mw + i);
  const med = ch => { const v = ring.map(k => ch[k]).sort((p, q) => p - q), m = v.length >> 1; return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; };
  const wl = med(L), wa = med(A), wb = med(B), d = new Float32Array(n);
  let dmax = 1e-6;
  for (let i = 0; i < n; i++) { d[i] = Math.sqrt(0.5 * (L[i] - wl) ** 2 + (A[i] - wa) ** 2 + (B[i] - wb) ** 2); if (d[i] > dmax) dmax = d[i]; }
  const d8 = new Uint8Array(n), hist = new Float64Array(256);
  for (let i = 0; i < n; i++) { d8[i] = Math.min(255, Math.max(0, d[i] * 255 / dmax)) | 0; hist[d8[i]]++; }
  let sum = 0; for (let t = 0; t < 256; t++) sum += t * hist[t];
  let wB = 0, sB = 0, best = -1, thr = 0; // Otsu
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue; const wF = n - wB; if (!wF) break;
    sB += t * hist[t]; const mB = sB / wB, mF = (sum - sB) / wF, v = wB * wF * (mB - mF) ** 2;
    if (v > best) { best = v; thr = t; }
  }
  let m = new Uint8Array(n); for (let i = 0; i < n; i++) m[i] = d8[i] > thr ? 1 : 0;
  const morph = (src, dil) => { // 3x3, pixels outside the box ignored (OpenCV's default border for dilate / erode)
    const out = new Uint8Array(n);
    for (let j = 0; j < mh; j++) for (let i = 0; i < mw; i++) {
      let v = dil ? 0 : 1;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const jj = j + dj, ii = i + di; if (jj < 0 || jj >= mh || ii < 0 || ii >= mw) continue;
        if (dil) v |= src[jj * mw + ii]; else v &= src[jj * mw + ii];
      }
      out[j * mw + i] = v;
    }
    return out;
  };
  m = morph(morph(m, true), false);
  const lab = new Int32Array(n), stack = []; let bestK = 0, bestA = 0, k = 0; // largest 8-connected blob
  for (let s0 = 0; s0 < n; s0++) {
    if (!m[s0] || lab[s0]) continue;
    k++; let a = 0; lab[s0] = k; stack.push(s0);
    while (stack.length) {
      const p = stack.pop(), pj = (p / mw) | 0, pi = p - pj * mw; a++;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const jj = pj + dj, ii = pi + di; if (jj < 0 || jj >= mh || ii < 0 || ii >= mw) continue;
        const q = jj * mw + ii; if (m[q] && !lab[q]) { lab[q] = k; stack.push(q); }
      }
    }
    if (a > bestA) { bestA = a; bestK = k; }
  }
  if (!bestK) return null;
  const out = new Uint8Array(n), seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = lab[i] === bestK ? 1 : 0;
  for (let j = 0; j < mh; j++) for (let i = 0; i < mw; i++) if ((j === 0 || j === mh - 1 || i === 0 || i === mw - 1) && !out[j * mw + i] && !seen[j * mw + i]) { seen[j * mw + i] = 1; stack.push(j * mw + i); }
  while (stack.length) { // background reached from the border (4-connected, as floodFill); the rest are holes
    const p = stack.pop(), pj = (p / mw) | 0, pi = p - pj * mw;
    for (const [jj, ii] of [[pj - 1, pi], [pj + 1, pi], [pj, pi - 1], [pj, pi + 1]]) {
      if (jj < 0 || jj >= mh || ii < 0 || ii >= mw) continue;
      const q = jj * mw + ii; if (!out[q] && !seen[q]) { seen[q] = 1; stack.push(q); }
    }
  }
  let a = 0; for (let i = 0; i < n; i++) { if (!out[i] && !seen[i]) out[i] = 1; a += out[i]; }
  return a < 0.08 * n || a > 0.92 * n ? null : out;
}
