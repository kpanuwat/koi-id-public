// The heavy per-frame work (detector, canonical crops, embedder) behind a small message-style API.
// Runs inside worker.js (UI stays responsive) or, if workers/WebGPU-in-workers are unavailable, on the
// main thread through engine.js with exactly the same code.
import { Models, CROP_W, CROP_H } from './models.js';
import { isGlint } from './reid.js';

const toDataURL = async (canvas, type, q) => {
  const blob = await canvas.convertToBlob({ type, quality: q });
  return new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
};

// focus / motion blur of a whole frame: variance of the 4-neighbour Laplacian on a 160 px wide grey copy.
// Real pond video gives >= 1,200, the same frames blurred (sigma 3) 140-230 (train clips, 25 Oct 2026).
const small = new OffscreenCanvas(160, 90), sctx = small.getContext('2d', { willReadFrequently: true });
function sharpness(src) {
  const w = 160, h = Math.max(8, Math.round(160 * src.height / src.width)); small.height = h;
  sctx.drawImage(src, 0, 0, w, h);
  const d = sctx.getImageData(0, 0, w, h).data, g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
  let s = 0, s2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, l = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i]; s += l; s2 += l * l; n++;
  }
  return s2 / n - (s / n) ** 2;
}

export function createPipeline() {
  let models, frame, fctx, dets = [];
  const thumb = new OffscreenCanvas(112, 56), tctx = thumb.getContext('2d');
  const big = new OffscreenCanvas(CROP_W, CROP_H), bctx = big.getContext('2d');

  return {
    async load(onStatus) {
      models = await new Models().load(onStatus);
      return { embedTag: models.embedTag, probeTag: models.probeTag, backend: models.backend };
    },

    // bitmap: ImageBitmap of the current video frame (consumed). Returns dets without the heavy logits.
    async detect({ bitmap, W, H, snap }) {
      if (!frame || frame.width !== W || frame.height !== H) { frame = new OffscreenCanvas(W, H); fctx = frame.getContext('2d', { willReadFrequently: true }); }
      fctx.drawImage(bitmap, 0, 0, W, H); bitmap.close?.();
      dets = await models.detect(frame, W, H);
      dets.forEach((d, i) => { d.idx = i; });
      const frameBlob = snap && dets.length ? await frame.convertToBlob({ type: 'image/jpeg', quality: 0.85 }) : null;
      const frameSharp = frameBlob ? sharpness(frame) : null;
      return { dets: dets.map(d => ({ idx: d.idx, box: d.box, score: d.score, mask: d.mask, mw: d.mw, mh: d.mh, msc: d.msc, area: d.area })),
        hasPond: !!models.hasPond, pond: !!models.pond, glare: models.glare || 0, rawCount: models.rawCount, frameBlob, frameSharp };
    },

    // items: [{idx, geo:{len, cx, cy, straight, heading}, want:{thumb, img, crop}}] for detections of the last detect()
    // -> per item {glint, q, hist, emb?, thumb?, img?, crop?}; embeddings only for crops that pass the quality gate
    async analyze({ items }) {
      const out = [], xs = [];
      for (const it of items) {
        const d = dets[it.idx], g = it.geo;
        const c = models.crop(frame, d, g, g.heading);
        const glint = isGlint(c);
        const q = Math.min(1, g.len / 70) * Math.min(1, c.sharp / 25) * (1 - Math.min(1, 2 * c.glare)) * (g.straight > 0.85 ? 1 : 0.6) * d.score;
        const r = { glint, q, hist: c.hist };
        if (!glint && q >= 0.08) {
          xs.push([r, c.x]);
          if (it.want?.thumb) { tctx.drawImage(models.out, 0, 0, 112, 56); r.thumb = await toDataURL(thumb, 'image/jpeg', 0.8); }
          if (it.want?.img && q > 0.3) r.img = await models.out.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
          if (it.want?.crop) { bctx.drawImage(models.out, 0, 0); r.crop = await toDataURL(big, 'image/jpeg', 0.92); }
        }
        out.push(r);
      }
      const embs = await models.embed(xs.map(x => x[1]));
      xs.forEach(([r], i) => { r.emb = embs[i]; });
      return out;
    },

    // stored crops (Blob or data URL) -> embeddings with the current model (gallery migration)
    async warm() { return models.warm(); },
    async loadObjects() { return models.loadObjects(); },
    // everyday objects in the frame of the last detect() (sparse, see app.js)
    async objects() { return frame ? models.objects(frame, frame.width, frame.height) : []; },

    async embedImages({ srcs }) {
      return models.embed(await Promise.all(srcs.map(s => models.tensorFrom(s))));
    },
  };
}
