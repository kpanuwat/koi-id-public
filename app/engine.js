// Main-thread handle on the inference pipeline: a module worker when possible, same code in-thread otherwise.
import { createPipeline } from './pipeline.js';

export class Engine {
  async start(onStatus, { noWorker = false, query = '' } = {}) {
    if (!noWorker && typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined') {
      try {
        this.worker = new Worker(new URL(`./worker.js?${query}`, import.meta.url), { type: 'module' }); // model options (safe level) travel in the URL
        this.pending = new Map(); this.seq = 0;
        this.worker.onmessage = ({ data }) => {
          if (data.status) return onStatus(data.status);
          const p = this.pending.get(data.id); this.pending.delete(data.id);
          if (p) data.err ? p.reject(new Error(data.err)) : p.resolve(data.res);
        };
        // worker failed to load or died: settle every waiting call so startup falls back and frames report it
        const fail = err => { for (const p of this.pending.values()) p.reject(err); this.pending.clear(); };
        this.worker.onerror = e => { this.dead = new Error(e.message || 'inference worker stopped'); fail(this.dead); };
        this.worker.onmessageerror = () => fail(new Error('inference worker reply could not be read')); // one bad reply, worker still alive
        this.info = await this.call('load');
        this.mode = 'worker';
        return this;
      } catch (e) {
        console.warn('[koi] worker unavailable, running in page', e);
        this.worker?.terminate(); this.worker = null; this.dead = null;
      }
    }
    this.local = createPipeline();
    this.info = await this.local.load(onStatus);
    this.mode = 'page';
    return this;
  }

  // ONNX Runtime Web cannot run two inferences at once (fish models + objects model): one call at a time, in order
  call(op, arg, transfer = []) {
    const run = () => {
      if (!this.worker) return this.local[op](arg);
      if (this.dead) return Promise.reject(this.dead);
      return new Promise((resolve, reject) => {
        const id = ++this.seq; this.pending.set(id, { resolve, reject });
        this.worker.postMessage({ id, op, arg }, transfer);
      });
    };
    const p = (this.queue || Promise.resolve()).then(run, run);
    this.queue = p.catch(() => {});
    return p;
  }

  get embedTag() { return this.info.embedTag; }
  get probeTag() { return this.info.probeTag; }
  get backend() { return `${this.info.backend} · ${this.mode === 'worker' ? 'worker' : 'page'}`; }
  detect(bitmap, W, H, snap) { return this.call('detect', { bitmap, W, H, snap }, [bitmap]); }
  analyze(items) { return this.call('analyze', { items }); }
  embedImages(srcs) { return this.call('embedImages', { srcs }); }
  loadObjects() { return this.call('loadObjects'); }
  objects() { return this.call('objects'); }
}
