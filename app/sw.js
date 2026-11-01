// Offline cache + cross-origin isolation headers (lets onnxruntime use wasm threads).
const CACHE = 'koi-id-v0.068';
// models and the onnxruntime wasm (~140 MB on an iPhone) live in their own cache, named by a hash of their bytes, so a
// code-only release does not download them again (the owner tests on mobile data). deploy.sh keeps the name current.
const MODELS = 'koi-models-8308bccae246';
const SHELL = ['./', 'index.html', 'about.html', 'app.js', 'engine.js', 'pipeline.js', 'worker.js', 'models.js', 'tracker.js', 'follow.js', 'reid.js', 'samples.js', 'sync.js', 'manifest.webmanifest', 'icon.svg', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/maskable-512.png', 'icons/apple-touch-icon.png',
  'vendor/ort.webgpu.min.mjs', 'vendor/ort-wasm-simd-threaded.asyncify.mjs', 'models/parts.json', 'models/coco_classes.json'];
// detectors and the objects model are cached on first use: a device only downloads the ones it runs (iPhone: lite)
const BIG = ['vendor/ort-wasm-simd-threaded.asyncify.wasm', ...[0, 1, 2].map(i => `models/embed.onnx.part${i}`)];
const isBig = path => /\.(onnx|wasm|part\d+)$/.test(path);
// v63.2 and v63.3 kept these exact model bytes (artifacts-v63.2) in their code cache, and v63.23 only added a part
// (koi_yx320) to the models cache before it: move them instead of downloading
const SAME_MODELS = ['koi-id-v63.2', 'koi-id-v63.3', 'koi-models-20cfd7c3df46'];

async function fillModels() {
  const m = await caches.open(MODELS), ks = await caches.keys();
  for (const old of SAME_MODELS.filter(k => ks.includes(k))) {
    const o = await caches.open(old);
    for (const r of await o.keys()) if (isBig(new URL(r.url).pathname) && !(await m.match(r))) await m.put(r, await o.match(r));
  }
  await Promise.all(BIG.map(async u => (await m.match(u)) || m.add(u)));
}
self.addEventListener('install', e => { e.waitUntil(Promise.all([caches.open(CACHE).then(c => c.addAll(SHELL)), fillModels()]).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => ![CACHE, MODELS, 'koi-last-video'].includes(k)).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

function isolate(res) {
  if (!res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || e.request.headers.has('range') || url.pathname.includes('/api/')) return;
  // code: network first (fresh when online); models/wasm: cache first (big, versioned by MODELS)
  const big = isBig(url.pathname);
  e.respondWith((async () => {
    const c = await caches.open(big ? MODELS : CACHE);
    if (big) {
      const hit = await c.match(e.request, { ignoreSearch: true });
      if (hit) return isolate(hit);
      const r = await fetch(e.request);
      if (r.ok) c.put(e.request, r.clone());
      return isolate(r);
    }
    try {
      const r = await fetch(e.request);
      if (r.ok) c.put(e.request, r.clone());
      return isolate(r);
    } catch { return isolate(await c.match(e.request, { ignoreSearch: true })); }
  })());
});
