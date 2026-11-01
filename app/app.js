// Koi ID PWA: video/camera -> detect -> track -> identify -> overlay + gallery.
import { Engine } from './engine.js';
import { Tracker, iou, COUNT } from './tracker.js';
import { Identity, maskGeometry, saveGallery, loadGallery, mergeGallery, sharedFish, sameEmbedder, CFG, StillSession, stillGeo, pondByFish } from './reid.js';
import { Harvest } from './samples.js';
import { Sync } from './sync.js';
import { Follow } from './follow.js';

const $ = s => document.querySelector(s);
const video = $('#video'), view = $('#view'), vctx = view.getContext('2d');
const params = new URLSearchParams(location.search);
// ?count=1 (owner, 31 Oct 2026): count only with the small YOLOX detector (models.js): no embedder, no naming, and the
// gallery is never loaded, changed or synced (an embedder-less session must not touch the central store); logs still go up
const COUNT_ONLY = params.get('count') !== '0'; // the default since v0.064 (owner, 1 Nov 2026); ?count=0 = the old full app
// ?count=1&id=1: photos also get the pond gallery's permanent fish numbers (#n), matched read-only: nothing is learned or
// written, a fish the gallery does not know keeps a 2-digit hex for the session (StillSession)
const COUNT_ID = COUNT_ONLY && params.get('id') === '1';
let statusTimer; // declared first: status() can run before the rest of this module (e.g. no service worker)
const BUILD = 'v0.066';
const SLOW_EMBED = params.get('slowAdapt') === '1'; // 'count' eases only the counting rules on slow devices // keep in step with CACHE in sw.js; sent with every field report

// ---------- self-healing after a crash: the OS killing the tab (iPhone memory limit) leaves the stage it died in behind;
// the next start runs one step leaner (1: no everyday-object model, one wasm thread; 2: also fewer embeddings and no
// saved frames). Three clean sessions of a minute or more step back up. Every start and crash is reported (diag).
const lsNum = k => { try { return +localStorage.getItem(k) || 0; } catch { return 0; } };
const lsPut = (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, String(v)); } catch {} };
const crashedAt = (() => { try { return localStorage.getItem('koi.stage'); } catch { return null; } })();
const IOS = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 1);
let SAFE = lsNum('koi.safe');
// a new build may have fixed what crashed before: the ladder starts again from 0 (and climbs again if it still crashes)
if ((() => { try { return localStorage.getItem('koi.safeBuild'); } catch { return null; } })() !== BUILD) { SAFE = 0; lsPut('koi.safe', 0); lsPut('koi.safeBuild', BUILD); }
if (crashedAt) { SAFE = Math.min(2, SAFE + 1); lsPut('koi.safe', SAFE); lsPut('koi.cleanRuns', 0); }
const diag = { build: BUILD, qs: location.search.slice(0, 80), ua: navigator.userAgent.slice(0, 180), gpu: !!navigator.gpu, coi: self.crossOriginIsolated, ios: IOS, safe: SAFE, crashedAt, n: 0, raw: 0, ms: 0, det: 0, emb: 0, nEmb: 0 };
// what the GPU offers (fp16 shaders missing would push the fp16 models onto slow paths)
navigator.gpu?.requestAdapter().then(a => { diag.f16 = !!a?.features.has('shader-f16'); diag.gpuInfo = a?.info ? `${a.info.vendor} ${a.info.architecture}`.slice(0, 60) : null; diag.maxBuf = a?.limits.maxBufferSize; }).catch(() => {});
function markStage(s) { lsPut('koi.stage', s); } // set while loading or running, cleared on a clean stop or exit
addEventListener('pagehide', () => markStage(null));
let reports = 0;
function report(ev, extra = {}) { // never throws: an error during start-up (before sync exists) is simply not sent
  try { if (DUMP || !sync.token || reports++ > 40) return; sync.log({ ev, ...diag, ...extra }).catch(() => {}); } catch {}
}
addEventListener('error', e => report('error', { msg: String(e.message).slice(0, 300) }));
addEventListener('unhandledrejection', e => report('error', { msg: String(e.reason?.message || e.reason).slice(0, 300) }));

// a track gets a fresh embedding every ~0.27 s of video (8 frames at 30 fps) whatever the detector pass rate
// iPhones always run at the leanest level: field logs (v50) showed tabs killed at levels 0 and 1, stable at 2
const LEVEL = IOS ? Math.max(SAFE, 2) : SAFE;
// iPhones: frames go to the worker scaled to MAX_SIDE px on the long side. 540x960 clips run for minutes; the owner's
// 1920x1440 clips killed the tab now and then 20-30 s into a run (field log 588, v63.5). The detector sees 312 px anyway.
// Camera A/B on v63.6 (logs 607-653): 960 / ?follow=0 / ?passDelay=500 died by t=28 s, ?maxSide=640 ran to t=73 s.
// Boxes, masks and areas come back in view px; geometry sent back to the worker is scaled to its frame.
const MAX_SIDE = +(params.get('maxSide') ?? (LEVEL >= 2 ? 640 : 0)) || 0;
let frameScale = 1;
// iPhones grab through one reused canvas: createImageBitmap(video) made a new bitmap from the camera or decoder surface
// every pass, and camera runs died after 20-80 s with passes slowing down (v63.8, logs 674-683). ?grab=bitmap = old way.
const GRAB_CANVAS = (params.get('grab') ?? (IOS ? 'canvas' : 'bitmap')) === 'canvas';
let grabCv = null;
function grabFrame(w, h) {
  if (!GRAB_CANVAS) return w === video.videoWidth && h === video.videoHeight ? createImageBitmap(video) : createImageBitmap(video, { resizeWidth: w, resizeHeight: h, resizeQuality: 'medium' });
  if (!grabCv || grabCv.width !== w || grabCv.height !== h) grabCv = new OffscreenCanvas(w, h);
  grabCv.getContext('2d').drawImage(video, 0, 0, w, h);
  return Promise.resolve(grabCv.transferToImageBitmap());
}
const EMBED_S = (+params.get('embedEvery') || 8) / 30, EMBED_BUDGET = +params.get('embedBudget') || (LEVEL >= 2 ? 2 : 6), CONSOLIDATE_MS = 30000;
let runAt = 0, engine, det = { hasPond: false, pond: false, glare: 0, rawCount: 0 }, tracker = new Tracker(), id = new Identity(), running = false, busy = false, mode = 'realtime';
let session = 0, rvfc = 0, paused = false, lastLive = [], objectsOn = false, objs = null, objsAt = -Infinity, objZones = [], liveSaidAt = 0, capAt = 0, passMs = 0, countHist = [], maxTogether = 0, countNow = 0, namedNow = 0, looping = false, lastVideoT = 0, harvest = null, harvestQ = Promise.resolve(), frameNo = 0, lastConsolidate = performance.now(), fpsEMA = 0, mergePick = null, dirty = false;
const embedder = () => ({ embedTag: engine.embedTag, probeTag: engine.probeTag });
function setEmbedder(identity) { identity.embedTag = engine.embedTag; identity.probeTag = engine.probeTag; return identity; }
const stats = { frames: [], embs: [] }; // eval hook
const DUMP = params.has('dump'), DUMP_CROPS = params.get('dump') === 'crops';
// photo only (owner, 27 Oct 2026: live video kept dying on the iPhone): video file, live camera and library pick are hidden;
// ?full=1 (or a ?video= dev run) brings them back; count mode shows them too (owner, 1 Nov 2026: YOLOX only, files ran)
const FULL = params.has('full') || params.has('video') || COUNT_ONLY;
document.body.classList.toggle('still', !FULL);
if (COUNT_ONLY && !COUNT_ID) $('#empty > div').lastChild.textContent = 'กดกล้องเพื่อนับปลาต่อเนื่อง หรือถ่ายภาพ หรือเลือกไฟล์วิดีโอ';
// between detector passes the drawn boxes follow the fish frame by frame (follow.js); ?follow=0 turns it off (velocity only)
const follow = params.get('follow') === '0' ? null : new Follow({ hz: +params.get('followHz') || (IOS ? 15 : 30), /* iPhone: copyTo cost unmeasured there, half rate */ mask: params.get('followMask') !== '0', vf: params.get('followVF') !== '0', dump: DUMP });
// test only, off by default: ?passEvery=650 starts a detector pass at most every 650 ms and ?passDelay=400 holds each
// pass's result that long (together they mimic an iPhone, ~1.5 passes/s each ~0.55 s late, on a fast machine)
const PASS_EVERY = +params.get('passEvery') || 0, PASS_DELAY = +params.get('passDelay') || 0;
// iPhone camera: a pass at most every 10 s and no box follower, unless the URL says otherwise. With follow off (owner,
// field logs 708-749, 27 Oct 2026): back to back died in 10-30 s, 1 s at ~60 s, 2 s twice of three in 13-25 s, 3 s
// died too; only 10 s ran for minutes. Root cause unknown (camera only: file passes run back to back for minutes).
const IOS_CAM = () => IOS && !!video.srcObject;
const passEvery = () => PASS_EVERY || (IOS_CAM() ? 10000 : 0);
const followOn = () => !!follow && !(IOS_CAM() && !params.has('follow'));
let passAt = -Infinity, capSnap = null, capVT = 0, resVT = 0;
window.koi = { get tracker() { return tracker; }, get id() { return id; }, get objs() { return objs; }, get objShown() { return objShown; }, get capAt() { return capAt; }, get passMs() { return passMs; }, follow, stats, CFG };

// ---------- boot ----------
(async () => {
  if ('serviceWorker' in navigator) {
    await navigator.serviceWorker.register('./sw.js').catch(() => {});
    // threads need cross-origin isolation, which the service worker provides after one reload
    if (!self.crossOriginIsolated && navigator.serviceWorker.controller && !sessionStorage.getItem('coi')) {
      sessionStorage.setItem('coi', '1'); location.reload(); return;
    }
  }
  // without WebGPU the models fall back to wasm, which on iPhone is far too slow (measured in WebKit: no pass in 60 s)
  if (!navigator.gpu) status(IOS ? 'เบราว์เซอร์นี้บน iPhone ยังไม่มี WebGPU จึงตรวจจับปลาไม่ได้: เปิดลิงก์นี้ใน Safari (iOS 26 ขึ้นไป)' : 'เบราว์เซอร์นี้ไม่มี WebGPU: ใช้ได้แต่จะช้ากว่า แนะนำ Chrome หรือ Edge รุ่นล่าสุด', IOS);
  try {
    markStage('load'); diag.t0 = Date.now();
    // enrol now (cheap) so a slow or stuck first load still reports where it is, once a minute
    if (!DUMP) sync.enroll().catch(() => {});
    const beat = setInterval(() => report('loading', { sec: Math.round((Date.now() - diag.t0) / 1000), at: $('#status').textContent.slice(0, 80) }), 60e3);
    engine = await new Engine().start(s => status(s), { noWorker: params.has('noworker'), query: `safe=${SAFE}${['det', 'ep', 'embB', 'count', 'id', 'embEp'].map(k => params.get(k) ? `&${k}=${encodeURIComponent(params.get(k))}` : '').join('')}` });
    clearInterval(beat); markStage(null); diag.ep = engine.backend; diag.loadMs = Date.now() - diag.t0;
  } catch (e) { markStage(null); modelsReady(); status(`โหลดโมเดลไม่สำเร็จ: ${e.message} (ลองรีเฟรช หรือใช้ Chrome รุ่นล่าสุด)`, true); diag.loadErr = String(e.message).slice(0, 200); setTimeout(() => report('loadfail'), 8000); return; }
  if (COUNT_ONLY) {
    if (COUNT_ID) { const saved = await loadPondGallery(pond); if (saved) id.load(saved, embedder()); setEmbedder(id); }
    connect().then(() => { report('start'); if (COUNT_ID) idGallery(); });
  }
  else {
    const saved = await loadPondGallery(pond);
    if (saved) id.load(saved, embedder());
    setEmbedder(id);
    wireIdentity();
    harvest = await new Harvest().init(embedder(), pond.id, legacyPond).catch(e => (console.warn('[koi] harvest off', e), null));
    renderPonds(); pondByLocation();
    connect().then(() => report('start'));
    renderReview();
    if (id.needsReembed) await migrate().catch(e => { console.warn('[koi] migrate', e); status(`ปรับคลังปลากับโมเดลใหม่ไม่สำเร็จ: ${e.message}`, true); });
    renderGallery();
  }
  // everyday objects (person, bird, car...) are optional: load in the background once the fish models are up
  // everyday-object labels are a nice-to-have: iPhones (and any device after a crash) skip that model (62 MB + working memory)
  // not on iPhones: with it the tab was killed (v50 field logs: crashes at levels 0-1 with objects, stable without)
  if (!IOS && SAFE < 1 && !COUNT_ONLY) loadObjects();
  try {
    status(`พร้อม · ${BUILD} · ${engine.backend}`);
    $('#dock').classList.remove('disabled');
    readyAt = performance.now(); modelsReady(); setTimeout(warm, 5000); // after the load's garbage is gone
    if (FULL) resumeVideo().catch(() => {});
    if (params.get('video')) openUrl(params.get('video'), params.get('mode') || 'every', params.get('loop') === '1');
    else if (params.get('photo')) fetch(params.get('photo')).then(r => (r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`)))).then(photoRun).catch(e => status(`เปิดภาพไม่ได้: ${e.message}`, true));
  } catch (e) { status(`โหลดโมเดลไม่สำเร็จ: ${e.message}`, true); }
})();

function status(s, err) {
  const el = $('#status'); el.textContent = s; el.classList.toggle('err', !!err); el.classList.remove('fade');
  // progress messages ('กำลัง...', e.g. model download %) stay up until the next message; others fade after 4 s
  clearTimeout(statusTimer); if (!err && !s.startsWith('กำลัง')) statusTimer = setTimeout(() => el.classList.add('fade'), 4000);
}

// fit the canvas panel inside the stage (CSS % heights do not resolve in this layout), on any resize
const stage = $('#stage');
const panel = $('#panel');
function fit() {
  const r = stage.getBoundingClientRect(), sc = Math.min((r.width - 32) / view.width, (r.height - 14) / view.height);
  panel.style.width = `${Math.floor(view.width * sc)}px`; panel.style.height = `${Math.floor(view.height * sc)}px`;
}
new ResizeObserver(fit).observe(stage);

// library sheet
const sheet = $('#sheet');
// the sheet opens above the dock so หยุด stays reachable; a tap on the picture closes it
$('#lib').addEventListener('click', () => { sheet.style.bottom = `${$('#dock').offsetHeight}px`; sheet.classList.toggle('open'); });
stage.addEventListener('click', () => sheet.classList.remove('open'));
$('#close').addEventListener('click', () => sheet.classList.remove('open'));
addEventListener('keydown', e => { if (e.key === 'Escape') sheet.classList.remove('open'); });

// ---------- inputs ----------
// a chosen file plays on a loop until the user stops it
$('#file').addEventListener('change', e => { const f = e.target.files[0]; if (f) { video.play().catch(() => {}); /* inside the tap: unlocks later play() on iOS */ keepVideo(f); lsPut('koi.resumes', 0); openUrl(URL.createObjectURL(f), 'realtime', true); } e.target.value = ''; });
// self-healing: iOS kills the tab now and then (memory); the picked file is kept in Cache Storage and, after a kill
// during a run, reopened by itself once the models are ready. At most 3 resumes in a row (a run past ~65 s resets it).
// never read whole: arrayBuffer() of the owner's 50-120 MB iPhone clips, on top of the models, left the tab dead right
// after 'picked' (field log 565, 27 Oct 2026). WebKit: bytes streamed (a File body kept only a reference, gone with the
// page), clips up to KEEP_MAX only; a bigger pick drops the old one so a resume never reopens a different clip.
// Elsewhere the File is the body: Chrome copies it without JS (VM test: JS heap +122 MB with arrayBuffer, +67 streamed).
const KEEP_MAX = IOS ? 30e6 : 300e6;
const keepVideo = async f => {
  try {
    const c = await caches.open('koi-last-video');
    if (f.size > KEEP_MAX) { await c.delete('/last-video'); return; }
    await c.put('/last-video', new Response(IOS ? f.stream() : f, { headers: { 'content-type': f.type || 'video/mp4' } }));
  } catch {}
};
async function resumeVideo() {
  const n = lsNum('koi.resumes');
  if (crashedAt !== 'run' || params.get('video') || n >= 3) return;
  const r = await caches.open('koi-last-video').then(c => c.match('/last-video')).catch(() => null);
  if (!r || running) return;
  lsPut('koi.resumes', n + 1); report('resume', { k: n + 1 });
  openUrl(URL.createObjectURL(await r.blob()), 'realtime', true);
}
$('#cam').addEventListener('click', async () => {
  if (COUNT_ONLY && !params.has('live')) return autoCount(); // count mode: photo rounds back to back (autoCount)
  try {
    stop();
    await pondHere();
    await warm();
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('เบราว์เซอร์นี้เปิดกล้องไม่ได้ (ต้องเปิดผ่าน https หรือ localhost)');
    markStage('open'); // a kill from here to the run start shows up as crashedAt 'open' (camera crashes left no trace)
    // iPhones: 1280x720, the capture, display and follow buffers cost per pixel too (1080x1920 runs died, see MAX_SIDE)
    const camH = +params.get('cam') || (IOS ? 720 : 1080);
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: Math.round(camH * 16 / 9) }, height: { ideal: camH } }, audio: false });
    video.removeAttribute('src'); await video.play();
    report('cam', { step: 'on', w: video.videoWidth, h: video.videoHeight });
    await warm(); start('realtime');
  } catch (e) {
    markStage(null);
    const denied = e.name === 'NotAllowedError' || e.name === 'SecurityError';
    status(denied ? 'ไม่ได้รับอนุญาตให้ใช้กล้อง: อนุญาตกล้องในการตั้งค่าเบราว์เซอร์แล้วลองใหม่' : `เปิดกล้องไม่ได้: ${e.message}`, true);
  }
});
$('#stop').addEventListener('click', () => stop(true));

// pause / play: the picture and the analysis both hold; one more pass on the frozen frame puts the boxes exactly on it
const PAUSE_SVG = $('#pause').innerHTML, PLAY_SVG = '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M8 5l11 7-11 7z" fill="currentColor"/></svg>';
function pauseButton(p) {
  const b = $('#pause'), label = p ? 'เล่นต่อ' : 'หยุดชั่วคราว';
  b.innerHTML = p ? PLAY_SVG : PAUSE_SVG; b.setAttribute('aria-label', label); b.title = `${label} (Space)`; b.setAttribute('aria-pressed', p);
}
function setPaused(p) {
  paused = p; pauseButton(p);
  if (!running || mode === 'every') return; // the stepping loop waits on its own
  if (p) { video.pause(); follow?.clear(); if (!busy) { busy = true; process(session).catch(failed).finally(() => { busy = false; }); } }
  else { capAt = 0; video.play(); }
  status(p ? 'หยุดชั่วคราว' : 'เล่นต่อ');
}
$('#pause').addEventListener('click', () => setPaused(!paused));
// Chrome pauses a muted video in a hidden tab; when the tab comes back, carry on unless the user paused
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && running && !paused && mode !== 'every' && video.paused) video.play().catch(() => {});
});
addEventListener('keydown', e => {
  if (e.key === ' ' && running && !e.target.closest?.('input, button, textarea')) { e.preventDefault(); setPaused(!paused); }
});

// boxes on/off (the count stays): remembered on this device
let showBoxes = true;
try { showBoxes = localStorage.getItem('koi.boxes') !== '0'; } catch {}
const boxesBtn = $('#boxes');
boxesBtn.setAttribute('aria-pressed', showBoxes);
boxesBtn.addEventListener('click', () => {
  showBoxes = !showBoxes; boxesBtn.setAttribute('aria-pressed', showBoxes);
  try { localStorage.setItem('koi.boxes', showBoxes ? '1' : '0'); } catch {}
  if (running) draw(lastLive, 0, true);
  else if (photoOn && !photoBusy) drawPhoto();
});

// a file plays on a loop: restart by hand after a short rest, not with video.loop (its seek during frame grabs killed the
// iPhone tab at every loop). Tracks restart, fish identities stay.
// Loop restart, checked twice a second (an 'ended' event alone was once missed). iPhones killed the tab at the loop point
// both with video.loop and with a seek back to 0 (field logs 25 Oct 2026), so: stop grabbing frames, let the pass in
// flight finish, then load the file into the element again (fresh decoder, no seek). Each step is logged.
let restarting = false, loops = 0;
// the element has a frame: loadeddata or canplay (iOS may send only one), 'error' rejects. After ms: give up (onSlow
// unset) or call onSlow and keep waiting (a background tab defers media loading)
const whenLoaded = (ms, onSlow) => new Promise((res, rej) => {
  const done = () => { clearTimeout(t); if (video.onloadeddata === done) video.onloadeddata = video.oncanplay = video.onerror = null; res(); }; // a later wait owns the handlers
  video.onloadeddata = done; video.oncanplay = done; video.onerror = () => { clearTimeout(t); rej(new Error('bad file')); };
  const t = setTimeout(() => (video.readyState >= 2 ? done() : onSlow ? onSlow() : rej(new Error('timeout'))), ms);
});
let nearEndSent = 0;
const nearEnd = () => looping && mode === 'realtime' && video.duration > 1 && video.duration - video.currentTime < 0.6;
setInterval(async () => {
  if (!(running && looping && mode === 'realtime' && !paused) || restarting) return;
  // the iPhone tab died at the end of the clip whatever the restart method (native loop, seek, reload): the report never
  // left the device, so the end itself is suspect. Report 2 s ahead, then restart before the clip can reach 'ended'.
  if (video.duration - video.currentTime < 2 && nearEndSent !== loops + 1) { nearEndSent = loops + 1; report('loop', { k: loops + 1, step: 'near-end' }); }
  if (!(video.ended || video.duration - video.currentTime < 0.3)) return;
  restarting = true; loops++; const s = session;
  try {
    video.pause();
    report('loop', { k: loops, step: 'end' });
    for (let i = 0; busy && i < 40; i++) await new Promise(r => setTimeout(r, 100)); // the detector pass in flight
    const src = video.src; video.removeAttribute('src'); video.load(); // drop the old decoder before the new one
    await new Promise(r => setTimeout(r, 300));
    // iOS fetched a reloaded file only when asked: without load() and a time limit the loop hung for good (field v53)
    for (let i = 0; ; i++) {
      video.src = src; video.load();
      try { await whenLoaded(8000); break; } catch (e) { report('loop', { k: loops, step: 'retry', msg: e.message }); if (i || s !== session) throw e; }
    }
    if (s !== session) return; // a new file was opened meanwhile
    if (running && looping) { await video.play(); video.cancelVideoFrameCallback(rvfc); rvfc = video.requestVideoFrameCallback(onFrame); report('loop', { k: loops, step: 'playing' }); }
  } catch (e) { report('loop', { k: loops, step: 'fail', msg: String(e.message).slice(0, 120) }); }
  finally { restarting = false; }
}, 200); // 5 checks a second: the restart window is the last 0.3 s
async function openUrl(url, m, loop = false) {
  stop();
  await warm();
  if (video.src.startsWith('blob:')) URL.revokeObjectURL(video.src);
  video.srcObject = null; video.muted = true; video.loop = false; looping = loop; video.preload = 'auto';
  status('กำลังเปิดไฟล์วิดีโอ...'); report('file', { step: 'picked' });
  markStage('open'); // a kill while the file opens shows up as crashedAt 'open' (field log 565 left no trace)
  video.src = url; video.load(); // iOS may not fetch a picked file until asked (owner had to pick the same file 3 times)
  try {
    await whenLoaded(15000, () => report('file', { step: 'slow' }));
  } catch (e) { markStage(null); report('file', { step: 'fail', msg: e.message }); status('เปิดไฟล์วิดีโอนี้ไม่ได้: ลองไฟล์ .mp4 (H.264) หรือ .mov', true); return; }
  report('file', { step: 'loaded', w: video.videoWidth, h: video.videoHeight, dur: Math.round(video.duration) });
  await warm();
  status(`พร้อม · ${BUILD} · ${engine.backend}`); // the 'opening' note must not stay up for the whole run
  start(m);
}

// detector warm-up (models.js warm), 5 s after load and before the first run, then 5 s idle so the GC frees its peak
// before live passes start (WebKit freed load and warm-up garbage within 5 s of idle). Once only; a run waits for it.
let warming = null, readyAt = 0, modelsReady;
// camera and files open only after the models are loaded and warmed: opening during the load (the memory peak) or the
// warm-up killed iPhone tabs (field logs 609, 637, 642, 660: crashedAt open/run seconds after start)
const ready = new Promise(r => { modelsReady = r; });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const warm = () => warming ||= (async () => {
  await ready;
  status('กำลังเตรียมโมเดลตรวจจับ...');
  await sleep(readyAt + 5000 - performance.now()); // a file picked at once still waits for the load's garbage to go
  try { if (await engine.call('warm')) await sleep(5000); } catch (e) { console.warn('[koi] warm', e); }
  status(`พร้อม · ${BUILD} · ${engine.backend}`);
})();
// iOS refuses play() without a recent tap (a resume after a tab kill, a pick that waited for the warm-up, Low Power
// Mode): field v62 'NotAllowedError'. Then ask for one tap and play on it.
const playOrTap = () => video.play().catch(e => {
  if (e.name !== 'NotAllowedError') return console.warn('[koi] play', e);
  status('แตะหน้าจอเพื่อเล่นวิดีโอต่อ', true); report('tap', {}); // err style: stays up until played
  addEventListener('pointerdown', () => video.play().then(() => status(`พร้อม · ${BUILD} · ${engine.backend}`)).catch(() => {}), { once: true });
});
let objectsLoading = false;
function loadObjects() {
  if (objectsOn || objectsLoading) return; objectsLoading = true;
  engine.loadObjects().then(() => { objectsOn = true; }).catch(e => console.warn('[koi] objects off', e)).finally(() => { objectsLoading = false; });
}
function start(m) {
  exitPhoto();
  if (harvest) harvest.lastFrameAt = -Infinity;
  markStage('run'); runAt = Date.now();
  // heartbeat at 20 s, then every 15 s: the last one before a kill dates the crash (iPhone tabs die at varying times)
  const beat = (s, hb) => setTimeout(() => { if (!running || s !== session) return; if (hb >= 3) lsPut('koi.resumes', 0); report('run', { hb, t: Math.round(video.currentTime), k: loops, n: diag.n, raw: +(diag.raw / Math.max(1, diag.n)).toFixed(2), ms: Math.round(diag.ms / Math.max(1, diag.n)), det: Math.round(diag.det / Math.max(1, diag.n)), emb: Math.round(diag.emb / Math.max(1, diag.n)), nEmb: +(diag.nEmb / Math.max(1, diag.n)).toFixed(1), fish: id.fish.size }); beat(s, hb + 1); }, hb ? 15000 : 20000);
  beat(session + 1, 0); // session++ follows below
  // slow devices: counting rules eased (stride 20/30: count +30-100 %, named fish about equal); embedding rules only with ?slowAdapt=1
  session++; restarting = false; mode = m; running = true; tracker = new Tracker(); tracker.adaptSlow = params.get('slowAdapt') !== '0'; frameNo = 0; capAt = 0; passMs = 0; objZones = []; objShown = []; countHist = []; maxTogether = 0; stats.frames = []; stats.embs = []; stats.done = false;
  follow?.clear(); passAt = -Infinity; capSnap = null; if (DUMP) stats.follow = { err: [] };
  view.width = video.videoWidth; view.height = video.videoHeight; fit();
  $('#empty').hidden = true; $('#dock').classList.add('running'); paused = false; pauseButton(false); lastVideoT = 0;
  if (mode === 'every') { video.pause(); stepEvery(session); }
  else { playOrTap(); rvfc = video.requestVideoFrameCallback(onFrame); }
}

// a new session makes any frame still in flight from the old one drop its result
function stop(user) {
  if (!running) return;
  running = false; session++; video.cancelVideoFrameCallback(rvfc); video.pause(); follow?.clear();
  markStage(null);
  if (Date.now() - runAt > 60e3) { const c = lsNum('koi.cleanRuns') + 1; lsPut('koi.cleanRuns', c); if (c >= 3 && SAFE > 0) { lsPut('koi.safe', SAFE - 1); lsPut('koi.cleanRuns', 0); } }
  if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
  if (!COUNT_ONLY) { id.retire(tracker.tracks); id.consolidate(); persist(); renderGallery(); syncNow(); }
  stats.done = true; $('#dock').classList.remove('running'); paused = false; pauseButton(false);
  vctx.clearRect(0, 0, view.width, view.height); $('#empty').hidden = !!video.currentSrc; // camera gone: blank panel shows the start hint again
  if (user) status('หยุดแล้ว');
}

async function onFrame(_, meta) {
  if (!running) return;
  // no rest between passes: the v53-v54 GPU rest did not stop the crashes (the v58 warm-up did) and cost a third of the passes
  const now = performance.now(), go = !busy && now - passAt >= passEvery();
  // the box follower reads this frame (whole when a pass starts on it: its templates come from the frame the detector sees);
  // never while the video restarts or jumps (frame grabs there crashed WebKit)
  const fp = followOn() && !paused && !(restarting || nearEnd() || video.seeking || video.readyState < 2) ? follow.frame(video, now, go) : null;
  if (go) { busy = true; passAt = now; process(session).catch(failed).finally(() => { busy = false; }); }
  if (fp) { const s0 = session; await fp; if (!running || s0 !== session) return; } // stopped or restarted meanwhile: that loop owns the frames
  // every video frame: boxes follow the fish (follow.js) or move along their velocity (per detector pass, so elapsed time is
  // counted in passes); past ~3 passes they would sit on the wrong fish
  const age = now - capAt;
  draw(age < Math.max(600, 3 * passMs) ? tracker.tracks.filter(t => t.lost === 0 && t.counted && !t.glint) : [], passes(age));
  // probe (?dump): where the boxes are drawn on the frame this pass looks at, compared with what it finds (process)
  if (DUMP && go) capVT = meta?.mediaTime ?? video.currentTime;
  if (DUMP && go) capSnap = new Map(lastLive.filter(t => t.disp).map(t => [t, [(t.disp[0] + t.disp[2]) / 2, (t.disp[1] + t.disp[3]) / 2, (t.goal[0] + t.goal[2]) / 2, (t.goal[1] + t.goal[3]) / 2, (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2, !!follow?.pos(t, flPos), t.box.map(Math.round), resVT]]));
  if (DUMP) window.koi.onDrawn?.();
  if (video.ended && !looping) return stop(); // a looping file restarts by hand (setInterval above); frames resume after it
  rvfc = video.requestVideoFrameCallback(onFrame);
}

function failed(e) {
  console.error(e);
  status(engine.dead ? 'ตัวประมวลผลหยุดทำงาน: รีเฟรชหน้าเพื่อเริ่มใหม่' : `ประมวลผลผิดพลาด: ${e.message}`, true);
  if (engine.dead) stop();
}

const seek = t => new Promise(r => { video.addEventListener('seeked', r, { once: true }); video.currentTime = t; });
async function stepEvery(s) { // deterministic: every frame (or every stride-th, to mimic a slower device) processed, video advanced by seeking
  const dt = (+params.get('stride') || 1) / 30, max = +params.get('max') || Infinity;
  try {
    await seek(video.currentTime); // right after loadeddata Chrome has no grabbable frame yet; a seek decodes one
    while (s === session && frameNo < max) {
      while (paused && s === session) await new Promise(r => setTimeout(r, 100));
      if (video.currentTime >= video.duration - dt / 2) {
        if (!looping) break;
        await seek(0);
      }
      await process(s);
      const before = video.currentTime;
      await seek(Math.min(video.duration, before + dt));
      if (video.currentTime <= before) { status('วิดีโอนี้เลื่อนทีละเฟรมไม่ได้ (เซิร์ฟเวอร์ไม่รองรับ Range) ใช้โหมดเรียลไทม์แทน', true); break; }
    }
  } catch (e) { failed(e); }
  if (s === session) stop();
}

// ---------- per frame ----------
// plain numbers only cross to the worker
const pick = (g, s = 1) => ({ len: g.len * s, cx: g.cx * s, cy: g.cy * s, straight: g.straight, heading: g.heading });
async function process(s = session) {
  if (video.videoWidth && (video.videoWidth !== view.width || video.videoHeight !== view.height)) { // phone rotated: new frame size
    view.width = video.videoWidth; view.height = video.videoHeight; fit(); tracker = new Tracker(); countHist = [];
  }
  const t0 = performance.now(), W = view.width, H = view.height;
  if (video.currentTime + 0.5 < lastVideoT) { tracker = new Tracker(); countHist = []; } // looped back: positions jump, identities stay
  lastVideoT = video.currentTime;
  const at = video.srcObject ? performance.now() / 1000 : video.currentTime;
  const snap = !!harvest && harvest.frameDue(at); // sparse full frame for later detector retraining (local only)
  if (restarting || nearEnd() || video.seeking || video.readyState < 2) return; // no frame grab while the video restarts or jumps (WebKit crashed there)
  const fs = MAX_SIDE && Math.max(W, H) > MAX_SIDE ? MAX_SIDE / Math.max(W, H) : 1, fw = Math.round(W * fs), fh = Math.round(H * fs);
  const bmp = await grabFrame(fw, fh);
  const got = await engine.detect(bmp, fw, fh, snap);
  if (fs < 1) for (const d of got.dets) { d.box = d.box.map(v => v / fs); d.msc *= fs; d.area /= fs * fs; }
  frameScale = fs;
  if (PASS_DELAY) await sleep(PASS_DELAY); // test only (see PASS_DELAY)
  if (s !== session) return;
  det = got;
  const dets = det.dets;
  const tDet = performance.now() - t0;
  // counted tracks keep getting crops (so a glint verdict can be revised); only non-glint ones are fish
  // slow device (a pass every > 0.4 s): crops start at 2 passes, before counting, so a fish has enough views to be named
  const cand = tracker.update(dets, video.srcObject ? performance.now() / 1000 : video.currentTime).filter(t => t.counted || (tracker.slow && SLOW_EMBED && t.hits >= 2));
  if (mode !== 'every') { // boxes now describe the frame captured at t0; pass interval for the overlay (first one seeds it, a tab switch is capped)
    if (capAt) { const d = Math.min(t0 - capAt, 3 * (passMs || t0 - capAt)); passMs = passMs ? 0.8 * passMs + 0.2 * d : d; }
    capAt = t0;
    // the follower takes its templates from this pass (the frame captured at t0) and carries them to the current frame
    if (followOn() && !paused) follow.reset(tracker.tracks.filter(t => t.lost === 0 && t.counted));
    if (DUMP && stats.follow && params.get('dumpMasks')) stats.follow.passes = [...(stats.follow.passes || []), { vt: capVT, tracks: tracker.tracks.filter(t => t.lost === 0 && t.counted).map(t => ({ id: t.id, box: t.box.map(Math.round), mw: t.det?.mw, mh: t.det?.mh, msc: t.det?.msc, mask: t.det?.mask ? btoa(String.fromCharCode(...t.det.mask.map(v => v ? 1 : 0))) : null })) }];
    if (DUMP && capSnap) {
      const seen = tracker.tracks.filter(o => o.lost === 0).map(o => [(o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2]);
      for (const [t, c] of capSnap) if (t.lost === 0 && t.counted) {
        const cx = (t.box[0] + t.box[2]) / 2, cy = (t.box[1] + t.box[3]) / 2, r = v => +v.toFixed(1);
        stats.follow.err.push({ id: t.id, t: r(video.currentTime), d: r(Math.hypot(c[0] - cx, c[1] - cy)), g: r(Math.hypot(c[2] - cx, c[3] - cy)), stale: r(Math.hypot(c[4] - cx, c[5] - cy)), f: c[6], diag: r(Math.hypot(t.box[2] - t.box[0], t.box[3] - t.box[1])), c: [r(c[0]), r(c[1])], p: [r(c[4]), r(c[5])], box: t.box.map(Math.round), vt: +capVT.toFixed(3), pb: c[7], pvt: +c[8].toFixed(3), nd: r(Math.min(...seen.map(q => Math.hypot(q[0] - c[0], q[1] - c[1])))) });
      }
      capSnap = null;
    }
    if (DUMP) resVT = capVT;
  }
  frameNo++;
  const due = [];
  for (const t of tracker.tracks) t.obs = null;
  for (const t of cand) {
    const d = t.det, g = maskGeometry(d.mask, d.mw, d.mh);
    if (!g) continue;
    // mask px -> frame px
    const geo = { ...g, cx: d.box[0] + g.cx / d.msc, cy: d.box[1] + g.cy / d.msc, len: g.len / d.msc,
      edge: d.box[0] <= 2 || d.box[1] <= 2 || d.box[2] >= W - 2 || d.box[3] >= H - 2 };
    // head direction: swimming direction, else last known, else the wide end of the body
    const vx = (t.vel[0] + t.vel[2]) / 2, vy = (t.vel[1] + t.vel[3]) / 2, along = vx * Math.cos(g.theta) + vy * Math.sin(g.theta);
    t.headSign = Math.abs(along) > 0.4 ? Math.sign(along) : t.headSign || g.widthSign || 1;
    geo.heading = g.theta + (t.headSign < 0 ? Math.PI : 0);
    // partial view (head/tail only, rest under water or glare): much shorter than this fish usually is
    const f = id.fish.get(t.fishId), sz = id.size;
    const expLen = f && f.size.n >= 5 && sz.a != null ? Math.exp(sz.a + sz.b * geo.cy / H + f.size.m) : t.lenRef;
    geo.partial = !!expLen && geo.len < 0.6 * expLen;
    t.lenRef = Math.max((t.lenRef || 0) * 0.98, geo.len);
    t.obs = { geo };
    if (at - (t.embAt ?? -Infinity) >= EMBED_S - 1e-3) due.push(t);
  }
  // embed the stalest tracks first, within budget
  due.sort((a, b) => (a.embAt ?? -Infinity) - (b.embAt ?? -Infinity));
  // canonical crops + embeddings for the stalest tracks, done in the inference worker
  // a slow device embeds every due track: one embedding batch costs ~0.1 s against a ~1 s detector pass (iPhone logs)
  const asked = COUNT_ONLY ? [] : due.filter(t => t.obs.geo.len >= 18 && !t.obs.geo.edge).slice(0, tracker.slow && SLOW_EMBED ? Math.max(EMBED_BUDGET, 6) : EMBED_BUDGET);
  const t1 = performance.now();
  const results = COUNT_ONLY ? [] : await engine.analyze(asked.map(t => ({ idx: t.det.idx, geo: pick(t.obs.geo, frameScale),
    want: { thumb: true, img: !!harvest, crop: DUMP_CROPS } })));
  if (s !== session) return;
  const tEmb = performance.now() - t1;
  const batch = [];
  asked.forEach((t, i) => {
    const r = results[i];
    // sky glint on the water, not a fish: vote per crop; a track that is mostly glint is never counted or named
    t.glintVotes = Math.max(-3, Math.min(3, (t.glintVotes || 0) + (r.glint ? 1 : -1)));
    t.glint = t.glintVotes > 0;
    t.embAt = at;
    if (!r.emb) return;
    Object.assign(t.obs, { emb: r.emb, hist: r.hist, q: r.q, thumb: r.thumb && (!t.thumbQ || r.q > t.thumbQ + 0.05) ? r.thumb : undefined, img: r.img });
    batch.push(t);
    if (DUMP) stats.embs.push({ f: frameNo, track: t.id, q: +r.q.toFixed(3), len: +t.obs.geo.len.toFixed(1), hist: r.hist.map(v => +v.toFixed(3)), e: Array.from(r.emb, v => +v.toFixed(4)), crop: r.crop, s: t.sizes?.at(-1) ?? null });
  });
  const live = cand.filter(t => !t.glint);
  if (pondCheckedRun !== s && batch.length >= 2) { pondCheckedRun = s; rerunOnPond(batch.map(t => ({ emb: t.obs.emb, hist: t.obs.hist })), s).catch(e => console.warn('[koi] pond by fish', e)); }
  const before = id.fish.size;
  if (!COUNT_ONLY) { id.step(live, H); id.retire(tracker.removed); }
  if (id.fish.size !== before) { dirty = true; persist(); } // a new fish: get its central number and name now
  if (harvest) collect(batch, dets, frameWorthKeeping() ? det.frameBlob : null, W, H);
  // other objects every ~2 s (they rarely move; keeps the phone's detector free for the fish); positions follow camera pans
  if (objectsOn && Math.abs(at - objsAt) >= 2) {
    objsAt = at; const tr = tracker, cam0 = [...tr.cam], s0 = session, os = frameScale;
    engine.objects().then(list => {
      if (s0 !== session) return;
      if (os < 1) for (const o of list) { o.cx /= os; o.cy /= os; o.w /= os; o.h /= os; }
      // screen-fixed things (a video-call inset, captions, logos) stay put while the camera pans: they get no label
      const prev = objs?.tr === tr ? objs : null, moved = prev && Math.hypot(cam0[0] - prev.cam0[0], cam0[1] - prev.cam0[1]) > 40;
      const zones = objZones, near = (q, o) => q.name === o.name && Math.hypot(q.cx - o.cx, q.cy - o.cy) < 0.2 * Math.max(o.w, o.h);
      for (const o of list) {
        o.fixed = zones.some(z => near(z, o)) || (!!moved && prev.list.some(q => near(q, o)));
        if (o.fixed && !zones.some(z => near(z, o))) zones.push({ name: o.name, cx: o.cx, cy: o.cy }); // remembered for this session
      }
      objs = { list, cam0, tr, at: performance.now(), zones };
    }).catch(e => console.warn('[koi] objects', e));
  }
  if (!COUNT_ONLY && performance.now() - lastConsolidate > CONSOLIDATE_MS) {
    lastConsolidate = performance.now();
    if (id.consolidate()) dirty = true;
    persist('routine'); renderGallery();
    harvestQ = harvestQ.then(autoResolve).catch(e => console.warn('[koi] auto-resolve', e));
  } else if (dirty) { dirty = false; renderGallery(); }
  // count = fish inside the pond; median of the last 5 detector passes so one flicker does not change it
  // fish that vanished < 0.5 s ago (glint, short dive) still count, so the number does not dip on every flash
  const nowT = video.srcObject ? performance.now() / 1000 : video.currentTime;
  // a briefly lost fish counts only if nothing live sits where it should be (else it was re-detected as a new track)
  const liveNow = tracker.tracks.filter(t => t.lost === 0);
  const counted = tracker.tracks.filter(t => t.counted && !t.glint && (t.lost === 0 ||
    (nowT - t.seenAt <= 0.5 && !liveNow.some(o => iou(o.box, t.pred || t.box) > 0.1)))).length;
  countHist.push(counted); if (countHist.length > 5) countHist.shift();
  const inPond = [...countHist].sort((a, b) => a - b)[countHist.length >> 1];
  maxTogether = Math.max(maxTogether, inPond);
  countNow = inPond; namedNow = new Set(live.filter(t => id.named(t)).map(t => t.fishId)).size;
  glareHint();
  draw(live, mode === 'every' || paused ? 0 : passes(performance.now() - t0)); // the picture moved on while this pass ran
  const ms = performance.now() - t0;
  diag.n++; diag.raw += det.rawCount || 0; diag.ms += ms; diag.det += tDet; diag.emb += tEmb; diag.nEmb += batch.length;
  fpsEMA = fpsEMA ? 0.9 * fpsEMA + 0.1 * (1000 / ms) : 1000 / ms;
  if (DUMP) stats.frames.push({ named: namedNow, count: inPond, t: video.currentTime, ms, tDet, tEmb, nEmb: batch.length, n: dets.length, raw: det.rawCount, glare: +(det.glare || 0).toFixed(3), all: tracker.tracks.filter(t => t.lost === 0).map(t => ({ id: t.id, hits: t.hits, sc: +(t.scoreEMA || 0).toFixed(2), st: !!t.static, j: t.jitter == null ? null : +t.jitter.toFixed(2), c: !!t.counted })), tracks: live.map(t => ({ id: t.id, fish: t.fishId || null, sure: !!t.sure, box: t.box.map(Math.round), len: t.obs?.geo ? +t.obs.geo.len.toFixed(1) : null, s: t.sizes?.at(-1) ?? null })) });
  $('#fps').textContent = `${fpsEMA.toFixed(1)} fps · ${engine.backend}`;
  // screen readers: the count lives on the canvas, so mirror it in a polite live region (at most every 5 s)
  if (performance.now() - liveSaidAt > 5000) { liveSaidAt = performance.now(); $('#live').textContent = `ปลาในบ่อ ${countNow} ตัว ระบุชื่อได้ ${namedNow} ตัว`; }
}

// ---------- overlay ----------
const passes = ms => Math.min(2, ms / (passMs || 200)); // elapsed time in detector passes (velocity is per pass)
const hue = n => (n * 137.5) % 360;
// counted fish the system cannot tell apart yet get a temporary 2-digit hex ID, unique among the tracks alive now
// (256 values, far more than fish on screen); random, so it never reads as a running number
function hexId(t) {
  if (t.hex) return t.hex;
  const used = new Set(tracker.tracks.map(o => o.hex).filter(Boolean));
  if (used.size >= 256) return '';
  let h; do h = Math.floor(Math.random() * 256).toString(16).toUpperCase().padStart(2, '0'); while (used.has(h));
  return (t.hex = h);
}
// labels have no background: the text takes black or white, whichever contrasts with the picture right behind it.
// A small copy of the frame is read for that (the video pixels themselves are never touched).
const sample = document.createElement('canvas'), sctx = sample.getContext('2d', { willReadFrequently: true });
let samplePx = null;
let sampleAt = 0;
function grabSample() { // label ink needs only a rough luminance map: 4 times a second, the small canvas resized only when needed
  const now = performance.now(); if (now - sampleAt < 250) return; sampleAt = now;
  const h = Math.max(1, Math.round(96 * view.height / view.width));
  if (sample.width !== 96 || sample.height !== h) { sample.width = 96; sample.height = h; }
  try { sctx.drawImage(video, 0, 0, sample.width, sample.height); samplePx = sctx.getImageData(0, 0, sample.width, sample.height).data; } catch { samplePx = null; }
}
function inkFor(x, y, w, h) { // mean luminance under the label -> [text, outline]
  const fl = followOn() ? follow.luma(x, y, x + w, y + h, performance.now()) : -1; // the follower's fresh gray frame, if any
  if (fl >= 0) return fl > 140 ? ['#111', 'rgba(255,255,255,.75)'] : ['#fff', 'rgba(0,0,0,.75)'];
  if (!samplePx) return ['#fff', 'rgba(0,0,0,.7)'];
  const sx = sample.width / view.width, X0 = Math.max(0, Math.floor(x * sx)), X1 = Math.min(sample.width - 1, Math.ceil((x + w) * sx)), Y0 = Math.max(0, Math.floor(y * sx)), Y1 = Math.min(sample.height - 1, Math.ceil((y + h) * sx));
  let sum = 0, n = 0;
  for (let yy = Y0; yy <= Y1; yy++) for (let xx = X0; xx <= X1; xx++) { const i = 4 * (yy * sample.width + xx); sum += 0.2126 * samplePx[i] + 0.7152 * samplePx[i + 1] + 0.0722 * samplePx[i + 2]; n++; }
  return sum / Math.max(1, n) > 140 ? ['#111', 'rgba(255,255,255,.75)'] : ['#fff', 'rgba(0,0,0,.75)'];
}
function label(text, x, y, w, h, pad, k) {
  const [ink, halo] = inkFor(x, y, w, h);
  vctx.lineJoin = 'round'; vctx.lineWidth = 3 * k; vctx.strokeStyle = halo; vctx.strokeText(text, x + pad, y + h - pad); // thin halo for busy water
  vctx.fillStyle = ink; vctx.fillText(text, x + pad, y + h - pad);
}
// boxes and labels glide: each video frame they ease toward where the fish should be (time constant GLIDE_MS) instead of
// jumping on every detector pass; a far jump (track picked up elsewhere), a pause or a redraw snaps
// A followed box (follow.js) is already smooth frame to frame: it glides faster (FOLLOW_GLIDE_MS) so it does not trail the fish.
const GLIDE_MS = 150, FOLLOW_GLIDE_MS = +params.get('followGlide') || 60;
let drawAt = 0;
const flPos = [0, 0];
function draw(live, dt, snap = false) {
  const k = view.width / 540; // keep text legible whatever the video resolution
  const now = performance.now(), ms = now - drawAt, a = snap || paused || mode === 'every' ? 1 : 1 - Math.exp(-ms / GLIDE_MS);
  const af = a === 1 ? 1 : 1 - Math.exp(-ms / FOLLOW_GLIDE_MS);
  drawAt = now;
  lastLive = live;
  // label ink: the follower's gray frame when it is fresh, else a small copy of the frame (a video readback costs ~7 ms)
  if ((live.length || objShown.length || objs) && !(followOn() && follow.luma(0, 0, 1, 1, now) >= 0)) grabSample();
  // one name per fish on screen: if two tracks claim the same fish, only the clearest keeps the name, the other shows its hex ID
  const owner = new Map();
  for (const t of live) if (id.named(t) && (t.mEMA ?? 0) > (owner.get(t.fishId)?.mEMA ?? -1)) owner.set(t.fishId, t);
  vctx.clearRect(0, 0, view.width, view.height); // overlay only: the <video> underneath shows the picture
  vctx.lineWidth = 2 * k; vctx.font = `600 ${Math.round(15 * k)}px system-ui, sans-serif`; vctx.textBaseline = 'bottom';
  for (const t of live) {
    // followed: the box (display size) centred where the follower sees the fish; else moved along the track velocity
    const fl = followOn() && follow.pos(t, flPos), sb = t.sbox || t.box, d = t.disp;
    const target = fl ? [flPos[0] - (sb[2] - sb[0]) / 2, flPos[1] - (sb[3] - sb[1]) / 2, flPos[0] + (sb[2] - sb[0]) / 2, flPos[1] + (sb[3] - sb[1]) / 2]
      : sb.map((v, k) => v + t.vel[k] * dt), g = fl ? af : a;
    const far = d && Math.hypot(target[0] - d[0], target[1] - d[1]) > Math.hypot(target[2] - target[0], target[3] - target[1]);
    t.goal = target; t.disp = !d || far || g === 1 ? target : d.map((v, i) => v + (target[i] - v) * g);
    const [x0, y0, x1, y1] = t.disp, f = id.fish.get(t.fishId), named = owner.get(t.fishId) === t;
    // counted fish get a thin outline (unless boxes are off); a name only when the evidence tells this fish apart
    if (showBoxes) {
      vctx.strokeStyle = named ? `hsl(${hue(f.id)} 85% 55%)` : 'rgba(255,255,255,.55)';
      vctx.lineWidth = (named ? 2 : 1) * k;
      vctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      vctx.lineWidth = 2 * k;
    }
    // label sits on the box corner, or centred on the fish when boxes are hidden
    const at = (w, h) => { const [lx, ly] = showBoxes ? [x0, y0 - h] : [(x0 + x1 - w) / 2, (y0 + y1 - h) / 2]; // kept inside the picture
      return [Math.max(0, Math.min(view.width - w, lx)), Math.max(0, Math.min(view.height - h, ly))]; };
    if (!named) { // temporary hex ID, small and quiet
      const hx = hexId(t); if (!hx) continue;
      vctx.font = `700 ${Math.round(13 * k)}px ui-monospace, Menlo, monospace`;
      const w = vctx.measureText(hx).width + 6 * k, h = 16 * k, [lx, ly] = at(w, h);
      label(hx, lx, ly, w, h, 3 * k, k);
      vctx.font = `600 ${Math.round(15 * k)}px system-ui, sans-serif`;
      continue;
    }
    const cm = id.cm(f);
    const text = `${f.name}${cm ? ` · ${cm.toFixed(0)} ซม.` : ''}`;
    vctx.font = `700 ${Math.round(16 * k)}px system-ui, sans-serif`;
    const w = vctx.measureText(text).width + 8 * k, h = 20 * k, [lx, ly] = at(w, h);
    label(text, lx, ly, w, h, 4 * k, k);
    vctx.font = `600 ${Math.round(15 * k)}px system-ui, sans-serif`;
  }
  drawObjects(k, a, ms);
  drawCount(k);
}

// names only, no box, no ID: where the object was at its last check, shifted with the camera; dropped after 4 s
const OBJ_TH = { person: 'คน', bicycle: 'จักรยาน', car: 'รถยนต์', motorcycle: 'มอเตอร์ไซค์', airplane: 'เครื่องบิน', bus: 'รถบัส', train: 'รถไฟ', truck: 'รถบรรทุก',
  boat: 'เรือ', 'traffic light': 'ไฟจราจร', 'fire hydrant': 'หัวดับเพลิง', 'stop sign': 'ป้ายหยุด', 'parking meter': 'มิเตอร์จอดรถ', bench: 'ม้านั่ง', bird: 'นก',
  cat: 'แมว', dog: 'สุนัข', horse: 'ม้า', sheep: 'แกะ', cow: 'วัว', elephant: 'ช้าง', bear: 'หมี', zebra: 'ม้าลาย', giraffe: 'ยีราฟ', backpack: 'กระเป๋าเป้',
  umbrella: 'ร่ม', handbag: 'กระเป๋าถือ', tie: 'เนกไท', suitcase: 'กระเป๋าเดินทาง', frisbee: 'จานร่อน', skis: 'สกี', snowboard: 'สโนว์บอร์ด',
  'sports ball': 'ลูกบอล', kite: 'ว่าว', 'baseball bat': 'ไม้เบสบอล', 'baseball glove': 'ถุงมือเบสบอล', skateboard: 'สเกตบอร์ด', surfboard: 'กระดานโต้คลื่น',
  'tennis racket': 'ไม้เทนนิส', bottle: 'ขวด', 'wine glass': 'แก้วไวน์', cup: 'แก้ว', fork: 'ส้อม', knife: 'มีด', spoon: 'ช้อน', bowl: 'ชาม', banana: 'กล้วย',
  apple: 'แอปเปิล', sandwich: 'แซนด์วิช', orange: 'ส้ม', broccoli: 'บรอกโคลี', carrot: 'แครอท', 'hot dog': 'ฮอตดอก', pizza: 'พิซซ่า', donut: 'โดนัท',
  cake: 'เค้ก', chair: 'เก้าอี้', couch: 'โซฟา', 'potted plant': 'กระถางต้นไม้', bed: 'เตียง', 'dining table': 'โต๊ะ', toilet: 'ส้วม', tv: 'ทีวี',
  laptop: 'แล็ปท็อป', mouse: 'เมาส์', remote: 'รีโมต', keyboard: 'คีย์บอร์ด', 'cell phone': 'มือถือ', microwave: 'ไมโครเวฟ', oven: 'เตาอบ', toaster: 'เครื่องปิ้งขนมปัง',
  sink: 'อ่างล้างมือ', refrigerator: 'ตู้เย็น', book: 'หนังสือ', clock: 'นาฬิกา', vase: 'แจกัน', scissors: 'กรรไกร', 'teddy bear': 'ตุ๊กตาหมี',
  'hair drier': 'ไดร์เป่าผม', toothbrush: 'แปรงสีฟัน' };
// object labels glide like the fish labels: each one is matched to its label from the last check (same name, nearest),
// eases to the new place, and fades in / out instead of popping
let objShown = [];
function drawObjects(k, a, ms) {
  const live = objs && objs.tr === tracker && performance.now() - objs.at < 4000 ? objs.list.filter(o => !o.fixed) : [];
  const dx = live.length ? tracker.cam[0] - objs.cam0[0] : 0, dy = live.length ? tracker.cam[1] - objs.cam0[1] : 0, next = [];
  for (const o of live) {
    const name = OBJ_TH[o.name] || o.name, tx = o.cx + dx, ty = o.cy + dy;
    let best = null, bd = 200 * k;
    for (const q of objShown) if (!q.used && q.name === name) { const d = Math.hypot(q.tx - tx, q.ty - ty); if (d < bd) { bd = d; best = q; } }
    if (best) Object.assign(best, { used: true, tx, ty, fade: false });
    next.push(best || { name, x: tx, y: ty, tx, ty, alpha: 0, fade: false });
  }
  for (const q of objShown) if (!q.used && !next.includes(q)) { q.fade = true; next.push(q); }
  const fa = a === 1 ? 1 : 1 - Math.exp(-ms / 250);
  vctx.font = `600 ${Math.round(14 * k)}px system-ui, sans-serif`;
  for (const q of next) {
    q.used = false;
    q.x += (q.tx - q.x) * a; q.y += (q.ty - q.y) * a;
    q.alpha += ((q.fade ? 0 : 1) - q.alpha) * fa;
    if (q.alpha < 0.02) continue;
    const w = vctx.measureText(q.name).width + 6 * k, h = 18 * k;
    vctx.globalAlpha = q.alpha;
    label(q.name, Math.max(0, Math.min(view.width - w, q.x - w / 2)), Math.max(0, Math.min(view.height - h, q.y - h / 2)), w, h, 3 * k, k);
    vctx.globalAlpha = 1;
  }
  objShown = next.filter(q => !q.fade || q.alpha >= 0.02);
}

// strong sky reflection on the water hides fish: tell the user how to fix the shot (once per 20 s)
let glareEMA = 0, glareTold = 0;
function glareHint() {
  glareEMA = 0.8 * glareEMA + 0.2 * (det.glare || 0);
  if (glareEMA > 0.15 && performance.now() - glareTold > 20000) {
    glareTold = performance.now();
    status('แสงสะท้อนบนผิวน้ำมาก: เอียงกล้องให้มองลงชันขึ้น หรือหันหลบด้านที่สะท้อนท้องฟ้า');
  }
}

// live count drawn on the picture itself (top-left), so it reads at a glance and survives screenshots
function drawCount(k) {
  if (!running) return;
  countBox(k, `${countNow}${det.hasPond && !det.pond ? '?' : ''}`, 'ปลาในบ่อ', `ระบุชื่อได้ ${namedNow} · สูงสุด ${maxTogether}`);
}
function countBox(k, big, word, line2) {
  const x = 12 * k, y = 12 * k, pad = 10 * k;
  vctx.font = `700 ${Math.round(34 * k)}px system-ui, sans-serif`; const wb = vctx.measureText(big).width;
  vctx.font = `600 ${Math.round(15 * k)}px system-ui, sans-serif`; const wl = vctx.measureText(word).width, w2 = vctx.measureText(line2).width;
  const w = Math.max(wb + 8 * k + wl, w2) + 2 * pad, h = 66 * k;
  vctx.fillStyle = 'rgba(255,255,255,.9)';
  vctx.beginPath(); vctx.roundRect(x, y, w, h, 10 * k); vctx.fill();
  vctx.textBaseline = 'alphabetic';
  vctx.fillStyle = '#2B7051'; vctx.font = `700 ${Math.round(34 * k)}px system-ui, sans-serif`; vctx.fillText(big, x + pad, y + 38 * k);
  vctx.fillStyle = '#1b2a22'; vctx.font = `600 ${Math.round(15 * k)}px system-ui, sans-serif`; vctx.fillText(word, x + pad + wb + 8 * k, y + 37 * k);
  vctx.fillStyle = '#5f6b63'; vctx.fillText(line2, x + pad, y + 58 * k);
  vctx.textBaseline = 'bottom';
}

// ---------- photo mode (v63.13): one still from the in-app shutter or the library -> count + identify, frozen ----------
// Live camera runs died on iPhones (WebKit memory limit; field logs 27 Oct 2026 grew with passes while the stream was open):
// a still means no detector pass while the camera is open and one pass per tap. The still is processed at PHOTO_SIDE px on
// the long side (re-ID crops come from that frame; the detector itself sees 312 px) and the decoded original is closed at
// once. Read only: matched against the pond gallery, nothing is written to it or synced. Unknown fish keep a hex ID for
// the session (reid.js StillSession). A detection counts at the tracker's counting score (COUNT.score) or ?photoMin=.
const PHOTO_SIDE = +params.get('photoSide') || 1920, PHOTO_MIN = +params.get('photoMin') || COUNT.score;
let still = new StillSession(), photoOn = false, photoBusy = false;
// the result's photo (a copy: the detector consumes the bitmap) and its fish, so กรอบ can redraw it with or without boxes
let photoBase = null, photoArgs = null;
$('#photoLib').addEventListener('change', e => { const f = e.target.files[0]; e.target.value = ''; if (f) photoRun(f); });
$('#photoClose').addEventListener('click', () => exitPhoto());

// in-app shutter (v63.14): <input capture> opened iOS's own camera app, the page went to the background holding ~1.1 GB
// of models and iOS killed it 40-50 s after each shot (field logs 761-770). Now a full-screen live preview in the page:
// no detector pass, no follow.js; the shutter copies the frame at the stream's size and stops the camera at once.
const shooter = $('#shooter'), pv = $('#pv'), shutter = $('#shutter');
let pvStream = null, pvGen = 0;
function closePreview() {
  pvGen++; pvStream?.getTracks().forEach(t => t.stop()); pvStream = null; pv.srcObject = null;
  shooter.hidden = true; shutter.disabled = true;
}
async function openPreview() {
  if (photoBusy || !shooter.hidden) return;
  const g = ++pvGen;
  try {
    stop();
    // no GPS pond step here (v63.17 had one): field v63.18 (828-833) died twice in 'open' right after it, and a fix more
    // than 150 m from บ่อ 2 made an empty local pond, so the photo named 0 fish. The pond comes from boot (pondByLocation,
    // pullPonds) as in v63.15, which ran 30 photos without a kill (795-823).
    await warm();
    if (g !== pvGen || document.hidden) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('เบราว์เซอร์นี้เปิดกล้องไม่ได้ (ต้องเปิดผ่าน https หรือ localhost)');
    markStage('open'); // a kill while the preview runs shows up as crashedAt 'open'
    shooter.hidden = false;
    const h = +params.get('shotH') || 1080;
    const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: Math.round(h * 16 / 9) }, height: { ideal: h } }, audio: false });
    if (g !== pvGen) return s.getTracks().forEach(t => t.stop()); // cancelled while the camera opened
    pvStream = s; pv.srcObject = s; await pv.play();
    shutter.disabled = false;
    report('photo', { step: 'preview', w: pv.videoWidth, h: pv.videoHeight });
  } catch (e) {
    if (g !== pvGen) return;
    closePreview(); markStage(photoOn ? 'result' : null);
    report('photo', { step: 'fail', msg: String(e.message).slice(0, 200) });
    const denied = e.name === 'NotAllowedError' || e.name === 'SecurityError';
    status(denied ? 'ไม่ได้รับอนุญาตให้ใช้กล้อง: อนุญาตกล้องในการตั้งค่าเบราว์เซอร์แล้วลองใหม่' : `เปิดกล้องไม่ได้: ${e.message}`, true);
  }
}
function cancelPreview() {
  if (shooter.hidden) { pvGen++; return; } // an open still waiting for the warm-up is dropped too
  closePreview(); markStage(photoOn ? 'result' : null); report('photo', { step: 'cancel' });
}
$('#photo').addEventListener('click', openPreview);
$('#shootCancel').addEventListener('click', cancelPreview);
shutter.addEventListener('click', () => {
  if (!pvStream || pv.readyState < 2 || photoBusy) return;
  const c = new OffscreenCanvas(pv.videoWidth, pv.videoHeight);
  c.getContext('2d').drawImage(pv, 0, 0, c.width, c.height);
  closePreview();
  photoRun(c.transferToImageBitmap());
});
// iOS freezes a hidden page's camera anyway; a stream left open there costs memory for nothing
document.addEventListener('visibilitychange', () => { if (document.hidden) { cancelPreview(); stopAuto(); } });

// count + camera (owner, 1 Nov 2026): count, show, count again with no wait between rounds. The live camera died on the
// iPhone under live detector passes (even 1 pass/s: D1 1136-1156, 15-25 s each). Each round copies one camera frame and
// counts it as a photo. The stream stays open (v63.27 default: D1 1178-1198, 1471 rounds in 3.5 min, ~7/s, no kill);
// reopening it every round (v63.26) died after 8-20 rounds (D1 1169-1176), ?autoKeep=0 brings that back.
// ?autoWarm=<frames> skipped after the camera opens (exposure settles), ?live=1 brings back the old live camera.
let autoGen = 0, autoOn = false;
const autoCam = document.createElement('video'); // in the page, never shown (a display:none video may get no frames)
autoCam.muted = true; autoCam.playsInline = true; autoCam.setAttribute('playsinline', '');
Object.assign(autoCam.style, { position: 'fixed', left: '0', top: '0', width: '2px', height: '2px', opacity: '0', pointerEvents: 'none' });
document.body.append(autoCam);
const nextFrame = v => new Promise(r => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(() => r()) : setTimeout(r, 34)));
function stopAuto() {
  if (!autoOn) return;
  autoOn = false; autoGen++;
  autoCam.srcObject?.getTracks().forEach(t => t.stop()); autoCam.srcObject = null;
}
async function autoCount() {
  if (autoOn) return stopAuto();
  const g = ++autoGen; autoOn = true;
  const h = +params.get('shotH') || 1080, warmN = +(params.get('autoWarm') ?? 8), keep = params.get('autoKeep') !== '0';
  let rounds = 0, openMs = 0, countMs = 0, beatAt = performance.now();
  report('auto', { step: 'on', keep, warmN });
  try {
    stop();
    // a video file picked before stays decoded in #video after stop(): with the camera on top the iPhone killed the
    // tab 11 s later (field log 1213-1214, 1 Nov 2026), so the file is let go first
    if (video.getAttribute('src')) { if (video.src.startsWith('blob:')) URL.revokeObjectURL(video.src); video.removeAttribute('src'); video.load(); }
    await warm();
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('เบราว์เซอร์นี้เปิดกล้องไม่ได้ (ต้องเปิดผ่าน https หรือ localhost)');
    while (g === autoGen && !document.hidden) {
      const t0 = performance.now();
      if (!autoCam.srcObject) {
        markStage('open');
        autoCam.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: Math.round(h * 16 / 9) }, height: { ideal: h } }, audio: false });
        await autoCam.play();
        for (let i = 0; i < warmN && g === autoGen; i++) await nextFrame(autoCam);
      } else await nextFrame(autoCam);
      if (g !== autoGen || !autoCam.srcObject) break;
      const c = new OffscreenCanvas(autoCam.videoWidth, autoCam.videoHeight);
      c.getContext('2d').drawImage(autoCam, 0, 0, c.width, c.height);
      if (!keep) { autoCam.srcObject.getTracks().forEach(t => t.stop()); autoCam.srcObject = null; }
      const t1 = performance.now();
      await photoRun(c.transferToImageBitmap(), true);
      if (g !== autoGen) { photoOn = true; exitPhoto(); break; } // stopped (ปิด, page hidden) while this round counted: clear what it drew
      rounds++; openMs += t1 - t0; countMs += performance.now() - t1;
      if (performance.now() - beatAt > 15000) { beatAt = performance.now(); report('auto', { step: 'beat', rounds, openMs: Math.round(openMs / rounds), countMs: Math.round(countMs / rounds), n: +(photoArgs?.[2] ?? -1) }); }
    }
  } catch (e) {
    report('auto', { step: 'fail', rounds, msg: String(e.message).slice(0, 200) });
    const denied = e.name === 'NotAllowedError' || e.name === 'SecurityError';
    status(denied ? 'ไม่ได้รับอนุญาตให้ใช้กล้อง: อนุญาตกล้องในการตั้งค่าเบราว์เซอร์แล้วลองใหม่' : `เปิดกล้องไม่ได้: ${e.message}`, true);
  } finally {
    if (g === autoGen) stopAuto();
    report('auto', { step: 'off', rounds, openMs: Math.round(openMs / (rounds || 1)), countMs: Math.round(countMs / (rounds || 1)) });
  }
}
addEventListener('pagehide', cancelPreview);
addEventListener('keydown', e => { if (e.key === 'Escape') cancelPreview(); });

function exitPhoto() {
  closePreview(); stopAuto();
  if (!photoOn) return;
  markStage(null);
  photoOn = false; still = new StillSession(); $('#dock').classList.remove('photo');
  photoBase?.close(); photoBase = photoArgs = null;
  vctx.clearRect(0, 0, view.width, view.height); $('#empty').hidden = !!video.currentSrc;
  if (DUMP) window.koi.photo = null;
}
async function photoRun(blob, quiet = false) { // quiet: an autoCount round (its own heartbeat instead of per-photo logs)
  if (photoBusy) return;
  photoBusy = true;
  const dock = $('#dock');
  let full = null, bmp = null, nb = null, W = 0, H = 0, shown = false;
  try {
    stop();
    await warm();
    const t0 = performance.now();
    photoOn = true; dock.classList.add('photo'); if (!quiet) dock.classList.add('disabled'); /* auto rounds: ปิด stays tappable */ $('#empty').hidden = true;
    if (!quiet) status('กำลังนับปลาในภาพ...');
    markStage('photo'); // a kill while a photo is processed shows up as crashedAt 'photo'
    // a shutter frame comes as an ImageBitmap (upright already), a library pick as a file
    const cam = blob instanceof ImageBitmap;
    full = cam ? blob : await createImageBitmap(blob, { imageOrientation: 'from-image' }).catch(e => (e.name === 'TypeError' ? createImageBitmap(blob) : Promise.reject(e)));
    const sc = Math.min(1, PHOTO_SIDE / Math.max(full.width, full.height));
    W = Math.round(full.width * sc); H = Math.round(full.height * sc);
    if (!quiet) report('photo', { step: 'shot', src: cam ? 'cam' : 'file', w: full.width, h: full.height });
    if (sc < 1) { bmp = await scaledBitmap(full, W, H); full.close(); }
    else bmp = full;
    full = null;
    // an auto round keeps the last result on screen until its own is drawn (no blank photo in between)
    if (!quiet || view.width !== W || view.height !== H) { vctx.clearRect(0, 0, view.width, view.height); view.width = W; view.height = H; fit(); vctx.drawImage(bmp, 0, 0); }
    nb = await createImageBitmap(bmp);
    if (!quiet) { photoBase?.close(); photoArgs = null; photoBase = nb; nb = null; }
    // label ink reads a small copy of the photo (as grabSample does from the video)
    sample.width = 96; sample.height = Math.max(1, Math.round(96 * H / W));
    sctx.drawImage(bmp, 0, 0, sample.width, sample.height); samplePx = sctx.getImageData(0, 0, sample.width, sample.height).data; sampleAt = performance.now();
    const b = bmp; bmp = null;
    const got = await engine.detect(b, W, H, false); // the bitmap is consumed (transferred to the worker)
    const dets = got.dets.filter(d => d.score >= PHOTO_MIN);
    if (COUNT_ONLY) { // boxes and the number (+ gallery numbers with id=1) only: no pond change, no gallery learning
      const labels = new Map(); let fish = dets, matched = 0, nEmb = 0, embMs = 0;
      if (COUNT_ID) {
        const te = performance.now();
        const ask = dets.map(d => [d, stillGeo(d, W, H)]).filter(([, g]) => g && g.len >= 18 && !g.edge);
        const res = ask.length ? await engine.analyze(ask.map(([d, g]) => ({ idx: d.idx, geo: pick(g), want: {} }))) : [];
        const glint = new Set(), items = [], itemDet = [];
        ask.forEach(([d], i) => { const r = res[i]; if (r.glint) glint.add(d); else if (r.emb) { items.push({ emb: r.emb, hist: r.hist, q: r.q }); itemDet.push(d); } });
        const lab = id.needsReembed ? items.map(() => ({ fish: null, hex: still.hex() })) : still.label(id, items); // another embedder's gallery: no matching
        if (!items.length) still.shots++;
        itemDet.forEach((d, i) => labels.set(d, lab[i]));
        fish = dets.filter(d => !glint.has(d));
        for (const d of fish) if (!labels.get(d)?.fish && !labels.get(d)?.hex) labels.set(d, { fish: null, hex: still.hex() }); // edge / small: an ID for this photo
        matched = lab.filter(l => l.fish != null).length; nEmb = items.length; embMs = Math.round(performance.now() - te);
      } else still.shots++;
      const n = fish.length, ms = Math.round(performance.now() - t0);
      if (quiet) { photoBase?.close(); photoBase = nb; nb = null; }
      photoArgs = [fish, labels, `${n}`, matched]; drawPhoto();
      status(`ภาพนี้มีปลา ${n} ตัว${COUNT_ID ? ` · รู้จัก ${matched} ตัว` : ''} · ${(ms / 1000).toFixed(1)} วินาที`);
      $('#live').textContent = `ภาพนี้มีปลา ${n} ตัว${COUNT_ID ? ` รู้จัก ${matched} ตัว` : ''}`;
      if (!quiet) report('photo', { step: 'done', count: 1, ids: COUNT_ID ? 1 : 0, w: W, h: H, n, ms, raw: got.rawCount, shots: still.shots, matched, emb: nEmb, embMs, fishN: id.fish.size, gal: idReady, ep: engine.backend });
      shown = true; const k = still.shots, ss = still;
      if (!quiet) for (const sec of [10, 25]) setTimeout(() => { if (photoOn && still === ss && still.shots === k && shooter.hidden) report('photo', { step: 'idle', count: 1, sec, shots: k }); }, sec * 1000);
      if (DUMP) window.koi.photo = { n, w: W, h: H, ms, matched, emb: nEmb, embMs, fishN: id.fish.size, dets: fish.map(d => ({ box: d.box.map(v => +v.toFixed(1)), det: +d.score.toFixed(3), fish: labels.get(d)?.fish ?? null, hex: labels.get(d)?.hex ?? null })) };
      return;
    }
    // crops + embeddings for every detection that is whole and big enough, in one call (no per-pass budget here)
    const ask = dets.map(d => [d, stillGeo(d, W, H)]).filter(([, g]) => g && g.len >= 18 && !g.edge);
    const res = ask.length ? await engine.analyze(ask.map(([d, g]) => ({ idx: d.idx, geo: pick(g), want: { thumb: true } }))) : [];
    const glint = new Set(), items = [], itemDet = [];
    ask.forEach(([d], i) => { const r = res[i]; if (r.glint) glint.add(d); else if (r.emb) { items.push({ emb: r.emb, hist: r.hist, q: r.q, thumb: r.thumb }); itemDet.push(d); } });
    // the fish choose the pond: one that names clearly more of them than this one becomes the pond (before any learning)
    const to = await pondFromFish(items);
    if (to) await switchPond(to, ' (เลือกตามปลาที่พบ)'); // a new photo session there (switchPond)
    const labels = new Map(), lab = still.label(id, items);
    itemDet.forEach((d, i) => labels.set(d, { ...lab[i], q: items[i].q }));
    // photos teach the pond gallery (reid.js LEARN): clear matches add a view, an unknown seen in 2 photos becomes a new fish
    const { added, promoted, covis } = still.learn(id, items, lab, Date.now());
    if (promoted.length) { persist(); renderGallery(); } else if (added || covis) persist(true);
    const fish = dets.filter(d => !glint.has(d)), n = fish.length;
    // every fish shows a name or a 2-digit ID (owner, 27 Oct 2026): one without an embedding (edge, too small) gets an ID
    // for this photo only; a fish not accepted centrally yet keeps the hex it showed (session memory)
    for (const d of fish) {
      const l = labels.get(d), f = l?.fish != null ? id.fish.get(l.fish) : null;
      if (l?.hex || (f && f.c !== false)) continue;
      labels.set(d, { ...(l || { fish: null, score: -1 }), hex: f ? still.fishHex(f.id) : still.hex(), temp: !f });
    }
    const matched = lab.filter(l => l.fish != null).length;
    const ms = Math.round(performance.now() - t0);
    photoArgs = [fish, labels, `${n}${got.hasPond && !got.pond ? '?' : ''}`, matched]; drawPhoto();
    // what this photo taught the pond gallery, so the owner sees the learning (the status pill fades after 4 s)
    const taught = promoted.length ? ` · เข้าคลังใหม่ ${promoted.length} ตัว` : added ? ` · เรียนรู้เพิ่ม ${added} ตัว` : '';
    status(`ภาพนี้มีปลา ${n} ตัว · ระบุได้ ${matched} ตัว${taught} · ${(ms / 1000).toFixed(1)} วินาที`);
    $('#live').textContent = `ภาพนี้มีปลา ${n} ตัว ระบุชื่อได้ ${matched} ตัว รวมทุกภาพ ${still.tally} ตัว${taught}`;
    report('photo', { step: 'done', pond: pond.id.slice(0, 8), fishN: id.fish.size, w: W, h: H, n, matched, ms, emb: items.length, tally: still.tally, shots: still.shots, learned: added, newFish: promoted.length });
    // v63.14 field (774-780): the page restarted 20-30 s after 'done' while the result sat on screen, camera closed. The
    // stage stays 'result' (pagehide clears it, an OS kill does not) and two heartbeats date the kill.
    shown = true; const k = still.shots, ss = still;
    for (const sec of [10, 25]) setTimeout(() => { if (photoOn && still === ss && still.shots === k && shooter.hidden) report('photo', { step: 'idle', sec, shots: k }); }, sec * 1000);
    if (DUMP) window.koi.photo = { n, w: W, h: H, ms, tally: still.tally, shots: still.shots, learned: added, newFish: promoted.length, temp: fish.filter(d => labels.get(d)?.temp).length, dets: fish.map(d => { const l = labels.get(d), f = l?.fish != null ? id.fish.get(l.fish) : null;
      return { box: d.box.map(v => +v.toFixed(1)), det: +d.score.toFixed(3), fish: l?.fish ?? null, hex: l?.hex ?? null, name: f?.name ?? null, score: l ? +l.score.toFixed(3) : null, q: l?.q != null ? +l.q.toFixed(3) : null }; }) };
  } catch (e) {
    console.error(e);
    report('photo', { step: 'fail', w: W, h: H, msg: String(e.message).slice(0, 200) });
    status(engine?.dead ? 'ตัวประมวลผลหยุดทำงาน: รีเฟรชหน้าเพื่อเริ่มใหม่' : `อ่านภาพนี้ไม่ได้: ${e.message}`, true);
    if (DUMP) window.koi.photo = { err: String(e.message) };
  } finally {
    full?.close(); bmp?.close(); nb?.close(); markStage(shown ? 'result' : null); photoBusy = false; dock.classList.remove('disabled');
  }
}
// resize options unsupported (older WebKit): through a canvas instead
const scaledBitmap = (src, w, h) => createImageBitmap(src, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' }).catch(() => {
  const c = new OffscreenCanvas(w, h); c.getContext('2d').drawImage(src, 0, 0, w, h); return c.transferToImageBitmap();
});
// the photo, then boxes (unless กรอบ is off) + labels in the live overlay's style on top, then the count box
function drawPhoto() {
  if (!photoBase || !photoArgs) return;
  const [fish, labels, big, matched] = photoArgs, k = view.width / 540;
  vctx.drawImage(photoBase, 0, 0);
  vctx.textBaseline = 'bottom';
  for (const d of fish) {
    const [x0, y0, x1, y1] = d.box, l = labels.get(d), f = l?.fish != null ? id.fish.get(l.fish) : null, named = !!f && f.c !== false;
    vctx.strokeStyle = named ? `hsl(${hue(f.id)} 85% 55%)` : 'rgba(255,255,255,.75)'; vctx.lineWidth = (named ? 2.5 : 1.5) * k;
    if (showBoxes) vctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (COUNT_ONLY && !COUNT_ID) continue; // count mode: boxes only, no labels
    const text = named ? (COUNT_ID ? `#${f.id}` : `${f.id} ${f.name}`) : l?.hex || '?';
    vctx.font = named ? `700 ${Math.round(16 * k)}px system-ui, sans-serif` : `700 ${Math.round(13 * k)}px ui-monospace, Menlo, monospace`;
    const pad = (named ? 4 : 3) * k, w = vctx.measureText(text).width + 2 * pad, h = (named ? 20 : 16) * k;
    const [lx, ly] = showBoxes ? [x0, y0 - h] : [(x0 + x1 - w) / 2, (y0 + y1 - h) / 2]; // without boxes the label sits on the fish
    label(text, Math.max(0, Math.min(view.width - w, lx)), Math.max(0, Math.min(view.height - h, ly)), w, h, pad, k);
  }
  countBox(k, big, 'ปลาในภาพ', COUNT_ID ? `รู้จัก ${matched} ตัว · ${still.shots} ภาพ` : COUNT_ONLY ? `โหมดนับอย่างเดียว · ${still.shots} ภาพ` : `ระบุชื่อได้ ${matched} · รวมทุกภาพ ${still.tally} ตัว (${still.shots} ภาพ)`);
}

// ---------- gallery ----------
// every change is shared at once (~1 s: new fish, owner answer, rename, merge, delete); routine learning within 15 s
function persist(routine) { if (COUNT_ONLY) return; galleryDirty = true; saveGallery(id, galleryKey(pond)).catch(e => console.warn(e)); kick(!routine); }

// ---------- ponds: every pond has its own fish names; the first pond keeps the data from before ponds existed ----------
const lsGet = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };
let ponds = lsGet('koi.ponds', []);
if (!ponds.length) { ponds = [{ id: crypto.randomUUID(), name: 'บ่อของฉัน' }]; lsSet('koi.ponds', ponds); }
const legacyPond = lsGet('koi.legacyPond', null) || (lsSet('koi.legacyPond', ponds[0].id), ponds[0].id);
let pond = ponds.find(p => p.id === lsGet('koi.pond', null)) || ponds[0];
const galleryKey = p => `gallery:${p.id}`;
async function loadPondGallery(p) { return (await loadGallery(galleryKey(p))) || (p.id === legacyPond ? loadGallery('gallery') : null); }

function renderPonds() {
  $('#pond').replaceChildren(...ponds.map(p => { const o = document.createElement('option'); o.value = p.id; o.textContent = p.name + (p.lat != null ? ' (มีตำแหน่ง)' : ''); o.selected = p.id === pond.id; return o; }));
}
async function switchPond(p, why = '') {
  if (p.id === pond.id) return;
  stop();
  await saveGallery(id, galleryKey(pond)).catch(e => console.warn(e));
  pondCache.set(pond.id, id); pondCache.delete(p.id);
  pond = p; lsSet('koi.pond', p.id);
  id = new Identity(); still = new StillSession(); // the photo session's fish numbers belong to the pond left
  let saved = await loadPondGallery(p);
  if (!saved && sync.token) saved = (await sync.pullPond(p.id).catch(() => null))?.gallery; // started on another device
  if (saved) id.load(saved, embedder());
  setEmbedder(id); wireIdentity();
  if (harvest) await harvest.init(embedder(), p.id, legacyPond);
  if (id.needsReembed) await migrate().catch(e => console.warn('[koi] migrate', e));
  galleryDirty = true;
  renderPonds(); renderGallery(); renderReview();
  status(`บ่อ: ${p.name}${why}`);
}
$('#pond').addEventListener('change', e => { const p = ponds.find(q => q.id === e.target.value); if (p) { pondPinned = true; switchPond(p); } });
// The fish choose the pond (owner, 30 Oct 2026: no manual step). Embeddings of one photo or video pass are scored against
// every known pond's gallery (reid.js pondByFish); a pond that names clearly more fish is returned and logged. Never creates
// a pond, never asks for a permission; a manual pick sticks for the session. Other ponds: the copy kept on this device,
// else the central one (GET /api/pond/:id), held in memory for the session.
let pondPinned = false;
const pondCache = new Map(); // pond id -> Identity (scoring only) of a pond this device is not on
async function pondGallery(p) {
  if (pondCache.has(p.id)) return pondCache.get(p.id);
  let j = await loadPondGallery(p);
  if (!j && p.central && sync.token) j = (await sync.pullPond(p.id)).gallery; // a failed pull is not cached: tried again later
  const g = j ? new Identity().load(j, embedder()) : null;
  pondCache.set(p.id, g);
  return g;
}
async function pondFromFish(items) {
  if (pondPinned || items.length < 2 || ponds.length < 2) return null;
  const others = [];
  for (const p of ponds) if (p.id !== pond.id) { const g = await pondGallery(p).catch(() => null); if (g?.fish.size) others.push([p, g]); }
  const r = pondByFish(items, id, others);
  if (!r) return null;
  console.info(`[koi] pond by fish: ${pond.name} (${r.matchedFrom}) -> ${r.key.name} (${r.matchedTo})`);
  report('pond', { from: pond.id, to: r.key.id, matchedFrom: r.matchedFrom, matchedTo: r.matchedTo });
  return r.key;
}
// video / camera: checked once per run, on its first pass with 2+ embeddings. A switch drops the run's tracks (labelled
// against the pond left, nothing of them is learned) and starts the run again on the new pond (the camera stream is kept).
let pondCheckedRun = 0;
async function rerunOnPond(items, s) {
  const to = await pondFromFish(items);
  if (!to || s !== session || !running) return;
  const m = mode, cam = video.srcObject;
  tracker = new Tracker();
  if (cam) video.srcObject = null; // stop() would end the camera stream
  await switchPond(to, ' (เลือกตามปลาที่พบ)');
  if (cam) { video.srcObject = cam; await video.play().catch(() => {}); }
  if (!running) start(m);
}
// the pond follows the phone (camera only: a video file may come from anywhere). Location ~100 m precision (3 decimals):
// enough to pick the pond, not a home address. Nearest known pond within 150 m -> that pond; the current pond has no
// location yet -> it is here; otherwise this is a new pond. No location (denied, indoors) -> stay on the current pond.
async function pondHere() {
  if (!navigator.geolocation || pondPinned) return;
  const c = await new Promise(res => navigator.geolocation.getCurrentPosition(p => res(p.coords), () => res(null), { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }));
  if (!c) return;
  const lat = +c.latitude.toFixed(3), lon = +c.longitude.toFixed(3);
  const near = ponds.filter(p => p.lat != null).map(p => [p, km(lat, lon, p.lat, p.lon)]).sort((x, y) => x[1] - y[1])[0];
  if (near && near[1] < 0.15) { if (near[0].id !== pond.id) await switchPond(near[0], ' (เลือกตามตำแหน่ง)'); return; }
  if (pond.lat == null) { pond.lat = lat; pond.lon = lon; lsSet('koi.ponds', ponds); persist(); renderPonds(); return; }
  const p = { id: crypto.randomUUID(), name: `บ่อ ${ponds.length + 1}`, lat, lon }; ponds.push(p); lsSet('koi.ponds', ponds);
  await switchPond(p, ' (บ่อใหม่ตามตำแหน่ง)');
}
const km = (a, b, c, d) => { const r = x => x * Math.PI / 180, h = Math.sin(r(c - a) / 2) ** 2 + Math.cos(r(a)) * Math.cos(r(c)) * Math.sin(r(d - b) / 2) ** 2; return 12742 * Math.asin(Math.sqrt(h)); };
async function pondByLocation() { // only when location is already allowed: never pops a permission prompt at start-up
  // only ponds the centre lists (they have fish): an empty pond made here by a GPS fix must not pull the device back
  if (!ponds.some(p => p.lat != null && p.central) || !navigator.permissions || !navigator.geolocation) return;
  try { if ((await navigator.permissions.query({ name: 'geolocation' })).state !== 'granted') return; } catch { return; }
  navigator.geolocation.getCurrentPosition(({ coords: c }) => {
    const near = ponds.filter(p => p.lat != null && p.central).map(p => [p, km(c.latitude, c.longitude, p.lat, p.lon)]).sort((x, y) => x[1] - y[1])[0];
    if (near && near[1] < 0.15 && near[0].id !== pond.id && !running && !pondPinned) switchPond(near[0], ' (เลือกตามตำแหน่ง)');
  }, () => {}, { maximumAge: 60000, timeout: 10000 });
}

// ---------- central sync (only with a sync code): pond galleries, labelled fish crops, sparse frames ----------
const sync = new Sync();
let syncing = false, galleryDirty = true;
const syncState = t => { $('#syncState').textContent = t; };
const packBits = m => { const b = new Uint8Array(Math.ceil(m.length / 8)); for (let i = 0; i < m.length; i++) if (m[i]) b[i >> 3] |= 1 << (i & 7); let s = ''; for (const v of b) s += String.fromCharCode(v); return btoa(s); };
let remotePonds = [];
async function pullPonds() {
  const all = await sync.ponds(), merged = new Map(all.filter(r => r.merged_into).map(r => [r.id, r.merged_into]));
  const remote = remotePonds = all.filter(r => !r.merged_into);
  // a pond found to be part of another one: this device moves over with its fish (once nothing is running)
  if (merged.has(pond.id) && !running) {
    let to = merged.get(pond.id); while (merged.has(to)) to = merged.get(to);
    const r = remote.find(x => x.id === to);
    if (r) await movePondTo({ id: r.id, name: r.name || 'บ่อ', ...(r.lat != null ? { lat: r.lat, lon: r.lon } : {}) });
  }
  if (ponds.some(p => merged.has(p.id) && p.id !== pond.id)) { ponds = ponds.filter(p => !merged.has(p.id) || p.id === pond.id); lsSet('koi.ponds', ponds); }
  // a device with nothing yet joins the pond worked on most recently instead of starting its own
  const best = remote.find(r => r.fish > 0); // newest pond that has fish
  if (!id.fish.size && best && best.id !== pond.id && !running) {
    const r = best, local = ponds.find(p => p.id === pond.id);
    if (local && !local.lat) ponds = ponds.filter(p => p !== local); // the untouched default pond goes away
    if (!ponds.some(p => p.id === r.id)) ponds.push({ id: r.id, name: r.name || 'บ่อ', ...(r.lat != null ? { lat: r.lat, lon: r.lon } : {}) });
    lsSet('koi.ponds', ponds); await switchPond(ponds.find(p => p.id === r.id), ' (จากส่วนกลาง)');
  }
  for (const r of remote) if (!ponds.some(p => p.id === r.id)) ponds.push({ id: r.id, name: r.name || 'บ่อ', ...(r.lat != null ? { lat: r.lat, lon: r.lon } : {}) });
  // ponds the centre knew and no longer lists (removed or merged away) leave this device's list too; ponds with local fish stay
  const ids = new Set(remote.map(r => r.id));
  for (const p of ponds) if (ids.has(p.id)) p.central = true;
  const drop = new Set();
  for (const p of ponds) if (p.id !== pond.id && !ids.has(p.id) && (p.central || !(await loadPondGallery(p))?.fish?.length)) drop.add(p.id); // also: listed before this rule, nothing local
  ponds = ponds.filter(p => !drop.has(p.id));
  lsSet('koi.ponds', ponds); renderPonds();
}
// Self-learning pond identity: two pond records within 300 m that share several fish are one body of water (a footbridge
// splits it from above, two phones started their own record...). The smaller one is merged into the bigger, centrally,
// and every device on it moves over. Checked at most every 10 min per pond.
const pondChecked = new Map();
async function checkPondMerge() {
  if (pond.lat == null || running || Date.now() - (pondChecked.get(pond.id) || 0) < (+params.get('pondCheckMs') || 600e3)) return;
  pondChecked.set(pond.id, Date.now());
  await pullPonds();
  for (const q of remotePonds) {
    if (q.id === pond.id || q.lat == null || km(pond.lat, pond.lon, q.lat, q.lon) > 0.3 || !q.fish) continue;
    const other = setEmbedder(new Identity().load((await sync.pullPond(q.id)).gallery, embedder()));
    const n = sharedFish(id, other);
    // measured 25 Oct 2026: the same fish filmed by two phones -> 3 of 11 pairs at >= mergeSim; disjoint fish of one pond
    // (look-alikes) -> 0 in 6 splits. ponytail: thresholds from one pond pair; retune when more ponds are synced
    if (n < Math.max(3, 0.2 * Math.min(id.fish.size, other.fish.size))) continue;
    const mineBigger = id.fish.size >= other.fish.size, gone = mineBigger ? q.id : pond.id, keep = mineBigger ? pond.id : q.id;
    await sync.mergePond(gone, keep).catch(e => console.warn('[koi] pond merge', e));
    console.info(`[koi] ponds ${gone} -> ${keep}: ${n} fish in common`);
    await pullPonds(); // moves this device if its pond was the one merged away
    return;
  }
}
// this device's pond was merged into `target`: its fish join the target's gallery as proposals (matched by look, else new
// central numbers), its kept crops and frames move over, and the merged-away pond disappears from the list
async function movePondTo(target) {
  const from = pond, old = id;
  for (const f of old.fish.values()) f.c = false;
  if (!ponds.some(p => p.id === target.id)) ponds.push(target);
  const ss = still;
  await switchPond(ponds.find(p => p.id === target.id), ' (รวมบ่อ: พบปลาตัวเดียวกัน)');
  still = ss;
  const own = new Set([...id.fish.values()].filter(f => f.c === false).map(f => f.id));
  const { gallery, map, added } = mergeGallery(id.toJSON(), old);
  const next = setEmbedder(new Identity().load(gallery, embedder()));
  for (const n of [...added, ...own]) if (next.fish.has(n)) next.fish.get(n).c = false; // not accepted centrally yet
  still.remap(map); // the photo session follows the merged numbers
  id = next; wireIdentity();
  if (harvest) { await harvest.movePond(from.id, target.id, map); await harvest.init(embedder(), target.id, legacyPond); }
  ponds = ponds.filter(p => p.id !== from.id); lsSet('koi.ponds', ponds);
  persist(); renderPonds(); renderGallery(); renderReview();
}
// The central gallery is the single source of truth (numbers, names, deletions, merges). Every sync pulls it, merges what
// this device learned into it (reid.js mergeGallery), takes the result at once and sends it back on top of the version it
// came from; if another device got there first the server refuses (409) and the merge runs again on the newer version.
const seenVer = new Map(); // pond id -> central version this device holds
async function syncGallery() {
  for (let tries = 0; tries < 3; tries++) {
    const p = pond;
    let r = await sync.pullPond(p.id, galleryDirty ? null : seenVer.get(p.id));
    if (p !== pond) return;
    if (!galleryDirty && r.ver === seenVer.get(p.id)) return; // nothing new on either side (the server sent no gallery)
    // the cheap 'unchanged' reply carries no gallery; if this device changed meanwhile, fetch the real one. Merging into
    // that empty reply once renumbered the whole pond and dropped every other device's fish (25 Oct 2026)
    if (!r.gallery && r.ver) { r = await sync.pullPond(p.id); if (p !== pond) return; }
    if (!r.gallery && !id.fish.size && !id.gone.size) return;  // an empty pond is not sent
    const send = galleryDirty || !r.gallery; galleryDirty = false;
    const { gallery, map, added } = mergeGallery(r.gallery, id);
    adopt(gallery, map, added);
    if (!send && !added.size && [...map].every(([a, b]) => a === b)) { seenVer.set(p.id, r.ver); return; }
    try { const w = await sync.pushPond(p.id, { base: r.ver ?? 0, name: p.name, lat: p.lat, lon: p.lon, gallery }); seenVer.set(p.id, w.ver); }
    catch (e) { galleryDirty = true; if (e.status === 409) continue; throw e; }
    if (p === pond) { for (const n of added) { const f = id.fish.get(n); if (f) f.c = true; } saveGallery(id, galleryKey(p)).catch(e => console.warn(e)); renderGallery(); }
    checkPondMerge().catch(e => console.warn('[koi] pond check', e));
    return;
  }
}
// take the merged central gallery as this device's own: live tracks and kept crops follow the new numbers; fish numbered
// just now stay proposals (hex ID, no name) until the server has accepted them
function adopt(gallery, map, added) {
  const next = setEmbedder(new Identity().load(gallery, embedder()));
  next.suggestions = id.suggestions.filter(x => map.get(x.a) === x.a && map.get(x.b) === x.b);
  for (const n of added) next.fish.get(n).c = false;
  still.remap(map); // the photo session's fish keys and hex memory follow the new numbers
  const m = x => (map.has(x) ? map.get(x) : x);
  for (const t of tracker.tracks) {
    if (t.fishId != null) { t.fishId = m(t.fishId); if (t.fishId == null) t.sure = false; }
    if (t.prior != null) t.prior = m(t.prior);
    if (t.votes) t.votes = [...t.votes].reduce((v, [k, x]) => (k = m(k), k == null ? v : v.set(k, (v.get(k) || 0) + x)), new Map());
    if (t.notFish) t.notFish = new Set([...t.notFish].map(m).filter(x => x != null));
  }
  // kept crops: two steps through temporary numbers, a fish may take a number another one gives up in the same sync
  const moves = [...map].filter(([a, b]) => a !== b);
  if (harvest && moves.length) harvestQ = harvestQ.then(async () => {
    for (const [a, b] of moves) if (b == null) await harvest.dropFish(a);
    const live = moves.filter(([, b]) => b != null);
    for (const [i, [a]] of live.entries()) await harvest.relabel(a, -1e9 - i);
    for (const [i, [, b]] of live.entries()) await harvest.relabel(-1e9 - i, b);
  }).catch(e => console.warn('[koi] harvest', e));
  id = next; wireIdentity();
  saveGallery(id, galleryKey(pond)).catch(e => console.warn(e)); renderGallery();
}
let enrollTry = 0;
async function syncNow(user) {
  if (COUNT_ONLY) return;
  // not enrolled yet (first try failed: offline, server busy): try again every 30 s, never give up for the session
  if (!sync.token && !DUMP && engine && Date.now() - enrollTry > 30e3) { enrollTry = Date.now(); return connect(); }
  if (syncing || !sync.token || !engine) return;
  if (!navigator.onLine) { if (user) syncState('ออฟไลน์: จะซิงก์เมื่อกลับมาออนไลน์'); return; }
  syncing = true;
  try {
    // an empty pond is not sent: every new device starts one, the central list would fill with empty duplicates
    await syncGallery();
    await harvestQ; // crops renumbered by the merge go up with their final fish number
    let sent = 0;
    for (let round = 0; round < 5 && harvest; round++) { // at most ~250 records per minute, the rest next time
      const { samples, frames } = await harvest.unsynced(40);
      if (!samples.length && !frames.length) break;
      for (const x of samples) {
        await sync.push('sample', { id: x.id, pond: x.pond, fish: x.fish, label: x.label, grade: x.label === 'owner' ? 'gold' : x.label === 'auto' ? 'bronze' : x.grade, q: x.q, len: x.len, s: x.s, view: x.view, tag: x.tag, hist: x.hist, at: x.at, session: x.session, cam: x.cam }, x.imgSynced ? null : x.img);
        await harvest.markSynced('samples', x); sent++;
      }
      for (const x of frames) {
        const dets = (x.dets || []).map(d => ({ box: d.box.map(Math.round), score: +d.score.toFixed(3), mw: d.mw, mh: d.mh, msc: d.msc, mask: packBits(d.mask) }));
        await sync.push('frame', { id: x.id, pond: x.pond, w: x.w, h: x.h, at: x.at, session: x.session, dets, quality: x.quality }, x.img);
        await harvest.markSynced('frames', x); sent++;
      }
    }
    lastSyncAt = Date.now();
    syncState(`ซิงก์อัตโนมัติ ล่าสุด ${new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' })} · บ่อ ${pond.name}${sent ? ` · ส่งภาพ ${sent} ภาพ` : ''} · ผู้ช่วยระดับ${helperLevel()}`);
  } catch (e) { galleryDirty = true; syncState(`ซิงก์ไม่สำเร็จ: ${e.message} (จะลองใหม่)`); }
  finally { syncing = false; }
}
// background sync, no setup: new trustworthy data goes up at once (~1 s, batched), routine learning within 15 s, and
// other devices' updates come down every 10 s while the app is open (a version check, the gallery only when it changed)
let kickT = 0, lastSyncAt = 0, helperDays = 0;
// optional helpers earn a level from use, not from asking: active days (server) + crops they labelled on this device
function helperLevel() {
  const score = helperDays + (harvest?.counts.owner || 0) / 10;
  return score >= 30 ? 'ทอง' : score >= 7 ? 'เงิน' : 'ทองแดง';
}
function kick(urgent) {
  const due = urgent ? 1000 : Math.max(1000, 15000 - (Date.now() - lastSyncAt));
  if (kickT && !urgent) return;
  clearTimeout(kickT); kickT = setTimeout(() => { kickT = 0; syncNow(); }, due);
}
setInterval(() => { if (!document.hidden) syncNow(); }, 10000);
addEventListener('online', () => connect());
document.addEventListener('visibilitychange', () => { if (!document.hidden) syncNow(); });
// count + id: the current pond's central gallery, read only (never saved, merged or sent back); ?gallery=<url> in tests
async function idGallery() {
  try {
    const g = DUMP && params.get('gallery') ? await (await fetch(params.get('gallery'))).json() : sync.token ? (await sync.pullPond(pond.id))?.gallery : null;
    if (g?.fish?.length) { id = setEmbedder(new Identity().load(g, embedder())); idReady = `${id.fish.size}`; }
  } catch (e) { console.warn('[koi] id gallery', e); idReady = 'error'; }
}
let idReady = '';
// first start (or a lost token): the device enrols itself and gets its own sync token
async function connect() {
  if (DUMP) { if (COUNT_ID) idGallery(); return; } // offline evaluation runs never touch the central store
  if (!navigator.onLine) return syncState('ออฟไลน์: ข้อมูลอยู่ในเครื่อง จะซิงก์เองเมื่อกลับมาออนไลน์');
  try {
    await sync.enroll(); id.central = !!sync.token;
    if (COUNT_ONLY) return;
    sync.me().then(m => { helperDays = m.days; }).catch(() => {});
    await pullPonds(); await syncNow();
  }
  catch (e) { syncState(`ซิงก์ไม่สำเร็จ: ${e.message} (จะลองใหม่)`); }
}

function renderGallery() {
  const g = $('#gallery'), fish = [...id.fish.values()].sort((a, b) => b.last - a.last);
  $('#count').textContent = id.fish.size;
  if (g.querySelector('.name:focus')) { dirty = true; return; } // owner is typing a name: rebuild once they leave the field
  g.replaceChildren(...fish.map(f => {
    const el = document.createElement('article'); el.className = 'card' + (mergePick === f.id ? ' pick' : '');
    const cm = id.cm(f), rel = f.size.n >= 3 ? `×${Math.exp(f.size.m).toFixed(2)}` : '–';
    el.innerHTML = `<img alt=""><div class="meta">
      <input class="name" value="${esc(f.name)}" maxlength="10" aria-label="ชื่อเล่น (ไม่เกิน 10 ตัวอักษร)">
      <span class="sub" title="สายพันธุ์ประเมินจากสี">${esc(f.variety)}</span>
      <span class="sub" title="ขนาดเทียบค่าเฉลี่ยของฝูง (หรือ ซม. ถ้าตั้งสเกลแล้ว)">ขนาด ${cm ? cm.toFixed(0) + ' ซม.' : rel}</span>
      <span class="sub" title="จำนวนมุมมองที่จำไว้ / จำนวนครั้งที่พบ">มุม ${f.ex.length} · พบ ${f.seen}</span></div>
      <div class="acts"><button data-a="merge" title="เลือก 2 ตัวเพื่อรวมเป็นตัวเดียวกัน">รวม</button><button data-a="del" title="กดซ้ำเพื่อยืนยัน">ลบ</button></div>`;
    if (/^data:image\/(jpeg|png);base64,/.test(f.thumb || '')) el.querySelector('img').src = f.thumb;
    el.style.setProperty('--c', `hsl(${hue(f.id)} 85% 55%)`);
    el.querySelector('.name').addEventListener('change', e => { const v = e.target.value.trim(), cur = id.fish.get(id.resolve(f.id)); if (v && cur) { cur.name = v; cur.autoName = false; cur.namedAt = Date.now(); } persist(); });
    el.querySelector('[data-a=merge]').addEventListener('click', () => {
      if (mergePick && mergePick !== f.id) { id.merge(mergePick, f.id); mergePick = null; persist(); }
      else mergePick = mergePick === f.id ? null : f.id;
      renderGallery();
    });
    const del = el.querySelector('[data-a=del]');
    del.addEventListener('click', () => {
      if (del.dataset.armed) { id.remove(f.id); harvestQ = harvestQ.then(() => harvest?.dropFish(f.id)).catch(e => console.warn('[koi] harvest', e)); persist(); renderGallery(); }
      else { del.dataset.armed = 1; del.textContent = 'ยืนยัน?'; setTimeout(() => { delete del.dataset.armed; del.textContent = 'ลบ'; }, 3000); }
    });
    return el;
  }));
  const s = $('#suggest');
  s.replaceChildren(...id.suggestions.filter(x => id.fish.has(x.a) && id.fish.has(x.b)).map(x => {
    const a = id.fish.get(x.a), b = id.fish.get(x.b), li = document.createElement('li');
    li.innerHTML = `<span>${esc(a.name)} กับ ${esc(b.name)} อาจเป็นตัวเดียวกัน (${(x.sim * 100).toFixed(0)}%)</span><button data-y>ใช่ รวม</button><button data-n>ไม่ใช่</button>`;
    li.querySelector('[data-y]').onclick = () => { id.merge(x.a, x.b); id.suggestions = id.suggestions.filter(y => y !== x); persist(); renderGallery(); };
    li.querySelector('[data-n]').onclick = () => { id.covis.add(x.a < x.b ? `${x.a}|${x.b}` : `${x.b}|${x.a}`); id.suggestions = id.suggestions.filter(y => y !== x); persist(); renderGallery(); };
    return li;
  }));
  $('#suggestBox').hidden = !s.children.length;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));


// ---------- local data harvest + owner labelling (IndexedDB only, nothing is uploaded) ----------
function wireIdentity() { id.central = !!sync.token; id.onMerge = (keep, gone) => { harvestQ = harvestQ.then(() => harvest?.relabel(gone, keep)).catch(e => console.warn('[koi] harvest', e)); }; }

// a full frame is kept for detector training only if it is sharp, not washed out by sky glare, big enough, and shows
// no person or screen overlay (privacy; the owner's rule: unwanted regions are not labelled)
const SHARP_MIN = 600, GLARE_MAX = 0.3, SIDE_MIN = 480;
function frameWorthKeeping() {
  if (LEVEL >= 2) return false; // leanest mode (iPhones, or after crashes): no full frames kept
  if (!det.frameBlob) return false;
  const recentObjs = objs && performance.now() - objs.at < 4000 ? objs.list : [];
  return det.frameSharp >= SHARP_MIN && (det.glare || 0) <= GLARE_MAX && Math.min(view.width, view.height) >= SIDE_MIN &&
    !recentObjs.some(o => o.name === 'person' || o.fixed) && !objZones.length;
}
// what the picture came from: for analysis of data quality later, never a reason to keep or drop data
function cameraInfo() {
  return { src: video.srcObject ? 'camera' : 'file', vw: video.videoWidth, vh: video.videoHeight, pass_ms: Math.round(passMs || 0),
    mobile: !!(navigator.userAgentData?.mobile ?? /Mobi|Android|iPhone|iPad/.test(navigator.userAgent)),
    platform: navigator.userAgentData?.platform || navigator.platform || '' };
}

// self-healing review queue: a crop the system could not place gets its label automatically once the gallery has learned
// enough to place it clearly (score and margin above the live thresholds); the owner may still answer, nothing waits on
// them. Crops nobody could place within 7 days are dropped.
async function autoResolve() {
  if (!harvest) return;
  let n = 0;
  for (const x of await harvest.pending()) {
    if (!sameEmbedder(x.tag, embedder())) continue;
    const [fid, sc, second] = id.matchOne(Float32Array.from(x.emb), x.hist);
    if (fid != null && sc >= id.cfg.matchSim + 0.05 && sc - second >= 0.05) { await harvest.resolve(x, fid, 'auto'); n++; }
    else if (Date.now() - x.at > 7 * 864e5) { await harvest.resolve(x, null); n++; }
  }
  if (n) reviewDirty = true;
}

function collect(tracks, dets, frameBlob, W, H) {
  const frameDets = frameBlob ? dets.map(d => ({ box: d.box, score: d.score, mask: d.mask, mw: d.mw, mh: d.mh, msc: d.msc })) : null;
  // copy what each track shows now: by the time the queue gets here the next frame has replaced t.obs
  const items = tracks.map(t => ({ t, o: t.obs, fishId: t.fishId, sure: t.sure, grade: id.grade(t), s: t.sizes?.at(-1) ?? null, ripe: t.hits >= 45 && t.embN >= 3 }));
  harvestQ = harvestQ.then(async () => {
    if (frameBlob) { await harvest.addFrame(frameBlob, W, H, frameDets, { sharp: Math.round(det.frameSharp), glare: +(det.glare || 0).toFixed(3), ...cameraInfo() }); reviewDirty = true; }
    for (const { t, o, fishId, sure, grade, s, ripe } of items) {
      if (!o?.emb || !o.img) continue;
      const rec = { emb: o.emb, hist: o.hist, q: o.q, len: o.geo.len, s, view: Math.round(o.geo.heading / (Math.PI / 2)) & 3, grade, cam: cameraInfo() };
      if (sure) { if (grade && id.fish.has(fishId) && await harvest.addConfident(fishId, rec, await o.img)) reviewDirty = true; } // red-flagged: no data
      else if (!t.pendingSaved && ripe) { // lingering undecided track: ask the owner
        t.pendingSaved = true;
        const cand = [...id.fish.values()].map(f => [f.id, id.score(t, f)]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(x => x[0]);
        if (await harvest.addPending(cand, rec, await o.img)) reviewDirty = true;
      }
    }
  }).catch(e => console.warn('[koi] harvest', e));
}

let reviewDirty = false, reviewUrls = [], reviewBusy = false;
setInterval(() => { if (reviewDirty && !reviewBusy) { reviewDirty = false; renderReview(); } }, 2000);

async function renderReview() {
  if (!harvest) return;
  const items = (await harvest.pending()).sort((a, b) => b.q - a.q).slice(0, 6);
  reviewUrls.forEach(URL.revokeObjectURL); reviewUrls = [];
  $('#review').replaceChildren(...items.map(x => {
    const li = document.createElement('li'), img = document.createElement('img'), url = URL.createObjectURL(x.img);
    reviewUrls.push(url); img.src = url; img.alt = 'ภาพปลาที่ระบบไม่แน่ใจ';
    const btns = document.createElement('div');
    // one answer per picture: a double tap must not make two fish
    const btn = (text, fn, cls) => { const b = document.createElement('button'); b.textContent = text; if (cls) b.className = cls; b.onclick = async () => { for (const o of btns.children) o.disabled = true; reviewBusy = true; try { await fn(); } finally { reviewBusy = false; renderReview(); renderGallery(); } }; btns.append(b); };
    // a picture saved under an older embedder is re-embedded before it teaches the gallery
    const fresh = async () => (sameEmbedder(x.tag, embedder()) ? x : { ...x, emb: (await engine.embedImages([x.img]))[0], tag: engine.embedTag });
    for (const fid of x.cand) {
      const f = id.fish.get(id.resolve(fid));
      if (f) btn(f.name, async () => { const y = await fresh(); id.addExemplar(f, y.emb, y.view, 1); await harvest.resolve(y, f.id); persist(); });
    }
    btn('ตัวใหม่', async () => {
      const y = await fresh();
      const f = id.newFish({ id: 'owner', embs: [y.emb], hist: y.hist, sizes: y.s != null ? [y.s] : [], thumb: await blobUrl(y.img), thumbQ: y.q }, Date.now());
      await harvest.resolve(y, f.id); persist();
    });
    btn('ไม่ใช่ปลา / ข้าม', () => harvest.resolve(x, null), 'quiet');
    li.append(img, btns);
    return li;
  }));
  const c = harvest.counts;
  $('#reviewBox').hidden = !items.length;
  $('#dataCount').textContent = `ข้อมูลในเครื่อง: ภาพปลา ${c.samples} · เฟรม ${c.frames} · ยืนยันโดยเจ้าของ ${c.owner}`;
}
const blobUrl = b => new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(b); });

// ---------- model update: rebuild each fish's views from crops kept on this device, names survive ----------
async function migrate() {
  const fish = [...id.fish.values()];
  let done = 0;
  for (const f of fish) {
    status(`อัปเดตโมเดลจำหน้าปลา: ปรับคลังปลา ${++done}/${fish.length}`);
    const kept = harvest ? (await harvest.ofFish(f.id)).sort((a, b) => b.q - a.q).slice(0, id.cfg.maxExemplars) : [];
    // Thumbnail embeddings collapse distinct fish and trigger wrong auto-merges; learn again when seen.
    if (!kept.length) continue;
    const srcs = kept.map(x => x.img);
    const embs = [];
    for (const s of srcs) { try { embs.push((await engine.embedImages([s]))[0]); } catch (e) { embs.push(null); console.warn('[koi] migrate crop', f.id, e); } } // a bad crop costs only itself
    embs.forEach((e, i) => e && id.addExemplar(f, e, kept[i]?.view ?? 0, kept[i]?.q ?? 0.3));
    if (harvest) for (const [i, x] of kept.entries()) if (embs[i]) await harvest.setEmb(x, embs[i]);
  }
  id.needsReembed = false;
  persist();
  if (harvest) await harvest.init(embedder(), pond.id, legacyPond); // reload the novelty cache with the new embeddings
  status(`ปรับคลังปลา ${fish.length} ตัวกับโมเดลใหม่แล้ว ชื่อเดิมยังอยู่`);
  await new Promise(r => setTimeout(r, 2500));
}
