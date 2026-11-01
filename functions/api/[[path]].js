// Koi ID sync API (Cloudflare Pages Function). Every call needs `Authorization: Bearer <token>`: a device token from
// POST /api/enroll (each install enrols itself, nobody types a code) or the admin SYNC_TOKEN (Pages secret, tools).
// D1 (env.DB) keeps everything: the index tables, and in `blobs` the images, pond galleries and frame pseudo-labels
// (one object = rows (key, part) of at most PART bytes each, joined in part order).
//   POST /api/enroll     JSON {device}  -> {token}   (no auth; at most ENROLL_PER_IP new devices per IP per day)
//   GET  /api/ping                     -> {ok}
//   POST /api/log        JSON {ev, ...}  -> {ok}   (field diagnostics, <= 4 KB; limits: LIMITS below, 429 when over)
//   GET  /api/me                       -> {days, admin}   (trust: days this device was active)
//   GET  /api/ponds                    -> [{id, name, lat, lon, fish, updated_at, merged_into}]
//   POST /api/pond/:id/merge  JSON {into}  -> {ok}   (this pond is the same water as `into`; devices move over)
//   GET  /api/pond/:id[?since=ver]     -> {updated_at, ver, gallery} (gallery null when since = ver)
//   PUT  /api/pond/:id   JSON {base, name, lat, lon, gallery} -> {ver}; {stale, ver} if base is not the current version
//   POST /api/sample     form {meta: JSON, img?: jpeg}   (upsert; no img = label update)
//   POST /api/frame      form {meta: JSON (incl. dets), img: jpeg}
const ID = /^[0-9a-zA-Z-]{8,64}$/, MAX_IMG = 2 << 20, MAX_JSON = 8 << 20, MAX_DETS = 256 << 10, MAX_LOG = 4096;
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const bad = (msg, status = 400) => json({ error: msg }, status);
// an expected refusal deep inside a route: thrown, turned into bad(msg, status) by the handler
const fail = (msg, status = 400) => { throw Object.assign(new Error(msg), { status }); };
const galleryKey = (pond, ver) => (ver ? `ponds/${pond}/gallery-v${ver}.json` : `ponds/${pond}/gallery.json`);
const num = v => (v !== null && v !== '' && Number.isFinite(+v) ? +v : null);
// pond position: inside the valid range, rounded to 3 decimals (~110 m) like the client does; anything else = unknown
const geo = (v, lim) => { const x = num(v); return x === null || Math.abs(x) > lim ? null : Math.round(x * 1000) / 1000; };
// a body larger than the declared header is still cut by the caller; this refuses a big upload before reading it
const tooBig = (request, max) => +(request.headers.get('content-length') || 0) > max;

const ENROLL_PER_IP = 10, DAY = 864e5, HOUR = 36e5, KEEP_VERSIONS = 30;
// Abuse limits (docs/public/HARDENING.md). Counted per device per UTC day (puth: per hour) in the D1 table `quota`;
// over a limit the call gets 429 and the client retries on its next sync. '*' rows are limits for all devices together.
const LIMITS = {
  log: { n: 1000 },                        // client sends <= 41 per page load
  sample: { n: 3000, bytes: 100e6 },       // labelled crops (small jpegs); device keeps <= 3000
  frame: { n: 300, bytes: 150e6 },         // full frames, one per 10 s while running; device keeps <= 300
  put: { n: 1500 },                        // gallery saves (~1 per 15-30 s while running)
  puth: { n: 300, win: HOUR },
  newpond: { n: 20 },                      // ponds created
  merge: { n: 10 },
};
const GLOBAL = { blob: { bytes: 300e6 }, enroll: { n: 300 } }; // frame + sample bytes, new devices: all devices per day
const MAX_FISH = 1000, MAX_FISH_ID = 1e7, MAX_LIST = 100000, NEXT_FISH_SLACK = 1000;
const hex = buf => [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
const sha = async s => hex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)));
function same(got, want) { // constant time
  if (!want || got.length !== want.length) return false;
  const a = new TextEncoder().encode(got), b = new TextEncoder().encode(want);
  let diff = 0; for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}
// -> {admin: true} | {device, days} | null. Device token = "d.<device id>.<secret>"; a device's trust = days it was active
async function authorized(request, env, now) {
  const got = (request.headers.get('authorization') || '').replace(/^Bearer\s+/i, '');
  if (same(got, env.SYNC_TOKEN || '')) return { admin: true };
  const [kind, id, secret] = got.split('.');
  if (kind !== 'd' || !ID.test(id || '') || !secret) return null;
  const row = await env.DB.prepare('SELECT token_hash, seen, days FROM devices WHERE id = ?').bind(id).first();
  if (!row || !same(await sha(secret), row.token_hash)) return null;
  const newDay = Math.floor(now / DAY) > Math.floor((row.seen || 0) / DAY);
  if (now - (row.seen || 0) > 60e3) await env.DB.prepare('UPDATE devices SET seen = ?, days = days + ? WHERE id = ?').bind(now, newDay ? 1 : 0, id).run();
  return { device: id, days: row.days + (newDay ? 1 : 0) };
}

// object store on D1: a row holds at most 2 MB, so an object is split into parts; readers accept any part size
const PART = 1000000;
async function putBlob(env, key, bytes, type) {
  const now = Date.now(), ins = env.DB.prepare('INSERT INTO blobs (key, part, type, data, at) VALUES (?, ?, ?, ?, ?)');
  const stmts = [env.DB.prepare('DELETE FROM blobs WHERE key = ?').bind(key)];
  for (let i = 0, part = 0; i < bytes.length || part === 0; i += PART, part++) stmts.push(ins.bind(key, part, type, bytes.subarray(i, i + PART), now));
  await env.DB.batch(stmts);
}
// -> Uint8Array | null; one part per query keeps memory near the object size
async function getBlob(env, key) {
  const { results } = await env.DB.prepare('SELECT part, length(data) AS n FROM blobs WHERE key = ? ORDER BY part').bind(key).all();
  if (!results.length) return null;
  const out = new Uint8Array(results.reduce((t, r) => t + r.n, 0));
  let at = 0;
  for (const r of results) {
    const row = await env.DB.prepare('SELECT data FROM blobs WHERE key = ? AND part = ?').bind(key, r.part).first();
    const d = new Uint8Array(row.data);
    out.set(d, at);
    at += d.length;
  }
  return out;
}
async function delBlobs(env, keys) {
  if (keys.length) await env.DB.batch(keys.map(k => env.DB.prepare('DELETE FROM blobs WHERE key = ?').bind(k)));
}

async function image(form) {
  const img = form.get('img');
  if (!img || typeof img === 'string') return null;
  if (img.size > MAX_IMG) fail('image must be under 2 MB', 413);
  if (!/^image\/(jpeg|png)$/.test(img.type)) fail('image must be a jpeg or png');
  return img;
}

// count one call (and its bytes) against a limit; refuses with 429 once over. The counter is bumped first in one
// statement (no race between two calls of the same device), so refused attempts count too.
async function spend(env, who, kind, now, { add = 1, bytes = 0, lim = LIMITS[kind] } = {}) {
  const win = Math.floor(now / (lim.win || DAY));
  const row = await env.DB.prepare(`INSERT INTO quota (who, kind, win, n, bytes, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
    ON CONFLICT(who, kind, win) DO UPDATE SET n = n + ?4, bytes = bytes + ?5 RETURNING n, bytes`).bind(who, kind, win, add, bytes, now).first();
  if ((lim.n && row.n > lim.n) || (lim.bytes && row.bytes > lim.bytes)) fail(`limit reached (${kind}), try again later`, 429);
}

// a gallery as the client writes it (app/reid.js toJSON): refuse shapes that no client makes, they could crash phones
function galleryProblem(g) {
  if (!g || typeof g !== 'object' || g.v !== 1 || !Array.isArray(g.fish)) return 'bad gallery';
  if (g.fish.length > MAX_FISH) return 'too many fish';
  for (const [k, v] of Object.entries(g)) if (Array.isArray(v) && v.length > MAX_LIST) return `gallery ${k} too long`;
  for (const f of g.fish) {
    if (!f || typeof f !== 'object' || !Number.isInteger(f.id) || f.id < 1 || f.id > MAX_FISH_ID) return 'bad fish id';
    if (f.ex != null && (!Array.isArray(f.ex) || f.ex.length > 32 || f.ex.some(x => !x || !Array.isArray(x.e) || x.e.length > 2048))) return 'bad fish exemplars';
    if (f.hist != null && (!Array.isArray(f.hist) || f.hist.length > 64)) return 'bad fish hist';
    if (f.thumb != null && (typeof f.thumb !== 'string' || f.thumb.length > 256e3)) return 'bad fish thumb';
    if (f.name != null && (typeof f.name !== 'string' || f.name.length > 200)) return 'bad fish name';
  }
  return null;
}

// roll-back history that junk saves cannot flush: besides the last KEEP_VERSIONS versions, keep each device's newest
// 2 versions (the 50 devices seen most recently, last 30 days) and the newest version of each of the last 14 days.
// A vandal's saves then push out only the vandal's own versions; every honest device's last good gallery stays.
async function prune(env, pond, now) {
  const { results: rows } = await env.DB.prepare('SELECT ver, key, device, at FROM gallery_keys WHERE pond = ? ORDER BY ver DESC').bind(pond).all();
  const keep = new Set(rows.slice(0, KEEP_VERSIONS).map(r => r.key)), perDev = new Map(), perDay = new Set();
  for (const r of rows) {
    if (!r.at || now - r.at > 30 * DAY) continue;
    const day = Math.floor(r.at / DAY);
    if (now - r.at <= 14 * DAY && !perDay.has(day)) { perDay.add(day); keep.add(r.key); }
    const d = r.device || '?', n = perDev.get(d) || 0;
    if (n < 2 && (perDev.has(d) || perDev.size < 50)) { perDev.set(d, n + 1); keep.add(r.key); }
  }
  const old = rows.filter(r => !keep.has(r.key)).map(r => r.key);
  if (!old.length) return;
  await delBlobs(env, old);
  await env.DB.batch(old.map(k => env.DB.prepare('DELETE FROM gallery_keys WHERE pond = ? AND key = ?').bind(pond, k)));
}

// schema the API needs, applied by the API itself once per isolate (the tool tokens cannot write D1):
// migrations/0009_blobs.sql and 0010_limits.sql. ALTERs fail harmlessly once the column exists.
let schemaReady = false, cleanedAt = 0;
async function schema(env) {
  await env.DB.batch([
    env.DB.prepare('CREATE TABLE IF NOT EXISTS blobs (key TEXT NOT NULL, part INTEGER NOT NULL, type TEXT, data BLOB NOT NULL, at INTEGER, PRIMARY KEY (key, part))'),
    env.DB.prepare('CREATE TABLE IF NOT EXISTS quota (who TEXT NOT NULL, kind TEXT NOT NULL, win INTEGER NOT NULL, n INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0, at INTEGER, PRIMARY KEY (who, kind, win))'),
    env.DB.prepare('CREATE INDEX IF NOT EXISTS quota_at ON quota(at)'),
  ]);
  for (const col of ['device TEXT', 'at INTEGER']) await env.DB.prepare(`ALTER TABLE gallery_keys ADD COLUMN ${col}`).run().catch(() => {});
  schemaReady = true;
}
// retention, at most once an hour per isolate, after the reply: field logs 90 days, quota counters 3 days
async function cleanup(env, now) {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM logs WHERE at < ?').bind(now - 90 * DAY),
    env.DB.prepare('DELETE FROM quota WHERE at < ?').bind(now - 3 * DAY),
  ]);
}

export async function onRequest(context) {
  const { request, env, params } = context, [route, arg, sub] = params.path || [], m = request.method, now = Date.now();
  try {
    if (!schemaReady) await schema(env);
    if (now - cleanedAt > HOUR) { cleanedAt = now; const job = cleanup(env, now).catch(e => console.error('cleanup', e)); context.waitUntil?.(job); }
    if (route === 'enroll' && m === 'POST') {
      if (tooBig(request, 1024)) return bad('too large', 413);
      const b = await request.json().catch(() => ({})), ipHash = await sha(`${request.headers.get('cf-connecting-ip') || ''}|${env.SYNC_TOKEN}`);
      if (!ID.test(b.device || '')) return bad('bad device');
      const n = await env.DB.prepare('SELECT COUNT(*) AS n FROM devices WHERE ip_hash = ? AND created > ?').bind(ipHash, now - DAY).first();
      if (n.n >= ENROLL_PER_IP) return bad('too many new devices from this network today', 429);
      await spend(env, '*', 'enroll', now, { lim: GLOBAL.enroll });
      const secret = hex(crypto.getRandomValues(new Uint8Array(32)));
      const res = await env.DB.prepare('INSERT INTO devices (id, token_hash, ip_hash, created, seen) VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING')
        .bind(b.device, await sha(secret), ipHash, now, now).run();
      if (!res.meta.changes) return bad('device already enrolled', 409);
      return json({ token: `d.${b.device}.${secret}` });
    }
    const who = await authorized(request, env, now);
    if (!who) return bad('unauthorized', 401);
    // this device's limits (the admin token has none)
    const mine = (kind, opt) => (who.admin ? null : spend(env, who.device, kind, now, opt));
    if (route === 'ping') return json({ ok: true });
    // admin write of one part of an object: PUT /api/blob/<part>?key=<key>&type=<mime>, raw body (tools; used for the R2 import)
    if (route === 'blob' && m === 'PUT') {
      if (!who.admin) return bad('admin only', 403);
      const u = new URL(request.url), key = u.searchParams.get('key') || '', part = Number(arg);
      if (!/^ponds\/[0-9a-zA-Z-]{8,64}\/[\w./-]{1,200}$/.test(key) || key.includes('..') || !Number.isInteger(part) || part < 0) return bad('bad key or part');
      const data = new Uint8Array(await request.arrayBuffer());
      if (!data.length || data.length > PART) return bad('part size', 413);
      await env.DB.prepare('INSERT OR REPLACE INTO blobs (key, part, type, data, at) VALUES (?, ?, ?, ?, ?)').bind(key, part, String(u.searchParams.get('type') || '').slice(0, 40), data, now).run();
      return json({ ok: true, key, part, n: data.length });
    }
    if (route === 'log' && m === 'POST') {
      if (tooBig(request, 3 * MAX_LOG)) return bad('log too large', 413);
      const text = await request.text();
      if (text.length > MAX_LOG) return bad('log too large', 413);
      const data = JSON.parse(text);
      if (!data || typeof data !== 'object' || Array.isArray(data)) return bad('bad log');
      await mine('log');
      await env.DB.prepare('INSERT INTO logs (device, at, ev, data) VALUES (?, ?, ?, ?)').bind(who.device || 'admin', now, String(data.ev || '').slice(0, 20), text).run();
      return json({ ok: true });
    }
    if (route === 'me') return json({ days: who.admin ? 999 : who.days, admin: !!who.admin });
    if (route === 'pond' && sub === 'merge' && m === 'POST' && ID.test(arg || '')) {
      const into = String((await request.json().catch(() => ({}))).into || '');
      if (!ID.test(into) || into === arg) return bad('bad target');
      await mine('merge');
      const res = await env.DB.prepare(`UPDATE ponds SET merged_into = ?2, updated_at = ?3 WHERE id = ?1 AND merged_into IS NULL
        AND EXISTS (SELECT 1 FROM ponds WHERE id = ?2 AND merged_into IS NULL)`).bind(arg, into, now).run();
      return res.meta.changes ? json({ ok: true }) : bad('pond or target missing or already merged', 409);
    }
    if (route === 'ponds' && m === 'GET') {
      const { results } = await env.DB.prepare('SELECT id, name, lat, lon, fish, updated_at, merged_into FROM ponds ORDER BY updated_at DESC').all();
      return json(results);
    }
    if (route === 'pond' && ID.test(arg || '')) {
      if (m === 'GET') {
        const row = await env.DB.prepare('SELECT updated_at, ver, gkey FROM ponds WHERE id = ?').bind(arg).first();
        const since = new URL(request.url).searchParams.get('since');
        if (row && since !== null && +since === row.ver) return json({ updated_at: row.updated_at, ver: row.ver, gallery: null }); // unchanged: no download
        const obj = row && await getBlob(env, row.gkey || galleryKey(arg, row.ver));
        return json(obj ? { updated_at: row.updated_at, ver: row.ver, gallery: JSON.parse(new TextDecoder().decode(obj)) } : { updated_at: 0, ver: row?.ver ?? 0, gallery: null });
      }
      if (m === 'PUT') {
        if (tooBig(request, MAX_JSON)) return bad('gallery too large', 413);
        const text = await request.text();
        if (text.length > MAX_JSON) return bad('gallery too large', 413);
        const b = JSON.parse(text), g = b?.gallery, problem = galleryProblem(g);
        if (problem) return bad(problem);
        // compare-and-swap: each version is its own object; the row moves to it only if nobody moved it first
        const row = await env.DB.prepare('SELECT ver, fish, next_fish FROM ponds WHERE id = ?').bind(arg).first(), cur = row?.ver ?? 0;
        if (b.base !== cur) return json({ stale: true, ver: cur });
        // fish numbers are never reused: a gallery that counts from lower down was renumbered by a broken client
        // (25 Oct 2026 one did, merging into an empty reply) and is refused, whoever sends it
        if (!Number.isInteger(g.nextFish)) return bad('bad nextFish');
        if (!(g.nextFish >= (row?.next_fish || 0))) return bad('fish numbering went backwards', 409);
        // ... and may not jump far ahead either (one huge number would block every honest client)
        const top = Math.max(row?.next_fish || 0, g.fish.length + 1, ...g.fish.map(f => f.id + 1));
        if (g.nextFish > top + NEXT_FISH_SLACK) return bad('fish numbering jumps too far');
        // a device active fewer than 3 days may not remove many fish (vandalism / a broken client), counted per pond per
        // day across all its saves; merges of a few duplicates still go through. Old versions are kept (prune), so any
        // damage can be rolled back.
        const had = row?.fish || 0, drop = Math.max(0, had - g.fish.length), newbie = !who.admin && who.days < 3;
        if (newbie && drop) {
          const d = await env.DB.prepare('SELECT n FROM quota WHERE who = ? AND kind = ? AND win = ?').bind(who.device, `drop:${arg}`, Math.floor(now / DAY)).first();
          if ((d?.n || 0) + drop > Math.max(3, 0.2 * had)) return bad('too many fish removed for a new device', 403);
        }
        await mine('puth');
        await mine('put');
        if (!row) await mine('newpond');
        // a key of its own per attempt: a writer that loses the race deletes only its own object, never the winner's
        const ver = cur + 1, key = `ponds/${arg}/gallery-v${ver}-${crypto.randomUUID().slice(0, 8)}.json`, device = who.device || 'admin';
        await putBlob(env, key, new TextEncoder().encode(JSON.stringify(g)), 'application/json');
        const vals = [arg, String(b.name || '').slice(0, 60), geo(b.lat, 90), geo(b.lon, 180), g.fish.length, String(g.embedTag || '').slice(0, 80), device, now, ver, cur, g.nextFish, key];
        // name and position are set once (the first save that has them); only the admin token changes them later, so
        // another device cannot rename a pond or move it onto someone else's spot (the client picks ponds by position)
        const keepOld = who.admin ? 'COALESCE(NULLIF(?2, \'\'), name), lat = COALESCE(?3, lat), lon = COALESCE(?4, lon)'
          : 'COALESCE(NULLIF(name, \'\'), NULLIF(?2, \'\')), lat = COALESCE(lat, ?3), lon = CASE WHEN lat IS NULL THEN COALESCE(?4, lon) ELSE lon END';
        const res = row
          ? await env.DB.prepare(`UPDATE ponds SET name = ${keepOld}, fish = ?5, embed_tag = ?6, device = ?7, updated_at = ?8, ver = ?9, next_fish = ?11, gkey = ?12 WHERE id = ?1 AND ver = ?10`).bind(...vals).run()
          : await env.DB.prepare('INSERT INTO ponds (id, name, lat, lon, fish, embed_tag, device, updated_at, ver, next_fish, gkey) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) ON CONFLICT(id) DO NOTHING').bind(...vals.slice(0, 9), vals[10], key).run();
        if (!res.meta.changes) { await delBlobs(env, [key]); return json({ stale: true, ver: cur }); }
        if (newbie && drop) await spend(env, who.device, `drop:${arg}`, now, { add: drop, lim: {} });
        await env.DB.prepare('INSERT INTO gallery_keys (pond, ver, key, device, at) VALUES (?, ?, ?, ?, ?)').bind(arg, ver, key, device, now).run();
        await prune(env, arg, now);
        return json({ ok: true, updated_at: now, ver });
      }
    }
    // samples and frames: the row records the authenticated device (not the one the client names), and a device may
    // overwrite only its own records
    const owned = async (table, id) => {
      const r = await env.DB.prepare(`SELECT device FROM ${table} WHERE id = ?`).bind(id).first();
      if (r && !who.admin && r.device !== who.device) fail('id taken', 403);
    };
    if (route === 'sample' && m === 'POST') {
      if (tooBig(request, MAX_IMG + (64 << 10))) return bad('sample too large', 413);
      const form = await request.formData(), x = JSON.parse(form.get('meta') || '{}');
      if (!ID.test(x.id || '') || !ID.test(x.pond || '')) return bad('bad id');
      const img = await image(form), key = `ponds/${x.pond}/samples/${x.id}.jpg`;
      await owned('samples', x.id);
      await mine('sample', { bytes: img?.size || 0 });
      if (img && !who.admin) await spend(env, '*', 'blob', now, { add: 0, bytes: img.size, lim: GLOBAL.blob });
      if (img) await putBlob(env, key, new Uint8Array(await img.arrayBuffer()), img.type);
      // an owner answer counts as gold only from a device with some history (>= 3 active days); a new device's answers are silver
      let grade = ['gold', 'silver', 'bronze'].includes(x.grade) ? x.grade : null;
      if (grade === 'gold' && !who.admin && who.days < 3) grade = 'silver';
      await env.DB.prepare(`INSERT INTO samples (id, pond, fish, label, q, len, s, view, tag, hist, at, session, device, img_key, updated_at, grade, cam)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
        ON CONFLICT(id) DO UPDATE SET fish = ?3, label = ?4, img_key = COALESCE(?14, img_key), updated_at = ?15, grade = COALESCE(?16, grade)`)
        .bind(x.id, x.pond, num(x.fish), String(x.label || '').slice(0, 16), num(x.q), num(x.len), num(x.s), num(x.view), String(x.tag || '').slice(0, 80),
          JSON.stringify((Array.isArray(x.hist) ? x.hist : []).slice(0, 7).map(num)), num(x.at), String(x.session || '').slice(0, 64), who.device || String(x.device || '').slice(0, 64), img ? key : null, now,
          grade, JSON.stringify(x.cam || {}).slice(0, 500)).run();
      return json({ ok: true });
    }
    if (route === 'frame' && m === 'POST') {
      if (tooBig(request, MAX_IMG + MAX_DETS + (64 << 10))) return bad('frame too large', 413);
      const form = await request.formData(), x = JSON.parse(form.get('meta') || '{}');
      if (!ID.test(x.id || '') || !ID.test(x.pond || '')) return bad('bad id');
      const dets = JSON.stringify(Array.isArray(x.dets) ? x.dets : []);
      if (dets.length > MAX_DETS) return bad('dets too large', 413);
      const img = await image(form);
      if (!img) return bad('frame needs an image');
      await owned('frames', x.id);
      await mine('frame', { bytes: img.size + dets.length });
      if (!who.admin) await spend(env, '*', 'blob', now, { add: 0, bytes: img.size + dets.length, lim: GLOBAL.blob });
      const base = `ponds/${x.pond}/frames/${x.id}`;
      await putBlob(env, `${base}.jpg`, new Uint8Array(await img.arrayBuffer()), img.type);
      await putBlob(env, `${base}.json`, new TextEncoder().encode(dets), 'application/json');
      await env.DB.prepare(`INSERT OR REPLACE INTO frames (id, pond, w, h, at, session, device, img_key, dets_key, updated_at, quality) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
        .bind(x.id, x.pond, num(x.w), num(x.h), num(x.at), String(x.session || '').slice(0, 64), who.device || String(x.device || '').slice(0, 64), `${base}.jpg`, `${base}.json`, now,
          JSON.stringify(x.quality || {}).slice(0, 500)).run();
      return json({ ok: true });
    }
    return bad('not found', 404);
  } catch (e) {
    if (e.status) return bad(e.message, e.status);
    // internal details (D1/SQL text) stay in the Pages log, the caller gets a generic message
    console.error('api', route, e);
    return bad(e instanceof SyntaxError ? 'bad json' : 'request failed', e instanceof SyntaxError ? 400 : 500);
  }
}
