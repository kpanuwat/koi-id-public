#!/usr/bin/env bash
# Local test of the API abuse limits (docs/public/HARDENING.md). Never touches production: local D1 in a temp dir,
# a random local-only SYNC_TOKEN in .dev.vars (removed at the end), wrangler pages dev on 127.0.0.1.
# Usage: scripts/api_limits_test.sh [port]
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${1:-8799}
STATE=$(mktemp -d /tmp/koi-limits-XXXX)
[ -e .dev.vars ] && { echo ".dev.vars exists; move it away first" >&2; exit 1; }
printf 'SYNC_TOKEN=%s\n' "$(openssl rand -hex 32)" > .dev.vars
PID=
cleanup() { [ -n "$PID" ] && kill -- -"$PID" 2>/dev/null; wait "$PID" 2>/dev/null || true; rm -f .dev.vars; rm -rf "$STATE"; }
trap cleanup EXIT

for f in migrations/*.sql; do
  npx --yes wrangler@4 d1 execute koi-id --local --persist-to "$STATE" --file "$f" >/dev/null
done
setsid npx --yes wrangler@4 pages dev app --ip 127.0.0.1 --port "$PORT" --persist-to "$STATE" >"$STATE/wrangler.log" 2>&1 &
PID=$!
for _ in $(seq 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/api/ping" && break; sleep 1; done

status=0
ADMIN=$(sed -n 's/^SYNC_TOKEN=//p' .dev.vars) BASE="http://127.0.0.1:$PORT/api" node --input-type=module <<'EOF' || status=1
const { BASE, ADMIN } = process.env;
let fails = 0;
const ok = (cond, what) => { console.log(`${cond ? 'PASS' : 'FAIL'} ${what}`); if (!cond) fails++; };
const call = async (path, token, init = {}) => {
  const r = await fetch(`${BASE}/${path}`, { ...init, headers: { ...(init.headers || {}), ...(token ? { authorization: `Bearer ${token}` } : {}) } });
  let body = null; try { body = await r.json(); } catch {}
  return { status: r.status, body };
};
const post = (path, token, data, method = 'POST') => call(path, token, { method, headers: { 'content-type': 'application/json' }, body: typeof data === 'string' ? data : JSON.stringify(data) });
const enrol = async () => (await post('enroll', null, { device: crypto.randomUUID() })).body.token;
const fish = (n, from = 1) => Array.from({ length: n }, (_, i) => ({ id: from + i, name: `ปลา ${from + i}`, ex: [{ e: [0.1, 0.2, 0.3, 0.4], q: 1, view: 0 }], hist: [0, 0, 0, 0, 0, 0, 0], thumb: '' }));
const gal = (fs, nextFish) => ({ v: 1, embedTag: 't', nextFish: nextFish ?? Math.max(0, ...fs.map(f => f.id)) + 1, covis: [], alias: [], gone: [], joins: [], fish: fs });
const frameForm = (id, pond, dets, kb = 20) => {
  const f = new FormData();
  f.append('meta', JSON.stringify({ id, pond, w: 100, h: 100, at: Date.now(), dets }));
  f.append('img', new Blob([new Uint8Array(kb << 10)], { type: 'image/jpeg' }), 'x.jpg');
  return f;
};

// normal flow
const A = await enrol(), B = await enrol();
ok(A && B, 'enrol two devices');
ok((await call('ping', A)).status === 200, 'ping');
ok((await post('log', A, { ev: 'start', b: 'v0.064' })).status === 200, 'log');
const pond = crypto.randomUUID();
let r = await post(`pond/${pond}`, A, { base: 0, name: 'บ่อ 1', lat: 13.7456789, lon: 100.5345678, gallery: gal(fish(10)) }, 'PUT');
ok(r.status === 200 && r.body.ver === 1, `pond create ${r.status} ${JSON.stringify(r.body)}`);
r = await call(`pond/${pond}`, A);
ok(r.status === 200 && r.body.gallery?.fish.length === 10, 'pond pull');
r = await call(`pond/${pond}?since=1`, A);
ok(r.body.gallery === null && r.body.ver === 1, 'pond pull unchanged');
r = await post(`pond/${pond}`, A, { base: 1, name: 'บ่อ 1', lat: 13.746, lon: 100.535, gallery: gal(fish(12)) }, 'PUT');
ok(r.status === 200 && r.body.ver === 2, 'gallery save within limits');
r = await call('ponds', A);
const row = r.body.find(p => p.id === pond);
ok(row && row.lat === 13.746 && row.lon === 100.535 && row.name === 'บ่อ 1', `ponds list, lat/lon rounded ${JSON.stringify(row)}`);
ok((await post(`pond/${pond}`, B, { base: 0, gallery: gal(fish(12)) }, 'PUT')).body.stale === true, 'stale save still {stale}');
ok((await call('frame', A, { method: 'POST', body: frameForm(crypto.randomUUID(), pond, [{ box: [1, 2, 3, 4], score: 0.9 }]) })).status === 200, 'frame within limits');

// abuse
r = await post(`pond/${pond}`, B, { base: 2, gallery: gal(fish(12), 1e300) }, 'PUT');
ok(r.status === 400, `nextFish 1e300 -> ${r.status} ${r.body?.error}`);
r = await post(`pond/${pond}`, B, { base: 2, gallery: gal(fish(12), 13.5) }, 'PUT');
ok(r.status === 400, `nextFish 13.5 -> ${r.status}`);
r = await post(`pond/${pond}`, B, { base: 2, gallery: { ...gal(fish(12)), fish: [...fish(11), { id: 12, ex: [{ e: new Array(5000).fill(0) }] }] } }, 'PUT');
ok(r.status === 400, `oversized exemplar -> ${r.status} ${r.body?.error}`);
r = await post(`pond/${pond}`, B, { base: 2, name: 'hijack', lat: 1, lon: 1, gallery: gal(fish(12)) }, 'PUT');
ok(r.status === 200, 'B saves (allowed: autonomous ponds)');
const after = (await call('ponds', A)).body.find(p => p.id === pond);
ok(after.name === 'บ่อ 1' && after.lat === 13.746 && after.lon === 100.535, `name/lat/lon not changed by another device ${JSON.stringify(after)}`);
let ver = r.body.ver;
// new device removes fish a few at a time: daily total is capped, not per request
const drops = [];
for (const n of [10, 8, 6]) { r = await post(`pond/${pond}`, B, { base: ver, gallery: gal(fish(n), 13) }, 'PUT'); drops.push(r.status); if (r.body?.ver) ver = r.body.ver; }
ok(drops[0] === 200 && drops.at(-1) === 403, `new device drops 2,2,2 of 12 fish -> ${drops.join(',')}`);
// junk saves cannot flush roll-back history: A's last good version survives 40 junk saves by B
const start = ver;
for (let i = 0; i < 40; i++) { r = await post(`pond/${pond}`, B, { base: ver, gallery: gal([...fish(10)].map(f => ({ ...f, name: `junk${i}` })), 13) }, 'PUT'); ver = r.body.ver ?? ver; }
ok(ver - start === 40, `40 junk saves accepted (ver ${start} -> ${ver})`);
console.log(`POND ${pond}`);
// oversized dets
r = await call('frame', A, { method: 'POST', body: frameForm(crypto.randomUUID(), pond, [{ mask: 'x'.repeat(300 << 10) }]) });
ok(r.status === 413, `dets 300 KB -> ${r.status}`);
// another device cannot overwrite A's frame
const fid = crypto.randomUUID();
await call('frame', A, { method: 'POST', body: frameForm(fid, pond, []) });
r = await call('frame', B, { method: 'POST', body: frameForm(fid, pond, []) });
ok(r.status === 403, `overwrite another device's frame -> ${r.status}`);
// oversized log
r = await post('log', A, { ev: 'x', pad: 'y'.repeat(5000) });
ok(r.status === 413, `log 5 KB -> ${r.status}`);
// log quota: 1000 per device per day
const C = await enrol();
let last = 0, firstRefused = 0;
for (let i = 1; i <= 1005; i += 15) {
  const batch = await Promise.all(Array.from({ length: Math.min(15, 1006 - i) }, () => post('log', C, { ev: 'flood' })));
  for (const b of batch) { last = b.status; if (b.status === 429 && !firstRefused) firstRefused = i; }
}
ok(last === 429 && firstRefused > 985, `over-quota logs -> 429 (first refused near call ${firstRefused})`);
ok((await post('log', A, { ev: 'still' })).status === 200, 'other device still logs');
// bad json -> generic 400, no internal text
r = await post('log', A, '{nope');
ok(r.status === 400 && r.body.error === 'bad json', `bad json -> ${r.status} ${r.body?.error}`);
r = await call('me', ADMIN);
ok(r.body.admin === true, 'admin token works');
process.exitCode = fails ? 1 : 0;
console.log(fails ? `${fails} FAILED` : 'ALL PASSED');
EOF
# roll-back history check straight from the local D1: version 2 (device A's last good save) must still be stored
KEYS=$(npx --yes wrangler@4 d1 execute koi-id --local --persist-to "$STATE" --json --command \
  "SELECT g.ver, g.device, (SELECT COUNT(*) FROM blobs b WHERE b.key = g.key) AS parts FROM gallery_keys g ORDER BY g.ver" 2>/dev/null)
echo "$KEYS" | node -e '
let s = ""; process.stdin.on("data", d => s += d).on("end", () => {
  const rows = JSON.parse(s)[0].results, v2 = rows.find(r => r.ver === 2);
  console.log(`${v2 && v2.parts > 0 ? "PASS" : "FAIL"} roll-back: device A version 2 kept after junk saves (${rows.length} versions kept: ${rows.map(r => r.ver).join(",")})`);
  process.exit(v2 && v2.parts > 0 ? 0 : 1);
});' || status=1
exit $status
