// Central sync with the Koi ID API (Cloudflare Pages Functions, D1 + private R2; see functions/api).
// Pushes each pond's gallery, labelled fish crops and sparse frames; pulls galleries so another device can carry on.
// No setup: each device enrols itself on first start (its own token) and syncs in the background.
const get = k => { try { return localStorage.getItem(k) || ''; } catch { return ''; } };
const set = (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch {} };

export const deviceId = get('koi.device') || (() => { const d = crypto.randomUUID(); set('koi.device', d); return d; })();

export class Sync {
  constructor(base = new URL('./api', location.href).href) { this.base = base; }
  // an admin code typed earlier wins; otherwise the device's own token from enrol()
  get token() { return get('koi.syncToken') || get('koi.deviceToken'); }
  async enroll() {
    if (this.token) return;
    const r = await fetch(`${this.base}/enroll`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ device: deviceId }) });
    if (!r.ok) throw new Error(`enroll: HTTP ${r.status}`);
    set('koi.deviceToken', (await r.json()).token);
  }
  async req(path, init = {}) {
    const r = await fetch(`${this.base}/${path}`, { ...init, headers: { ...(init.headers || {}), authorization: `Bearer ${this.token}` } });
    if (r.status === 401) throw new Error('รหัสซิงก์ไม่ถูกต้อง');
    if (!r.ok) throw Object.assign(new Error(`${path}: HTTP ${r.status}`), { status: r.status });
    return r.json();
  }
  ping() { return this.req('ping'); }
  ponds() { return this.req('ponds'); }
  me() { return this.req('me'); }
  log(data) { return this.req('log', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(data) }); }
  mergePond(id, into) { return this.req(`pond/${id}/merge`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ into }) }); }
  pullPond(id, since) { return this.req(`pond/${id}${since != null ? `?since=${since}` : ''}`); } // same version -> no gallery sent
  async pushPond(id, body) { // another device got there first -> status 409, the caller merges again
    const r = await this.req(`pond/${id}`, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ...body, device: deviceId }) });
    if (r.stale) throw Object.assign(new Error('stale'), { status: 409 });
    return r;
  }
  push(kind, meta, img) { // kind: 'sample' | 'frame'
    const f = new FormData();
    f.append('meta', JSON.stringify({ ...meta, device: deviceId }));
    if (img) f.append('img', img, `${meta.id}.jpg`);
    return this.req(kind, { method: 'POST', body: f });
  }
}
