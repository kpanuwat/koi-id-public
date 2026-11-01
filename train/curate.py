"""Build a training set from the central store (D1 index + private R2 images), with green/red-flag curation.
    .venv/bin/python curate.py [--with-bronze] [--pond <id>]      # -> data/central/<pond>/<fish>/<id>.jpg + data/central/manifest.jsonl
    .venv/bin/python curate.py --selftest
Tiers: gold = owner confirmed, silver = confirmed + followed >= 2 s + >= 3 views + told apart (no red flag), bronze = confirmed only
(skipped unless --with-bronze). Red-flagged crops never reach the server (app.js).
Curation per pond and fish:
  1. near-duplicates across devices dropped (64-bit dHash, Hamming <= 4), keeping the better grade / quality
  2. crops that do not look like their fish dropped (embedder cosine to the fish's gold-first centroid below
     max(0.5, median - 3 MAD))
  3. weight = grade (gold 2, silver 1, bronze 0.5) x device factor (device mean quality / overall mean, clamped 0.5..1.5)
Needs `npx wrangler login`. ponytail: one wrangler call per image; switch to the R2 S3 API with access keys past a few thousand.
"""
import json, shlex, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'central'
GRADE_W = {'gold': 2.0, 'silver': 1.0, 'bronze': 0.5}
RANK = {'gold': 0, 'silver': 1, 'bronze': 2}


def wrangler(*args, binary=False):
    cmd = ['npx', '--yes', 'wrangler@4', *args]
    r = subprocess.run(['zsh', '-lc', 'source ~/.nvm/nvm.sh >/dev/null; ' + ' '.join(shlex.quote(a) for a in cmd)], capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    if r.returncode: raise RuntimeError((r.stderr or r.stdout).decode()[-400:])
    return r.stdout if binary else r.stdout.decode()


def d1(sql):
    out = wrangler('d1', 'execute', 'koi-id', '--remote', '--json', '--command', sql)
    return json.loads(out[out.index('['):])[0]['results']


def dhash(img):  # img: HxW grey uint8
    import cv2
    s = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def dedup(items, hashes, max_dist=4):
    """items sorted best first; returns indices kept (a later item within max_dist bits of a kept one is a duplicate)."""
    kept = []
    for i, h in enumerate(hashes):
        if all(bin(h ^ hashes[k]).count('1') > max_dist for k in kept): kept.append(i)
    return kept


def outliers(embs, grades, floor=0.5):
    """embs: N x D (L2-normalised). Centroid from gold crops when there are >= 3, else all. Returns bool mask of outliers."""
    e = np.asarray(embs, np.float32)
    g = np.array([x == 'gold' for x in grades])
    c = e[g].mean(0) if g.sum() >= 3 else e.mean(0)
    c /= np.linalg.norm(c) + 1e-9
    cos = e @ c
    med = np.median(cos); mad = np.median(np.abs(cos - med)) + 1e-6
    return cos < max(floor, med - 3 * 1.4826 * mad)


def device_factors(rows):
    q = np.array([r['q'] or 0 for r in rows], float); overall = q.mean() if len(q) else 1
    by = {}
    for r in rows: by.setdefault(r['device'], []).append(r['q'] or 0)
    return {d: float(np.clip(np.mean(v) / max(overall, 1e-6), 0.5, 1.5)) for d, v in by.items()}


def embedder():
    import onnxruntime as ort
    s = ort.InferenceSession(str(ROOT.parent / 'app' / 'models' / 'embed.onnx'), providers=['CPUExecutionProvider'])
    def run(bgr):
        import cv2
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 112)).astype(np.float32) / 255
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        v = s.run(None, {'pixel_values': x})[0][0]
        return v / (np.linalg.norm(v) + 1e-9)
    return run


def main():
    import cv2
    tiers = ('gold', 'silver', 'bronze') if '--with-bronze' in sys.argv else ('gold', 'silver')
    pond = sys.argv[sys.argv.index('--pond') + 1] if '--pond' in sys.argv else None
    where = f"grade IN ({','.join(repr(t) for t in tiers)}) AND img_key IS NOT NULL AND fish IS NOT NULL" + (f" AND pond = '{pond}'" if pond else '')
    rows = d1(f'SELECT id, pond, fish, grade, q, device, img_key FROM samples WHERE {where}')
    print(len(rows), 'labelled crops in the central store')
    if not rows: return
    dev = device_factors(rows); emb = embedder(); manifest = []
    for (pd, fish) in sorted({(r['pond'], r['fish']) for r in rows}):
        group = sorted([r for r in rows if r['pond'] == pd and r['fish'] == fish], key=lambda r: (RANK[r['grade']], -(r['q'] or 0)))
        imgs = []
        for r in group:
            path = OUT / pd / str(fish) / f"{r['id']}.jpg"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(wrangler('r2', 'object', 'get', f"koi-id-data/{r['img_key']}", '--remote', '--pipe', binary=True))
            imgs.append(cv2.imread(str(path)))
        ok = [i for i, im in enumerate(imgs) if im is not None]
        keep = set(ok[k] for k in dedup([group[i] for i in ok], [dhash(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)) for i in ok]))
        kk = sorted(keep)
        bad = set(np.array(kk)[outliers([emb(imgs[i]) for i in kk], [group[i]['grade'] for i in kk])].tolist()) if len(kk) >= 4 else set()
        for i, r in enumerate(group):
            why = 'unreadable' if i not in ok else 'duplicate' if i not in keep else 'outlier' if i in bad else None
            manifest.append({**r, 'path': str((OUT / pd / str(fish) / f"{r['id']}.jpg").relative_to(ROOT)), 'keep': why is None, 'drop': why,
                             'weight': round(GRADE_W[r['grade']] * dev.get(r['device'], 1.0), 3)})
    for pd in {r['pond'] for r in rows}:  # covis (fish seen together = distinct) for the trainer's negatives
        (OUT / pd).mkdir(parents=True, exist_ok=True)
        (OUT / pd / 'gallery.json').write_bytes(wrangler('r2', 'object', 'get', f'koi-id-data/ponds/{pd}/gallery.json', '--remote', '--pipe', binary=True))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.jsonl').write_text('\n'.join(json.dumps(m, ensure_ascii=False) for m in manifest))
    from collections import Counter
    print('kept', sum(m['keep'] for m in manifest), 'of', len(manifest), dict(Counter(m['drop'] for m in manifest if m['drop'])), '->', OUT / 'manifest.jsonl')


def selftest():
    import cv2
    rng = np.random.default_rng(0)
    a = (rng.random((112, 224)) * 255).astype(np.uint8); b = a.copy(); b[:2] = 0; c = (rng.random((112, 224)) * 255).astype(np.uint8)
    h = [dhash(a), dhash(b), dhash(c)]
    assert dedup([0, 1, 2], h) == [0, 2], 'near-identical crop from another device dropped'
    e = rng.normal(size=(10, 16)); base = e[0] / np.linalg.norm(e[0])
    embs = [base + 0.05 * rng.normal(size=16) for _ in range(9)] + [rng.normal(size=16)]
    embs = [v / np.linalg.norm(v) for v in embs]
    m = outliers(embs, ['gold'] * 3 + ['silver'] * 7)
    assert m[-1] and not m[:-1].any(), 'the crop of another fish is the only outlier'
    f = device_factors([{'q': 0.8, 'device': 'good'}, {'q': 0.8, 'device': 'good'}, {'q': 0.2, 'device': 'poor'}])
    assert f['good'] > 1 > f['poor'] >= 0.5, f
    print('selftest ok')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
