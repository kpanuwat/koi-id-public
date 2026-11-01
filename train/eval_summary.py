# Per-clip summary of eval_app.py runs: count, named share, sure-dup, tracks, fish, ms, varieties.  cd train && .venv/bin/python eval_summary.py eval_v11 [...]
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / 'runs'


def metrics(tag):
    """{clip: {count, max, named, dup, tracks, len, fish, ms, varieties}} for one eval run."""
    out = {}
    for c, d in json.load(open(RUNS / tag / 'result.json')).items():
        F = d['frames']
        L = {}
        for i, f in enumerate(F):
            for t in f['tracks']: L.setdefault(t['id'], []).append(i)
        ms = sorted(f['ms'] for f in F)
        out[c] = dict(frames=len(F), count=float(np.median([f.get('count', 0) for f in F])), max=max(f.get('count', 0) for f in F),
                      named=float(np.mean([f.get('named', 0) / max(1, len(f['tracks'])) for f in F])),
                      dup=float(np.mean([(lambda ids: (len(ids) - len(set(ids))) / max(1, len(ids)))([t['fish'] for t in f['tracks'] if t['fish'] and t['sure']]) for f in F])),
                      tracks=len(L), len=float(np.median([len(v) for v in L.values()])), fish=len(d['fish']), ms=round(ms[len(ms) // 2]),
                      varieties=dict(Counter(f['variety'] for f in d['fish'])))
    return out


def console_errors(tag):
    p = RUNS / tag / 'console.log'
    return sum('error' in line.lower() for line in p.read_text().splitlines()) if p.exists() else 0


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        for c, m in metrics(tag).items():
            print(tag, c, 'frames', m['frames'], 'count med', m['count'], 'max', m['max'], '| named %.2f' % m['named'], '| sure-dup %.3f' % m['dup'],
                  '| tracks', m['tracks'], 'med len', m['len'], '| fish', m['fish'], '| ms med', m['ms'], m['varieties'])
