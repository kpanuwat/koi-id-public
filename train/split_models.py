"""Split app/models/*.onnx into <=20 MB parts + parts.json (Cloudflare Pages serves files up to 25 MiB).
The app loads the parts and joins them (models.js). Re-run after exporting a new model.
    .venv/bin/python split_models.py
"""
import json
from pathlib import Path

M = Path(__file__).resolve().parent.parent / 'app' / 'models'
PART = 20 * 1024 * 1024
index = {}
for f in sorted(M.glob('*.onnx')):
    b = f.read_bytes(); n = 0
    for old in M.glob(f'{f.name}.part*'): old.unlink()  # stale parts of an older export
    for i in range(0, len(b), PART):
        (M / f'{f.name}.part{n}').write_bytes(b[i:i + PART]); n += 1
    index[f.name] = {'parts': n, 'bytes': len(b)}
(M / 'parts.json').write_text(json.dumps(index, indent=1))
print(index)
