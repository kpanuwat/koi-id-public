"""Build docs/Koi_ID_Developer_Manual.html from docs/dev/manual_src.html: {{img:NAME}} -> docs/dev/img/NAME.png as an
embedded JPEG data URL (one self-contained file), and the same page as app/about.html.     train/.venv/bin/python -I docs/dev/build_manual.py"""
import base64, re
from pathlib import Path

import cv2

D = Path(__file__).resolve().parent


def data_url(name, width=520):
    im = cv2.imread(str(D / 'img' / f'{name}.png'))
    if im.shape[1] > width:
        im = cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 'data:image/jpeg;base64,' + base64.b64encode(b.tobytes()).decode()


src = (D / 'manual_src.html').read_text()
out = re.sub(r'\{\{img:(\w+)\}\}', lambda m: data_url(m.group(1)), src)
bad = [c for c in ['—', '§', '─', '│', '├', '└'] if c in out]  # house style: none of these
assert not bad, f'banned characters: {bad}'
(D.parent / 'Koi_ID_Developer_Manual.html').write_text(out)
print('wrote', D.parent / 'Koi_ID_Developer_Manual.html', len(out) // 1024, 'KB')
# the same manual is the app's /about page, with a way back to the app
about = out.replace('<body>', '<body>\n<p class="meta" style="max-width:1200px;margin:12px auto 0;padding:0 16px"><a href="./">กลับไปที่แอป (back to the app)</a></p>', 1)
assert about != out
(D.parents[1] / 'app' / 'about.html').write_text(about)
print('wrote app/about.html')
