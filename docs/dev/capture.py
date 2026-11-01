"""Screenshots for the developer manual from the live app (read-only: /api aborted, dump=1).
    train/.venv/bin/python -I docs/dev/capture.py     (needs /tmp/fakecam.y4m: a pond clip as the fake camera)
-> docs/dev/img/{start,auto,photo}.png"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / 'img'
BASE = 'https://koi-id.pages.dev'
UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1"
TEST = Path(__file__).resolve().parents[2] / 'pending' / 'photo' / 'test'
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--use-file-for-fake-video-capture=/tmp/fakecam.y4m"])
    def page():
        pg = b.new_context(user_agent=UA, viewport={"width": 390, "height": 844}, device_scale_factor=2, service_workers="block", permissions=["camera"]).new_page()
        pg.route("**/api/**", lambda r: r.abort())
        pg.route("**/uat/*", lambda r: r.fulfill(path=str(TEST / r.request.url.rsplit('/', 1)[-1])))
        return pg
    ready = "() => !document.querySelector('#dock').classList.contains('disabled')"
    pg = page(); pg.goto(BASE + '/?count=1&dump=1'); pg.wait_for_function(ready, timeout=300000); time.sleep(1)
    pg.screenshot(path=str(OUT / 'start.png'))
    pg.click('#cam'); pg.wait_for_function("() => window.koi.photo", timeout=120000); time.sleep(4)
    pg.screenshot(path=str(OUT / 'auto.png')); pg.click('#photoClose'); pg.close()
    pg = page(); pg.goto(BASE + '/?count=1&dump=1&photo=uat/land_4161.jpg'); pg.wait_for_function("() => window.koi && window.koi.photo", timeout=300000); time.sleep(1)
    pg.screenshot(path=str(OUT / 'photo.png')); pg.close()
    b.close()
