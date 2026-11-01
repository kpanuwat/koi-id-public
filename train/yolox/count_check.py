"""Count mode (?count=1, v63.23) check: Chromium with the iPhone UA on a local server (wasm on the VM).
    train/.venv/bin/python -I pending/yolox/count_check.py <base URL> <image under app/> [...]
    EXTRA_QS='&id=1&gallery=devgal/pond2.json' (count + id; the gallery file is a read-only copy of /api/pond/<id>)
    IMG_DIR=pending/photo/test ... https://koi-id.pages.dev uat/land_4161.jpg   # production (read-only: dump=1 sends nothing)
PASS per still: n > 0, boxes inside the image, no page/console errors, no /api request, no embedder or RF-DETR model
fetched, the status line names the build. Screenshot: $CLAUDE_JOB_DIR/tmp/count_<name>.png (or /tmp)."""
import os, sys
from playwright.sync_api import sync_playwright

base, imgs = sys.argv[1].rstrip("/"), sys.argv[2:]
IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1"
shots = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/"), "tmp")
ok_all = True
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    for img in imgs:
        r = {"errs": [], "api": [], "models": []}
        pg = b.new_context(user_agent=IPHONE, viewport={"width": 390, "height": 844}, service_workers="block").new_page()
        pg.on("pageerror", lambda e: r["errs"].append("pageerror: " + str(e)[:200]))
        pg.on("console", lambda m: m.type == "error" and r["errs"].append("console: " + m.text[:200]))
        pg.on("request", lambda q: ("/api" in q.url and r["api"].append(q.url)) or (".onnx" in q.url and r["models"].append(q.url.rsplit("/", 1)[-1])))
        if os.environ.get("IMG_DIR"):  # production UAT: /uat/<name> answered from a local folder, nothing uploaded
            pg.route("**/uat/*", lambda route: route.fulfill(path=os.path.join(os.environ["IMG_DIR"], route.request.url.rsplit("/", 1)[-1])))
        pg.goto(f"{base}/?photo={img}&count=1&dump=1{os.environ.get('EXTRA_QS', '')}")
        pg.wait_for_function("() => window.koi && window.koi.photo", timeout=900_000, polling=2000)
        ph = pg.evaluate("() => window.koi.photo"); st = pg.evaluate("() => document.querySelector('#status').textContent")
        name = img.rsplit("/", 1)[-1]
        pg.screenshot(path=f"{shots}/count_{name}.png")
        inside = all(0 <= d["box"][0] < d["box"][2] <= ph["w"] + 0.5 and 0 <= d["box"][1] < d["box"][3] <= ph["h"] + 0.5 for d in ph.get("dets", []))
        bad_models = [m for m in r["models"] if not m.startswith("koi_yx320") and not (os.environ.get("EXTRA_QS") and m.startswith("embed.onnx"))]
        ok = ph.get("n", 0) > 0 and inside and not r["errs"] and not r["api"] and not bad_models and "err" not in ph
        ok_all &= ok
        print(name, "PASS" if ok else "FAIL", {"n": ph.get("n"), "ms": ph.get("ms"), "matched": ph.get("matched"), "emb": ph.get("emb"), "embMs": ph.get("embMs"), "fishN": ph.get("fishN"), "labels": [d.get("fish") or d.get("hex") for d in ph.get("dets", [])], "w": ph.get("w"), "h": ph.get("h"), "status": st[:80],
              "models": sorted(set(r["models"])), "api": r["api"][:3], "errs": r["errs"][:3]})
        pg.context.close()
    b.close()
sys.exit(0 if ok_all else 1)
