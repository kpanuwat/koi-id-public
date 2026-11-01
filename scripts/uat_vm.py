"""Read-only UAT that runs on the Linux VM (no GPU: the models run on wasm, ~3 min per detector pass).

    train/.venv/bin/python scripts/uat_vm.py <base URL> <BUILD> [iphone|chromium|webkit ...] [--clip=...] [--passes=2] [--emb] [--timeout=3600]
    e.g. scripts/uat_vm.py https://koi-id.pages.dev v63.3 iphone chromium --clip=pending/embedfix/out/clipA_long.mp4

Per browser (iphone = Chromium with the iPhone UA: the iPhone code path, LEVEL 2, lite16, fixed embed batch; chromium =
desktop; webkit = iPhone UA, but WPE on the VM cannot decode H.264, so the video never plays there).
Use a clip much longer than a pass (~3-4 min on wasm): at each loop the app reloads the video and drops the running pass
(ffmpeg concat of clipA x20 = 10.6 min).
loads <base>/?dump=1 (dump: the app never calls /api), waits for 'พร้อม · <BUILD>', picks the clip through #file like a
user, waits for <passes> detector passes with embeddings, then checks 0 console errors and 0 /api requests.
Speed and memory are NOT measured here (wasm, not WebGPU): the phone test covers those. Exit 1 on any failure.
"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
base, build, browsers = args[0].rstrip("/"), args[1], args[2:] or ["iphone", "chromium"]
clip = next((a[7:] for a in sys.argv if a.startswith("--clip=")), "app/models/clipA.mp4")
passes = int(next((a[9:] for a in sys.argv if a.startswith("--passes=")), "2"))
need_emb = "--emb" in sys.argv  # also require a pass with embeddings (on wasm that is usually the 3rd pass)
limit = int(next((a[10:] for a in sys.argv if a.startswith("--timeout=")), "3600"))
ok_all = True


def check(name):
    errs, api = [], []
    with sync_playwright() as p:
        b = (p.webkit if name == "webkit" else p.chromium).launch(headless=True)
        ctx = b.new_context(viewport={"width": 390, "height": 844}, user_agent=IPHONE) if name in ("webkit", "iphone") else b.new_context(viewport={"width": 1280, "height": 1000})
        pg = ctx.new_page()
        pg.on("console", lambda m: m.type == "error" and errs.append(m.text[:200]))
        pg.on("pageerror", lambda e: errs.append(f"pageerror {e}"[:200]))
        ctx.on("request", lambda r: "/api/" in r.url and api.append(r.url))
        t0 = time.time()
        say = lambda s: print(f"[{name} {int(time.time() - t0)}s] {s}", flush=True)
        pg.goto(base + "/?dump=1&full=1")
        pg.wait_for_function("() => document.querySelector('#status')?.textContent.startsWith('พร้อม')", timeout=900_000)
        st = pg.evaluate("() => document.querySelector('#status').textContent")
        say(f"ready: {st}")
        pg.set_input_files("#file", clip)
        # progress every 60 s (status, video state, passes) so a stall shows; a pass on wasm takes minutes
        PROG = "() => { const v = document.querySelector('video'), f = window.koi?.stats?.frames || []; return {status: document.querySelector('#status')?.textContent, ready: v?.readyState, t: v && +v.currentTime.toFixed(1), paused: v?.paused, err: v?.error?.code, passes: f.length, emb: f.filter(x => x.nEmb > 0).length} }"
        timed_out = False
        while True:
            pr = pg.evaluate(PROG)
            if pr["passes"] >= passes and (pr["emb"] >= 1 or not need_emb): break
            if time.time() - t0 > limit:
                timed_out = True; say(f"TIMEOUT after {limit} s: {pr}"); break
            say(f"progress {pr}")
            pg.wait_for_timeout(60_000)
        fr = pg.evaluate("() => koi.stats.frames.map(f => ({n: f.n, count: f.count, named: f.named, nEmb: f.nEmb, ms: Math.round(f.ms)}))")
        st2 = pg.evaluate("() => document.querySelector('#status').textContent")
        # service worker caches: code under koi-id-<BUILD>, models + wasm under koi-models-<hash> (kept across code releases)
        cs = pg.evaluate("async () => { const o = {}; for (const k of await caches.keys()) o[k] = (await (await caches.open(k)).keys()).filter(r => /\\.(part\\d+|wasm)$/.test(r.url)).length; return o; }")
        say(f"caches (model files): {cs}")
        models_cached = any(k.startswith("koi-models-") and n >= 4 for k, n in cs.items())
        say(f"passes {len(fr)}: {fr[:6]} | status: {st2}")
        b.close()
    ok = build in st and not errs and not api and not timed_out and models_cached and any(f["n"] > 0 for f in fr)
    say(f"build {'ok' if build in st else 'WRONG'} | console errors {len(errs)} {errs[:3]} | /api requests {len(api)} {api[:3]} | "
        f"models cache {'ok' if models_cached else 'MISSING'} | {'TIMEOUT' if timed_out else 'in time'} -> {'PASS' if ok else 'FAIL'}")
    return ok


for name in browsers:
    try:
        ok_all &= check(name)
    except Exception as e:
        print(f"[{name}] FAIL {str(e)[:300]}", flush=True)
        ok_all = False
print("UAT:", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
