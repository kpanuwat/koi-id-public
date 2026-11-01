"""Drive the PWA in headless Chrome on a clip, dump per-frame tracks + gallery, save overlay frames.

    .venv/bin/python eval_app.py clipA [clipB ...] [--max=N] [--crops] [--out=dir] [--port=8766] [--qs=k=v]   # app must be served at http://127.0.0.1:<port>
Clips play in order in ONE browser profile, so the gallery carries over (cross-session test).
"""
import json, sys, time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "runs" / "eval"


def main():
    global MAX
    clips = [a for a in sys.argv[1:] if not a.startswith("--")]
    MAX = next((int(a[6:]) for a in sys.argv[1:] if a.startswith("--max=")), 0)
    global DUMP, EXTRA, OUT
    DUMP = "crops" if "--crops" in sys.argv else "1"
    EXTRA = "&embedEvery=3&embedBudget=12" if DUMP == "crops" else ""
    OUT = OUT.parent / next((a[6:] for a in sys.argv[1:] if a.startswith("--out=")), OUT.name)
    port = next((a[7:] for a in sys.argv[1:] if a.startswith("--port=")), "8766")
    EXTRA += "".join("&" + a[5:] for a in sys.argv[1:] if a.startswith("--qs="))  # extra URL params, e.g. --qs=noworker=1
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=metal"])
        ctx = b.new_context(viewport={"width": 1280, "height": 1100})
        page = ctx.new_page()
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        results = {}
        for clip in clips:
            page.goto(f"http://127.0.0.1:{port}/?video=models/{clip}.mp4&mode=every&dump={DUMP}" + (f"&max={MAX}" if MAX else "") + EXTRA)
            t0, shots = time.time(), 0
            while True:
                time.sleep(2)
                st = page.evaluate("() => window.koi && ({n: koi.stats.frames.length, done: !!koi.stats.done, status: document.querySelector('#status').textContent})")
                if st and st["n"] // 150 > shots:
                    shots = st["n"] // 150
                    page.locator("#panel").screenshot(path=str(OUT / f"{clip}_{shots:02d}.png"))
                if st and st["done"]:
                    break
                if not st and time.time() - t0 > 60:  # app never started (wrong server root, 404s): fail fast, not after 30 min
                    sys.exit(f"app did not start: {[l for l in logs if 'error' in l][:3]}")
                if time.time() - t0 > 1800 or (st and "ไม่สำเร็จ" in st["status"]):
                    print("timeout/err", st); break
            data = page.evaluate("""() => ({frames: koi.stats.frames, backend: document.querySelector('#status').textContent,
                fish: [...koi.id.fish.values()].map(f => ({id: f.id, name: f.name, variety: f.variety, ex: f.ex.length, seen: f.seen, size: f.size, hist: f.hist})),
                log: koi.id.log, suggestions: koi.id.suggestions, embs: koi.stats.embs})""")
            results[clip] = data
            page.screenshot(path=str(OUT / f"{clip}_page.png"), full_page=True)
            ms = sorted(f["ms"] for f in data["frames"])
            print(clip, "frames", len(ms), "median ms", ms[len(ms) // 2] if ms else None, "fish", len(data["fish"]), data["backend"])
        (OUT / "result.json").write_text(json.dumps(results, ensure_ascii=False))
        (OUT / "console.log").write_text("\n".join(logs))
        print("console errors:", sum("error" in l for l in logs))
        b.close()


if __name__ == "__main__":
    main()
