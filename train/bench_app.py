"""Benchmark ONNX models in headless Chrome WebGPU.  .venv/bin/python bench_app.py "m=models/x.onnx:1,3,432,432&verbose=1" """
import sys
from playwright.sync_api import sync_playwright
q = sys.argv[1] if len(sys.argv) > 1 else ''
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-webgpu", "--use-angle=metal"])
    pg = b.new_page(); logs = []
    pg.on("console", lambda m: logs.append(m.text))
    pg.goto(f"http://127.0.0.1:8766/dev/bench.html?{q}")
    pg.wait_for_function("document.getElementById('o').textContent.includes('done')", timeout=600000)
    print(pg.inner_text('#o'))
    if 'verbose' in q:
        open('runs/ort_verbose.log', 'w').write('\n'.join(logs))
    b.close()
