# Setup and deployment / การติดตั้งและ deploy

**ไทย:** ขั้นตอนรันแอปบนเครื่อง ทดสอบ เทรน และ deploy instance ของคุณเองบน Cloudflare
**English:** run the app locally, test it, train models, and deploy your own instance on Cloudflare (free tier works).

## 1. Prerequisites / สิ่งที่ต้องมี

| Item | Version used | Note |
|---|---|---|
| Linux or macOS | Ubuntu 24.04 (arm64), macOS 26 | macOS only for WebKit memory probes |
| Node.js | 22 or 24 | `npx` runs wrangler@4 and http-server@14.1.1 |
| Python | 3.12 | `train/` tools, Playwright checks |
| ffmpeg | any recent | test clips (optional) |

## 2. Get the code and models / โค้ดและโมเดล

```bash
git clone https://github.com/kpanuwat/koi-id-public koi-id && cd koi-id
scripts/fetch_models.sh      # koi_yx320 + embed parts -> app/models/
```

`app/models/parts.json` lists every model the code knows. Only `koi_yx320.onnx` (count mode, the default) and
`embed.onnx` (`?id=1`, `?count=0`) are published. The full app (`?count=0`) also needs an RF-DETR detector
(`koi_det_lite16.onnx` / `koi_det.onnx`) that you train yourself with `train/train_detector.py`.

## 3. Run locally / รันบนเครื่อง

```bash
cd app && npx --yes http-server@14.1.1 -p 8766 -a 127.0.0.1 -c-1 -s     # http://127.0.0.1:8766
```

- The camera needs HTTPS or localhost. WebGPU needs a recent Chrome/Edge (desktop) or Safari 26+ (iOS).
- Add `?dump=1` while testing: the app then never calls `/api`.

## 4. Python tools and checks / เครื่องมือทดสอบ

```bash
python3.12 -m venv train/.venv
train/.venv/bin/pip install -r train/requirements.txt
train/.venv/bin/python -m playwright install chromium chrome   # eval_app.py launches Google Chrome (channel "chrome")
# count a still in headless Chromium with the iPhone user agent (PASS/FAIL per image)
train/.venv/bin/python -I train/yolox/count_check.py http://127.0.0.1:8766 path/under/app/to/image.jpg
# drive the full app on a clip (put the clip at app/models/<name>.mp4), per-frame results in train/runs/<out>/
train/.venv/bin/python train/eval_app.py <name> --out=eval_check --qs=count=0
```

Syntax-check a module as an `.mjs` copy: `cp app/app.js /tmp/app.mjs && node --check /tmp/app.mjs`.

## 5. Deploy your own instance / deploy instance ของคุณเอง

```bash
npx wrangler@4 login
npx wrangler@4 d1 create koi-id                      # put the printed database_id into wrangler.toml
for f in migrations/*.sql; do npx wrangler@4 d1 execute koi-id --remote --file "$f"; done
npx wrangler@4 pages project create koi-id --production-branch main
openssl rand -hex 24 | npx wrangler@4 pages secret put SYNC_TOKEN --project-name koi-id   # admin token, keep it private
KOI_PROJECT=koi-id ./deploy.sh
```

- Before every deploy bump `BUILD` in `app/app.js` and `CACHE` in `app/sw.js` (running numbers v0.064, v0.065, ...).
  `deploy.sh` sets `MODELS` in `sw.js` from the model bytes; never edit it by hand (commit the change it makes, run again).
- On CI or a server without OAuth, export `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` (optionally from
  `~/.config/koi-id/cf.env`, which `deploy.sh` loads if present).
- Field diagnostics land in the D1 `logs` table:
  `npx wrangler@4 d1 execute koi-id --remote --command "SELECT id, at, ev, data FROM logs ORDER BY id DESC LIMIT 50" --json`

## 6. Training / การเทรน

- Count-mode detector: `train/yolox/` (YOLOX-tiny on a Colab GPU, ~15 min on an L4; `colab.sh` shows the full recipe).
- Full-app detector and embedder: `train/make_dataset.py` (OWLv2 + SAM2 auto labels), `train/train_detector.py`,
  `train/train_reid.py`, exports in `train/export_*.py`, then `train/split_models.py` writes the <= 20 MB parts and
  `app/models/parts.json`.
- You need your own pond footage; the footage behind the published models is not shared.
