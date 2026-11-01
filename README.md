# Koi ID

แอปนับปลาคาร์ฟ (koi) ในบ่อจากกล้องมือถือ ทำงานบนเครื่องทั้งหมด (on-device) เป็น PWA ที่เปิดในเบราว์เซอร์ได้ทันที ไม่ต้องติดตั้ง
An on-device PWA that counts koi in a pond from the phone camera, a photo or a video file.

- เดโม (reference deployment): https://koi-id.pages.dev
- คู่มือนักพัฒนา (developer manual): [`docs/Koi_ID_Developer_Manual.html`](docs/Koi_ID_Developer_Manual.html) (download and open in a browser)
- สัญญาอนุญาต (licence): Apache-2.0, see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

## สถานะ (Status, v0.064)

| โหมด (mode) | URL | สถานะ (status) |
|---|---|---|
| นับต่อเนื่องจากกล้อง (continuous camera count) | `/` then **กล้อง** | เสถียรบน iPhone: ~7 รอบ/วินาที นาน 3.5 นาทีไม่ crash (stable) |
| นับจากภาพถ่าย (photo count) | `/` then **ถ่ายภาพ** | เสถียร (stable) |
| นับจากไฟล์วิดีโอ (video file) | `/` then **ไฟล์วิดีโอ** | ใช้ได้ เคย crash 1 ครั้งบน iPhone (works, one iPhone crash seen) |
| เลขประจำตัวปลา (permanent fish numbers) | `/?id=1` | ทดลอง: iPhone crash หลัง 4-5 ภาพ (experimental) |
| แอปเต็มแบบเดิม (full app: tracking, naming, gallery) | `/?count=0` | ทดลอง: crash บน iPhone (experimental) |

แต่ละรอบนับปลาที่เห็นในเฟรมนั้น ยังไม่ใช่ยอดรวมทั้งบ่อ (each round counts the fish in that frame, not a pond total yet).

## เริ่มใช้งานบนเครื่องตัวเอง (Quick start)

```bash
git clone https://github.com/kpanuwat/koi-id-public koi-id && cd koi-id
scripts/fetch_models.sh        # model parts from GitHub Releases -> app/models/
cd app && npx http-server@14.1.1 -p 8766 -a 127.0.0.1 -c-1 -s
# open http://127.0.0.1:8766 in Chrome or Edge (WebGPU), Safari 26+ on iPhone needs HTTPS (deploy, see SETUP.md)
```

The server must support HTTP Range requests (Python's `http.server` does not seek videos frame by frame).

## ทำงานอย่างไร (How it works)

```
camera / photo / video  ->  page (app.js: UI, round loop, drawing)
                              |  one frame as an ImageBitmap
                              v
                          inference worker (pipeline.js, models.js)
                          ONNX Runtime Web: WebGPU, falls back to wasm
                          YOLOX-tiny koi_yx320: letterbox 320 px, decode, NMS
                              |  boxes + scores
                              v
                          page draws boxes + count, then the next round starts at once
```

- โมเดลแบ่งเป็นไฟล์ย่อย <= 20 MB (`app/models/parts.json`, `train/split_models.py`) เพราะ Cloudflare Pages จำกัดขนาดไฟล์ และ service worker (`app/sw.js`) cache ไว้ใช้ offline
- ไม่ส่งภาพออกจากเครื่อง ในการ deploy อ้างอิง แอปส่งเฉพาะ log วินิจฉัยขนาดเล็ก (เวลา จำนวน รุ่นเบราว์เซอร์) ไปที่ `/api/log`
- Backend (optional): Cloudflare Pages Functions (`functions/api/[[path]].js`) + D1 (`migrations/`) for logs and the gallery sync of the full app

รายละเอียดทั้งหมด (modules, URL parameters, API, data flow, iPhone memory findings, roadmap): see the developer manual.

## โครงสร้าง repo (Layout)

| Path | What |
|---|---|
| `app/` | the PWA (no build step): `app.js` UI and loops, `pipeline.js` + `models.js` inference (worker), `tracker.js`, `reid.js`, `follow.js`, `sync.js`, `sw.js` |
| `app/vendor/` | ONNX Runtime Web 1.29.0 |
| `functions/`, `migrations/`, `wrangler.toml` | Cloudflare Pages Functions API and D1 schema |
| `train/` | dataset building, training, export, evaluation (`eval_app.py` drives the real app in headless Chromium) |
| `train/yolox/` | YOLOX-tiny training and export for the count-mode detector (Colab script) |
| `scripts/` | `fetch_models.sh`, UAT helpers |
| `docs/` | developer manual (HTML) and its sources |

## ร่วมพัฒนา (Contributing)

ยินดีรับ issue และ pull request อ่าน [`CONTRIBUTING.md`](CONTRIBUTING.md) ก่อน รายงานช่องโหว่ตาม [`SECURITY.md`](SECURITY.md)
Issues and pull requests are welcome; see `CONTRIBUTING.md`. Report vulnerabilities privately as described in `SECURITY.md`.
