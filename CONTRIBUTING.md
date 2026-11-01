# Contributing / การร่วมพัฒนา

ขอบคุณที่สนใจร่วมพัฒนา Koi ID / Thank you for helping.

## Before you start / ก่อนเริ่ม
- Read the developer manual `docs/Koi_ID_Developer_Manual.html`, in particular "iPhone crash findings" and "rules and traps".
- Open an issue first for anything bigger than a small fix, so the approach can be agreed.

## Ground rules / กฎหลัก
1. **Do not regress the stable path.** The default count mode (camera rounds through `autoCount()` and `photoRun()` in
   `app/app.js`) is the stable base on iPhones. Changes there need evidence from a real iPhone (Safari/Chrome on iOS 26+)
   or a clear statement that it was not tested on one.
2. **Memory is the constraint on iPhones.** WebKit kills the tab near 1.25-1.5 GB. Avoid a second WebGPU session, per-frame
   allocations (canvases, bitmaps), and long-lived camera frame grabs in a `requestVideoFrameCallback` loop.
3. **No code after a trailing `//` comment on the same line.** It silently swallowed real statements twice in this project.
4. Match the surrounding style: plain ES modules, no build step, no framework, comments explain why (with evidence).
5. Changes that touch detection or counting come with a before/after measurement (`train/eval_app.py`,
   `train/eval_summary.py`, or `train/yolox/count_check.py`) in the pull request.

## Pull requests / การส่ง PR
- One topic per PR; describe what changed, why, and how it was tested (browser, device, numbers).
- `node --check` each changed module as an `.mjs` copy; run `node app/pond.test.mjs` and `node app/tracker.test.mjs` (the latter fails on v0.064, a known issue) if
  you touched those areas.
- Do not commit model weights, videos, photos of people, tokens or `.dev.vars` / `.env` files.
- By contributing you agree that your contribution is licensed under the Apache License 2.0.

## Language / ภาษา
Issues and PRs in Thai or English are both fine. ใช้ภาษาไทยหรืออังกฤษก็ได้
