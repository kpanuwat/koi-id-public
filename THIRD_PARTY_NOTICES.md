# Third-party notices / ส่วนประกอบจากภายนอก

Koi ID itself is Apache-2.0 (see `LICENSE`, `NOTICE`). This file lists what it builds on.
Koi ID ใช้สัญญาอนุญาต Apache-2.0 รายการด้านล่างคือส่วนประกอบจากภายนอกและเงื่อนไขของแต่ละรายการ

## Bundled in this repository / รวมอยู่ใน repo

| Component | Path | Version | Licence | Source |
|---|---|---|---|---|
| ONNX Runtime Web (JS, wasm) | `app/vendor/` | 1.29.0 | MIT | https://github.com/microsoft/onnxruntime |

## Model files in GitHub Releases / ไฟล์โมเดลใน Releases

| File | What it is | Derived from | Licence of the base |
|---|---|---|---|
| `koi_yx320.onnx` | koi detector used by the default count mode (YOLOX-tiny, 320 px, 1 class) | YOLOX-tiny `yolox_tiny.pth` (COCO-pretrained), fine-tuned on koi pond frames | Apache-2.0 (Megvii) |
| `embed.onnx` | koi re-identification embedder (256-d, 112x224 crops), used by `?id=1` and `?count=0` | DINOv2 ViT-S/14 (`facebook/dinov2-small`), fine-tuned on koi crops | Apache-2.0 (Meta) |

The fine-tuned weights are released under Apache-2.0 by the Koi ID Authors. Pretrained bases were trained on public
datasets such as COCO (annotations CC BY 4.0; images under their own Flickr terms); check those terms for your use.
The pond footage used for fine-tuning is not published.
ไฟล์โมเดลที่ fine-tune แล้วเผยแพร่ภายใต้ Apache-2.0 ส่วนวิดีโอบ่อที่ใช้เทรนไม่ได้เผยแพร่

## Used by the training scripts (not shipped) / ใช้ในสคริปต์เทรน ไม่ได้แจกจ่าย

| Tool or model | Used in | Licence |
|---|---|---|
| RF-DETR (rfdetr) | `train/train_detector.py`, `train/export_coco.py` | Apache-2.0 |
| YOLOX | `train/yolox/` | Apache-2.0 |
| OWLv2 `google/owlv2-base-patch16-ensemble` | auto-labelling (`train/make_dataset.py`, `train/add_pond.py`, `train/colab_hardneg.py`) | Apache-2.0 |
| SAM 2.1 `facebook/sam2.1-hiera-small` | auto-labelling masks | Apache-2.0 |
| PyTorch, torchvision, transformers, onnx, onnxruntime, OpenCV, SciPy, Pillow, Playwright | `train/requirements.txt` | BSD / Apache-2.0 / MIT (see each project) |
| Cloudflare Wrangler | `deploy.sh` | MIT OR Apache-2.0 |

## External services (optional) / บริการภายนอก (ไม่บังคับ)

Cloudflare Pages, Functions, D1 and R2 host the reference deployment; Google Colab runs training. Neither is required
to run the app locally. Each has its own terms of service.
