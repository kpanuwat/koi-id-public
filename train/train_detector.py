"""Fine-tune RF-DETR-Seg Nano on the auto-labelled koi set, export ONNX for the PWA.

    .venv/bin/python train_detector.py [epochs]     # -> runs/det/, ../app/models/koi_det.onnx
    .venv/bin/python train_detector.py --export runs/det/checkpoint_best_ema.pth   # export only
    RES=312 DST=koi_det_lite.onnx .venv/bin/python train_detector.py --export <ckpt>   # lighter detector for iPhones
"""
import os, sys
from pathlib import Path

from rfdetr import RFDETRSegNano

ROOT = Path(__file__).resolve().parent
RES = int(os.environ.get("RES", 432))  # multiple of 12; small far fish need the pixels (312 default loses them)
BATCH = int(os.environ.get("BATCH", 2))


def main():
    out = ROOT / "runs" / "det"
    if len(sys.argv) > 2 and sys.argv[1] == "--export":  # export a given checkpoint only
        return export(RFDETRSegNano(resolution=RES, pretrain_weights=sys.argv[2]), out, sys.argv[2])
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    model = RFDETRSegNano(resolution=RES)
    # 16 GB Mac: batch 4 + expanded scales hit 14 GB unified memory and swapped; batch 2 x accum 4 keeps effective 8
    model.train(dataset_dir=str(ROOT / "data" / "koi"), output_dir=str(out), epochs=epochs, batch_size=BATCH,
                grad_accum_steps=8 // BATCH, lr=1e-4, num_workers=0, checkpoint_interval=5, multi_scale=os.environ.get("MS", "0") == "1", expanded_scales=False, progress_bar="tqdm")
    # newest "best" file: runs/det is reused, an older run's best_total must not win
    best = max((p for p in [out / "checkpoint_best_total.pth", out / "checkpoint_best_ema.pth", out / "checkpoint_best_regular.pth"] if p.exists()),
               key=lambda p: p.stat().st_mtime, default=None)
    if best:
        model = RFDETRSegNano(resolution=RES, pretrain_weights=str(best))
    export(model, out, best)


def export(model, out, best):
    onnx = model.export(output_dir=str(out / "export"), format="onnx", opset_version=17)
    # app gets fp16 weights (half the download, same detections: IoU>0.8 for 135/135 fish on valid frames)
    import onnx as ox
    from onnxruntime.transformers.float16 import convert_float_to_float16
    # SOFTMAX16=1: attention Softmax in fp16 too (no fp32 cast islands around the 677x677 maps: about a third of the bytes
    # moved per pass on iPhones, whose detector is memory-bound; Opus review 25 Oct 2026)
    BLOCK = ["GridSample", "TopK", "Resize", "ReduceMax", "LayerNormalization"] + ([] if os.environ.get("SOFTMAX16") == "1" else ["Softmax"])
    dst = ROOT.parent / "app" / "models" / os.environ.get("DST", "koi_det.onnx")  # DST=koi_det_lite.onnx: low-res copy for iPhones
    ox.save(convert_float_to_float16(ox.load(str(onnx)), keep_io_types=True,
                                     op_block_list=BLOCK), str(dst))
    print("exported", dst, f"{dst.stat().st_size / 1e6:.1f} MB (fp16)", "from", best, "fp32 at", onnx)


if __name__ == "__main__":
    main()
