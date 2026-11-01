"""Export the COCO-pretrained RF-DETR Nano (Apache-2.0, 80 everyday object classes) for the PWA's "other objects" labels.
    .venv/bin/python export_coco.py      # -> app/models/coco_det.onnx (fp16) + app/models/coco_classes.json
Then run split_models.py. Downloads the public pretrained weights on first run.
"""
import json
from pathlib import Path

from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

ROOT = Path(__file__).resolve().parent
model = RFDETRNano()
onnx = model.export(output_dir=str(ROOT / "runs" / "coco_export"), format="onnx", opset_version=17)
import onnx as ox
from onnxruntime.transformers.float16 import convert_float_to_float16
dst = ROOT.parent / "app" / "models" / "coco_det.onnx"
ox.save(convert_float_to_float16(ox.load(str(onnx)), keep_io_types=True,
                                 op_block_list=["GridSample", "TopK", "Resize", "ReduceMax", "Softmax", "LayerNormalization"]), str(dst))
(dst.parent / "coco_classes.json").write_text(json.dumps({int(k): v for k, v in COCO_CLASSES.items()}))
print("exported", dst, f"{dst.stat().st_size / 1e6:.1f} MB", "resolution", model.model.resolution if hasattr(model, 'model') else '?')
