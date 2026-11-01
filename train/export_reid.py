"""Export the fine-tuned re-ID embedder (runs/reid.pt) for the PWA: fp16 weights, fp32 io, no Resize op.

The crop size is fixed (112x224), so DINOv2's bicubic position-embedding interpolation is a constant:
it is computed once here and baked in (WebGPU's bicubic Resize shader fails in ORT-web 1.29).

    .venv/bin/python export_reid.py      # -> ../app/models/embed.onnx
"""
from pathlib import Path

import onnx
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16

from train_reid import Embed

ROOT = Path(__file__).resolve().parent


def main():
    model = Embed().eval()
    model.load_state_dict(torch.load(ROOT / "runs" / "reid.pt", map_location="cpu"))
    emb = model.m.embeddings
    with torch.no_grad():
        pe = emb.interpolate_pos_encoding(torch.zeros(1, 1 + 8 * 16, emb.position_embeddings.shape[-1]), 112, 224).clone()
    emb.interpolate_pos_encoding = lambda embeddings, height, width: pe
    fp32 = ROOT / "runs" / "embed_fp32.onnx"
    torch.onnx.export(model, torch.randn(2, 3, 112, 224), fp32, input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
    m = onnx.load(str(fp32))
    assert not any(n.op_type == "Resize" for n in m.graph.node), "Resize still in graph"
    dst = ROOT.parent / "app" / "models" / "embed.onnx"
    onnx.save(convert_float_to_float16(m, keep_io_types=True, op_block_list=["LayerNormalization", "Softmax", "ReduceMean"]), str(dst))
    print("exported", dst, f"{dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
