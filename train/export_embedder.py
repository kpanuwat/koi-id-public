"""DINOv2-S/14 -> ONNX re-ID embedder for the PWA.

Input  pixel_values [N,3,112,224] (canonical koi crop: body axis horizontal, head right,
       ImageNet-normalised, background zeroed). Output embedding [N,768] =
       concat(CLS, mean patch token), L2-normalised.

    .venv/bin/python export_embedder.py        # writes ../app/models/embed.onnx (+ embed_int8.onnx)
"""
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel

OUT = Path(__file__).resolve().parent.parent / "app" / "models"


class Embed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        return torch.nn.functional.normalize(e, dim=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "embed.onnx"
    torch.onnx.export(Embed(), torch.randn(2, 3, 112, 224), f, input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
    quantize_dynamic(f, OUT / "embed_int8.onnx", weight_type=QuantType.QInt8)
    for p in (f, OUT / "embed_int8.onnx"):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
