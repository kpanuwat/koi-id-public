"""Hard-negative retrain of the koi detector on a Colab GPU (owner frames: unlabelled leaves/reflections become background).

Inputs (next to this script, uploaded from the VM by pending/hardneg/colab.sh):
  hn/frames/<frame>.jpg          owner-clip frames at full res
  hn/final.json                  {frame: {"split": "train"|"valid", "K": [[x0, y0, x1, y1], ...]}}  reviewed koi boxes (duplicates allowed)
  data/koi/{train,valid}/        the existing clipA/clipB COCO set (kept as is)
  ckpt.pth                       detector checkpoint to fine-tune from

    python colab_hardneg.py build [N]     # SAM2 masks + pond -> data/koi2 (whole frame at 960 px + 2x2 tiles, 15 % overlap)
    python colab_hardneg.py train [EPOCHS]  # fine-tune ckpt.pth on data/koi2 -> runs/hn/checkpoint_best_*.pth
"""
import json, os, shutil, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
HN, OUT = ROOT / "hn", ROOT / "data" / "koi2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LONG, OVL, KEEP, DUP = 960, 0.15, 0.3, 0.6  # whole-frame long side; tile overlap; a clipped koi stays if >= 30 % of it is in the tile; mask dup share
CATS = [{"id": 0, "name": "fish", "supercategory": "none"}, {"id": 1, "name": "koi", "supercategory": "fish"},
        {"id": 2, "name": "pond", "supercategory": "fish"}]


def polys(mask, min_area=40, eps=1.0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea) if cs else None
    if c is None or cv2.contourArea(c) < min_area:
        return None
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return c.astype(float).flatten().tolist() if len(c) >= 3 else None


def ann(S, iid, cat, mask, eps=1.0):
    p = polys(mask, eps=eps)
    if p is None:
        return
    xs, ys = p[0::2], p[1::2]
    S["annotations"].append({"id": len(S["annotations"]), "image_id": iid, "category_id": cat,
                             "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "area": float(mask.sum()),
                             "segmentation": [p], "iscrowd": 0})


def views(W, H):
    """(name suffix, x0, y0, x1, y1, scale): the whole frame at LONG px, then 2x2 tiles at native scale (as photo mode)."""
    tw, th = int(W * (1 + OVL) / 2), int(H * (1 + OVL) / 2)
    out = [("w", 0, 0, W, H, LONG / max(W, H))]
    for k, (x, y) in enumerate([(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]):
        out.append((f"t{k}", x, y, x + tw, y + th, 1.0))
    return out


def build(limit=0):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor
    op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
    sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
    sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(DEV).eval()
    final = json.load(open(HN / "final.json"))
    if OUT.exists():
        shutil.rmtree(OUT)
    splits = {}
    for d in ("train", "valid"):  # start from the existing clipA/clipB set
        shutil.copytree(ROOT / "data" / "koi" / d, OUT / d)
        splits[d] = json.load(open(OUT / d / "_annotations.coco.json"))
    stats = {"frames": 0, "images": 0, "koi": 0, "no_pond": 0}
    for n, (name, r) in enumerate(sorted(final.items())):
        if limit and n >= limit:
            break
        img = cv2.cvtColor(cv2.imread(str(HN / "frames" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        pil = Image.fromarray(img)
        with torch.no_grad():
            km = []
            if r["K"]:
                inp = sp(images=pil, input_boxes=[r["K"]], return_tensors="pt").to(DEV)
                m = sp.post_process_masks(sm(**inp, multimask_output=False).pred_masks.cpu(), inp["original_sizes"].cpu())[0]
                km = sorted((x[0].numpy().astype(np.uint8) for x in m), key=lambda x: -int(x.sum()))
                # reviewers mark several boxes on one fish (half fish, tail): drop a mask mostly inside a bigger kept one.
                # Box containment would also drop koi swimming over each other; their SAM masks stay apart.
                keep = []
                for x in km:
                    if x.sum() and all((x & k).sum() < DUP * x.sum() for k in keep):
                        keep.append(x)
                km = keep
            inp = op(text=[["a pond of water", "water surface"]], images=pil, return_tensors="pt").to(DEV)
            res = op.post_process_grounded_object_detection(om(**inp), threshold=0.05, target_sizes=torch.tensor([[max(H, W)] * 2]))[0]
            box = res["boxes"][res["scores"].argmax()].cpu().tolist() if len(res["boxes"]) else [0, 0, W, H]
            inp = sp(images=pil, input_boxes=[[box]], return_tensors="pt").to(DEV)
            ms = sp.post_process_masks(sm(**inp, multimask_output=True).pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy()
        cen = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in r["K"]]).reshape(-1, 2).astype(int)
        cx, cy = cen[:, 0].clip(0, W - 1), cen[:, 1].clip(0, H - 1)
        pond = ms[max(range(len(ms)), key=lambda i: (round(float(ms[i][cy, cx].mean()) if len(cen) else 0, 2), ms[i].mean()))].astype(np.uint8)
        if pond.mean() < 0.05:
            pond = None; stats["no_pond"] += 1
        S = splits[r["split"]]
        for suf, x0, y0, x1, y1, s in views(W, H):
            crop = img[y0:y1, x0:x1]
            size = (round((x1 - x0) * s), round((y1 - y0) * s))
            fn = f"{name}_{suf}.jpg"
            cv2.imwrite(str(OUT / r["split"] / fn), cv2.cvtColor(cv2.resize(crop, size, interpolation=cv2.INTER_AREA) if s != 1 else crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            iid = max([i["id"] for i in S["images"]], default=-1) + 1
            S["images"].append({"id": iid, "file_name": fn, "width": size[0], "height": size[1]})
            stats["images"] += 1
            for m in km:
                full = m.sum()
                part = m[y0:y1, x0:x1]
                if full == 0 or part.sum() < KEEP * full:
                    continue
                ann(S, iid, 1, cv2.resize(part, size, interpolation=cv2.INTER_NEAREST) if s != 1 else part)
                stats["koi"] += 1
            if pond is not None:
                pp = pond[y0:y1, x0:x1]
                ann(S, iid, 2, cv2.resize(pp, size, interpolation=cv2.INTER_NEAREST) if s != 1 else pp, eps=2.0)
        stats["frames"] += 1
        print(name, r["split"], "koi", len(km), "pond", pond is not None, flush=True)
    for d, S in splits.items():
        for i, a in enumerate(S["annotations"]):
            a["id"] = i
        S["categories"] = CATS
        (OUT / d / "_annotations.coco.json").write_text(json.dumps(S))
    print("built", stats, {d: len(S["images"]) for d, S in splits.items()})


def train(epochs):
    from rfdetr import RFDETRSegNano
    out = ROOT / "runs" / "hn"
    model = RFDETRSegNano(resolution=int(os.environ.get("RES", 432)), pretrain_weights=str(ROOT / "ckpt.pth"))
    bs = int(os.environ.get("BATCH", 8))
    model.train(dataset_dir=str(OUT), output_dir=str(out), epochs=epochs, batch_size=bs, grad_accum_steps=max(1, 8 // bs),
                lr=float(os.environ.get("LR", 5e-5)), num_workers=2, checkpoint_interval=5, multi_scale=False,
                expanded_scales=False, progress_bar="tqdm")
    print("trained", sorted(p.name for p in out.glob("checkpoint_best*")))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
