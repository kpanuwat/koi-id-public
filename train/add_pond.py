"""Add a "pond" (water surface) instance to every labelled frame and drop fish labels outside it.

OWLv2 ("a pond of water") gives the pond box, SAM2.1 turns it into a mask; the mask kept is the one
containing most fish centroids. Fish whose centre lies outside the pond (leaves, plants, rim) are
removed from the training labels. Pixels are never changed. Resumable via data/koi/pond.jsonl.

    .venv/bin/python add_pond.py      # then rebuilds data/koi/{train,valid}/_annotations.coco.json
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "koi"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
INSIDE = 0.5  # fish kept if its mask is at least half inside the pond (dilated)


def main():
    rows = [json.loads(l) for l in (OUT / "labels.jsonl").read_text().splitlines()]
    pj = OUT / "pond.jsonl"
    done = {json.loads(l)["name"] for l in pj.read_text().splitlines()} if pj.exists() else set()
    todo = [r for r in rows if r["name"] not in done]
    if todo:
        op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
        sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
        sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(DEV).eval()
    with pj.open("a") as fo:
        for r in todo:
            img = cv2.cvtColor(cv2.imread(str(OUT / r["split"] / r["name"])), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            with torch.no_grad():
                inp = op(text=[["a pond of water", "water surface"]], images=Image.fromarray(img), return_tensors="pt").to(DEV)
                res = op.post_process_grounded_object_detection(om(**inp), threshold=0.05, target_sizes=torch.tensor([[max(H, W)] * 2]))[0]
                box = res["boxes"][res["scores"].argmax()].cpu().tolist() if len(res["boxes"]) else [0, 0, W, H]
                inp = sp(images=Image.fromarray(img), input_boxes=[[box]], return_tensors="pt").to(DEV)
                ms = sp.post_process_masks(sm(**inp, multimask_output=True).pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy()
            fish = np.array([[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2] for a in r["anns"]]).reshape(-1, 2)
            fx, fy = fish[:, 0].astype(int).clip(0, W - 1), fish[:, 1].astype(int).clip(0, H - 1)
            k = max(range(len(ms)), key=lambda i: (round(float(ms[i][fy, fx].mean()) if len(fish) else 0, 2), ms[i].mean()))
            m = ms[k].astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea) if cs else None
            poly = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).astype(float).flatten().tolist() if c is not None and cv2.contourArea(c) > 0.05 * H * W else None
            fo.write(json.dumps({"name": r["name"], "pond": poly, "area": float(m.mean())}) + "\n"); fo.flush()
            if DEV == "mps":
                torch.mps.empty_cache()
            print(r["name"], "pond" if poly else "NO POND", f"{m.mean():.2f}", flush=True)
    build(rows, {json.loads(l)["name"]: json.loads(l) for l in pj.read_text().splitlines()})


def build(rows, pond):
    cats = [{"id": 0, "name": "fish", "supercategory": "none"}, {"id": 1, "name": "koi", "supercategory": "fish"},
            {"id": 2, "name": "pond", "supercategory": "fish"}]
    splits = {k: {"images": [], "annotations": []} for k in ("train", "valid")}
    dropped = kept = 0
    for r in rows:
        S, p = splits[r["split"]], pond[r["name"]]["pond"]
        iid = len(S["images"])
        S["images"].append({"id": iid, "file_name": r["name"], "width": r["w"], "height": r["h"]})
        pm = None
        if p:
            pm = np.zeros((r["h"], r["w"]), np.uint8)
            cv2.fillPoly(pm, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
            pm = cv2.dilate(pm, np.ones((9, 9), np.uint8))  # fish at the very edge still count
        for a in r["anns"]:
            if pm is not None:
                fm = np.zeros_like(pm); cv2.fillPoly(fm, [np.array(a["poly"]).reshape(-1, 2).astype(np.int32)], 1)
                if (fm & pm).sum() < INSIDE * max(1, fm.sum()):
                    dropped += 1; continue
            kept += 1
            S["annotations"].append({"id": len(S["annotations"]), "image_id": iid, "category_id": 1, "bbox": a["bbox"],
                                     "area": a["area"], "segmentation": [a["poly"]], "iscrowd": 0})
        if p:
            xs, ys = p[0::2], p[1::2]
            S["annotations"].append({"id": len(S["annotations"]), "image_id": iid, "category_id": 2,
                                     "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                                     "area": float(cv2.contourArea(np.array(p).reshape(-1, 1, 2).astype(np.float32))),
                                     "segmentation": [p], "iscrowd": 0})
    for d, S in splits.items():
        (OUT / d / "_annotations.coco.json").write_text(json.dumps({**S, "categories": cats}))
    print(f"fish kept {kept}, dropped outside pond {dropped}; frames without pond {sum(1 for v in pond.values() if not v['pond'])}")


if __name__ == "__main__":
    main()
