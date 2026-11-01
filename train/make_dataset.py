"""Auto-label koi in the pond videos -> COCO instance-seg dataset for RF-DETR.

OWLv2 ("koi fish") on the full frame + two square tiles gives boxes, SAM2.1 turns
each box into a mask. Last 20% of each clip is the valid split (no time leakage).

    .venv/bin/python make_dataset.py            # writes data/koi/{train,valid,test}
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parent
VIDEOS = {
    "A": (ROOT.parent / "814441436.111085.mp4", None),
    # clip B carries a video-call overlay top-right: pixels stay original, no labels inside it
    "B": (ROOT.parent / "814441508.105654.mp4", (330, 0, 540, 280)),
}
STEP = 4            # every 4th frame (~7.5 fps of samples)
THRESH = 0.14       # OWLv2 score; calibration knob, raise if reflections get boxed
OUT = ROOT / "data" / "koi"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def frames(path):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


class Labeler:
    def __init__(self):
        self.op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
        self.sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
        self.sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(DEV).eval()

    @torch.no_grad()
    def owl(self, img):
        h, w = img.shape[:2]
        inp = self.op(text=[["a photo of a koi fish"]], images=Image.fromarray(img), return_tensors="pt").to(DEV)
        out = self.om(**inp)
        side = max(h, w)  # OWLv2 pads to square
        r = self.op.post_process_grounded_object_detection(out, threshold=THRESH, target_sizes=torch.tensor([[side, side]]))[0]
        return r["boxes"].cpu(), r["scores"].cpu()

    def boxes(self, img):
        h, w = img.shape[:2]
        allb, alls = [], []
        # full frame + top/bottom square tiles: far (small) fish need more pixels
        for y0, y1 in [(0, h), (max(0, h // 2 - w), h // 2 + w // 2), (h - w, h)]:
            b, s = self.owl(img[y0:y1])
            b[:, [1, 3]] += y0
            allb.append(b); alls.append(s)
        b, s = torch.cat(allb), torch.cat(alls)
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
        keep = nms(b, s, 0.5)
        b, s = b[keep], s[keep]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ok = (area > 60) & (area < 0.12 * w * h)
        b, s, area = b[ok], s[ok], area[ok]
        # drop "group" boxes that swallow 2+ other fish
        x0 = torch.max(b[:, None, 0], b[None, :, 0]); y0 = torch.max(b[:, None, 1], b[None, :, 1])
        x1 = torch.min(b[:, None, 2], b[None, :, 2]); y1 = torch.min(b[:, None, 3], b[None, :, 3])
        inter = (x1 - x0).clamp(0) * (y1 - y0).clamp(0)
        ioa = inter / area[None, :]          # fraction of box j inside box i
        ioa.fill_diagonal_(0)
        group = (ioa > 0.8).sum(1) >= 2
        return b[~group], s[~group]

    @torch.no_grad()
    def masks(self, img, boxes):
        if len(boxes) == 0:
            return []
        inp = self.sp(images=Image.fromarray(img), input_boxes=[boxes.tolist()], return_tensors="pt").to(DEV)
        out = self.sm(**inp, multimask_output=False)
        m = self.sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        return [x[0].numpy().astype(np.uint8) for x in m]


def polygon(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c.astype(float).flatten().tolist() if len(c) >= 3 else None


def main():
    # pass 1: label frames, one json line each (resumable: MPS memory creeps, restarts are cheap)
    OUT.mkdir(parents=True, exist_ok=True)
    jl = OUT / "labels.jsonl"
    done = {json.loads(l)["name"] for l in jl.read_text().splitlines()} if jl.exists() else set()
    lab = None
    with jl.open("a") as fo:
        for clip, (path, pip) in VIDEOS.items():
            n = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
            for i, img in frames(path):
                name = f"{clip}_{i:05d}.jpg"
                if name in done:
                    continue
                lab = lab or Labeler()
                split = "valid" if i >= 0.8 * n else "train"
                (OUT / split).mkdir(exist_ok=True)
                cv2.imwrite(str(OUT / split / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                anns = []
                boxes = lab.boxes(img)[0]
                if pip:  # drop boxes centred in the overlay
                    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
                    boxes = boxes[~((cx >= pip[0]) & (cx <= pip[2]) & (cy >= pip[1]) & (cy <= pip[3]))]
                for m in lab.masks(img, boxes):
                    poly = polygon(m)
                    if poly is None:
                        continue
                    ys, xs = np.nonzero(m)
                    anns.append({"bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                                 "area": float(m.sum()), "poly": poly})
                fo.write(json.dumps({"name": name, "split": split, "w": img.shape[1], "h": img.shape[0], "anns": anns}) + "\n"); fo.flush()
                if DEV == "mps":
                    torch.mps.empty_cache()
                print(clip, i, split, len(anns), flush=True)
    # pass 2: COCO json per split + review sheet
    splits = {k: {"images": [], "annotations": []} for k in ("train", "valid")}
    sheet = []
    for k, l in enumerate(jl.read_text().splitlines()):
        r = json.loads(l)
        S = splits[r["split"]]
        iid = len(S["images"])
        S["images"].append({"id": iid, "file_name": r["name"], "width": r["w"], "height": r["h"]})
        for a in r["anns"]:
            S["annotations"].append({"id": len(S["annotations"]), "image_id": iid, "category_id": 1, "bbox": a["bbox"],
                                     "area": a["area"], "segmentation": [a["poly"]], "iscrowd": 0})
        if k % 30 == 0:
            vis = cv2.imread(str(OUT / r["split"] / r["name"]))
            for a in r["anns"]:
                cv2.polylines(vis, [np.array(a["poly"]).reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 1)
            sheet.append(cv2.cvtColor(cv2.resize(vis, (270, 480)), cv2.COLOR_BGR2RGB))
    cats = [{"id": 0, "name": "fish", "supercategory": "none"}, {"id": 1, "name": "koi", "supercategory": "fish"}]
    for d, S in splits.items():
        (OUT / d / "_annotations.coco.json").write_text(json.dumps({"images": S["images"], "annotations": S["annotations"], "categories": cats}))
        print(d, len(S["images"]), "images", len(S["annotations"]), "masks")
    # rfdetr wants a test split; reuse valid
    test = OUT / "test"
    if not test.exists():
        test.symlink_to(OUT / "valid", target_is_directory=True)
    rows = [np.hstack(sheet[r:r + 6] + [np.zeros_like(sheet[0])] * (6 - len(sheet[r:r + 6]))) for r in range(0, len(sheet), 6)]
    cv2.imwrite(str(ROOT / "data" / "labels_sheet.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    sys.exit(main())
