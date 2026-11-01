"""data/koi2 (RF-DETR COCO: fish/koi/pond) -> /content/koiyx in YOLOX's COCO layout, koi boxes only (pond dropped)."""
import json, os, shutil
from pathlib import Path

SRC, DST = Path(os.environ.get("SRC", "/content/data/koi2")), Path(os.environ.get("DST", "/content/koiyx"))
(DST / "annotations").mkdir(parents=True, exist_ok=True)
for split, name, ann in (("train", "train2017", "train.json"), ("valid", "val2017", "val.json")):
    d = json.load(open(SRC / split / "_annotations.coco.json"))
    (DST / name).mkdir(exist_ok=True)
    for im in d["images"]:
        dst = DST / name / im["file_name"]
        if not dst.exists():
            os.link(SRC / split / im["file_name"], dst)
    anns = [{**{k: a[k] for k in ("id", "image_id", "bbox", "area", "iscrowd")}, "category_id": 1}
            for a in d["annotations"] if a["category_id"] == 1]
    json.dump({"images": d["images"], "annotations": anns, "categories": [{"id": 1, "name": "koi"}]},
              open(DST / "annotations" / ann, "w"))
    print(split, len(d["images"]), "images", len(anns), "koi")
