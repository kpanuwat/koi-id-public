"""Fine-tune the koi re-ID embedder without manual labels, export ONNX for the PWA.

Labels come from the app itself (eval_app.py --crops): crops of one track = same fish (positives),
tracks visible in the same frames = different fish (negatives, guaranteed). Each batch is drawn
from one time window so every negative in it is a real negative.
Augmentations target what changes between sessions: light (morning/evening), white balance,
blur, resolution and JPEG of different phones, small view changes, and partly visible fish
(head/mid/tail windows zoomed like the app does, random occlusion). No horizontal flip (left
and right patterns differ).

    .venv/bin/python train_reid.py              # held-out check A->B and B->A, then final on both
    .venv/bin/python train_reid.py 1200 --central   # final also learns from the curated central store (curate.py):
        crops of one pond fish = positives, fish of the same pond seen together on screen (gallery covis) = negatives
    -> runs/reid.pt, runs/reid_report.json, ../app/models/embed.onnx (via export_reid.py)
"""
import base64, io, json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms as T
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GAP = 15        # positives at least this many frames apart (different moment / view)
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1200
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def load(clip, res):
    r = res[clip]
    live = defaultdict(list)
    for i, fr in enumerate(r["frames"]):
        for t in fr["tracks"]:
            live[t["id"]].append(i + 1)
    span = {t: (min(v), max(v)) for t, v in live.items()}
    by = defaultdict(list)
    for x in r["embs"]:
        if x.get("crop"):
            by[x["track"]].append(x)
    by = {t: xs for t, xs in by.items() if len(xs) >= 3 and xs[-1]["f"] - xs[0]["f"] >= 6}
    return by, span


def decode(x):
    if "path" in x:  # central store crop on disk (curate.py)
        return Image.open(ROOT / x["path"]).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(x["crop"].split(",", 1)[1]))).convert("RGB")


class Jpeg:  # phone-to-phone compression differences
    def __call__(self, im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(35, 95)); return Image.open(b).convert("RGB")


class LowRes:  # small/far fish and low-res sensors
    def __call__(self, im):
        if random.random() < 0.5:
            s = random.uniform(0.35, 0.9)
            im = im.resize((max(8, int(224 * s)), max(4, int(112 * s))), Image.BILINEAR).resize((224, 112), Image.BILINEAR)
        return im


class Partial:  # fish only partly visible (head, mid-body, tail; rest under water/glare): the app scales the
    # visible part to the crop width, so cut a window along the body axis and zoom it to full size the same way
    def __call__(self, im):
        if random.random() < 0.5:
            w = random.uniform(0.4, 0.9); x0 = random.uniform(0.06, 0.94 - w * 0.88) * 224
            cw, ch = w * 0.88 * 224 / 0.88, w * 112
            im = im.crop((int(x0 - (cw - w * 0.88 * 224) / 2), int(56 - ch / 2), int(x0 - (cw - w * 0.88 * 224) / 2 + cw), int(56 + ch / 2))).resize((224, 112), Image.BILINEAR)
        return im


AUG = T.Compose([
    Partial(),
    T.RandomApply([T.ColorJitter(0.45, 0.4, 0.35, 0.04)], p=0.9),
    T.RandomApply([T.Lambda(lambda im: im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.6))))], p=0.4),
    LowRes(), Jpeg(),
    T.RandomAffine(degrees=8, translate=(0.04, 0.06), scale=(0.88, 1.12), fill=(124, 116, 104)),
    T.ToTensor(), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.3, scale=(0.04, 0.2), value=0),  # occlusion by glare / another fish (0 = mean grey after Normalize)
])
PLAIN = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class Embed(torch.nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.m = AutoModel.from_pretrained("facebook/dinov2-small")
        self.head = torch.nn.Linear(768, dim)
        for p in self.m.parameters():
            p.requires_grad = False
        for blk in self.m.encoder.layer[-4:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.m.layernorm.parameters():
            p.requires_grad = True

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state
        return F.normalize(self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)), dim=1)


class Baseline(torch.nn.Module):  # what the app shipped before: frozen DINOv2 CLS+mean
    def __init__(self):
        super().__init__(); self.m = AutoModel.from_pretrained("facebook/dinov2-small")

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state
        return F.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1), dim=1)


def supcon(z, y, tau=0.1):
    sim = z @ z.T / tau
    eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def clip_co(span):  # tracks alive at the same moment => distinct fish
    def co(a, tracks):
        s0, s1 = span[a]; f0 = random.randint(s0, s1)
        return [t for t in tracks if t in span and span[t][0] <= f0 + 10 and span[t][1] >= f0 - 10]
    return co


def central_co(covis):  # pond fish confirmed on screen together (gallery covis) => distinct fish
    def co(a, tracks):
        return [t for t in tracks if t[0] == "C" and t[1] == a[1] and (t == a or (a[1], *sorted((a[2], t[2]))) in covis)]
    return co


def batches(by, co_of, P=16, K=4):
    tracks = list(by)
    while True:
        a = random.choice(tracks)
        co = co_of(a, tracks)
        if len(co) < 4:
            continue
        pick = random.sample(co, min(P, len(co)))
        ims, ys = [], []
        for k, t in enumerate(pick):
            for x in random.choices(by[t], weights=[x.get("w", 1) for x in by[t]], k=K):  # curated weight: gold > silver, good devices > poor
                ims.append(AUG(decode(x))); ys.append(k)
        yield torch.stack(ims), torch.tensor(ys)


@torch.no_grad()
def evaluate(model, by, span):
    model.eval()
    keys, embs = [], []
    items = [(t, x) for t, xs in by.items() for x in xs]
    for i in range(0, len(items), 64):
        chunk = items[i:i + 64]
        embs.append(model(torch.stack([PLAIN(decode(x)) for _, x in chunk]).to(DEV)).cpu())
        keys += [(t, x["f"]) for t, x in chunk]
    E = torch.cat(embs).numpy()
    idx = defaultdict(list)
    for i, (t, f) in enumerate(keys):
        idx[t].append((i, f))
    pos, neg, rng = [], [], np.random.default_rng(0)
    for t, v in idx.items():
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if v[b][1] - v[a][1] >= GAP:
                    pos.append(float(E[v[a][0]] @ E[v[b][0]]))
    ts = list(idx)
    for a in range(len(ts)):
        for b in range(a + 1, len(ts)):
            A, B = span[ts[a]], span[ts[b]]
            if min(A[1], B[1]) - max(A[0], B[0]) >= 5:
                for _ in range(4):
                    i, j = idx[ts[a]][rng.integers(len(idx[ts[a]]))][0], idx[ts[b]][rng.integers(len(idx[ts[b]]))][0]
                    neg.append(float(E[i] @ E[j]))
    pos, neg = np.array(pos), np.array(neg)
    auc = float((pos[:, None] > neg[None, rng.integers(len(neg), size=min(4000, len(neg)))]).mean())
    out = {"auc": round(auc, 3), "pos": len(pos), "neg": len(neg)}
    for fpr in (0.01, 0.05):
        th = float(np.quantile(neg, 1 - fpr)); out[f"tpr@{fpr}"] = round(float((pos >= th).mean()), 3); out[f"thr@{fpr}"] = round(th, 3)
    model.train()
    return out


def train(by, co_of, iters=ITERS):
    model = Embed().to(DEV)
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": 1e-3},
                             {"params": [p for p in model.m.parameters() if p.requires_grad], "lr": 2e-5}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 2e-5], total_steps=iters, pct_start=0.1)
    gen = batches(by, co_of)
    for it in range(iters):
        x, y = next(gen)
        loss = supcon(model(x.to(DEV)), y.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 100 == 0:
            print(f"  it {it} loss {loss.item():.3f}", flush=True)
        if DEV == "mps" and it % 50 == 0:
            torch.mps.empty_cache()
    return model


def load_central():
    """curated central crops: {("C", pond, fish): [{path, w}]} and covis {(pond, a, b)} from each pond's gallery.json"""
    man, by, covis = ROOT / "data" / "central" / "manifest.jsonl", defaultdict(list), set()
    if not man.exists():
        return {}, covis
    for line in man.read_text().splitlines():
        m = json.loads(line)
        if m["keep"]:
            by[("C", m["pond"], m["fish"])].append({"path": m["path"], "w": m["weight"]})
    for g in (ROOT / "data" / "central").glob("*/gallery.json"):
        for pr in json.loads(g.read_text()).get("covis", []):
            a, b = sorted(int(x) for x in pr.split("|")); covis.add((g.parent.name, a, b))
    return {k: v for k, v in by.items() if len(v) >= 3}, covis


def main():
    res = json.load(open(ROOT / "runs" / "crops" / "result.json"))
    data = {c: load(c, res) for c in res}
    for c, (by, _) in data.items():
        print(c, len(by), "tracks", sum(len(v) for v in by.values()), "crops")
    report = {}
    base = Baseline().to(DEV)
    for tr, te in (("clipA", "clipB"), ("clipB", "clipA")):
        b = evaluate(base, *data[te])
        m = evaluate(train(data[tr][0], clip_co(data[tr][1])), *data[te])
        report[f"{tr}->{te}"] = {"baseline_dinov2": b, "finetuned": m}
        print(tr, "->", te, "baseline", b, "\n   finetuned", m, flush=True)
    # final: both clips, export
    by = {**{("A", t): v for t, v in data["clipA"][0].items()}, **{("B", t): v for t, v in data["clipB"][0].items()}}
    span = {**{("A", t): v for t, v in data["clipA"][1].items()}, **{("B", t): v for t, v in data["clipB"][1].items()}}
    co_of = clip_co(span)
    if "--central" in sys.argv:
        cb, covis = load_central()
        print("central store:", len(cb), "pond fish", sum(len(v) for v in cb.values()), "crops")
        if cb:
            by.update(cb); cco = central_co(covis)
            co_of = lambda a, tracks, c=co_of: cco(a, tracks) if a[0] == "C" else c(a, [t for t in tracks if t[0] != "C"])
            report["central"] = {"fish": len(cb), "crops": sum(len(v) for v in cb.values()), "pairs": len(covis)}
    model = train(by, co_of).to("cpu").eval()
    torch.save(model.state_dict(), ROOT / "runs" / "reid.pt")
    (ROOT / "runs" / "reid_report.json").write_text(json.dumps(report, indent=1))
    from export_reid import main as export  # fp16 + baked position embeddings for WebGPU
    export()


if __name__ == "__main__":
    main()
