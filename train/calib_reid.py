"""Self-supervised threshold check from an eval_app.py run (no manual labels).

positives = two views of the same track >= GAP frames apart; negatives = tracks seen in the same frame
(cannot be the same fish). Prints score distributions (same formula as reid.js) and size stability.

    .venv/bin/python calib_reid.py [runs/eval/result.json]
"""
import json, sys
from collections import defaultdict

import numpy as np

GAP = 15


def score(a, b):
    return 0.8 * float(np.dot(a["e"], b["e"])) + 0.2 * (1 - np.abs(np.array(a["hist"]) - np.array(b["hist"])).sum() / 2)


def main():
    res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/eval/result.json"))
    for clip, r in res.items():
        E = r["embs"]
        for x in E:
            x["e"] = np.array(x["e"], dtype=np.float32)
        live = defaultdict(set)  # frame index -> live track ids
        for i, fr in enumerate(r["frames"]):
            for t in fr["tracks"]:
                live[i + 1].add(t["id"])
        span = defaultdict(lambda: [1e9, -1])
        for f, ids in live.items():
            for t in ids:
                span[t][0] = min(span[t][0], f); span[t][1] = max(span[t][1], f)
        by = defaultdict(list)
        for x in E:
            by[x["track"]].append(x)
        pos, neg = [], []
        for t, xs in by.items():
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    if xs[j]["f"] - xs[i]["f"] >= GAP:
                        pos.append(score(xs[i], xs[j]))
        tracks = list(by)
        rng = np.random.default_rng(0)
        for a in range(len(tracks)):
            for b in range(a + 1, len(tracks)):
                A, B = span[tracks[a]], span[tracks[b]]
                if min(A[1], B[1]) - max(A[0], B[0]) >= 5:  # co-visible => different fish
                    xa, xb = by[tracks[a]], by[tracks[b]]
                    for _ in range(min(6, len(xa) * len(xb))):
                        neg.append(score(xa[rng.integers(len(xa))], xb[rng.integers(len(xb))]))
        pos, neg = np.array(pos), np.array(neg)
        q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")
        print(f"\n{clip}: {len(E)} embeddings, {len(by)} tracks, {len(pos)} pos pairs, {len(neg)} neg pairs")
        print("  pos  p5 %.3f  p25 %.3f  p50 %.3f" % (q(pos, 5), q(pos, 25), q(pos, 50)))
        print("  neg  p50 %.3f  p95 %.3f  p99 %.3f  max %.3f" % (q(neg, 50), q(neg, 95), q(neg, 99), neg.max() if len(neg) else float("nan")))
        for th in np.arange(0.5, 0.95, 0.05):
            print(f"  thr {th:.2f}: pos kept {np.mean(pos >= th):.2f}  neg passed {np.mean(neg >= th):.3f}")
        # size stability: per-track spread of log relative size (0.10 ~ 10% length jitter)
        sz = defaultdict(list)
        for fr in r["frames"]:
            for t in fr["tracks"]:
                if t.get("s") is not None:
                    sz[t["id"]].append(t["s"])
        sd = [np.std(v) for v in sz.values() if len(v) >= 10]
        if sd:
            print(f"  size: {len(sd)} tracks with >=10 measures, median std(log size) {np.median(sd):.3f}, p75 {np.percentile(sd, 75):.3f}")
            allm = [np.median(v) for v in sz.values() if len(v) >= 10]
            print(f"  size spread across fish: std of track medians {np.std(allm):.3f}")
        print(f"  fish created {len(r['fish'])}: " + ", ".join(f"{f['name']}({f['variety']},{f['ex']}v)" for f in r["fish"][:40]))


if __name__ == "__main__":
    main()
