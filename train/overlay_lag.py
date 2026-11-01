"""How well do overlay boxes from one detector pass sit on the fish one pass later (the realtime display lag)?
Compares predictors on a strided eval run (eval_app.py --qs=stride=S): none, box velocity, camera shift
(phase correlation of the real video frames), camera shift + fish's own velocity.
    .venv/bin/python overlay_lag.py runs/eval_r_s5/result.json [lag_passes=1]
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

res = json.load(open(sys.argv[1])); LAG = int(sys.argv[2]) if len(sys.argv) > 2 else 1
VID = {'clipA': '../app/models/clipA.mp4', 'clipB': '../app/models/clipB.mp4'}

def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

def gray(cap, t, W):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = 160 / g.shape[1]; return cv2.resize(g, None, fx=s, fy=s), g.shape[1] / 160

for clip, d in res.items():
    F = d['frames']; cap = cv2.VideoCapture(VID[clip]); cache = {}
    def frame(i):
        if i not in cache: cache[i] = gray(cap, F[i]['t'], None)
        return cache[i]
    win = None; scores = {k: [] for k in ['none', 'vel', 'cam', 'cam+fish']}
    for k in range(1, len(F) - LAG):
        a, b, prev = F[k], F[k + LAG], F[k - 1]
        if b['t'] < a['t']: continue  # loop wrap
        ga, sc = frame(k); gb, _ = frame(k + LAG); gp, _ = frame(k - 1)
        if win is None: win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win); cam = np.array([dx, dy, dx, dy]) * sc
        (px, py), _ = cv2.phaseCorrelate(gp, ga, win); camPrev = np.array([px, py, px, py]) * sc
        A = {t['id']: np.array(t['box'], float) for t in a['tracks']}; B = {t['id']: np.array(t['box'], float) for t in b['tracks']}
        P = {t['id']: np.array(t['box'], float) for t in prev['tracks']}
        for tid, box in A.items():
            if tid not in B: continue
            vel = (box - P[tid]) if tid in P else np.zeros(4)
            fish = (vel - camPrev) if tid in P else np.zeros(4)  # own swimming per pass
            scores['none'].append(iou(box, B[tid]))
            scores['vel'].append(iou(box + vel * LAG, B[tid]))
            scores['cam'].append(iou(box + cam, B[tid]))
            scores['cam+fish'].append(iou(box + cam + fish * LAG, B[tid]))
    dt = (F[1 + LAG]['t'] - F[1]['t'])
    print(clip, f'lag {LAG} pass(es) = {dt:.2f} s, pairs {len(scores["none"])}')
    for k, v in scores.items():
        v = np.array(v); print(f'  {k:9s} mean IoU {v.mean():.2f}  on-fish (IoU>0.5) {np.mean(v > 0.5):.0%}  lost (IoU<0.1) {np.mean(v < 0.1):.0%}')
