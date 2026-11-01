"""Release gate that survives eval noise: compare MEANS over several strides, tolerance from the measured spread.

    .venv/bin/python robust_gate.py <base prefix> <candidate prefix> [strides=8,9,10,11,12]
    e.g. robust_gate.py eval_v63 eval_t16   -> eval_v63_s8 .. eval_v63_s12 vs eval_t16_s8 .. eval_t16_s12

Why: the named share of one clip at one stride moves +-6 points between near-identical detectors (koi_det_lite vs
koi_det_lite16, 26 Oct 2026: clipA 0.17 -> 0.11, clipB 0.33 -> 0.37), so release_gate.py's -3 point rule on a
single stride passes or fails by chance. Each stride is a different sample of the clip, so the spread across
strides measures that chaos. Rules per clip (mean over strides):
  named  >= base - max(0.03, 2 * se)      count >= base - max(1.0, 2 * se)      se = sqrt(sd_b^2/n + sd_c^2/n)
  sure-dup <= 2 % on EVERY stride;  0 console errors on every stride.
Exit code 1 if any rule fails.
"""
import sys

import numpy as np

from eval_summary import console_errors, metrics

DUP_MAX = 0.02


def stats(prefix, strides):
    per = {}
    for s in strides:
        for clip, m in metrics(f"{prefix}_s{s}").items():
            per.setdefault(clip, []).append(m)
    return per


def main():
    base, cand = sys.argv[1], sys.argv[2]
    strides = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "8,9,10,11,12").split(",")]
    B, C = stats(base, strides), stats(cand, strides)
    errs = sum(console_errors(f"{cand}_s{s}") for s in strides)
    ok = errs == 0
    print(f"robust gate {base} -> {cand}, strides {strides}, console errors {errs}")
    for clip in C:
        b, c = B.get(clip, []), C[clip]
        n = min(len(b), len(c))
        row = []
        for k, floor in (("named", 0.03), ("count", 1.0)):
            vb, vc = np.array([x[k] for x in b]), np.array([x[k] for x in c])
            se = float(np.sqrt(vb.var(ddof=1) / len(vb) + vc.var(ddof=1) / len(vc))) if n > 1 else 0.0
            tol = max(floor, 2 * se)
            good = vc.mean() >= vb.mean() - tol
            ok &= bool(good)
            fmt = (lambda v: f"{100 * v:.1f}%") if k == "named" else (lambda v: f"{v:.1f}")
            row.append(f"{k} {fmt(vb.mean())}±{fmt(vb.std(ddof=1)) if len(vb) > 1 else '-'} -> {fmt(vc.mean())}±{fmt(vc.std(ddof=1)) if len(vc) > 1 else '-'} "
                       f"(tol {fmt(tol)}) {'ok' if good else 'FAIL'}")
        dmax = max(x["dup"] for x in c)
        ok &= dmax <= DUP_MAX
        row.append(f"sure-dup max {100 * dmax:.1f}% {'ok' if dmax <= DUP_MAX else 'FAIL'}")
        print(f"  {clip} (n={n}): " + " | ".join(row))
    print("ROBUST GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
