"""Release gate: a new model or tracking change ships only if it is not worse than the current one on the fixed test set.
Test set = the clips eval_app.py plays (this pond: clipA + clipB, played in one browser profile); add each pond's clips as they come.
    .venv/bin/python release_gate.py <base prefix> <candidate prefix> [strides=1,5,10]
    e.g. release_gate.py eval_rc4 eval_rc5   -> compares eval_rc4_s1/_s5/_s10 with eval_rc5_s1/_s5/_s10
Rules per clip and stride: name clashes (sure-dup) <= 2 %; named share >= base - 3 points; count >= base - 1; 0 console errors.
Exit code 1 if any rule fails.
"""
import sys

from eval_summary import console_errors, metrics

DUP_MAX, NAMED_DROP, COUNT_DROP = 0.02, 0.03, 1.0


def gate(base, cand, strides):
    rows, ok = [], True
    for s in strides:
        b, c = f'{base}_s{s}', f'{cand}_s{s}'
        mb, mc = metrics(b), metrics(c)
        errs = console_errors(c)
        for clip in mc:
            x, y = mb.get(clip), mc[clip]
            checks = {'dup': y['dup'] <= DUP_MAX, 'named': x is None or y['named'] >= x['named'] - NAMED_DROP,
                      'count': x is None or y['count'] >= x['count'] - COUNT_DROP, 'errors': errs == 0}
            ok &= all(checks.values())
            rows.append((s, clip, x, y, errs, checks))
    for s, clip, x, y, errs, ch in rows:
        f = lambda m, k, p=False: '-' if m is None else (f'{100 * m[k]:.1f}%' if p else f'{m[k]:g}')
        print(f"stride {s:<2} {clip}: dup {f(y, 'dup', 1)} | named {f(x, 'named', 1)} -> {f(y, 'named', 1)} | count {f(x, 'count')} -> {f(y, 'count')} "
              f"| fish {f(x, 'fish')} -> {f(y, 'fish')} | errors {errs} | {'PASS' if all(ch.values()) else 'FAIL ' + ','.join(k for k, v in ch.items() if not v)}")
    print('RELEASE GATE:', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    strides = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,5,10').split(',')]
    sys.exit(0 if gate(sys.argv[1], sys.argv[2], strides) else 1)
