"""
test_patterns.py — correctness tests for every registered detector.

Two failure modes matter here, and neither shows up in ordinary use.

LOOKAHEAD. patterns.py documents a real bug found and fixed in
detect_rectangle(): swings were admitted whose confirming bar had not yet
printed, on 40% of evaluation points, making every downstream backtest number
quietly optimistic. It survived an empirical sweep by luck, because median
clustering is robust to one stray point. Test 1 makes that class of bug fail
deterministically: run the detector on the full series, re-run it on data
truncated at the evaluation bar, and demand an identical result. Anything
using future information cannot pass.

DEAD DETECTOR. A detector that never fires is indistinguishable from a market
containing no such pattern — both give an empty list and a clean log. Test 5
builds a textbook instance of each shape and requires detection.

Fixtures use np.interp over anchors, NOT chained linspace: chaining repeats
the turning-point value on consecutive bars, find_swing_points() requires a
STRICT extremum, so the shape has no swings and every detector correctly
returns nothing. The fixture must contain the pattern before an empty result
can be blamed on the detector.

    python test_patterns.py
"""
import numpy as np
import pandas as pd
import patterns as pt

CFG = dict(swing_window=3, lookback=60, min_bars_before=70, min_touches=2,
           min_r2=0.55, min_slope_pct=0.02, min_convergence=0.8,
           tolerance_pct=1.5, min_span_bars=10, breakout_buffer_pct=0.3,
           peak_tolerance_pct=3.0, min_depth_pct=2.0, min_separation_bars=6)


def synth(n=400, seed=7):
    r = np.random.default_rng(seed)
    close = np.maximum(100 + np.cumsum(r.normal(0, 1.2, n)), 5)
    return pd.DataFrame({
        "Open": close, "Close": close,
        "High": close * (1 + np.abs(r.normal(0, .008, n))),
        "Low": close * (1 - np.abs(r.normal(0, .008, n))),
        "Volume": r.integers(1e6, 5e6, n)})


def zigzag(anchors, n):
    c = np.interp(np.arange(n), [a[0] for a in anchors], [a[1] for a in anchors])
    return pd.DataFrame({"Open": c, "Close": c, "High": c * 1.004,
                         "Low": c * 0.996, "Volume": np.full(n, 2_000_000)})


fails = 0
print("=" * 72)
print("TEST 1 — NO LOOKAHEAD (truncation equivalence)")
print("=" * 72)
for name in sorted(pt.SCANNERS):
    checked = leaked = 0
    for seed in range(6):
        df = synth(seed=seed)
        full = {h["bar_index"]: h for h in pt.scan(name, df, CFG)}
        for k in sorted(full)[:6]:
            trunc = df.iloc[:k + 1].reset_index(drop=True)
            got = {h["bar_index"]: h for h in pt.scan(name, trunc, CFG)}
            checked += 1
            a = {x: v for x, v in full[k].items() if x != "bar_index"}
            b = {x: v for x, v in got.get(k, {}).items() if x != "bar_index"}
            if a != b:
                leaked += 1
                if leaked == 1:
                    print(f"   {name}: MISMATCH at bar {k}, fields "
                          f"{ {x for x in set(a) | set(b) if a.get(x) != b.get(x)} }")
    fails += leaked
    print(f"   {name:10} {checked:3} detections re-verified truncated -> "
          f"{'PASS' if not leaked else f'FAIL ({leaked})'}")

print("\n" + "=" * 72)
print("TEST 2 — LEVEL ORDERING (stop/entry/target)")
print("=" * 72)
for name in sorted(pt.SCANNERS):
    hits = [h for s in range(6) for h in pt.scan(name, synth(seed=s), CFG)]
    bad = sum(1 for h in hits
              if not ((h["stop"] < h["entry"] < h["target"])
                      if h["direction"] == "Bullish"
                      else (h["target"] < h["entry"] < h["stop"])))
    fails += bad
    print(f"   {name:10} {len(hits):4} detections -> "
          f"{'PASS' if not bad else f'FAIL ({bad})'}")

print("\n" + "=" * 72)
print("TEST 3 — r2 GUARD REJECTS SCATTER")
print("=" * 72)
noise = synth(seed=99)
strict = len(pt.scan("trendline", noise, {**CFG, "min_r2": 0.90}))
loose = len(pt.scan("trendline", noise, {**CFG, "min_r2": 0.10}))
print(f"   noise @ min_r2=0.90: {strict}   @ min_r2=0.10: {loose}")
if strict >= loose:
    print("   FAIL — r2 guard is not filtering scatter fits")
    fails += 1
else:
    print(f"   -> guard removes {loose - strict} scatter fits: PASS")

print("\n" + "=" * 72)
print("TEST 4 — CHANNEL AND TRIANGLE ARE DISJOINT")
print("=" * 72)
overlap = 0
for s in range(6):
    df = synth(seed=s)
    overlap += len({h["bar_index"] for h in pt.scan("trendline", df, CFG)}
                   & {h["bar_index"] for h in pt.scan("triangle", df, CFG)})
print(f"   shared detection bars across 6 series: {overlap}")
if overlap:
    print("   FAIL — the same breakout is counted by two families")
    fails += 1
else:
    print("   -> disjoint: PASS")

print("\n" + "=" * 72)
print("TEST 5 — EVERY DETECTOR FIRES ON ITS TEXTBOOK SHAPE")
print("=" * 72)


def shape_check(name, df, cfg, want_dir, want_sub=None):
    global fails
    hits = pt.scan(name, df, cfg)
    if not hits:
        print(f"   {name:10} FAIL — no detection on a textbook {name}")
        fails += 1
        return
    h = hits[0]
    sub = f" {h['subtype']}" if "subtype" in h else ""
    ok = h["direction"] == want_dir and (not want_sub or h.get("subtype") == want_sub)
    fails += 0 if ok else 1
    print(f"   {name:10} {len(hits):3} hits · bar {h['bar_index']} "
          f"{h['direction']}{sub} -> {'PASS' if ok else 'FAIL'}")


n, amp, a, x = 240, 16.0, [(0, 100.0)], 60
for k in range(9):
    a.append((x, 100.0 + (amp if k % 2 == 0 else -amp)))
    amp *= 0.80
    x += 9
a += [(x + 6, 122.0), (n - 1, 122.0)]
shape_check("triangle", zigzag(a, n),
            dict(swing_window=3, lookback=100, min_bars_before=100,
                 min_touches=3, min_r2=0.70, min_convergence=0.75,
                 breakout_buffer_pct=0.3, min_span_bars=15), "Bullish")

a, x = [(0, 100.0)], 60
for k in range(9):
    a.append((x, 100.0 + 0.30 * x + (7 if k % 2 == 0 else -7)))
    x += 9
a += [(x + 6, 100.0 + 0.30 * (x + 6) + 26), (239, 100.0 + 0.30 * 240 + 26)]
shape_check("trendline", zigzag(a, 240),
            dict(swing_window=3, lookback=100, min_bars_before=100,
                 min_touches=3, min_r2=0.70, min_slope_pct=0.02,
                 breakout_buffer_pct=0.3, min_span_bars=15), "Bullish")

dbl = dict(swing_window=3, lookback=90, min_bars_before=80,
           peak_tolerance_pct=2.0, min_separation_bars=8,
           min_depth_pct=3.0, breakout_buffer_pct=0.3)
shape_check("double", zigzag([(0, 100.), (40, 100.), (70, 120.), (95, 108.),
                              (125, 120.), (150, 103.), (199, 103.)], 200),
            dbl, "Bearish", "double_top")
shape_check("double", zigzag([(0, 120.), (40, 120.), (70, 100.), (95, 112.),
                              (125, 100.), (150, 118.), (199, 118.)], 200),
            dbl, "Bullish", "double_bottom")

a, x = [(0, 100.0)], 50
for k in range(8):
    a.append((x, 110.0 if k % 2 == 0 else 100.0))
    x += 9
a += [(x + 6, 118.0), (219, 118.0)]
shape_check("rectangle", zigzag(a, 220),
            dict(swing_window=3, lookback=90, min_bars_before=95,
                 min_touches=2, tolerance_pct=1.5, min_span_bars=10,
                 breakout_buffer_pct=0.3), "Bullish")

print("\n" + "=" * 72)
print(f"{'ALL TESTS PASSED' if not fails else f'{fails} FAILURE(S)'}")
print("=" * 72)
raise SystemExit(1 if fails else 0)
