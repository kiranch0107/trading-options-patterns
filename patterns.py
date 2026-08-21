"""
patterns.py — Chart pattern detectors
======================================
Pure geometry. Each detector answers ONLY "does this shape exist at bar i?"
and, if so, what levels the pattern implies (entry/stop/target per the
standard technical-analysis convention: breakout = entry, opposite side of the
range = stop, pattern height projected from the breakout = target).

None of this file expresses an opinion about whether the pattern is tradeable.
That question belongs entirely to pattern_backtest.py, which is the only
component allowed to say whether a detected pattern has a real edge.

WHY RECTANGLES FIRST
---------------------
Of all chart patterns, a trading range has the cleanest, least subjective
definition: two roughly horizontal boundaries, price touching each one
multiple times, for a meaningful stretch of bars. "Roughly horizontal" and
"touching" both become simple numeric thresholds (tolerance %, touch count),
so two people implementing this from the same definition would write
essentially the same detector — which is what makes a backtest result on it
trustworthy rather than an artifact of one person's coding choices.

Reversal patterns like head-and-shoulders are deliberately NOT here yet: their
definitions ("similar peak heights", "a clear neckline") are inherently
fuzzier, and fuzzy definitions make backtest results hard to trust either way.

SWING POINT DETECTION
----------------------
A bar is a swing HIGH if its High is the max within a symmetric window on both
sides (and likewise for swing LOW). This is the simplest, most defensible pivot
rule — no smoothing, no adaptive lookback, nothing that could be tuned to
flatter a backtest after the fact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# SWING POINTS
# ══════════════════════════════════════════════════════════════════════
def find_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3
                      ) -> tuple[list[int], list[int]]:
    """
    Returns (swing_high_indices, swing_low_indices) as positions into df.

    A bar at position i is a swing high if df['High'][i] is the strict maximum
    over the window [i-left, i+right]. Swing low is the mirror on 'Low'.

    `right` bars of lookahead means a swing point at position i is only
    CONFIRMED once bar i+right has occurred — a swing detected at i is never
    used by the caller before that point, so this does not leak the future
    into a decision made before the swing was knowable. The backtest driver
    enforces this by only ever looking at swings whose confirming bar has
    already passed relative to the evaluation index.
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    swing_high, swing_low = [], []
    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        if highs[i] == window_h.max() and np.sum(window_h == highs[i]) == 1:
            swing_high.append(i)
        window_l = lows[i - left: i + right + 1]
        if lows[i] == window_l.min() and np.sum(window_l == lows[i]) == 1:
            swing_low.append(i)
    return swing_high, swing_low


# ══════════════════════════════════════════════════════════════════════
# RECTANGLE DETECTOR
# ══════════════════════════════════════════════════════════════════════
def detect_rectangle(df: pd.DataFrame, as_of: int,
                     swing_high_idx: list[int], swing_low_idx: list[int],
                     lookback: int = 60,
                     tolerance_pct: float = 1.5,
                     min_touches: int = 2,
                     min_span_bars: int = 10,
                     breakout_buffer_pct: float = 0.3,
                     right_bars: int = 3,
                     stop_mode: str = "far") -> dict | None:
    """
    Look for a horizontal trading range in the `lookback` bars strictly before
    `as_of`, and test whether bar `as_of` is breaking out of it.

    NO LOOKAHEAD: only swing points and bars with index < as_of are used to
    define the range. The breakout test uses ONLY df.iloc[as_of] itself — the
    bar being evaluated, which is legitimate exactly the same way
    backtest.py's evaluate_signal() uses the current bar's close.

    Definition:
      - Take swing highs and swing lows within [as_of - lookback, as_of).
      - Cluster swing highs within `tolerance_pct` of their median; same for
        swing lows. Need >= min_touches on EACH side.
      - The two clusters must span >= min_span_bars.
      - The clustered-high level and clustered-low level must not overlap
        (top of the low cluster's tolerance below the bottom of the high
        cluster's tolerance) — otherwise it is not a range, it is noise.
      - Breakout: as_of's Close outside the range by more than
        `breakout_buffer_pct`, in either direction.

    Returns None if no valid rectangle or no breakout at `as_of`.
    """
    lo_bound = max(0, as_of - lookback)
    # ── LOOKAHEAD FIX ──
    # The filter used to be `i < as_of`, which is NOT sufficient. A swing at
    # index i is only identifiable once bar i+right has printed, so a swing at
    # i = as_of-1 required bars as_of..as_of+right-1 — the future relative to
    # this decision. Measured on 5 years of bars, 40% of evaluation points were
    # admitting at least one such unconfirmed swing.
    #
    # It survived an empirical truncation sweep only because median clustering
    # is robust: one stray edge swing rarely moves the median enough to flip a
    # result. That is luck, not correctness — with min_touches=2 and a tight
    # tolerance it can and will flip, and every downstream backtest number
    # would be quietly optimistic.
    #
    # Correct rule: the swing's CONFIRMING bar must also be strictly in the
    # past, i.e. i + right < as_of.
    confirm_lag = right_bars
    highs_in = [i for i in swing_high_idx
                if lo_bound <= i and i + confirm_lag < as_of]
    lows_in  = [i for i in swing_low_idx
                if lo_bound <= i and i + confirm_lag < as_of]
    if len(highs_in) < min_touches or len(lows_in) < min_touches:
        return None

    high_vals = df["High"].values[highs_in]
    low_vals  = df["Low"].values[lows_in]

    # Cluster around the median — the median is robust to one stray touch,
    # which matters because a single outlier swing shouldn't define a level.
    res_level = float(np.median(high_vals))
    sup_level = float(np.median(low_vals))
    if res_level <= sup_level:
        return None

    res_tol = res_level * tolerance_pct / 100
    sup_tol = sup_level * tolerance_pct / 100

    res_touches = int(np.sum(np.abs(high_vals - res_level) <= res_tol))
    sup_touches = int(np.sum(np.abs(low_vals - sup_level) <= sup_tol))
    if res_touches < min_touches or sup_touches < min_touches:
        return None

    # Range must be a real gap, not two clusters whose tolerance bands overlap
    if (sup_level + sup_tol) >= (res_level - res_tol):
        return None

    span = as_of - min(highs_in + lows_in)
    if span < min_span_bars:
        return None

    close = float(df["Close"].iloc[as_of])
    buf_up = res_level * (1 + breakout_buffer_pct / 100)
    buf_dn = sup_level * (1 - breakout_buffer_pct / 100)

    if close > buf_up:
        direction = "Bullish"
    elif close < buf_dn:
        direction = "Bearish"
    else:
        return None   # inside the range — no breakout yet

    height = res_level - sup_level

    # ── STOP CONVENTION ──
    # "far"  — stop at the OPPOSITE boundary. The classic textbook placement:
    #          the pattern is only void if price traverses the whole range back.
    #          Safe from noise, but risk = full pattern height, so with the
    #          target also one height away the trade is ~1:1. That is what the
    #          first backtest measured: 52.5% wins yet NEGATIVE expectancy,
    #          because timeout exits book small gains while stops are full size.
    #
    # "near"  — stop just back inside the BROKEN boundary. Also a standard
    #          convention (a failed breakout is proven by re-entering the
    #          range), and it cuts risk to roughly the buffer distance, lifting
    #          R:R toward 3:1+. The cost is more stop-outs on noise.
    #
    # Which wins is an empirical question, not a preference — that is the
    # single comparison this option exists to answer. It is NOT a knob to
    # sweep alongside tolerance/lookback/min_touches until something turns
    # positive; with that many degrees of freedom a positive result would be
    # fitted to this sample and mean nothing.
    if direction == "Bullish":
        entry = close
        if stop_mode == "near":
            # just under the broken resistance, offset by the same buffer
            stop = res_level * (1 - breakout_buffer_pct / 100)
        else:
            stop = sup_level                   # opposite side of the range
        target = close + height                # height projected from breakout
    else:
        entry = close
        if stop_mode == "near":
            stop = sup_level * (1 + breakout_buffer_pct / 100)
        else:
            stop = res_level
        target = close - height

    # A "near" stop can land on the wrong side of entry if the breakout close
    # ran a long way past the boundary in one bar. Degenerate levels would
    # corrupt the R calculation, so reject rather than silently clamp.
    if direction == "Bullish" and not (stop < entry < target):
        return None
    if direction == "Bearish" and not (target < entry < stop):
        return None

    return {
        "pattern":       "rectangle",
        "direction":     direction,
        "stop_mode":     stop_mode,
        "resistance":    round(res_level, 4),
        "support":       round(sup_level, 4),
        "height":        round(height, 4),
        "span_bars":     span,
        "res_touches":   res_touches,
        "sup_touches":   sup_touches,
        "entry":         round(entry, 4),
        "stop":          round(stop, 4),
        "target":        round(target, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# DRIVER HELPER — precompute swings once per ticker, scan every bar
# ══════════════════════════════════════════════════════════════════════
def scan_rectangles(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Runs detect_rectangle() at every bar with enough history, returning one
    dict per bar where a breakout was detected (empty list if none).

    Swing points are computed ONCE for the whole series here, which is faster
    than recomputing per-bar. That is only safe because detect_rectangle()
    discards any swing whose CONFIRMING bar (i + right) has not yet passed
    relative to as_of.

    An earlier version filtered on `i < as_of` alone and claimed precomputing
    was "equivalent to recomputing fresh at each bar" — it was not. That
    admitted swings which could only have been identified using future bars,
    on 40% of evaluation points. Fixed in detect_rectangle(); the right_bars
    argument below is what makes the correction possible, so it must stay in
    sync with the swing window used here.
    """
    swing_w = cfg.get("swing_window", 3)
    swing_high, swing_low = find_swing_points(df, left=swing_w, right=swing_w)

    hits = []
    start = cfg.get("min_bars_before", 70)
    for i in range(start, len(df)):
        r = detect_rectangle(
            df, i, swing_high, swing_low,
            lookback=cfg.get("lookback", 60),
            tolerance_pct=cfg.get("tolerance_pct", 1.5),
            min_touches=cfg.get("min_touches", 2),
            min_span_bars=cfg.get("min_span_bars", 10),
            breakout_buffer_pct=cfg.get("breakout_buffer_pct", 0.3),
            right_bars=swing_w,
            stop_mode=cfg.get("stop_mode", "far"),
        )
        if r:
            r["bar_index"] = i
            hits.append(r)
    return hits


# ══════════════════════════════════════════════════════════════════════
# FLAG / PENNANT DETECTOR
# ══════════════════════════════════════════════════════════════════════
def detect_flag(df: pd.DataFrame, as_of: int,
                pole_bars: int = 10,
                flag_bars: int = 8,
                pole_min_pct: float = 8.0,
                max_retrace_pct: float = 50.0,
                max_flag_range_pct: float = 6.0,
                breakout_buffer_pct: float = 0.3) -> dict | None:
    """
    A flag is a sharp directional move (the POLE) followed by a tight
    consolidation (the FLAG), breaking out in the SAME direction as the pole.

    WHY THIS IS A DIFFERENT TEST FROM RECTANGLES — not a rerun:

    1. CONTINUATION, NOT DIRECTION-AGNOSTIC. A rectangle sits in a range and we
       trade whichever way it exits, i.e. we predict direction from nothing.
       Here a move has ALREADY happened and we only trade its resumption. That
       is a weaker, more modest claim and generally better evidenced.

    2. THE PAYOFF IS ASYMMETRIC BY CONSTRUCTION. Rectangle target = pattern
       height and stop = pattern height, giving ~1:1 — which is precisely why
       52.3% wins still lost money there. Here target = POLE height while stop
       = the flag's tight low, so R:R lands near 3:1 with nothing tightened.
       The failure mode that killed rectangles does not apply.

    3. NO SWING-POINT DEPENDENCY. Rectangles needed confirmed pivots, which is
       where the lookahead bug lived. A flag is measured directly from two
       adjacent bar windows, so that whole class of bug is structurally absent.

    Windows (both STRICTLY before as_of; only as_of's close tests the breakout):
        pole : [as_of - flag_bars - pole_bars, as_of - flag_bars)
        flag : [as_of - flag_bars, as_of)

    Rejection criteria are what stop this from firing on noise:
      - pole move below pole_min_pct        -> a drift, not a pole
      - flag range wider than max_flag_range_pct -> not a consolidation
      - retracement beyond max_retrace_pct  -> a reversal, not a pause
    """
    flag_start = as_of - flag_bars
    pole_start = flag_start - pole_bars
    if pole_start < 0 or as_of >= len(df):
        return None

    pole = df.iloc[pole_start:flag_start]
    flag = df.iloc[flag_start:as_of]
    if len(pole) < pole_bars or len(flag) < flag_bars:
        return None

    p0 = float(pole["Close"].iloc[0])
    p1 = float(pole["Close"].iloc[-1])
    if p0 <= 0:
        return None
    pole_ret = (p1 - p0) / p0 * 100
    if abs(pole_ret) < pole_min_pct:
        return None
    direction = "Bullish" if pole_ret > 0 else "Bearish"
    pole_height = abs(p1 - p0)
    if pole_height <= 0:
        return None

    fh = float(flag["High"].max())
    fl = float(flag["Low"].min())
    if fl <= 0:
        return None
    flag_range_pct = (fh - fl) / fl * 100
    if flag_range_pct > max_flag_range_pct:
        return None

    # How much of the pole was given back during the consolidation?
    if direction == "Bullish":
        retrace = (p1 - fl) / pole_height * 100
    else:
        retrace = (fh - p1) / pole_height * 100
    if retrace < 0 or retrace > max_retrace_pct:
        return None

    close = float(df["Close"].iloc[as_of])
    if direction == "Bullish":
        if close <= fh * (1 + breakout_buffer_pct / 100):
            return None
        entry, stop, target = close, fl, close + pole_height
        if not (stop < entry < target):
            return None
    else:
        if close >= fl * (1 - breakout_buffer_pct / 100):
            return None
        entry, stop, target = close, fh, close - pole_height
        if not (target < entry < stop):
            return None

    return {
        "pattern":        "flag",
        "direction":      direction,
        "pole_ret_pct":   round(pole_ret, 2),
        "pole_height":    round(pole_height, 4),
        "flag_high":      round(fh, 4),
        "flag_low":       round(fl, 4),
        "flag_range_pct": round(flag_range_pct, 2),
        "retrace_pct":    round(retrace, 1),
        "entry":          round(entry, 4),
        "stop":           round(stop, 4),
        "target":         round(target, 4),
    }


def scan_flags(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Run detect_flag() at every bar with enough history."""
    hits = []
    pole_bars = cfg.get("pole_bars", 10)
    flag_bars = cfg.get("flag_bars", 8)
    start = max(cfg.get("min_bars_before", 70), pole_bars + flag_bars + 1)
    for i in range(start, len(df)):
        r = detect_flag(
            df, i,
            pole_bars=pole_bars,
            flag_bars=flag_bars,
            pole_min_pct=cfg.get("pole_min_pct", 8.0),
            max_retrace_pct=cfg.get("max_retrace_pct", 50.0),
            max_flag_range_pct=cfg.get("max_flag_range_pct", 6.0),
            breakout_buffer_pct=cfg.get("breakout_buffer_pct", 0.3),
        )
        if r:
            r["bar_index"] = i
            hits.append(r)
    return hits
