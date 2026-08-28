"""
pattern_charts.py — draw what the detector actually saw
========================================================
Every detector in patterns.py returns the geometry it used to make its call:
the levels, the fitted lines, the swing counts. This module renders exactly
those numbers — it never re-derives a level for display.

WHY THAT RULE MATTERS
----------------------
A chart that recomputes its own support line is not showing you the detector;
it is showing you a second opinion that happens to sit nearby. When the two
disagree you would be debugging the chart while believing you were debugging
the detector. Every line drawn here comes from the hit dict, so if the picture
looks wrong, the detector IS wrong.

This is the fastest available check on a geometry bug. A detector can pass its
level-ordering assertions and still be finding nonsense — a "resistance" fitted
through three unrelated highs satisfies every numeric guard in the file. Eyes
catch that in one glance and no assertion catches it at all.

Built on Altair, already in requirements.txt and already used by app_patterns.py.
No new dependency.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Semantic colours, consistent across every chart in this module.
C_UP = "#2E7D32"       # bullish / target
C_DOWN = "#C62828"     # bearish / stop
C_RES = "#EF6C00"      # resistance, upper bound
C_SUP = "#1565C0"      # support, lower bound
C_NECK = "#6A1B9A"     # neckline
C_ENTRY = "#37474F"    # entry / breakout bar


def _window(df: pd.DataFrame, hit: dict, pad_before: int, pad_after: int
            ) -> tuple[pd.DataFrame, int, int]:
    """Slice the bars around a detection, returning (frame, lo, break_index)."""
    b = hit["bar_index"]
    span = hit.get("span_bars", 40)
    lo = max(0, b - span - pad_before)
    hi = min(len(df), b + pad_after + 1)
    w = df.iloc[lo:hi].copy().reset_index(drop=True)
    w["bar"] = np.arange(lo, hi)
    return w, lo, b


def _price_layer(w: pd.DataFrame) -> alt.Chart:
    """Candle-style high/low range plus the close line."""
    base = alt.Chart(w)
    wick = base.mark_rule(color="#90A4AE", size=1).encode(
        x=alt.X("bar:Q", title="bar", scale=alt.Scale(zero=False)),
        y=alt.Y("Low:Q", title="price", scale=alt.Scale(zero=False)),
        y2="High:Q",
    )
    line = base.mark_line(color="#263238", size=1.5).encode(
        x="bar:Q", y="Close:Q")
    return wick + line


def _hline(w: pd.DataFrame, value: float, color: str, label: str,
           dash: list[int] | None = None) -> alt.Chart:
    d = pd.DataFrame({"bar": [w["bar"].min(), w["bar"].max()],
                      "y": [value, value], "lbl": [label, label]})
    mark = dict(color=color, size=2)
    if dash:
        mark["strokeDash"] = dash
    return alt.Chart(d).mark_line(**mark).encode(x="bar:Q", y="y:Q",
                                                 tooltip=["lbl", "y"])


def _sloped(w: pd.DataFrame, slope_pct: float, at_break: float, brk: int,
            close: float, color: str, label: str) -> alt.Chart:
    """
    Re-project a fitted line across the window from the value the detector
    recorded AT THE BREAK BAR. slope_pct is % of price per bar, matching how
    detect_trendline() normalises, so this reverses that normalisation rather
    than refitting anything.
    """
    slope = slope_pct / 100 * close
    bars = np.array([w["bar"].min(), w["bar"].max()], dtype=float)
    ys = at_break + slope * (bars - brk)
    d = pd.DataFrame({"bar": bars, "y": ys, "lbl": [label, label]})
    return alt.Chart(d).mark_line(color=color, size=2).encode(
        x="bar:Q", y="y:Q", tooltip=["lbl", "y"])


def _trade_levels(w: pd.DataFrame, hit: dict) -> list[alt.Chart]:
    """Entry, stop and target — identical treatment for every pattern, so a
    chart reads the same way regardless of which detector produced it."""
    brk = hit["bar_index"]
    entry_rule = alt.Chart(pd.DataFrame({"bar": [brk]})).mark_rule(
        color=C_ENTRY, size=2, strokeDash=[2, 2]).encode(x="bar:Q")
    return [
        entry_rule,
        _hline(w, hit["entry"], C_ENTRY, "entry", [6, 3]),
        _hline(w, hit["stop"], C_DOWN, "stop", [6, 3]),
        _hline(w, hit["target"], C_UP, "target", [6, 3]),
    ]


def chart_hit(df: pd.DataFrame, hit: dict, title: str = "",
              pad_before: int = 10, pad_after: int = 25) -> alt.Chart:
    """
    Render one detection with the geometry the detector recorded.

    Works for every registered pattern; the pattern-specific part is only
    which structural lines get drawn.
    """
    w, lo, brk = _window(df, hit, pad_before, pad_after)
    layers = [_price_layer(w)]
    p = hit["pattern"]
    close = float(hit["entry"])

    if p == "rectangle":
        layers += [_hline(w, hit["resistance"], C_RES,
                          f"resistance ({hit['res_touches']} touches)"),
                   _hline(w, hit["support"], C_SUP,
                          f"support ({hit['sup_touches']} touches)")]
    elif p in ("trendline", "triangle"):
        layers += [
            _sloped(w, hit["upper_slope_pct"] if p == "trendline"
                    else _slope_pct(hit, "upper"),
                    hit["upper_at_break"], brk, close, C_RES,
                    f"upper (r2 {hit['upper_r2']})"),
            _sloped(w, hit["lower_slope_pct"] if p == "trendline"
                    else _slope_pct(hit, "lower"),
                    hit["lower_at_break"], brk, close, C_SUP,
                    f"lower (r2 {hit['lower_r2']})"),
        ]
    elif p == "double":
        layers += [_hline(w, (hit["peak_1"] + hit["peak_2"]) / 2, C_RES,
                          f"{hit['subtype']} level"),
                   _hline(w, hit["neckline"], C_NECK,
                          f"neckline (depth {hit['depth_pct']}%)")]
    elif p == "flag":
        layers += [_hline(w, hit["flag_high"], C_RES, "flag high"),
                   _hline(w, hit["flag_low"], C_SUP, "flag low")]

    layers += _trade_levels(w, hit)
    ttl = title or (f"{p} · {hit['direction']} · bar {brk}"
                    + (f" · {hit['subtype']}" if "subtype" in hit else ""))
    return alt.layer(*layers).properties(
        title=ttl, width="container", height=380).interactive()


def _slope_pct(hit: dict, side: str) -> float:
    """
    Triangles record widths rather than slopes. Recover the per-bar slope from
    the two widths and the span — algebra on the detector's own numbers, not a
    refit, so the drawn line is still the line that was tested.
    """
    span = max(1, hit["span_bars"])
    # width shrinks linearly; split the change evenly between the two lines
    # unless the subtype pins one of them flat.
    d_width = (hit["width_now"] - hit["width_start"]) / span
    entry = float(hit["entry"])
    if hit["subtype"] == "ascending":       # flat top, rising bottom
        return 0.0 if side == "upper" else -d_width / entry * 100
    if hit["subtype"] == "descending":      # falling top, flat bottom
        return d_width / entry * 100 if side == "upper" else 0.0
    half = d_width / 2
    return (half if side == "upper" else -half) / entry * 100


def summarize_hits(hits: list[dict]) -> pd.DataFrame:
    """Flat table of detections — the scan-level view before drilling in."""
    if not hits:
        return pd.DataFrame(columns=["bar_index", "pattern", "direction",
                                     "entry", "stop", "target", "rr"])
    rows = []
    for h in hits:
        risk = abs(h["entry"] - h["stop"])
        rows.append({
            "bar_index": h["bar_index"], "pattern": h["pattern"],
            "direction": h["direction"], "subtype": h.get("subtype", ""),
            "entry": h["entry"], "stop": h["stop"], "target": h["target"],
            "rr": round(abs(h["target"] - h["entry"]) / risk, 2) if risk else 0,
            "span_bars": h.get("span_bars", ""),
        })
    return pd.DataFrame(rows)


def render_streamlit(st, df: pd.DataFrame, hits: list[dict],
                     max_charts: int = 5) -> None:
    """
    Drop-in for app_patterns.py: a summary table plus the most recent
    detections drawn. Capped, because thirty charts is not review, it is
    scrolling.
    """
    if not hits:
        st.info("No detections in this window.")
        return
    st.dataframe(summarize_hits(hits), use_container_width=True)
    st.caption(f"Showing the {min(max_charts, len(hits))} most recent of "
               f"{len(hits)} detections.")
    for h in sorted(hits, key=lambda x: -x["bar_index"])[:max_charts]:
        st.altair_chart(chart_hit(df, h), use_container_width=True)
