"""
pattern_backtest.py — Does the rectangle breakout have a real edge?
=====================================================================
Reuses backtest.py's validated walk-forward engine (no-lookahead proven,
next-bar-open fills, stop-first conservative resolution) and swaps in the
rectangle detector from patterns.py in place of the indicator-based signal.
Same methodology as universe_diagnostic.py, so results are directly
comparable to everything already measured on the indicator signal.

WHAT WOULD COUNT AS A REAL RESULT HERE
----------------------------------------
The bar is the same skeptical one applied everywhere else in this project:
  - Profit factor meaningfully above 1.0, not just barely over
  - A large enough sample that the result isn't 20 lucky trades
  - Consistency across MULTIPLE tickers, not one name carrying the average
    (exactly the trap the universe diagnostic caught with META/NVDA/AAPL)
A single positive number on one ticker is not evidence. A distribution of
positive results across a real universe would be.

Run
---
    pip install yfinance pandas numpy scipy ta
    python pattern_backtest.py
    python pattern_backtest.py --tickers TSLA,NVDA,AAPL --years 5
    python pattern_backtest.py --tolerance 1.0 --min-touches 3   # stricter rectangle
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import backtest as bt          # the SAME validated engine — download/compute/
                                # simulate_trade/stats are reused unmodified
import patterns as pt

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, **kw):
        out = ["  ".join(str(h) for h in headers)]
        for r in rows:
            out.append("  ".join(str(c) for c in r))
        return "\n".join(out)


DEFAULT_PATTERN_CFG = dict(
    swing_window        = 3,
    lookback             = 60,
    tolerance_pct        = 1.5,
    min_touches          = 2,
    min_span_bars        = 10,
    breakout_buffer_pct  = 0.3,
    min_bars_before      = 70,
)

# Trade-management params reused as-is from backtest.py's DEFAULTS, since the
# fill/exit logic is identical regardless of what generated the signal.
TRADE_CFG = dict(
    max_hold      = bt.DEFAULTS["max_hold"],
    slippage_bps  = bt.DEFAULTS["slippage_bps"],
    commission    = bt.DEFAULTS["commission"],
    cooldown_bars = bt.DEFAULTS["cooldown_bars"],
    min_rr        = 0.5,   # gate applied below, same spirit as the indicator test
)


def backtest_rectangle_ticker(df: pd.DataFrame, pattern_cfg: dict,
                              trade_cfg: dict) -> list[dict]:
    """
    Mirrors backtest.py's backtest_ticker(), but the "signal" at each bar comes
    from the rectangle detector instead of evaluate_signal(). Cooldown after a
    trade is enforced the same way, so overlapping detections on the same
    breakout don't get double-counted.
    """
    hits = pt.scan_rectangles(df, pattern_cfg)
    hits_by_bar = {h["bar_index"]: h for h in hits}

    trades = []
    i = 0
    n = len(df)
    while i < n - 1:
        h = hits_by_bar.get(i)
        if h:
            risk = abs(h["entry"] - h["stop"])
            reward = abs(h["target"] - h["entry"])
            rr = round(reward / risk, 2) if risk > 0 else 0
            if rr >= trade_cfg["min_rr"]:
                trade = {"trend": h["direction"], "stop": h["stop"],
                         "target": h["target"], "rr": rr}
                res = bt.simulate_trade(df, i, trade, trade_cfg)
                if res.get("filled"):
                    res["pattern_meta"] = h
                    trades.append(res)
                    i += trade_cfg["cooldown_bars"] + 1
                    continue
        i += 1
    return trades


def run(cfg: dict) -> None:
    print("=" * 78)
    print("RECTANGLE PATTERN BACKTEST")
    print("=" * 78)
    print(f"Tickers    : {', '.join(cfg['tickers'])}")
    print(f"History    : {cfg['years']} years daily")
    print(f"Pattern    : lookback {cfg['pattern']['lookback']}, "
          f"tolerance {cfg['pattern']['tolerance_pct']}%, "
          f"min touches {cfg['pattern']['min_touches']}, "
          f"min span {cfg['pattern']['min_span_bars']} bars, "
          f"breakout buffer {cfg['pattern']['breakout_buffer_pct']}%")
    print(f"Trade mgmt : max hold {cfg['trade']['max_hold']} bars, "
          f"min R:R {cfg['trade']['min_rr']}, "
          f"slippage {cfg['trade']['slippage_bps']}bps")
    print("Engine     : backtest.py (no-lookahead proven, next-bar-open fills, "
          "stop-first conservative)")
    print("=" * 78)

    per_ticker_rows = []
    all_trades = []

    for tk in cfg["tickers"]:
        raw = bt.download(tk, cfg["years"])
        if raw is None:
            print(f"\n{tk}: no data — skipped")
            continue
        df = bt.compute(raw)
        if len(df) < bt.MIN_BARS_AFTER:
            print(f"\n{tk}: insufficient history after warm-up — skipped")
            continue
        tail = raw.tail(len(df)).reset_index(drop=True)
        for col in ("Open", "Date"):
            if col in tail.columns:
                df[col] = tail[col].values

        trades = backtest_rectangle_ticker(df, cfg["pattern"], cfg["trade"])
        s = bt.stats(trades)
        all_trades.extend(trades)

        if s.get("trades", 0) == 0:
            per_ticker_rows.append([tk, 0, "—", "—", "—", "—", "—"])
        else:
            per_ticker_rows.append([
                tk, s["trades"], f"{s['win_rate']:.0f}%",
                f"{s['avg_r']:+.3f}", f"{s['total_r']:+.1f}",
                f"{s['pf']:.2f}", f"{s['avg_hold']:.0f}",
            ])

    print("\nPER-TICKER RESULTS")
    print(tabulate(per_ticker_rows,
                   headers=["Ticker", "Trades", "Win%", "Avg R", "Total R",
                            "PF", "Hold"],
                   tablefmt="simple"))

    agg = bt.stats(all_trades)
    print("\n" + "=" * 78)
    print("AGGREGATE (all tickers combined)")
    print("=" * 78)
    if agg.get("trades", 0) == 0:
        print("No trades generated. The detector found no qualifying breakouts —")
        print("try --lookback, --tolerance or --min-touches to widen the search,")
        print("or accept that rectangles are rare on this universe/window.")
        return
    print(f"  Total trades   : {agg['trades']}")
    print(f"  Win rate       : {agg['win_rate']:.1f}%")
    print(f"  Expectancy     : {agg['avg_r']:+.3f} R per trade")
    print(f"  Total return   : {agg['total_r']:+.1f} R")
    print(f"  Profit factor  : {agg['pf']:.2f}")
    print(f"  Max drawdown   : {agg['max_dd']:+.1f} R")
    print(f"  Avg hold       : {agg['avg_hold']:.1f} bars")

    n_positive = sum(1 for row in per_ticker_rows
                    if row[1] and row[1] != 0 and str(row[3]).startswith("+"))
    n_with_trades = sum(1 for row in per_ticker_rows if row[1])
    print(f"\n  Tickers with a positive average R: {n_positive}/{n_with_trades}")

    print("\nINTERPRETATION")
    exp, pf = agg["avg_r"], agg["pf"]
    if agg["trades"] < 100:
        print(f"  🟡 Only {agg['trades']} trades total — too small a sample to")
        print("     trust regardless of the sign. Widen --years or the ticker list.")
    elif exp > 0 and pf > 1.15 and n_with_trades and n_positive / n_with_trades >= 0.6:
        print(f"  ✅ Positive expectancy ({exp:+.3f} R/trade, PF {pf:.2f}) AND")
        print(f"     consistent across {n_positive}/{n_with_trades} tickers — not")
        print("     one name carrying the average. This clears the bar for")
        print("     'worth building further' rather than 'looks fine on paper'.")
    elif exp > 0:
        print(f"  🟡 Marginally positive ({exp:+.3f} R/trade, PF {pf:.2f}) but not")
        print(f"     both strong AND broad ({n_positive}/{n_with_trades} tickers")
        print("     positive). Treat as inconclusive, not confirmed.")
    else:
        print(f"  🔴 Negative expectancy ({exp:+.3f} R/trade). Same conclusion as")
        print("     the indicator signal: no edge detected in this pattern, on")
        print("     this universe, over this window.")
    print("\n  Reminder: past performance is not predictive. This is share-based;")
    print("  options add theta/spread on top, as measured for the indicator signal.")


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Rectangle pattern backtest")
    p.add_argument("--tickers", type=str,
                   default="TSLA,NVDA,AAPL,MSFT,AMZN,META,ROKU,AMD,GOOGL,NFLX,"
                           "INTC,BABA,CSCO,QQQ")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--swing-window", type=int, default=DEFAULT_PATTERN_CFG["swing_window"])
    p.add_argument("--lookback", type=int, default=DEFAULT_PATTERN_CFG["lookback"])
    p.add_argument("--tolerance", type=float, default=DEFAULT_PATTERN_CFG["tolerance_pct"])
    p.add_argument("--min-touches", type=int, default=DEFAULT_PATTERN_CFG["min_touches"])
    p.add_argument("--min-span", type=int, default=DEFAULT_PATTERN_CFG["min_span_bars"])
    p.add_argument("--breakout-buffer", type=float,
                   default=DEFAULT_PATTERN_CFG["breakout_buffer_pct"])
    p.add_argument("--max-hold", type=int, default=TRADE_CFG["max_hold"])
    p.add_argument("--min-rr", type=float, default=TRADE_CFG["min_rr"])
    p.add_argument("--stop-mode", choices=["far", "near"], default="far",
                   help="far = stop at opposite boundary (textbook, ~1:1 R:R). "
                        "near = stop just inside the broken boundary (~3:1+, "
                        "but stops out more often).")
    p.add_argument("--compare-stops", action="store_true",
                   help="Run BOTH stop conventions and print them side by side. "
                        "This is the one structural comparison worth making.")
    a = p.parse_args()

    return dict(
        tickers=[t.strip().upper() for t in a.tickers.split(",") if t.strip()],
        years=a.years,
        pattern=dict(
            swing_window=a.swing_window, lookback=a.lookback,
            tolerance_pct=a.tolerance, min_touches=a.min_touches,
            min_span_bars=a.min_span, breakout_buffer_pct=a.breakout_buffer,
            min_bars_before=DEFAULT_PATTERN_CFG["min_bars_before"],
            stop_mode=a.stop_mode,
        ),
        compare_stops=a.compare_stops,
        trade=dict(
            max_hold=a.max_hold, slippage_bps=TRADE_CFG["slippage_bps"],
            commission=TRADE_CFG["commission"],
            cooldown_bars=TRADE_CFG["cooldown_bars"], min_rr=a.min_rr,
        ),
    )


def compare_stop_modes(cfg: dict) -> None:
    """
    Run both stop conventions on identical detections and compare.

    This is a MECHANISM test, not a parameter sweep. The first backtest showed
    52.5% wins with negative expectancy — a payoff-structure problem, not
    obviously a prediction problem. The far stop risks a full pattern height
    against a one-height target (~1:1); the near stop risks only the buffer
    (~3:1+) but is hit by noise more often. Whether the better payoff survives
    the lower win rate is exactly what this measures.

    Read BREADTH (tickers positive) at least as hard as expectancy. A real
    edge appears in most names; a thin edge in a few is what noise looks like.
    """
    rows = []
    for mode in ("far", "near"):
        pcfg = dict(cfg["pattern"], stop_mode=mode)
        all_trades = []
        for tk in cfg["tickers"]:
            raw = bt.download(tk, cfg["years"])
            if raw is None:
                continue
            df = bt.compute(raw)
            if len(df) < bt.MIN_BARS_AFTER:
                continue
            tail = raw.tail(len(df)).reset_index(drop=True)
            for col in ("Open", "Date"):
                if col in tail.columns:
                    df[col] = tail[col].values
            trades = backtest_rectangle_ticker(df, pcfg, cfg["trade"])
            for t in trades:
                t["ticker"] = tk
            all_trades += trades

        s = bt.stats(all_trades)
        if s.get("trades", 0) == 0:
            rows.append([mode, 0, "—", "—", "—", "—", "—"])
            continue
        npos = sum(1 for tk in cfg["tickers"]
                   if bt.stats([t for t in all_trades
                                if t.get("ticker") == tk]).get("avg_r", 0) > 0)
        nwith = sum(1 for tk in cfg["tickers"]
                    if bt.stats([t for t in all_trades
                                 if t.get("ticker") == tk]).get("trades", 0) > 0)
        rows.append([mode, s["trades"], f"{s['win_rate']:.1f}%",
                     f"{s['avg_r']:+.3f}", f"{s['total_r']:+.1f}",
                     f"{s['pf']:.2f}", f"{npos}/{nwith}"])

    print("\n" + "=" * 78)
    print("STOP CONVENTION COMPARISON")
    print("=" * 78)
    print(tabulate(rows, headers=["Stop", "Trades", "Win%", "Avg R",
                                  "Total R", "PF", "Tickers +"],
                   tablefmt="simple"))
    print("""
  far  = stop at the opposite boundary  (risk = full pattern height, ~1:1)
  near = stop just inside broken boundary (risk = buffer, ~3:1+, noisier)

  What would count as a real finding: 'near' positive on expectancy AND
  positive on MOST tickers. If expectancy improves but breadth stays under
  half, that is a payoff artefact on a signal that still does not predict —
  and rectangles are tested. Do not proceed to sweep tolerance/lookback/
  min-touches looking for a better number; with that many knobs across 14
  tickers you will always find one, and it will not survive out of sample.
""")


if __name__ == "__main__":
    _cfg = parse_args()
    if _cfg.pop("compare_stops", False):
        compare_stop_modes(_cfg)
    else:
        run(_cfg)
