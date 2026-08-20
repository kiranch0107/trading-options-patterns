"""
Trading Copilot ELITE — Historical Backtester
=============================================
Validates the EXACT signal logic from app.py against REAL historical daily
data for your watchlist. This is the "does the edge survive on real prices?"
test — distinct from the synthetic Monte-Carlo sweep used for parameter tuning.

What it does
------------
  • Downloads N years of real OHLCV per ticker (yfinance)
  • Computes the same indicators as app.py (ta library, Wilder smoothing)
  • Discards the indicator warm-up head (first 100 converged bars) exactly
    like app.py's compute()
  • Walks forward bar-by-bar with NO lookahead: at each bar it evaluates the
    signal using ONLY data up to and including that bar
  • On a signal, it enters at the NEXT bar's open (realistic — you can't fill
    on the close that generated the signal), then walks forward checking
    whether stop or target is hit first (stop checked first on ambiguous bars
    = conservative), with a max-hold timeout that marks to market
  • Applies optional slippage + commission per trade
  • Reports per-ticker and aggregate: trades, win rate, expectancy (avg R),
    total R, profit factor, max drawdown, avg hold — plus a monthly R curve

This mirrors app.py's analyze() base conditions, strength rule, ADX filter,
and the exact entry/stop/target math (single-reference, ATR-based caps,
structural-stop validation, relative zero-risk gate, MIN_RR gate).

Filters intentionally SIMPLIFIED for a clean historical test:
  • ADX filter: applied (same threshold)
  • Weekly alignment / earnings blackout / SPY regime: these need external
    calendars or a second data feed and would add lookahead/complexity, so
    they are OFF by default here. Turn USE_REGIME on to approximate the SPY
    macro filter using SPY's own 200-SMA (computed from the same download).

Run
---
  pip install yfinance pandas ta tabulate numpy
  python backtest.py

  # options:
  python backtest.py --years 5 --tickers TSLA,NVDA,AAPL
  python backtest.py --atr-stop 1.0 --atr-tgt 3.0 --adx-min 25
  python backtest.py --slippage-bps 5 --commission 0.65 --use-regime
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing yfinance. Run: pip install yfinance pandas ta tabulate numpy")
try:
    import ta
except ImportError:
    raise SystemExit("Missing ta. Run: pip install ta")
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, **kw):   # minimal fallback
        out = ["  ".join(str(h) for h in headers)]
        for r in rows:
            out.append("  ".join(str(c) for c in r))
        return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════
# CONFIG — defaults mirror app.py's sidebar defaults
# ══════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    tickers       = ["TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "SPY"],
    years         = 5,
    adx_min       = 25,
    atr_stop_mult = 1.0,      # app.py default
    atr_tgt_mult  = 3.0,      # app.py default (updated from 2.5)
    min_rr        = 0.5,
    volume_mult   = 1.0,      # for the "Strong" strength tag only
    max_hold      = 20,       # bars to hold before timeout mark-to-market
    slippage_bps  = 2.0,      # per side, in basis points of price
    commission    = 0.0,      # $ per trade (round trip), for share trades
    use_regime    = False,    # approximate SPY 200-SMA macro filter
    cooldown_bars = 3,        # bars to wait after a trade before re-entering
)

WARMUP_BARS       = 100       # matches app.py INDICATOR_WARMUP_BARS
MIN_BARS_AFTER    = 40        # matches app.py MIN_BARS_AFTER_WARMUP


# ══════════════════════════════════════════════════════════════════════
# INDICATORS — identical to app.py compute()
# ══════════════════════════════════════════════════════════════════════
def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["EMA20"]     = ta.trend.ema_indicator(c, window=20)
    df["EMA50"]     = ta.trend.ema_indicator(c, window=50)
    macd            = ta.trend.MACD(c)
    df["MACD"]      = macd.macd()
    df["Signal"]    = macd.macd_signal()
    df["RSI"]       = ta.momentum.rsi(c, window=14)
    df["ATR"]       = ta.volatility.average_true_range(h, l, c, window=14)
    df["ADX"]       = ta.trend.adx(h, l, c, window=14)
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    df = df.dropna(subset=["EMA20", "EMA50", "MACD", "Signal", "RSI",
                           "ATR", "ADX", "VOL_AVG20"])
    # discard warm-up head exactly like app.py
    if len(df) > WARMUP_BARS + MIN_BARS_AFTER:
        df = df.iloc[WARMUP_BARS:]
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# SIGNAL — mirrors app.py analyze() base conditions + ADX + levels
# ══════════════════════════════════════════════════════════════════════
def evaluate_signal(df: pd.DataFrame, i: int, cfg: dict,
                    regime: str | None = None) -> dict | None:
    """
    Evaluate the signal at bar i using ONLY rows 0..i (no lookahead).
    Returns a trade dict {trend, stop, target, rr, atr} or None.
    """
    row = df.iloc[i]
    price  = float(row["Close"])
    ema20  = float(row["EMA20"]); ema50 = float(row["EMA50"])
    macd   = float(row["MACD"]);  signal = float(row["Signal"])
    rsi    = float(row["RSI"]);   adx_v  = float(row["ADX"])
    atr    = float(row["ATR"])
    volume = float(row["Volume"]); vol_avg = float(row["VOL_AVG20"])

    vol_soft_ok = volume >= vol_avg * 0.70

    # ── Base conditions (identical to app.py) ──
    if price > ema20 > ema50 and macd > signal and 30 < rsi < 75 and vol_soft_ok:
        trend = "Bullish"
    elif price < ema20 < ema50 and macd < signal and 25 < rsi < 70 and vol_soft_ok:
        trend = "Bearish"
    else:
        return None

    # ── ADX enhancement filter ──
    if adx_v < cfg["adx_min"]:
        return None

    # ── Optional SPY macro regime approximation ──
    if regime is not None:
        if trend == "Bullish" and regime == "Bear":
            return None
        if trend == "Bearish" and regime == "Bull":
            return None

    # ── Levels (identical single-reference logic to app.py) ──
    window = df.iloc[max(0, i - 19): i + 1]     # up to 20 bars, no lookahead
    swing_low_10  = float(df["Low"].iloc[max(0, i - 9): i + 1].min())
    swing_high_10 = float(df["High"].iloc[max(0, i - 9): i + 1].max())

    if trend == "Bullish":
        entry = round(price, 2)
        atr_stop = price - (atr * cfg["atr_stop_mult"])
        structural_stop = swing_low_10 - (atr * 0.10)
        stop = max(structural_stop, atr_stop) if structural_stop < price else atr_stop
        stop = round(min(stop, entry - 0.01), 2)

        raw_target = price + (atr * cfg["atr_tgt_mult"])
        resistance = float(window["High"].max())
        if resistance >= entry + (atr * 1.0):
            target = round(min(raw_target, resistance * 0.995), 2)
        else:
            target = round(raw_target, 2)
        target = round(max(target, entry + 0.02), 2)
    else:
        entry = round(price, 2)
        atr_stop = price + (atr * cfg["atr_stop_mult"])
        structural_stop = swing_high_10 + (atr * 0.10)
        stop = min(structural_stop, atr_stop) if structural_stop > price else atr_stop
        stop = round(max(stop, entry + 0.01), 2)

        raw_target = price - (atr * cfg["atr_tgt_mult"])
        support = float(window["Low"].min())
        if support <= entry - (atr * 1.0):
            target = round(max(raw_target, support * 1.005), 2)
        else:
            target = round(raw_target, 2)
        target = round(min(target, entry - 0.02), 2)

    # ── Risk / RR gates (identical to app.py) ──
    risk = abs(entry - stop)
    min_risk = max(0.05, price * 0.003)
    if risk < min_risk:
        return None
    rr = round(abs(target - entry) / risk, 2)
    if rr < cfg["min_rr"]:
        return None

    return {"trend": trend, "entry": entry, "stop": stop,
            "target": target, "rr": rr, "atr": atr}


# ══════════════════════════════════════════════════════════════════════
# TRADE SIMULATION — enter next open, stop-first fills, timeout m2m
# ══════════════════════════════════════════════════════════════════════
def simulate_trade(df: pd.DataFrame, signal_i: int, trade: dict,
                   cfg: dict) -> dict:
    """
    Enter at the OPEN of bar signal_i+1 (no same-bar fill). Walk forward until
    stop or target is hit (stop checked first on ambiguous bars = conservative),
    or max_hold is reached (mark to market at that close).
    Returns realised R multiple net of slippage/commission.
    """
    n = len(df)
    entry_i = signal_i + 1
    if entry_i >= n:
        return {"filled": False}

    # Realistic entry: next bar open, adjusted for slippage
    raw_entry = float(df["Open"].iloc[entry_i]) if "Open" in df.columns \
        else float(df["Close"].iloc[signal_i])
    slip = raw_entry * cfg["slippage_bps"] / 10_000.0
    trend = trade["trend"]
    entry = raw_entry + slip if trend == "Bullish" else raw_entry - slip

    stop, target = trade["stop"], trade["target"]
    risk = abs(entry - stop)
    if risk <= 0:
        return {"filled": False}

    exit_i = None; exit_px = None; outcome = None
    for j in range(entry_i, min(entry_i + cfg["max_hold"], n)):
        hi = float(df["High"].iloc[j]); lo = float(df["Low"].iloc[j])
        if trend == "Bullish":
            if lo <= stop:                       # stop first (conservative)
                exit_px, outcome, exit_i = stop, "loss", j; break
            if hi >= target:
                exit_px, outcome, exit_i = target, "win", j; break
        else:
            if hi >= stop:
                exit_px, outcome, exit_i = stop, "loss", j; break
            if lo <= target:
                exit_px, outcome, exit_i = target, "win", j; break

    if exit_i is None:                           # timeout — mark to market
        exit_i = min(entry_i + cfg["max_hold"] - 1, n - 1)
        exit_px = float(df["Close"].iloc[exit_i])
        outcome = "timeout"

    # Exit slippage (opposite direction)
    exit_slip = exit_px * cfg["slippage_bps"] / 10_000.0
    exit_fill = exit_px - exit_slip if trend == "Bullish" else exit_px + exit_slip

    pnl = (exit_fill - entry) if trend == "Bullish" else (entry - exit_fill)
    r_multiple = pnl / risk

    # Commission expressed in R (approx: commission / dollar-risk-per-share
    # is negligible for share trades; included for completeness)
    if cfg["commission"] > 0:
        r_multiple -= cfg["commission"] / (risk * 100)   # ~1 contract-ish scale

    return {
        "filled": True, "trend": trend, "outcome": outcome,
        "r": r_multiple, "rr_planned": trade["rr"],
        "hold": exit_i - entry_i,
        "entry_date": df["Date"].iloc[entry_i] if "Date" in df.columns else entry_i,
    }


# ══════════════════════════════════════════════════════════════════════
# BACKTEST DRIVER
# ══════════════════════════════════════════════════════════════════════
def backtest_ticker(df: pd.DataFrame, cfg: dict,
                    regime_series: pd.Series | None = None) -> list[dict]:
    trades = []
    i = 0
    n = len(df)
    while i < n - 1:
        regime = None
        if regime_series is not None and i < len(regime_series):
            regime = regime_series.iloc[i]
        sig = evaluate_signal(df, i, cfg, regime=regime)
        if sig:
            res = simulate_trade(df, i, sig, cfg)
            if res.get("filled"):
                trades.append(res)
                i += cfg["cooldown_bars"] + 1     # cooldown after a trade
                continue
        i += 1
    return trades


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    r = np.array([t["r"] for t in trades])
    wins = r[r > 0]; losses = r[r < 0]
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())
    gp = wins.sum(); gl = abs(losses.sum())
    return {
        "trades":     len(r),
        "win_rate":   len(wins) / len(r) * 100,
        "avg_r":      float(r.mean()),
        "total_r":    float(r.sum()),
        "pf":         (gp / gl) if gl > 0 else float("inf"),
        "max_dd":     max_dd,
        "avg_hold":   float(np.mean([t["hold"] for t in trades])),
        "best":       float(r.max()),
        "worst":      float(r.min()),
    }


def build_regime_series(spy_df: pd.DataFrame) -> pd.Series:
    """Approximate app.py's SPY macro filter: price vs its own 200-SMA."""
    sma200 = spy_df["Close"].rolling(200, min_periods=50).mean()
    out = pd.Series("Neutral", index=spy_df.index)
    out[spy_df["Close"] > sma200] = "Bull"
    out[spy_df["Close"] < sma200] = "Bear"
    return out.reset_index(drop=True)


def download(ticker: str, years: int) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=f"{years}y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        # flatten possible multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        keep = {"Date": "Date", "Open": "Open", "High": "High",
                "Low": "Low", "Close": "Close", "Volume": "Volume"}
        df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
        return df
    except Exception as e:
        print(f"  ! download failed for {ticker}: {e}")
        return None


def run(cfg: dict) -> None:
    print("=" * 78)
    print("TRADING COPILOT ELITE — HISTORICAL BACKTEST (real data)")
    print("=" * 78)
    print(f"Tickers      : {', '.join(cfg['tickers'])}")
    print(f"History      : {cfg['years']} years daily")
    print(f"ADX min      : {cfg['adx_min']}   ATR stop×: {cfg['atr_stop_mult']}   "
          f"ATR tgt×: {cfg['atr_tgt_mult']}   Min R:R: {cfg['min_rr']}")
    print(f"Max hold     : {cfg['max_hold']} bars   Slippage: {cfg['slippage_bps']}bps/side   "
          f"Regime filter: {'ON (SPY 200-SMA)' if cfg['use_regime'] else 'OFF'}")
    print("=" * 78)

    # Optional regime series from SPY
    regime_by_date = None
    if cfg["use_regime"]:
        spy_raw = download("SPY", cfg["years"])
        if spy_raw is not None:
            spy_c = compute(spy_raw)
            spy_c = spy_c.merge(spy_raw[["Date"]], left_index=True,
                                right_index=True, how="left")
            reg = build_regime_series(spy_raw)
            regime_by_date = dict(zip(spy_raw["Date"], reg))

    per_ticker_rows = []
    all_trades = []

    for tk in cfg["tickers"]:
        raw = download(tk, cfg["years"])
        if raw is None:
            print(f"\n{tk}: no data — skipped")
            continue
        df = compute(raw)
        # attach dates back for reporting/regime mapping
        df = df.copy()
        if "Date" in raw.columns:
            df = df.merge(raw[["Date", "Open"]], on=None, how="left") \
                if False else df   # (Open already dropped; re-attach below)
        # re-attach Open + Date aligned by tail length
        tail = raw.tail(len(df)).reset_index(drop=True)
        for col in ["Open", "Date"]:
            if col in tail.columns:
                df[col] = tail[col].values

        # per-bar regime lookup
        reg_series = None
        if regime_by_date is not None and "Date" in df.columns:
            reg_series = df["Date"].map(regime_by_date).fillna("Neutral").reset_index(drop=True)

        trades = backtest_ticker(df, cfg, regime_series=reg_series)
        s = stats(trades)
        all_trades.extend(trades)

        if s["trades"] == 0:
            per_ticker_rows.append([tk, 0, "—", "—", "—", "—", "—", "—"])
        else:
            per_ticker_rows.append([
                tk, s["trades"], f"{s['win_rate']:.0f}%",
                f"{s['avg_r']:+.3f}", f"{s['total_r']:+.1f}",
                f"{s['pf']:.2f}", f"{s['max_dd']:+.1f}", f"{s['avg_hold']:.0f}",
            ])

    print("\nPER-TICKER RESULTS")
    print(tabulate(
        per_ticker_rows,
        headers=["Ticker", "Trades", "Win%", "Avg R", "Total R",
                 "PF", "MaxDD", "Hold"],
        tablefmt="simple",
    ))

    agg = stats(all_trades)
    print("\n" + "=" * 78)
    print("AGGREGATE (all tickers combined)")
    print("=" * 78)
    if agg["trades"] == 0:
        print("No trades generated. Try --adx-min 20 or a longer --years window.")
        return
    print(f"  Total trades   : {agg['trades']}")
    print(f"  Win rate       : {agg['win_rate']:.1f}%")
    print(f"  Expectancy     : {agg['avg_r']:+.3f} R per trade")
    print(f"  Total return   : {agg['total_r']:+.1f} R")
    print(f"  Profit factor  : {agg['pf']:.2f}")
    print(f"  Max drawdown   : {agg['max_dd']:+.1f} R")
    print(f"  Avg hold       : {agg['avg_hold']:.1f} bars")
    print(f"  Best / worst   : {agg['best']:+.2f} R / {agg['worst']:+.2f} R")

    # Interpretation
    print("\nINTERPRETATION")
    exp = agg["avg_r"]; pf = agg["pf"]
    if exp > 0 and pf > 1.1:
        print(f"  ✅ Positive expectancy ({exp:+.3f} R/trade, PF {pf:.2f}) on REAL data.")
        print("     The edge that showed up in the Monte-Carlo sweep survives on")
        print("     actual historical prices. This is the result you want to see.")
    elif exp > 0:
        print(f"  🟡 Marginally positive ({exp:+.3f} R/trade, PF {pf:.2f}). The edge is")
        print("     real but thin — transaction costs and slippage matter a lot here.")
    else:
        print(f"  🔴 Negative expectancy ({exp:+.3f} R/trade) on real data. The synthetic")
        print("     edge did NOT survive. Do not trade this as-is — investigate which")
        print("     tickers/periods dragged it down before risking capital.")
    print("\n  Reminder: past performance is not predictive. Stops are not guaranteed")
    print("  (overnight gaps). Options add theta/slippage this share-based test omits.")


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Trading Copilot historical backtest")
    p.add_argument("--tickers", type=str, default=",".join(DEFAULTS["tickers"]))
    p.add_argument("--years", type=int, default=DEFAULTS["years"])
    p.add_argument("--adx-min", type=float, default=DEFAULTS["adx_min"])
    p.add_argument("--atr-stop", type=float, default=DEFAULTS["atr_stop_mult"])
    p.add_argument("--atr-tgt", type=float, default=DEFAULTS["atr_tgt_mult"])
    p.add_argument("--min-rr", type=float, default=DEFAULTS["min_rr"])
    p.add_argument("--max-hold", type=int, default=DEFAULTS["max_hold"])
    p.add_argument("--slippage-bps", type=float, default=DEFAULTS["slippage_bps"])
    p.add_argument("--commission", type=float, default=DEFAULTS["commission"])
    p.add_argument("--cooldown", type=int, default=DEFAULTS["cooldown_bars"])
    p.add_argument("--use-regime", action="store_true", default=DEFAULTS["use_regime"])
    a = p.parse_args()
    return dict(
        tickers       = [t.strip().upper() for t in a.tickers.split(",") if t.strip()],
        years         = a.years,
        adx_min       = a.adx_min,
        atr_stop_mult = a.atr_stop,
        atr_tgt_mult  = a.atr_tgt,
        min_rr        = a.min_rr,
        volume_mult   = DEFAULTS["volume_mult"],
        max_hold      = a.max_hold,
        slippage_bps  = a.slippage_bps,
        commission    = a.commission,
        use_regime    = a.use_regime,
        cooldown_bars = a.cooldown,
    )


if __name__ == "__main__":
    run(parse_args())
