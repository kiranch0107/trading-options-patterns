"""
research_ledger.py — pre-registration and multiple-comparisons accounting
==========================================================================
Five independent tests on the indicator signal came back no-edge, and that
finding was accepted rather than tuned around. Pattern families are now test
six, seven and eight.

THE PROBLEM THIS SOLVES
------------------------
With enough pattern families, one clears any fixed threshold by chance. Test
twenty independent patterns at "profit factor > 1.3" and roughly one will pass
on noise alone. Nothing about that run looks suspicious from inside it: the
engine is correct, the detector is honest, the number is real. What is missing
is the denominator — the other nineteen.

Two mechanisms, both borrowed from how oos_validate.lock.json already works in
the trading-copilot repo:

  1. PRE-REGISTRATION. Criteria and detector params are frozen to a lock file
     BEFORE the backtest runs. Afterwards, the run is checked against what was
     frozen. Loosening a threshold once results are visible is the single
     easiest way to manufacture an edge, and it never feels like cheating in
     the moment — it feels like refining the question.

  2. MULTIPLE-COMPARISONS CORRECTION. Every completed test is counted. The
     bar for significance rises with the count via a Šidák correction, so the
     sixth family must clear a materially higher hurdle than the first did.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
It does not stop you running a test, and it does not veto a result. It records
what was asked and what the honest threshold was at that moment. A positive
finding that clears a corrected bar, on params frozen in advance, is worth
believing. The same number without either is worth nothing, and the only
difference between the two is bookkeeping nobody does after the fact.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER_PATH = Path(__file__).with_name("research_ledger.json")

# Uncorrected per-test false-positive rate. Everything below flows from this.
ALPHA = 0.05

# The criteria a pattern family must clear to count as a real finding. These
# mirror oos_validate.lock.json's shape so results stay comparable across both
# repos rather than reflecting two different definitions of "worked".
BASE_CRITERIA = {
    "min_trades": 60,
    "min_profit_factor": 1.3,
    "min_expectancy_r": 0.2,
    "require_ci_above_zero": True,
    "min_tickers_positive_frac": 0.5,
}


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def load() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {"tests": [], "created": datetime.now(timezone.utc).isoformat()}


def save(led: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(led, indent=2) + "\n")


def completed_count(led: dict) -> int:
    return sum(1 for t in led["tests"] if t.get("status") == "complete")


def corrected_alpha(n_tests: int) -> float:
    """
    Šidák: 1 - (1 - alpha)^(1/n). Less conservative than Bonferroni while
    still controlling the family-wise error rate, which matters because
    over-correcting is its own failure — it makes a real edge undetectable
    and turns the whole exercise into theatre.
    """
    n = max(1, n_tests)
    return 1 - (1 - ALPHA) ** (1 / n)


def register(family: str, pattern_cfg: dict, trade_cfg: dict,
             tickers: list[str], years: int, note: str = "") -> dict:
    """
    Freeze a test BEFORE running it. Returns the registration record.

    Re-registering the same family with different params creates a NEW entry
    rather than overwriting — that is the point. Three registrations of
    "trendline" with three parameter sets is three tests, and the correction
    must know that. Silently overwriting would hide a parameter sweep behind
    a single-test label, which is the exact failure this file exists to catch.
    """
    led = load()
    prior = completed_count(led)
    rec = {
        "family": family,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": "registered",
        "config_hash": _hash({"p": pattern_cfg, "t": trade_cfg,
                              "tk": sorted(tickers), "y": years}),
        "frozen": {
            "pattern_cfg": pattern_cfg,
            "trade_cfg": trade_cfg,
            "tickers": sorted(tickers),
            "years": years,
        },
        "criteria": dict(BASE_CRITERIA),
        "tests_completed_before_this": prior,
        "corrected_alpha": round(corrected_alpha(prior + 1), 5),
        "note": note,
    }
    led["tests"].append(rec)
    save(led)
    return rec


def complete(config_hash: str, result: dict) -> dict:
    """
    Attach results to a registration and evaluate against what was frozen.

    The verdict is computed here, from the frozen criteria, rather than being
    passed in — so the thresholds a run is judged against are necessarily the
    ones recorded before it ran.
    """
    led = load()
    rec = next((t for t in led["tests"]
                if t["config_hash"] == config_hash
                and t["status"] == "registered"), None)
    if rec is None:
        raise ValueError(
            f"no open registration with hash {config_hash}. Register the test "
            f"before running it — that ordering is the whole mechanism."
        )
    c = rec["criteria"]
    checks = {
        "trades": (result.get("trades", 0) >= c["min_trades"],
                   f"{result.get('trades', 0)} vs {c['min_trades']} min"),
        "profit_factor": (result.get("profit_factor", 0) >= c["min_profit_factor"],
                          f"{result.get('profit_factor', 0)} vs {c['min_profit_factor']} min"),
        "expectancy_r": (result.get("expectancy_r", 0) >= c["min_expectancy_r"],
                         f"{result.get('expectancy_r', 0)} vs {c['min_expectancy_r']} min"),
        "tickers_positive": (
            result.get("tickers_positive_frac", 0) >= c["min_tickers_positive_frac"],
            f"{result.get('tickers_positive_frac', 0)} vs "
            f"{c['min_tickers_positive_frac']} min"),
    }
    if c["require_ci_above_zero"]:
        checks["ci_above_zero"] = (bool(result.get("ci_low", -1) > 0),
                                   f"CI low {result.get('ci_low')}")

    rec["status"] = "complete"
    rec["completed_at"] = datetime.now(timezone.utc).isoformat()
    rec["result"] = result
    rec["checks"] = {k: {"pass": v[0], "detail": v[1]} for k, v in checks.items()}
    rec["verdict"] = "EDGE" if all(v[0] for v in checks.values()) else "NO EDGE"
    save(led)
    return rec


def report() -> str:
    led = load()
    n = completed_count(led)
    out = ["=" * 72,
           "RESEARCH LEDGER — pattern family tests",
           "=" * 72,
           f"Completed tests : {n}",
           f"Uncorrected alpha: {ALPHA}",
           f"Corrected alpha  : {corrected_alpha(n):.5f} (Sidak, n={max(1, n)})",
           ""]
    if not led["tests"]:
        out.append("  (nothing registered yet)")
    for t in led["tests"]:
        v = t.get("verdict", "-")
        out.append(f"  [{t['status']:10}] {t['family']:12} {t['config_hash']} "
                   f"-> {v}")
        if t.get("result"):
            r = t["result"]
            out.append(f"               trades={r.get('trades')} "
                       f"pf={r.get('profit_factor')} "
                       f"exp={r.get('expectancy_r')}R")
        if t.get("note"):
            out.append(f"               note: {t['note']}")
    out += ["",
            "A positive verdict here means the family cleared PRE-REGISTERED",
            "criteria. With several families tested, weigh any single positive",
            "against the corrected alpha above, not the uncorrected one.",
            "=" * 72]
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
