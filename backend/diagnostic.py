"""
diagnostic.py — Empirical Forward Return Analysis
===================================================
STOP GUESSING. MEASURE FIRST.

This script does NOT trade. It measures what actually happens to price
in the 5, 10, 20 bars AFTER each condition is true on your real Kite data.

For each of 20 conditions it calculates:
  - How often does it occur? (frequency)
  - Average forward return at +5, +10, +20 bars
  - Win rate at each horizon (% of times price was higher)
  - Whether it's better than random (baseline = unconditional return)

If a condition has forward win_rate > 55% at +10 bars consistently across
multiple stocks, THAT is worth building a strategy around.
If every condition shows ~50% win rate, daily bar signals on these stocks
have no edge and we need to move to intraday or use weekly bars.

Usage:
    python diagnostic.py --years 2
    python diagnostic.py --years 2 --symbol SUZLON   # single stock deep dive

Output:
    backtest_results/diagnostic_report.html
    backtest_results/diagnostic_data.csv
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("diagnostic")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

KITE_API_KEY      = os.getenv("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
RESULTS_DIR       = Path(__file__).parent.parent / "backtest_results"

UNIVERSE = [
    "TATAPOWER", "JSWENERGY", "ADANIPOWER", "SUZLON", "NHPC",
    "SJVN", "TORNTPOWER", "CESC", "RPOWER", "INOXWIND",
    "BEL", "BEML", "COCHINSHIP", "MAZDOCK", "GRSE",
    "RVNL", "IRCON", "TITAGARH", "RAILTEL", "IRFC",
    "CUMMINSIND", "SIEMENS", "ABB", "CGPOWER", "KEI",
    "POLYCAB", "APARINDS", "THERMAX",
    "TATAMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSON",
    "TVSMOTOR",
    "AUBANK", "FEDERALBNK", "BANKBARODA", "PNB",
    "RECLTD", "PFC", "CHOLAFIN",
    "DEEPAKNTR", "AARTIIND", "SRF",
    "LUPIN", "AUROPHARMA", "LAURUSLABS", "GLENMARK",
]


# ─── Kite fetch ───────────────────────────────────────────────────────────────

def fetch_history(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    from kiteconnect import KiteConnect
    kite  = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)
    ltp   = kite.ltp([f"NSE:{symbol}"])
    token = list(ltp.values())[0]["instrument_token"]
    data  = kite.historical_data(token, from_date, to_date, "day")
    df    = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


# ─── Indicators ───────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # EMAs
    df["ema9"]  = close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # RSI
    d = close.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / l.replace(0, np.nan))

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct_price"] = df["atr"] / close * 100   # ATR as % of price

    # ADX
    atr14 = tr.rolling(14).mean()
    up = high.diff(); dn = -low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=close.index).rolling(14).mean() / atr14
    ndi = 100 * pd.Series(ndm, index=close.index).rolling(14).mean() / atr14
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()

    # Stochastic RSI
    rsi14 = df["rsi"]
    lo_r = rsi14.rolling(14).min()
    hi_r = rsi14.rolling(14).max()
    df["stoch_k"] = 100 * (rsi14 - lo_r) / (hi_r - lo_r).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R
    hi14 = high.rolling(14).max()
    lo14 = low.rolling(14).min()
    df["willr"] = -100 * (hi14 - close) / (hi14 - lo14).replace(0, np.nan)

    # Bollinger Band Width
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bbw"] = (2 * std) / mid.replace(0, np.nan)
    df["bbw_avg"] = df["bbw"].rolling(20).mean()

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()
    df["obv_slope3"] = df["obv"] - df["obv"].shift(3)

    # Volume ratio
    df["vol_ratio"] = volume / volume.rolling(20).mean()

    # Rolling returns (forward) — what we're measuring
    for h in [3, 5, 10, 20]:
        df[f"fwd_ret_{h}"] = close.shift(-h) / close - 1

    # Price vs EMAs
    df["pct_above_ema21"] = (close - df["ema21"]) / df["ema21"] * 100
    df["pct_above_ema50"] = (close - df["ema50"]) / df["ema50"] * 100

    # 20d low proximity
    df["low20"] = low.rolling(20).min()
    df["pct_from_low20"] = (close - df["low20"]) / df["low20"] * 100

    return df


# ─── Condition definitions ────────────────────────────────────────────────────

def evaluate_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define 20 testable conditions. For each bar, True/False.
    These cover every major signal type we've tried.
    """
    c = pd.DataFrame(index=df.index)

    # RSI conditions
    c["rsi_lt_30"]          = df["rsi"] < 30
    c["rsi_lt_40"]          = df["rsi"] < 40
    c["rsi_40_50"]          = (df["rsi"] >= 40) & (df["rsi"] < 50)
    c["rsi_cross_up_40"]    = (df["rsi"].shift(1) < 40) & (df["rsi"] >= 40)

    # StochRSI conditions
    c["stoch_lt_20"]        = df["stoch_k"] < 20
    c["stoch_cross_up_20"]  = (df["stoch_k"].shift(1) < 20) & (df["stoch_k"] >= 20)
    c["stoch_k_above_d"]    = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] < 50)

    # EMA conditions
    c["price_cross_ema21"]  = (df["close"].shift(1) < df["ema21"].shift(1)) & (df["close"] >= df["ema21"])
    c["ema9_above_ema21"]   = df["ema9"] > df["ema21"]
    c["full_ema_align"]     = (df["ema9"] > df["ema21"]) & (df["ema21"] > df["ema50"])
    c["price_below_ema21"]  = df["close"] < df["ema21"]

    # ADX conditions
    c["adx_gt_25"]          = df["adx"] > 25
    c["adx_lt_20"]          = df["adx"] < 20

    # Swing low proximity
    c["near_20d_low"]       = df["pct_from_low20"] < 3.0
    c["near_20d_low_tight"] = df["pct_from_low20"] < 1.5

    # Volume conditions
    c["high_volume"]        = df["vol_ratio"] > 1.5
    c["obv_rising"]         = df["obv_slope3"] > 0

    # Volatility / BB
    c["bb_squeeze"]         = df["bbw"] < df["bbw_avg"]
    c["atr_expanding"]      = df["atr"] > df["atr"].shift(3)

    # Williams %R
    c["willr_oversold"]     = df["willr"] < -80
    c["willr_cross_up_70"]  = (df["willr"].shift(1) < -70) & (df["willr"] >= -70)

    return c


# ─── Forward return analysis ──────────────────────────────────────────────────

def analyze_condition(df: pd.DataFrame, condition: pd.Series, name: str) -> dict:
    """
    For all bars where condition is True, measure forward returns.
    Compare against unconditional (baseline) returns.
    """
    horizons = [3, 5, 10, 20]

    # Baseline: unconditional mean forward returns
    baseline = {}
    for h in horizons:
        col = f"fwd_ret_{h}"
        valid = df[col].dropna()
        baseline[h] = {
            "mean_ret": float(valid.mean() * 100),
            "win_rate": float((valid > 0).mean() * 100),
            "n":        len(valid),
        }

    # Conditional: forward returns when condition was True
    mask = condition & df["rsi"].notna()   # require indicator warmup
    cond_df = df[mask]

    if len(cond_df) < 5:
        return {"name": name, "frequency": 0, "error": "too few occurrences"}

    results = {
        "name":      name,
        "frequency": int(mask.sum()),
        "freq_pct":  round(float(mask.mean() * 100), 1),
    }

    for h in horizons:
        col = f"fwd_ret_{h}"
        fwd = cond_df[col].dropna()
        if len(fwd) < 3:
            continue
        mean_ret  = float(fwd.mean() * 100)
        win_rate  = float((fwd > 0).mean() * 100)
        edge      = mean_ret - baseline[h]["mean_ret"]
        n         = len(fwd)

        results[f"h{h}_mean_ret"]  = round(mean_ret, 2)
        results[f"h{h}_win_rate"]  = round(win_rate, 1)
        results[f"h{h}_edge"]      = round(edge, 2)    # vs baseline
        results[f"h{h}_n"]         = n

    return results


# ─── Main analysis loop ───────────────────────────────────────────────────────

async def run_diagnostic(symbols: list[str], years: int) -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    to_date   = date.today()
    from_date = to_date - timedelta(days=365 * years + 60)

    all_results = []   # per-condition, per-symbol

    for idx_s, symbol in enumerate(symbols):
        logger.info(f"[{idx_s+1}/{len(symbols)}] {symbol}...")
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=symbol: fetch_history(
                    s, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d")
                )
            )
            if len(df) < 100:
                logger.warning(f"  {symbol}: only {len(df)} bars, skipping")
                continue

            df = compute_indicators(df)
            conditions = evaluate_conditions(df)

            for cname in conditions.columns:
                result = analyze_condition(df, conditions[cname], cname)
                result["symbol"] = symbol
                all_results.append(result)

            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"  {symbol}: FAILED — {e}")
            continue

    if not all_results:
        logger.error("No results — check Kite token")
        return []

    # Aggregate across all symbols per condition
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "diagnostic_data.csv", index=False)

    # Aggregate: mean across symbols
    numeric_cols = [c for c in results_df.columns
                    if c not in ["name", "symbol", "error"]
                    and pd.api.types.is_numeric_dtype(results_df[c])]

    agg = results_df.groupby("name")[numeric_cols].mean().reset_index()
    agg = agg.sort_values("h10_win_rate", ascending=False)

    logger.info("\n" + "="*65)
    logger.info("DIAGNOSTIC RESULTS — Forward Win Rate at +10 bars")
    logger.info("="*65)
    logger.info(f"{'Condition':<28} {'Freq%':>6} {'WR5':>6} {'WR10':>6} {'WR20':>6} {'Edge10':>8}")
    logger.info("-"*65)
    for _, row in agg.iterrows():
        wr10 = row.get("h10_win_rate", 0)
        marker = " ◄ EDGE" if wr10 > 55 else (" ◄ weak" if wr10 > 52 else "")
        logger.info(
            f"{row['name']:<28} "
            f"{row.get('freq_pct',0):>6.1f}% "
            f"{row.get('h5_win_rate',0):>6.1f}% "
            f"{wr10:>6.1f}% "
            f"{row.get('h20_win_rate',0):>6.1f}% "
            f"{row.get('h10_edge',0):>+8.2f}%"
            f"{marker}"
        )

    # Generate HTML report
    generate_html(agg, results_df, symbols)
    logger.info(f"\nReport → backtest_results/diagnostic_report.html")
    logger.info(f"Data   → backtest_results/diagnostic_data.csv")

    return agg.to_dict("records")


def generate_html(agg: pd.DataFrame, detail: pd.DataFrame, symbols: list[str]) -> None:
    rows_good = ""
    rows_bad  = ""
    rows_neutral = ""

    for _, row in agg.iterrows():
        wr10 = row.get("h10_win_rate", 50)
        edge = row.get("h10_edge", 0)
        color = "#22c55e" if wr10 > 55 else ("#ef4444" if wr10 < 47 else "#f59e0b")
        verdict = "✓ USE" if wr10 > 55 else ("✗ AVOID" if wr10 < 47 else "~ WEAK")

        tr = f"""<tr style="border-bottom:1px solid #1e293b">
            <td style="padding:7px 12px;font-weight:600">{row['name']}</td>
            <td style="text-align:right">{row.get('freq_pct',0):.1f}%</td>
            <td style="text-align:right">{row.get('h5_win_rate',0):.1f}%</td>
            <td style="text-align:right;color:{color};font-weight:700">{wr10:.1f}%</td>
            <td style="text-align:right">{row.get('h20_win_rate',0):.1f}%</td>
            <td style="text-align:right;color:{color}">{edge:+.2f}%</td>
            <td style="text-align:right">{row.get('h10_mean_ret',0):+.2f}%</td>
            <td style="text-align:center;color:{color};font-weight:700">{verdict}</td>
        </tr>"""

        if wr10 > 55:
            rows_good += tr
        elif wr10 < 47:
            rows_bad += tr
        else:
            rows_neutral += tr

    html = f"""<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<title>NiftyLens Diagnostic — Forward Return Analysis</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
  h1{{color:#38bdf8;margin-bottom:6px}}
  h2{{color:#64748b;font-size:13px;font-weight:normal;margin-bottom:20px}}
  h3{{color:#94a3b8;font-size:14px;margin:24px 0 10px}}
  .callout{{background:#1e293b;border-radius:10px;padding:18px;margin:16px 0;border-left:4px solid}}
  .good{{border-color:#22c55e}} .bad{{border-color:#ef4444}} .info{{border-color:#38bdf8}}
  .callout p{{font-size:13px;color:#94a3b8;line-height:1.6}}
  .callout strong{{color:#e2e8f0}}
  table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:10px;font-size:13px;margin-bottom:20px}}
  th{{background:#334155;padding:8px 12px;text-align:right;color:#64748b;font-weight:600}}
  th:first-child{{text-align:left}}
</style>
</head><body>

<h1>NiftyLens Diagnostic — Empirical Forward Return Analysis</h1>
<h2>Measured on {len(symbols)} stocks · {date.today()} · Shows what ACTUALLY happens after each condition</h2>

<div class="callout info">
  <p><strong>How to read this:</strong> Each row shows the historical win rate
  (price higher) at 5, 10, 20 bars AFTER the condition was true.
  Random = 50%. Edge = consistently above 55% across multiple stocks.
  <strong>"Edge" column</strong> = win rate minus unconditional baseline for that stock.
  Green rows = conditions with genuine predictive power worth building on.</p>
</div>

<h3>✓ Conditions with Edge (WR10 > 55%) — BUILD STRATEGY AROUND THESE</h3>
<table>
  <thead><tr><th>Condition</th><th>Frequency</th><th>Win@+5</th><th>Win@+10</th><th>Win@+20</th><th>Edge@10</th><th>Mean Ret</th><th>Verdict</th></tr></thead>
  <tbody>{rows_good if rows_good else '<tr><td colspan="8" style="padding:12px;color:#64748b;text-align:center">No conditions above 55% — daily signals may lack edge on this universe</td></tr>'}</tbody>
</table>

<h3>~ Neutral Conditions (47–55%) — WEAK OR NO EDGE</h3>
<table>
  <thead><tr><th>Condition</th><th>Frequency</th><th>Win@+5</th><th>Win@+10</th><th>Win@+20</th><th>Edge@10</th><th>Mean Ret</th><th>Verdict</th></tr></thead>
  <tbody>{rows_neutral}</tbody>
</table>

<h3>✗ Conditions that HURT (WR10 < 47%) — THESE ARE ANTI-SIGNALS (or go SHORT)</h3>
<table>
  <thead><tr><th>Condition</th><th>Frequency</th><th>Win@+5</th><th>Win@+10</th><th>Win@+20</th><th>Edge@10</th><th>Mean Ret</th><th>Verdict</th></tr></thead>
  <tbody>{rows_bad if rows_bad else '<tr><td colspan="8" style="padding:12px;color:#64748b;text-align:center">None</td></tr>'}</tbody>
</table>

<div class="callout good" style="margin-top:24px">
  <p><strong>Next step:</strong> Take only conditions in the green table with WR10 > 55%.
  Combine 2-3 of them with AND logic — the intersection will have higher win rate but lower frequency.
  Target: 3-5 conditions combined giving WR10 > 60% on at least 50 occurrences per stock.
  That is your actual entry signal.</p>
</div>

</body></html>"""

    with open(RESULTS_DIR / "diagnostic_report.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NiftyLens Diagnostic")
    parser.add_argument("--years",  default=2,   type=int)
    parser.add_argument("--symbol", default=None)
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else UNIVERSE
    asyncio.run(run_diagnostic(symbols, args.years))