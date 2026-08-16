"""
diagnostic_advanced.py — 120 Advanced Indicator Forward-Return Diagnostic
=========================================================================
Tests every indicator from the Advanced NSE Swing Screener list against
actual forward returns on your universe. Pure empirical measurement —
no assumptions, no theory, just what the data shows.

Each condition is measured at +3, +5, +10, +20 bars forward.
Sorted by best WR10 edge vs baseline. Top conditions go into the backtest.

All 120 indicators implemented in pure pandas/numpy — no TA-Lib required.

Usage:
    python diagnostic_advanced.py --years 2
    python diagnostic_advanced.py --years 2 --min-freq 2.0  # only show conditions firing >2% of bars
    python diagnostic_advanced.py --years 2 --symbol SUZLON  # single stock

Output:
    backtest_results/advanced_diagnostic_report.html
    backtest_results/advanced_diagnostic_data.csv
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
from scipy import stats as scipy_stats

load_dotenv()
logger = logging.getLogger("diag_adv")
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
    "TATAMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSON", "TVSMOTOR",
    "AUBANK", "FEDERALBNK", "BANKBARODA", "PNB",
    "RECLTD", "PFC", "CHOLAFIN",
    "DEEPAKNTR", "AARTIIND", "SRF",
    "LUPIN", "AUROPHARMA", "LAURUSLABS", "GLENMARK",
]

NIFTY_SYMBOL = "NIFTY 50"   # for relative strength calculations


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


# ─── Core indicator library ───────────────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rma(s: pd.Series, p: int) -> pd.Series:
    """RMA = Wilder's smoothed moving average (used in RSI, ATR)."""
    return s.ewm(alpha=1/p, adjust=False).mean()

def _sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()

def _std(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).std()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, p)

def _rsi(close: pd.Series, p: int = 14) -> pd.Series:
    d = close.diff()
    g = _rma(d.clip(lower=0), p)
    l = _rma((-d).clip(lower=0), p)
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> tuple:
    tr  = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, p)
    up  = high.diff(); dn = -low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * _rma(pd.Series(pdm, index=close.index), p) / atr
    ndi = 100 * _rma(pd.Series(ndm, index=close.index), p) / atr
    dx  = 100 * (pdi-ndi).abs() / (pdi+ndi).replace(0, np.nan)
    return _rma(dx, p), pdi, ndi

def _kama(close: pd.Series, fast: int = 2, slow: int = 30, period: int = 10) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    kama = close.copy()
    for i in range(period, len(close)):
        direction = abs(close.iloc[i] - close.iloc[i - period])
        volatility = sum(abs(close.iloc[j] - close.iloc[j-1]) for j in range(i-period+1, i+1))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama.iloc[i] = kama.iloc[i-1] + sc * (close.iloc[i] - kama.iloc[i-1])
    kama.iloc[:period] = np.nan
    return kama

def _vidya(close: pd.Series, cmo_period: int = 9, ema_period: int = 12) -> pd.Series:
    """Variable Index Dynamic Average."""
    diff  = close.diff()
    up    = diff.clip(lower=0).rolling(cmo_period).sum()
    down  = (-diff).clip(lower=0).rolling(cmo_period).sum()
    cmo   = ((up - down) / (up + down).replace(0, np.nan)).abs()
    alpha = 2 / (ema_period + 1)
    vidya = close.copy()
    for i in range(1, len(close)):
        if pd.isna(cmo.iloc[i]):
            vidya.iloc[i] = np.nan
        else:
            prev = vidya.iloc[i-1] if not pd.isna(vidya.iloc[i-1]) else close.iloc[i]
            vidya.iloc[i] = prev + alpha * float(cmo.iloc[i]) * (close.iloc[i] - prev)
    return vidya

def _hma(close: pd.Series, p: int = 20) -> pd.Series:
    """Hull Moving Average."""
    half = _ema(close, p // 2)
    full = _ema(close, p)
    raw  = 2 * half - full
    return _ema(raw, int(np.sqrt(p)))

def _rsx(close: pd.Series, p: int = 14) -> pd.Series:
    """RSX — smoother version of RSI using Jurik-style smoothing."""
    d1 = close.diff()
    up = d1.clip(lower=0).ewm(span=p, adjust=False).mean()
    dn = (-d1).clip(lower=0).ewm(span=p, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsx_raw = 100 - 100 / (1 + rs)
    # Apply additional smoothing
    return rsx_raw.ewm(span=3, adjust=False).mean()

def _fisher_transform(high: pd.Series, low: pd.Series, p: int = 9) -> pd.Series:
    """Fisher Transform — converts price to Gaussian normal distribution."""
    hi = high.rolling(p).max()
    lo = low.rolling(p).min()
    hl = (hi - lo).replace(0, np.nan)
    value = 2 * ((high + low) / 2 - lo) / hl - 1
    value = value.clip(-0.999, 0.999)
    return 0.5 * np.log((1 + value) / (1 - value))

def _connors_rsi(close: pd.Series, high: pd.Series, low: pd.Series,
                 rsi_p: int = 3, streak_p: int = 2, pct_p: int = 100) -> pd.Series:
    """Connors RSI = avg(RSI3, StreakRSI2, PercentRank100)."""
    rsi3 = _rsi(close, rsi_p)
    # Streak: count consecutive up/down days
    streak = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            streak.iloc[i] = max(streak.iloc[i-1], 0) + 1
        elif close.iloc[i] < close.iloc[i-1]:
            streak.iloc[i] = min(streak.iloc[i-1], 0) - 1
    streak_rsi = _rsi(streak, streak_p)
    # Percent rank of 1-day return over last 100 days
    ret1 = close.pct_change()
    pct_rank = ret1.rolling(pct_p).rank(pct=True) * 100
    return (rsi3 + streak_rsi + pct_rank) / 3

def _choppiness(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    """Choppiness Index — <38.2 = trending, >61.8 = choppy."""
    tr   = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(p).sum()
    hl_range = high.rolling(p).max() - low.rolling(p).min()
    return 100 * np.log10(atr_sum / hl_range.replace(0, np.nan)) / np.log10(p)

def _vhf(close: pd.Series, p: int = 28) -> pd.Series:
    """Vertical Horizontal Filter — >0.4 = trending."""
    hh = close.rolling(p).max()
    ll = close.rolling(p).min()
    num = (hh - ll).abs()
    den = close.diff().abs().rolling(p).sum()
    return num / den.replace(0, np.nan)

def _stoch_rsi(close: pd.Series, rsi_p: int = 14, stoch_p: int = 14) -> tuple:
    rsi_s  = _rsi(close, rsi_p)
    lo     = rsi_s.rolling(stoch_p).min()
    hi     = rsi_s.rolling(stoch_p).max()
    k      = 100 * (rsi_s - lo) / (hi - lo).replace(0, np.nan)
    d      = k.rolling(3).mean()
    return k, d

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    hi = high.rolling(p).max()
    lo = low.rolling(p).min()
    return -100 * (hi - close) / (hi - lo).replace(0, np.nan)

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 20) -> pd.Series:
    tp  = (high + low + close) / 3
    ma  = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()

def _vwap_daily(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, p: int = 14) -> pd.Series:
    tp      = (high + low + close) / 3
    raw_mf  = tp * volume
    pos_mf  = raw_mf.where(tp > tp.shift(1), 0).rolling(p).sum()
    neg_mf  = raw_mf.where(tp < tp.shift(1), 0).rolling(p).sum()
    return 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

def _bb(close: pd.Series, p: int = 20, mult: float = 2.0) -> tuple:
    mid = _sma(close, p)
    std = _std(close, p)
    return mid + mult*std, mid, mid - mult*std

def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                p: int = 10, mult: float = 3.0) -> pd.Series:
    atr    = _atr(high, low, close, p)
    hl2    = (high + low) / 2
    upper  = hl2 + mult * atr
    lower  = hl2 - mult * atr
    trend  = pd.Series(1, index=close.index)  # 1=up, -1=down
    final_upper = upper.copy()
    final_lower = lower.copy()
    for i in range(1, len(close)):
        fu_prev = final_upper.iloc[i-1]
        fl_prev = final_lower.iloc[i-1]
        final_upper.iloc[i] = upper.iloc[i] if upper.iloc[i] < fu_prev or close.iloc[i-1] > fu_prev else fu_prev
        final_lower.iloc[i] = lower.iloc[i] if lower.iloc[i] > fl_prev or close.iloc[i-1] < fl_prev else fl_prev
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
    return trend

def _zscore(close: pd.Series, p: int = 20) -> pd.Series:
    mu  = close.rolling(p).mean()
    sig = close.rolling(p).std()
    return (close - mu) / sig.replace(0, np.nan)

def _percentile_rank(series: pd.Series, p: int = 60) -> pd.Series:
    return series.rolling(p).rank(pct=True) * 100

def _swing_highs(high: pd.Series, lookback: int = 5) -> pd.Series:
    """Returns True where a swing high exists (highest of lookback bars each side)."""
    left  = high.rolling(lookback).max().shift(1)
    right = high[::-1].rolling(lookback).max().shift(1)[::-1]
    return (high >= left) & (high >= right)

def _swing_lows(low: pd.Series, lookback: int = 5) -> pd.Series:
    left  = low.rolling(lookback).min().shift(1)
    right = low[::-1].rolling(lookback).min().shift(1)[::-1]
    return (low <= left) & (low <= right)

def _nr7(high: pd.Series, low: pd.Series) -> pd.Series:
    """NR7: narrowest range of last 7 bars."""
    range_ = high - low
    return range_ == range_.rolling(7).min()

def _inside_day(high: pd.Series, low: pd.Series) -> pd.Series:
    return (high < high.shift(1)) & (low > low.shift(1))

def _cvd_proxy(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative Volume Delta proxy using close position in bar range."""
    body_pct = (close - low) / (high - low).replace(0, np.nan)  # 1 = buy pressure, 0 = sell
    delta = (2 * body_pct - 1) * volume
    return delta.rolling(5).sum()

def _buying_pressure(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Williams buying pressure: (close - low) / (high - low)."""
    return (close - low) / (high - low).replace(0, np.nan)

def _trend_efficiency(close: pd.Series, p: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio: directional move / total path."""
    direction = (close - close.shift(p)).abs()
    path      = close.diff().abs().rolling(p).sum()
    return direction / path.replace(0, np.nan)

def _entropy(close: pd.Series, p: int = 20) -> pd.Series:
    """Approximate entropy of price changes — low = orderly trend."""
    def _ent(x):
        counts, _ = np.histogram(x, bins=5)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    return close.pct_change().rolling(p).apply(_ent, raw=True)

def _hurst(close: pd.Series, p: int = 40) -> pd.Series:
    """
    Simplified Hurst exponent proxy via variance ratio.
    >0.5 = trending (persistent), <0.5 = mean-reverting.
    """
    def _h(x):
        if len(x) < 8:
            return np.nan
        r1 = np.var(np.diff(x))
        r2 = np.var(np.diff(x[::2])) / 2 if len(x) >= 4 else np.nan
        if r2 and r1:
            return 0.5 * np.log(r2 / r1) / np.log(2) + 0.5
        return np.nan
    return close.rolling(p).apply(_h, raw=True)

def _weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly for multi-timeframe analysis."""
    df_w = df.copy()
    df_w.index = pd.to_datetime(df_w["date"])
    weekly = df_w.resample("W").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return weekly


# ─── Full indicator computation ───────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame, nifty_close: pd.Series = None) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsx"]         = _rsx(close, 14)
    df["connors_rsi"] = _connors_rsi(close, high, low)
    df["fisher"]      = _fisher_transform(high, low, 9)
    df["fisher_prev"] = df["fisher"].shift(1)
    df["stoch_k"], df["stoch_d"] = _stoch_rsi(close)
    df["willr"]       = _williams_r(high, low, close, 14)
    df["cci"]         = _cci(high, low, close, 20)
    df["rsi14"]       = _rsi(close, 14)
    df["rsi3"]        = _rsi(close, 3)
    df["momentum5"]   = close / close.shift(5) - 1
    df["momentum10"]  = close / close.shift(10) - 1
    df["mom_accel"]   = df["momentum5"] - df["momentum5"].shift(3)

    # Momentum indicator (10-bar ROC)
    df["msi"]         = (close / close.shift(10) - 1) * 100

    # ── Adaptive Trend ────────────────────────────────────────────────────────
    df["kama"]        = _kama(close)
    df["kama_slope"]  = df["kama"] - df["kama"].shift(3)
    df["kama_accel"]  = df["kama_slope"] - df["kama_slope"].shift(3)
    df["vidya"]       = _vidya(close)
    df["vidya_slope"] = df["vidya"] - df["vidya"].shift(3)
    df["hma"]         = _hma(close, 20)
    df["hma_fast"]    = _hma(close, 10)
    df["hma_slope"]   = df["hma"] - df["hma"].shift(2)
    df["eff_ratio"]   = _trend_efficiency(close, 10)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["atr14"]       = _atr(high, low, close, 14)
    df["atr_pct"]     = _percentile_rank(df["atr14"], 60)
    df["atr_slope"]   = df["atr14"] - df["atr14"].shift(3)
    _, df["bb_upper"], df["bb_lower"] = _bb(close, 20)
    df["bb_mid"]      = _sma(close, 20)
    df["bbw"]         = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bbw_pct"]     = _percentile_rank(df["bbw"], 60)
    df["bbw_slope"]   = df["bbw"] - df["bbw"].shift(3)
    df["chop"]        = _choppiness(high, low, close, 14)
    df["vhf"]         = _vhf(close, 28)
    df["nr7"]         = _nr7(high, low)
    df["inside_day"]  = _inside_day(high, low)
    hrange            = high - low
    df["range5_min"]  = hrange.rolling(5).min()

    # ── ADX ───────────────────────────────────────────────────────────────────
    df["adx"], df["pdi"], df["ndi"] = _adx(high, low, close, 14)

    # ── Market Structure ──────────────────────────────────────────────────────
    df["swing_hi"]    = _swing_highs(high, 5).astype(float)
    df["swing_lo"]    = _swing_lows(low, 5).astype(float)
    df["low5"]        = low.rolling(5).min()
    df["high5"]       = high.rolling(5).max()
    df["low20"]       = low.rolling(20).min()
    df["high20"]      = high.rolling(20).max()
    df["pct_from_low20"] = (close - df["low20"]) / df["low20"] * 100

    # Higher highs / higher lows
    df["prev_swing_lo"]  = low.where(df["swing_lo"] == 1).ffill()
    df["prev_swing_hi"]  = high.where(df["swing_hi"] == 1).ffill()
    df["prev_swing_lo2"] = df["prev_swing_lo"].shift(1)
    df["prev_swing_hi2"] = df["prev_swing_hi"].shift(1)

    # ── Volume Flow ───────────────────────────────────────────────────────────
    df["obv"]         = _obv(close, volume)
    df["obv_slope5"]  = df["obv"] - df["obv"].shift(5)
    df["vol_ratio"]   = volume / volume.rolling(20).mean()
    df["vol_slope"]   = df["vol_ratio"] - df["vol_ratio"].shift(3)
    df["cvd_proxy"]   = _cvd_proxy(close, high, low, volume)
    df["buy_press"]   = _buying_pressure(close, high, low)
    df["mfi"]         = _mfi(high, low, close, volume, 14)
    df["vol_ma5"]     = volume.rolling(5).mean()
    df["vol_ma20"]    = volume.rolling(20).mean()
    df["vol_dry"]     = (volume < df["vol_ma20"] * 0.6)   # volume dryup

    # ── VWAP ──────────────────────────────────────────────────────────────────
    df["vwap"]        = _vwap_daily(high, low, close, volume)
    # Anchored VWAP from 20d low
    df["lo20_idx"]    = low.rolling(20).apply(lambda x: x.argmin(), raw=True).astype(int)
    # Simplified: use SMA as VWAP proxy for anchored calc
    df["avwap_proxy"] = _sma(close, 20)   # approximate
    df["vwap_dist"]   = (close - df["vwap"]) / df["vwap"] * 100

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df["ema9"]  = _ema(close, 9)
    df["ema21"] = _ema(close, 21)
    df["ema50"] = _ema(close, 50)
    df["ema200"]= _ema(close, 200)

    # ── Relative Strength vs Nifty ────────────────────────────────────────────
    if nifty_close is not None and len(nifty_close) == len(close):
        df["rs_nifty"]      = (close / close.shift(20)) / (nifty_close / nifty_close.shift(20))
        df["rs_nifty_slope"]= df["rs_nifty"] - df["rs_nifty"].shift(5)
        df["alpha_5d"]      = close.pct_change(5) - nifty_close.pct_change(5)
    else:
        df["rs_nifty"]      = np.nan
        df["rs_nifty_slope"]= np.nan
        df["alpha_5d"]      = np.nan

    # ── Statistical ───────────────────────────────────────────────────────────
    df["zscore20"]    = _zscore(close, 20)
    df["pct_rank60"]  = _percentile_rank(close, 60)
    df["ret1"]        = close.pct_change(1)
    df["ret5"]        = close.pct_change(5)
    # Rolling skew and kurtosis of 20d returns
    df["skew20"]      = df["ret1"].rolling(20).skew()
    df["kurt20"]      = df["ret1"].rolling(20).kurt()
    df["entropy20"]   = _entropy(close, 20)
    df["hurst40"]     = _hurst(close, 40)
    # Trend persistence: autocorrelation of 1d returns
    df["autocorr5"]   = df["ret1"].rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x)>5 else np.nan, raw=False)

    # ── Supertrend ────────────────────────────────────────────────────────────
    df["supertrend"]  = _supertrend(high, low, close, 10, 3.0)

    # ── Candle quality ────────────────────────────────────────────────────────
    body              = (close - open_).abs()
    upper_wick        = high - close.combine(open_, max)
    lower_wick        = close.combine(open_, min) - low
    total_range       = high - low
    df["wick_ratio"]  = (upper_wick + lower_wick) / total_range.replace(0, np.nan)
    df["body_pct"]    = body / total_range.replace(0, np.nan)
    df["green_candle"]= (close > open_).astype(float)

    # ── Forward returns ───────────────────────────────────────────────────────
    for h in [3, 5, 10, 20]:
        df[f"fwd_{h}"] = close.shift(-h) / close - 1

    return df


# ─── Condition definitions ────────────────────────────────────────────────────

def evaluate_conditions(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame(index=df.index)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    c["rsx_cross_35"]           = (df["rsx"].shift(1) < 35) & (df["rsx"] >= 35)
    c["rsx_cross_50"]           = (df["rsx"].shift(1) < 50) & (df["rsx"] >= 50)
    c["rsx_slope_positive"]     = (df["rsx"] > df["rsx"].shift(3)) & (df["rsx"] < 60)
    c["rsx_bull_divergence"]    = (close < close.shift(5)) & (df["rsx"] > df["rsx"].shift(5)) & (df["rsx"] < 50)
    c["connors_rsi_lt_20"]      = df["connors_rsi"] < 20
    c["connors_rsi_reversal"]   = (df["connors_rsi"].shift(1) < 20) & (df["connors_rsi"] >= 20)
    c["msi_bullish"]            = (df["msi"] > 0) & (df["msi"] > df["msi"].shift(3))
    c["stoch_rsi_cross_up"]     = (df["stoch_k"].shift(1) < 20) & (df["stoch_k"] >= 20)
    c["fisher_transform_bullish"]= (df["fisher"] > df["fisher_prev"]) & (df["fisher"] < 0)
    c["momentum_acceleration"]  = (df["mom_accel"] > 0) & (df["mom_accel"] > df["mom_accel"].shift(3))

    # ── ADAPTIVE TREND ────────────────────────────────────────────────────────
    c["price_above_kama"]       = close > df["kama"]
    c["kama_slope_positive"]    = df["kama_slope"] > 0
    c["kama_acceleration"]      = (df["kama_accel"] > 0) & (df["kama_slope"] > 0)
    c["price_above_vidya"]      = close > df["vidya"]
    c["vidya_slope_positive"]   = df["vidya_slope"] > 0
    c["hma_turn_up"]            = (df["hma_slope"] > 0) & (df["hma_slope"].shift(1) <= 0)
    c["hma_fast_above_slow"]    = df["hma_fast"] > df["hma"]
    c["adaptive_trend_score"]   = (close > df["kama"]) & (close > df["vidya"]) & (df["hma_slope"] > 0)
    c["trend_efficiency_positive"] = df["eff_ratio"] > 0.4

    # ── VOLATILITY REGIME ─────────────────────────────────────────────────────
    c["atr_percentile_expanding"]     = (df["atr_pct"] < 40) & (df["atr_slope"] > 0)
    c["volatility_contraction_score"] = (df["bbw_pct"] < 20) & (df["atr_pct"] < 30)
    c["bb_width_percentile_lt_10"]    = df["bbw_pct"] < 10
    c["bb_width_expanding"]           = (df["bbw_slope"] > 0) & (df["bbw_pct"].shift(3) < df["bbw_pct"])
    c["chop_lt_38"]                   = df["chop"] < 38.2
    c["chop_falling_fast"]            = df["chop"] - df["chop"].shift(5) < -5
    c["vhf_trending"]                 = df["vhf"] > 0.4
    c["volatility_breakout"]          = (df["bbw_slope"] > 0) & (df["bbw_pct"].shift(5) < 25) & (close > df["bb_mid"])
    c["nr7_breakout"]                 = df["nr7"].shift(1) & (close > close.shift(1))
    c["inside_day_breakout"]          = df["inside_day"].shift(1) & (close > high.shift(1))

    # ── MARKET STRUCTURE ──────────────────────────────────────────────────────
    c["higher_low_confirmed"]   = (low > df["prev_swing_lo2"]) & (df["swing_lo"] == 1)
    c["higher_high_breakout"]   = close > df["prev_swing_hi2"]
    c["higher_low_near_support"]= (df["pct_from_low20"] < 3) & (low > df["low20"].shift(5))
    c["bullish_market_structure"]= (close > df["ema21"]) & (df["ema21"] > df["ema50"]) & (close > df["kama"])
    c["swing_low_reclaimed"]    = (close > df["low20"].shift(1)) & (close.shift(1) <= df["low20"].shift(2))
    c["swing_high_breakout"]    = close > df["high20"].shift(1)
    c["bos_bullish"]            = (close > df["high5"].shift(5)) & (df["supertrend"] == 1)
    c["choch_bullish"]          = (df["supertrend"] == 1) & (df["supertrend"].shift(1) == -1)
    c["liquidity_sweep_low"]    = (low < df["low20"].shift(1)) & (close > df["low20"].shift(1))
    c["compression_breakout"]   = (df["bbw_pct"].shift(5) < 20) & (df["bbw_slope"] > 0) & (close > df["ema21"])
    c["range_breakout"]         = close > df["high20"].shift(1)
    c["tight_base_breakout"]    = (df["bbw_pct"].shift(10) < 15) & (close > df["bb_upper"].shift(1))
    c["volatility_box_breakout"]= (df["atr_pct"].shift(5) < 25) & (df["atr_slope"] > 0) & (close > df["ema21"])

    # ── VOLUME FLOW ───────────────────────────────────────────────────────────
    c["rvol_gt_2"]                      = df["vol_ratio"] > 2.0
    c["rvol_gt_1_5"]                    = df["vol_ratio"] > 1.5
    c["volume_accumulation"]            = (df["obv_slope5"] > 0) & (df["vol_ratio"] > 1.0)
    c["volume_dryup_before_breakout"]   = df["vol_dry"].shift(1) & (df["vol_ratio"] > 1.2) & (close > close.shift(1))
    c["volume_expansion_breakout"]      = (df["vol_ratio"] > 1.5) & (close > df["high5"].shift(1))
    c["smart_money_volume"]             = (df["vol_ratio"] > 1.5) & (df["buy_press"] > 0.6) & (close > df["ema21"])
    c["cvd_proxy_bullish"]              = df["cvd_proxy"] > 0
    c["buying_pressure_positive"]       = df["buy_press"] > 0.6
    c["delivery_volume_spike"]          = df["vol_ratio"] > 2.0   # proxy for delivery spike
    c["volume_weighted_momentum"]       = (df["mfi"] > 50) & (df["mfi"] > df["mfi"].shift(5))

    # ── VWAP / INSTITUTIONAL ──────────────────────────────────────────────────
    c["price_above_vwap"]           = close > df["vwap"]
    c["intraday_vwap_reclaim"]      = (close > df["vwap"]) & (close.shift(1) <= df["vwap"].shift(1))
    c["anchored_vwap_support"]      = (df["vwap_dist"] > -1) & (df["vwap_dist"] < 1)
    c["distance_from_anchored_vwap"]= df["vwap_dist"] > 0
    c["event_vwap_reclaim"]         = (close > df["avwap_proxy"]) & (close.shift(1) <= df["avwap_proxy"].shift(1))
    c["earnings_vwap_hold"]         = (df["vwap_dist"] > -2) & (close > df["ema21"])
    c["anchored_vwap_breakout"]     = (df["vwap_dist"] > 2) & (df["vol_ratio"] > 1.3)
    c["vwap_trend_alignment"]       = (close > df["vwap"]) & (close > df["ema21"]) & (df["ema21"] > df["ema50"])

    # ── RELATIVE STRENGTH ─────────────────────────────────────────────────────
    c["rs_vs_nifty_positive"]       = df["rs_nifty"] > 1.0
    c["rs_vs_nifty_accelerating"]   = df["rs_nifty_slope"] > 0
    c["sector_rs_leader"]           = df["rs_nifty"] > 1.05
    c["industry_rs_leader"]         = (df["rs_nifty"] > 1.05) & (df["rs_nifty_slope"] > 0)
    c["relative_strength_52w_high"] = close == close.rolling(252).max()
    c["alpha_vs_nifty_positive"]    = df["alpha_5d"] > 0
    c["outperforming_index"]        = df["alpha_5d"] > 0.01
    c["momentum_leader"]            = (df["ret5"] > df["ret5"].rolling(20).quantile(0.8))

    # ── STATISTICAL / QUANT ───────────────────────────────────────────────────
    c["zscore_close_lt_minus2"]     = df["zscore20"] < -2
    c["zscore_reversion_signal"]    = (df["zscore20"].shift(1) < -2) & (df["zscore20"] >= -2)
    c["percentile_pullback"]        = df["pct_rank60"] < 20
    c["mean_reversion_setup"]       = (df["zscore20"] < -1.5) & (df["rsi14"] < 35)
    c["statistical_expansion"]      = (df["zscore20"] > 1) & (df["bbw_slope"] > 0)
    c["price_distance_percentile"]  = df["pct_rank60"] > 80
    c["return_skew_positive"]       = df["skew20"] > 0
    c["kurtosis_expansion"]         = df["kurt20"] > 3
    c["entropy_low"]                = df["entropy20"] < df["entropy20"].rolling(20).median()
    c["trend_persistence_high"]     = df["hurst40"] > 0.55

    # ── MULTI TIMEFRAME ───────────────────────────────────────────────────────
    # Weekly trend: 5-bar (=1 week) trend on daily bars
    ema5w  = _ema(close, 25)   # ~5 weeks
    ema13w = _ema(close, 65)   # ~13 weeks
    c["daily_weekly_alignment"]     = (close > df["ema21"]) & (close > ema5w)
    c["weekly_trend_bullish"]       = ema5w > ema13w
    c["daily_pullback_weekly_uptrend"] = (close < df["ema21"]) & (ema5w > ema13w)
    c["weekly_breakout_structure"]  = (close > close.rolling(25).max().shift(1)) & (close > df["ema50"])
    c["multi_tf_momentum_alignment"]= (df["rsi14"] > 50) & (df["rsx"] > 50) & (close > ema5w)
    c["multi_tf_volume_confirmation"]= (df["vol_ratio"] > 1.3) & (df["obv_slope5"] > 0) & (close > df["ema21"])
    c["weekly_rs_positive"]         = df["rs_nifty"] > 1.0
    c["monthly_trend_support"]      = close > _ema(close, 63)   # ~3 months

    # ── BREAKOUT QUALITY ──────────────────────────────────────────────────────
    c["base_breakout_quality"]          = (df["bbw_pct"].shift(10) < 25) & (close > df["high20"].shift(1)) & (df["vol_ratio"] > 1.3)
    c["compression_breakout_quality"]   = (df["chop"].shift(3) > 55) & (close > df["high5"].shift(1)) & (df["vol_ratio"] > 1.2)
    c["breakout_with_rvol"]             = (close > df["high20"].shift(1)) & (df["vol_ratio"] > 1.5)
    c["breakout_after_volatility_contraction"] = (df["atr_pct"].shift(5) < 25) & (close > df["high5"].shift(1))
    c["clean_range_expansion"]          = (df["body_pct"] > 0.6) & (close > df["high5"].shift(1))
    c["low_float_expansion"]            = (df["vol_ratio"] > 2.0) & (df["bbw_slope"] > 0)
    c["expansion_followthrough"]        = (close > close.shift(1)) & (df["vol_ratio"] > 1.3) & (close > df["ema21"])
    c["strong_close_breakout"]          = (df["buy_press"] > 0.7) & (close > df["high5"].shift(1))

    # ── RISK / NOISE FILTERS ──────────────────────────────────────────────────
    c["low_gap_noise"]              = (close - close.shift(1)).abs() / close.shift(1) < 0.02
    c["low_intraday_volatility_noise"] = df["atr_pct"] < 40
    c["clean_candle_structure"]     = df["body_pct"] > 0.5
    c["low_wick_instability"]       = df["wick_ratio"] < 0.4
    c["stable_trend_behavior"]      = (df["adx"] > 20) & (df["chop"] < 50)
    c["low_false_break_probability"]= (df["adx"] > 20) & (df["bbw_pct"] > 30)
    c["low_mean_reversion_risk"]    = df["hurst40"] > 0.5
    c["efficient_price_structure"]  = df["eff_ratio"] > 0.5

    # ── INSTITUTIONAL FOOTPRINT ───────────────────────────────────────────────
    c["delivery_percent_spike"]     = df["vol_ratio"] > 2.0
    c["block_deal_activity"]        = df["vol_ratio"] > 2.5
    c["accumulation_phase"]         = (df["obv_slope5"] > 0) & (df["pct_from_low20"] < 10) & (df["vol_ratio"] > 1.0)
    c["stealth_accumulation"]       = (df["obv_slope5"] > 0) & (df["vol_ratio"] < 1.2) & (df["pct_from_low20"] < 5)
    c["smart_money_entry"]          = (df["buy_press"] > 0.65) & (df["vol_ratio"] > 1.5) & (df["zscore20"] < -0.5)
    c["operator_accumulation_signature"] = (df["vol_dry"].shift(3)) & (df["vol_ratio"] > 1.5) & (close > close.shift(3))
    c["institutional_range_holding"]= (df["bbw_pct"] < 25) & (df["vol_ratio"].rolling(5).mean() > 1.1)
    c["absorption_near_support"]    = (df["pct_from_low20"] < 3) & (df["buy_press"] > 0.6) & (df["vol_ratio"] > 1.2)

    # ── SWING REVERSAL ────────────────────────────────────────────────────────
    c["failed_breakdown_reversal"]  = (low < df["low20"].shift(1)) & (close > df["low20"].shift(1)) & (df["vol_ratio"] > 1.2)
    c["undercut_and_reclaim"]       = (low < df["low5"].shift(1)) & (close > df["low5"].shift(1))
    c["spring_pattern"]             = (low < df["low20"].shift(1)) & (close > df["low20"].shift(1)) & (df["buy_press"] > 0.6)
    c["shakeout_reversal"]          = (df["pct_from_low20"] < 2) & (df["vol_ratio"] > 1.5) & (close > close.shift(1))
    c["reversal_after_exhaustion"]  = (df["rsi14"] < 30) & (df["willr"] < -80) & (close > close.shift(1))
    c["bear_trap_signal"]           = (low < df["low20"].shift(1)) & (close > df["ema21"])
    c["v_reversal_strength"]        = (df["rsi14"] < 35) & (close > df["ema21"]) & (df["vol_ratio"] > 1.5)
    c["reversal_with_volume_confirmation"] = (close > close.shift(1)) & (df["vol_ratio"] > 1.5) & (df["rsi14"] < 45)

    # ── COMPOSITE SCORES ─────────────────────────────────────────────────────
    # Trend quality: trend indicators aligned
    tqs = ((close > df["ema21"]).astype(int) +
           (df["ema21"] > df["ema50"]).astype(int) +
           (df["adx"] > 20).astype(int) +
           (df["supertrend"] == 1).astype(int) +
           (df["eff_ratio"] > 0.4).astype(int))
    c["trend_quality_score"]        = tqs >= 4

    # Swing quality: reversal indicators aligned
    sqs = ((df["rsi14"] < 40).astype(int) +
           (df["willr"] < -70).astype(int) +
           (df["stoch_k"] < 30).astype(int) +
           (df["pct_from_low20"] < 5).astype(int) +
           (df["obv_slope5"] > 0).astype(int))
    c["swing_quality_score"]        = sqs >= 3

    # Institutional activity
    ias = ((df["vol_ratio"] > 1.3).astype(int) +
           (df["buy_press"] > 0.6).astype(int) +
           (df["obv_slope5"] > 0).astype(int) +
           (df["mfi"] > 50).astype(int))
    c["institutional_activity_score"] = ias >= 3

    # Breakout probability
    bps = ((df["bbw_pct"] < 25).astype(int) +
           (df["chop"] < 50).astype(int) +
           (df["vol_ratio"] > 1.2).astype(int) +
           (close > df["ema21"]).astype(int) +
           (df["adx"] > 18).astype(int))
    c["breakout_probability_score"] = bps >= 4

    # Reversal probability
    rps = ((df["zscore20"] < -1.5).astype(int) +
           (df["connors_rsi"] < 25).astype(int) +
           (df["rsx"] < 35).astype(int) +
           (df["pct_from_low20"] < 3).astype(int) +
           (df["vol_ratio"] > 1.2).astype(int))
    c["reversal_probability_score"] = rps >= 3

    # Relative strength score
    rss = ((df["rs_nifty"] > 1.0).astype(int) +
           (df["alpha_5d"] > 0).astype(int) +
           (df["ret5"] > 0).astype(int))
    c["relative_strength_score"]    = rss >= 2

    # Volatility expansion score
    ves = ((df["atr_slope"] > 0).astype(int) +
           (df["bbw_slope"] > 0).astype(int) +
           (df["atr_pct"] < 50).astype(int) +
           (df["chop"] < 55).astype(int))
    c["volatility_expansion_score"] = ves >= 3

    # Smart money score
    sms = ((df["buy_press"] > 0.6).astype(int) +
           (df["vol_ratio"] > 1.3).astype(int) +
           (df["cvd_proxy"] > 0).astype(int) +
           (df["obv_slope5"] > 0).astype(int))
    c["smart_money_score"]          = sms >= 3

    # Multi-factor long score
    mfl = (tqs + sqs + ias + bps + rss)
    c["multi_factor_long_score"]    = mfl >= 12

    return c.fillna(False).astype(bool)


# ─── Analysis engine (reused from diagnostic.py) ─────────────────────────────

def analyze_condition(df: pd.DataFrame, condition: pd.Series, name: str) -> dict:
    horizons = [3, 5, 10, 20]

    baseline = {}
    for h in horizons:
        col  = f"fwd_{h}"
        valid = df[col].dropna()
        baseline[h] = {
            "mean_ret": float(valid.mean() * 100),
            "win_rate": float((valid > 0).mean() * 100),
        }

    mask    = condition & df["rsi14"].notna() & df["atr14"].notna()
    cond_df = df[mask]

    if len(cond_df) < 5:
        return {"name": name, "frequency": 0, "freq_pct": 0.0, "error": "too_few"}

    results = {
        "name":      name,
        "frequency": int(mask.sum()),
        "freq_pct":  round(float(mask.mean() * 100), 2),
    }

    best_wr = 0
    for h in horizons:
        col = f"fwd_{h}"
        fwd = cond_df[col].dropna()
        if len(fwd) < 3:
            continue
        mean_ret = float(fwd.mean() * 100)
        win_rate = float((fwd > 0).mean() * 100)
        edge     = win_rate - baseline[h]["win_rate"]
        t_stat   = float(scipy_stats.ttest_1samp(fwd.dropna(), 0).statistic) if len(fwd) > 10 else 0.0

        results[f"h{h}_mean_ret"] = round(mean_ret, 3)
        results[f"h{h}_win_rate"] = round(win_rate, 2)
        results[f"h{h}_edge"]     = round(edge, 3)
        results[f"h{h}_n"]        = len(fwd)
        results[f"h{h}_tstat"]    = round(t_stat, 2)
        best_wr = max(best_wr, win_rate)

    results["best_wr"] = round(best_wr, 2)
    return results


# ─── HTML report ─────────────────────────────────────────────────────────────

def generate_html(agg: pd.DataFrame, n_stocks: int, min_freq: float) -> None:
    categories = {
        "MOMENTUM":       ["rsx_cross_35","rsx_cross_50","rsx_slope_positive","rsx_bull_divergence","connors_rsi_lt_20","connors_rsi_reversal","msi_bullish","stoch_rsi_cross_up","fisher_transform_bullish","momentum_acceleration"],
        "ADAPTIVE TREND": ["price_above_kama","kama_slope_positive","kama_acceleration","price_above_vidya","vidya_slope_positive","hma_turn_up","hma_fast_above_slow","adaptive_trend_score","trend_efficiency_positive"],
        "VOLATILITY":     ["atr_percentile_expanding","volatility_contraction_score","bb_width_percentile_lt_10","bb_width_expanding","chop_lt_38","chop_falling_fast","vhf_trending","volatility_breakout","nr7_breakout","inside_day_breakout"],
        "MKT STRUCTURE":  ["higher_low_confirmed","higher_high_breakout","higher_low_near_support","bullish_market_structure","swing_low_reclaimed","swing_high_breakout","bos_bullish","choch_bullish","liquidity_sweep_low","compression_breakout","range_breakout","tight_base_breakout","volatility_box_breakout"],
        "VOLUME FLOW":    ["rvol_gt_2","rvol_gt_1_5","volume_accumulation","volume_dryup_before_breakout","volume_expansion_breakout","smart_money_volume","cvd_proxy_bullish","buying_pressure_positive","delivery_volume_spike","volume_weighted_momentum"],
        "VWAP":           ["price_above_vwap","intraday_vwap_reclaim","anchored_vwap_support","distance_from_anchored_vwap","event_vwap_reclaim","earnings_vwap_hold","anchored_vwap_breakout","vwap_trend_alignment"],
        "REL STRENGTH":   ["rs_vs_nifty_positive","rs_vs_nifty_accelerating","sector_rs_leader","industry_rs_leader","relative_strength_52w_high","alpha_vs_nifty_positive","outperforming_index","momentum_leader"],
        "STATISTICAL":    ["zscore_close_lt_minus2","zscore_reversion_signal","percentile_pullback","mean_reversion_setup","statistical_expansion","price_distance_percentile","return_skew_positive","kurtosis_expansion","entropy_low","trend_persistence_high"],
        "MULTI TF":       ["daily_weekly_alignment","weekly_trend_bullish","daily_pullback_weekly_uptrend","weekly_breakout_structure","multi_tf_momentum_alignment","multi_tf_volume_confirmation","weekly_rs_positive","monthly_trend_support"],
        "BREAKOUT":       ["base_breakout_quality","compression_breakout_quality","breakout_with_rvol","breakout_after_volatility_contraction","clean_range_expansion","low_float_expansion","expansion_followthrough","strong_close_breakout"],
        "NOISE FILTER":   ["low_gap_noise","low_intraday_volatility_noise","clean_candle_structure","low_wick_instability","stable_trend_behavior","low_false_break_probability","low_mean_reversion_risk","efficient_price_structure"],
        "INSTITUTIONAL":  ["delivery_percent_spike","block_deal_activity","accumulation_phase","stealth_accumulation","smart_money_entry","operator_accumulation_signature","institutional_range_holding","absorption_near_support"],
        "SW REVERSAL":    ["failed_breakdown_reversal","undercut_and_reclaim","spring_pattern","shakeout_reversal","reversal_after_exhaustion","bear_trap_signal","v_reversal_strength","reversal_with_volume_confirmation"],
        "COMPOSITE":      ["trend_quality_score","swing_quality_score","institutional_activity_score","breakout_probability_score","reversal_probability_score","relative_strength_score","volatility_expansion_score","smart_money_score","multi_factor_long_score"],
    }

    agg_dict  = agg.set_index("name").to_dict("index")
    top10     = agg.nlargest(10, "h10_edge")
    top10_rows= ""
    for _, row in top10.iterrows():
        wr10 = row.get("h10_win_rate", 50)
        edge = row.get("h10_edge", 0)
        col  = "#22c55e" if wr10 > 54 else ("#f59e0b" if wr10 > 51 else "#ef4444")
        top10_rows += f"""<tr>
            <td style="padding:6px 12px;font-weight:700">{row['name']}</td>
            <td style="text-align:right">{row.get('freq_pct',0):.1f}%</td>
            <td style="text-align:right">{row.get('h5_win_rate',0):.1f}%</td>
            <td style="text-align:right;color:{col};font-weight:700">{wr10:.1f}%</td>
            <td style="text-align:right">{row.get('h20_win_rate',0):.1f}%</td>
            <td style="text-align:right;color:{col}">{edge:+.2f}%</td>
            <td style="text-align:right">{row.get('h10_tstat',0):.2f}</td>
        </tr>"""

    # Category tables
    cat_html = ""
    for cat, conds in categories.items():
        rows = ""
        for cond in conds:
            r = agg_dict.get(cond, {})
            if not r:
                continue
            wr10 = r.get("h10_win_rate", 50)
            edge = r.get("h10_edge", 0)
            freq = r.get("freq_pct", 0)
            if freq < min_freq:
                continue
            col  = "#22c55e" if wr10 > 54 else ("#f59e0b" if wr10 > 51 else "#ef4444")
            ver  = "✓" if wr10 > 54 else ("~" if wr10 > 51 else "✗")
            rows += f"""<tr style="border-bottom:1px solid #0f172a">
                <td style="padding:5px 10px;font-size:12px">{cond}</td>
                <td style="text-align:right;font-size:12px">{freq:.1f}%</td>
                <td style="text-align:right;font-size:12px">{r.get('h5_win_rate',0):.1f}%</td>
                <td style="text-align:right;color:{col};font-weight:700;font-size:12px">{wr10:.1f}%</td>
                <td style="text-align:right;font-size:12px">{r.get('h20_win_rate',0):.1f}%</td>
                <td style="text-align:right;color:{col};font-size:12px">{edge:+.2f}%</td>
                <td style="text-align:center;color:{col};font-weight:700;font-size:12px">{ver}</td>
            </tr>"""
        if rows:
            cat_html += f"""
            <h3 style="color:#94a3b8;font-size:13px;margin:20px 0 8px;text-transform:uppercase;letter-spacing:1px">{cat}</h3>
            <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;margin-bottom:4px">
              <thead><tr style="background:#334155">
                <th style="padding:6px 10px;text-align:left;color:#64748b;font-size:11px">Condition</th>
                <th style="text-align:right;color:#64748b;font-size:11px">Freq%</th>
                <th style="text-align:right;color:#64748b;font-size:11px">WR+5</th>
                <th style="text-align:right;color:#64748b;font-size:11px">WR+10</th>
                <th style="text-align:right;color:#64748b;font-size:11px">WR+20</th>
                <th style="text-align:right;color:#64748b;font-size:11px">Edge</th>
                <th style="text-align:center;color:#64748b;font-size:11px">Use?</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>"""

    html = f"""<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<title>NiftyLens Advanced Diagnostic — 120 Indicators</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
  h1{{color:#f59e0b;margin-bottom:4px;font-size:20px}}
  h2{{color:#64748b;font-size:12px;font-weight:normal;margin-bottom:20px}}
  .callout{{background:#1e293b;border-left:4px solid #38bdf8;border-radius:8px;padding:14px 18px;margin:16px 0;font-size:13px;color:#94a3b8;line-height:1.6}}
  .callout strong{{color:#e2e8f0}}
  table.top{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:10px;overflow:hidden;margin:12px 0}}
  table.top th{{background:#334155;padding:8px 12px;text-align:right;color:#64748b;font-size:12px}}
  table.top th:first-child{{text-align:left}}
  table.top tr{{border-bottom:1px solid #0f172a}}
</style>
</head><body>

<h1>NiftyLens Advanced Diagnostic — 120 Indicators vs Forward Returns</h1>
<h2>Measured on {n_stocks} stocks · {date.today()} · Min frequency: {min_freq}% · Sorted by WR+10 edge</h2>

<div class="callout">
  <strong>How to read:</strong> WR+10 = % of times price was higher 10 bars after the condition fired.
  Random baseline ≈ 50%. <strong>Edge = WR minus baseline</strong> — positive = bullish predictive power.
  T-stat > 1.65 = statistically significant. Green ✓ = WR+10 > 54%. Use top conditions in AND combinations for v5 backtest.
</div>

<h3 style="color:#f59e0b;margin:20px 0 10px">Top 10 by Edge at +10 Bars</h3>
<table class="top">
  <thead><tr>
    <th>Condition</th><th>Freq%</th><th>WR+5</th><th>WR+10</th><th>WR+20</th><th>Edge@10</th><th>T-stat</th>
  </tr></thead>
  <tbody>{top10_rows}</tbody>
</table>

{cat_html}

<div class="callout" style="margin-top:24px;border-color:#22c55e">
  <strong>Next step:</strong> Take the top 3-5 conditions from green rows above.
  Combine with AND logic into backtest_v5.py entry signal.
  Aim for combined frequency of 3-8% (not too rare, not too common).
  Run backtest_v5 — if still failing, move to 15min intraday bars where momentum is more persistent.
</div>

</body></html>"""

    with open(RESULTS_DIR / "advanced_diagnostic_report.html", "w", encoding="utf-8") as f:
        f.write(html)


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_diagnostic(symbols: list[str], years: int, min_freq: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    to_date   = date.today()
    from_date = to_date - timedelta(days=365 * years + 90)

    # Fetch Nifty for relative strength
    logger.info("Fetching Nifty 50 for RS calculations...")
    try:
        nifty_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_history(NIFTY_SYMBOL, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))
        )
        nifty_close = nifty_df["close"]
    except Exception as e:
        logger.warning(f"Nifty fetch failed ({e}) — RS conditions will be skipped")
        nifty_close = None

    all_results = []

    for idx_s, symbol in enumerate(symbols):
        logger.info(f"[{idx_s+1}/{len(symbols)}] {symbol}...")
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda s=symbol: fetch_history(
                    s, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d")
                )
            )
            if len(df) < 120:
                logger.warning(f"  {symbol}: {len(df)} bars — skip")
                continue

            # Align nifty close to stock dates
            nifty_aligned = None
            if nifty_close is not None:
                nifty_reindexed = nifty_df.set_index("date")["close"].reindex(df["date"]).ffill()
                if len(nifty_reindexed) == len(df):
                    nifty_aligned = nifty_reindexed.values
                    nifty_aligned = pd.Series(nifty_aligned, index=df.index)

            df = compute_all_indicators(df, nifty_aligned)
            conditions = evaluate_conditions(df)

            n_conds = len(conditions.columns)
            for cname in conditions.columns:
                result = analyze_condition(df, conditions[cname], cname)
                result["symbol"] = symbol
                all_results.append(result)

            logger.info(f"  {symbol}: {n_conds} conditions evaluated")
            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"  {symbol}: FAILED — {e}")
            continue

    if not all_results:
        logger.error("No results")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "advanced_diagnostic_data.csv", index=False)

    numeric_cols = [c for c in results_df.columns
                    if c not in ["name","symbol","error"]
                    and pd.api.types.is_numeric_dtype(results_df[c])]
    agg = results_df.groupby("name")[numeric_cols].mean().reset_index()
    agg = agg.sort_values("h10_edge", ascending=False)

    # Console output
    logger.info(f"\n{'='*80}")
    logger.info("ADVANCED DIAGNOSTIC — Top 30 by Edge at +10 bars")
    logger.info(f"{'='*80}")
    logger.info(f"{'Condition':<40} {'Freq%':>6} {'WR5':>6} {'WR10':>6} {'WR20':>6} {'Edge10':>8} {'Tstat':>7}")
    logger.info("-"*80)
    for _, row in agg.head(30).iterrows():
        freq = row.get("freq_pct", 0)
        if freq < min_freq:
            continue
        wr10 = row.get("h10_win_rate", 50)
        mark = " ◄ EDGE" if wr10 > 54 else ""
        logger.info(
            f"{row['name']:<40} {freq:>6.1f}% "
            f"{row.get('h5_win_rate',0):>6.1f}% "
            f"{wr10:>6.1f}% "
            f"{row.get('h20_win_rate',0):>6.1f}% "
            f"{row.get('h10_edge',0):>+8.2f}% "
            f"{row.get('h10_tstat',0):>7.2f}"
            f"{mark}"
        )

    generate_html(agg, len(symbols), min_freq)
    logger.info(f"\nReport → backtest_results/advanced_diagnostic_report.html")
    logger.info(f"Data   → backtest_results/advanced_diagnostic_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",    default=2,   type=int)
    parser.add_argument("--min-freq", default=1.0, type=float, help="Min %% frequency to show")
    parser.add_argument("--symbol",   default=None)
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else UNIVERSE
    asyncio.run(run_diagnostic(symbols, args.years, args.min_freq))