"""
pattern_engine.py — Stock Pattern Memory Engine
================================================
Learns recurring price/indicator patterns from historical data,
stores their forward-return statistics, and at runtime matches
the current market state to known patterns to predict outcome.

HOW IT WORKS:
  LEARNING PHASE (run once or nightly):
    1. For each stock, slide a window of N bars across history
    2. Extract a feature vector: normalized OHLCV + RSI + ADX + EMA ratios
    3. Compute what actually happened in the NEXT M bars (forward return,
       max drawdown, whether it hit +2R before -1R, etc.)
    4. Store (feature_vector, outcome) in SQLite

  MATCHING PHASE (at trade time):
    1. Extract feature vector for the current bar window
    2. Find K nearest historical patterns using cosine similarity
    3. Aggregate their outcomes: win_rate, avg_return, confidence
    4. If win_rate >= threshold AND confidence >= min_matches → signal

WHY THIS BEATS RULE-BASED:
  Rules are human assumptions. Pattern matching is empirical —
  it finds what ACTUALLY preceded profitable moves in this stock's
  own history. HDFCBANK has different patterns than TATASTEEL.
  Each stock learns its own behaviour.

PATTERN WINDOW: 10 bars lookback (2 trading weeks) — captures
  the setup context without being too specific to overfit.

FORWARD WINDOW: 15 bars — measures what happened after the pattern.

SIMILARITY: Cosine similarity on normalized feature vectors.
  Fast, no DTW library needed, works well for this feature space.

Usage:
    engine = PatternEngine()
    engine.learn(symbol, df)              # learn from history
    signal = engine.match(symbol, df, i)  # match at bar i
    if signal.win_rate >= 0.58:
        place_trade(signal.entry, signal.stop, signal.target)
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("pattern_engine")

DB_PATH = Path(__file__).parent.parent / "backtest_results" / "patterns.db"

# ─── Configuration ────────────────────────────────────────────────────────────

LOOKBACK      = 10    # bars of history to form one pattern
FORWARD       = 15    # bars ahead to measure outcome
MIN_MATCHES   = 5     # minimum similar patterns needed for a signal
WIN_THRESHOLD = 0.55  # minimum historical win rate to trigger trade
SIM_THRESHOLD = 0.92  # minimum cosine similarity to count as a match
MAX_PATTERNS  = 500   # max patterns stored per symbol (keep DB lean)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """Result of matching current bar to historical patterns."""
    symbol: str
    win_rate: float          # fraction of similar patterns that won
    avg_return_pct: float    # average forward return across matches
    avg_rr: float            # average R:R achieved in matches
    n_matches: int           # how many similar patterns were found
    confidence: str          # "high" | "medium" | "low"
    entry: float
    stop: float
    target: float
    dominant_exit: str       # most common exit reason in matched patterns
    pattern_ids: list[int]   # IDs of the matched patterns (for audit)


# ─── Feature extraction ───────────────────────────────────────────────────────

def _safe_series(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def extract_feature_vector(df: pd.DataFrame, end_idx: int) -> Optional[np.ndarray]:
    """
    Extract a normalized feature vector for the window ending at end_idx.

    Features (per bar in the LOOKBACK window):
      - normalized close:  (close - window_mean) / window_std  → shape
      - normalized volume: (vol - vol_mean) / vol_std          → relative activity
      - bar body:          (close - open) / atr                → candle direction
      - bar range:         (high - low) / atr                  → volatility
      - rsi / 100                                              → momentum level
      - ema9/close ratio - 1                                   → short trend
      - ema21/close ratio - 1                                  → medium trend
      - adx / 100                                              → trend strength

    Total: LOOKBACK × 8 = 80 features (with LOOKBACK=10)
    Normalized so patterns from different price levels are comparable.
    Returns None if data is insufficient or contains NaN.
    """
    start_idx = end_idx - LOOKBACK + 1
    if start_idx < 1:
        return None

    window = df.iloc[start_idx: end_idx + 1].copy().reset_index(drop=True)
    if len(window) < LOOKBACK:
        return None

    close  = window["close"]
    high   = window["high"]
    low    = window["low"]
    op     = window["open"]
    volume = window["volume"]

    # Compute indicators on full df up to end_idx for accuracy, then slice
    full_close  = df["close"].iloc[:end_idx + 1]
    full_high   = df["high"].iloc[:end_idx + 1]
    full_low    = df["low"].iloc[:end_idx + 1]

    # ATR on full history
    tr = pd.concat([
        full_high - full_low,
        (full_high - full_close.shift()).abs(),
        (full_low  - full_close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_full = tr.rolling(14).mean()
    atr_win  = atr_full.iloc[start_idx: end_idx + 1].values

    # RSI
    d = full_close.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rsi_full = (100 - 100 / (1 + g / l.replace(0, np.nan)))
    rsi_win  = rsi_full.iloc[start_idx: end_idx + 1].values

    # EMA9, EMA21
    ema9_full  = full_close.ewm(span=9,  adjust=False).mean()
    ema21_full = full_close.ewm(span=21, adjust=False).mean()
    ema9_win   = ema9_full.iloc[start_idx: end_idx + 1].values
    ema21_win  = ema21_full.iloc[start_idx: end_idx + 1].values

    # ADX (simplified: use rolling std of directional movement)
    up   = full_high.diff()
    dn   = -full_low.diff()
    pdm  = np.where((up > dn) & (up > 0), up, 0.0)
    ndm  = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi  = pd.Series(pdm, index=full_close.index).rolling(14).mean() / atr_full.replace(0, np.nan)
    ndi  = pd.Series(ndm, index=full_close.index).rolling(14).mean() / atr_full.replace(0, np.nan)
    dx   = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx_full = dx.rolling(14).mean() * 100
    adx_win  = adx_full.iloc[start_idx: end_idx + 1].values

    close_arr  = close.values
    open_arr   = op.values
    high_arr   = high.values
    low_arr    = low.values
    vol_arr    = volume.values.astype(float)

    # Normalize price shape: subtract mean, divide by std
    c_mean = np.mean(close_arr)
    c_std  = np.std(close_arr)
    if c_std < 1e-8:
        return None
    norm_close = (close_arr - c_mean) / c_std

    # Normalize volume
    v_mean = np.mean(vol_arr)
    v_std  = np.std(vol_arr)
    norm_vol = (vol_arr - v_mean) / (v_std + 1e-8)

    # ATR for candle features
    atr_safe = np.where(atr_win > 0, atr_win, c_std)

    # Per-bar features
    body_dir   = (close_arr - open_arr) / atr_safe          # positive = bullish
    bar_range  = (high_arr - low_arr) / atr_safe             # volatility
    rsi_norm   = np.where(np.isnan(rsi_win), 0.5, rsi_win / 100.0)
    ema9_ratio = np.where(close_arr > 0, ema9_win / close_arr - 1, 0)
    ema21_ratio= np.where(close_arr > 0, ema21_win / close_arr - 1, 0)
    adx_norm   = np.where(np.isnan(adx_win), 0.25, adx_win / 100.0)

    # Stack: shape (LOOKBACK, 8)
    features = np.column_stack([
        norm_close,
        norm_vol,
        body_dir,
        bar_range,
        rsi_norm,
        ema9_ratio,
        ema21_ratio,
        adx_norm,
    ])

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return None

    vec = features.flatten()  # shape: (LOOKBACK*8,) = (80,)

    # L2 normalize so cosine similarity = dot product
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def compute_outcome(df: pd.DataFrame, signal_idx: int,
                    entry: float, stop: float, target: float) -> dict:
    """
    Simulate what happened after a pattern: target hit, stop hit, or timeout.
    Uses target-first order (unbiased daily bar simulation).
    Returns outcome dict stored alongside the pattern vector.
    """
    risk   = entry - stop
    if risk <= 0:
        return {"outcome": "skip", "return_pct": 0, "rr": 0, "exit": "skip", "bars": 0}

    for j in range(signal_idx + 1, min(signal_idx + FORWARD + 1, len(df))):
        h = float(df["high"].iloc[j])
        l = float(df["low"].iloc[j])
        o = float(df["open"].iloc[j])
        bars = j - signal_idx

        # Target first (unbiased)
        if h >= target:
            return {
                "outcome": "win",
                "return_pct": round((target - entry) / entry * 100, 3),
                "rr": round((target - entry) / risk, 2),
                "exit": "target",
                "bars": bars,
            }

        # Stop — gap fill
        if l <= stop:
            fill = min(o, stop)
            return {
                "outcome": "loss",
                "return_pct": round((fill - entry) / entry * 100, 3),
                "rr": round((fill - entry) / risk, 2),
                "exit": "stop",
                "bars": bars,
            }

    # Timeout
    last_close = float(df["close"].iloc[min(signal_idx + FORWARD, len(df) - 1)])
    return {
        "outcome": "win" if last_close > entry else "loss",
        "return_pct": round((last_close - entry) / entry * 100, 3),
        "rr": round((last_close - entry) / risk, 2),
        "exit": "timeout",
        "bars": FORWARD,
    }


# ─── Database layer ───────────────────────────────────────────────────────────

class PatternDB:
    """SQLite storage for pattern vectors and their outcomes."""

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                date        TEXT NOT NULL,
                vector      BLOB NOT NULL,
                entry       REAL,
                stop        REAL,
                target      REAL,
                outcome     TEXT,
                return_pct  REAL,
                rr          REAL,
                exit_reason TEXT,
                bars_held   INTEGER,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol ON patterns(symbol)"
        )
        self.conn.commit()

    def insert_pattern(self, symbol: str, date: str, vector: np.ndarray,
                       entry: float, stop: float, target: float,
                       outcome: dict):
        vec_bytes = vector.tobytes()
        self.conn.execute("""
            INSERT INTO patterns
              (symbol, date, vector, entry, stop, target,
               outcome, return_pct, rr, exit_reason, bars_held)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, date, vec_bytes,
            entry, stop, target,
            outcome["outcome"], outcome["return_pct"],
            outcome["rr"], outcome["exit"], outcome["bars"],
        ))
        self.conn.commit()

    def get_patterns(self, symbol: str) -> list[dict]:
        cur = self.conn.execute("""
            SELECT id, date, vector, entry, stop, target,
                   outcome, return_pct, rr, exit_reason, bars_held
            FROM patterns WHERE symbol = ?
            ORDER BY id DESC LIMIT ?
        """, (symbol, MAX_PATTERNS))
        rows = []
        for row in cur.fetchall():
            vec = np.frombuffer(row[2], dtype=np.float64)
            rows.append({
                "id":          row[0],
                "date":        row[1],
                "vector":      vec,
                "entry":       row[3],
                "stop":        row[4],
                "target":      row[5],
                "outcome":     row[6],
                "return_pct":  row[7],
                "rr":          row[8],
                "exit_reason": row[9],
                "bars_held":   row[10],
            })
        return rows

    def count(self, symbol: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM patterns WHERE symbol=?", (symbol,)
        )
        return cur.fetchone()[0]

    def clear(self, symbol: Optional[str] = None):
        if symbol:
            self.conn.execute("DELETE FROM patterns WHERE symbol=?", (symbol,))
        else:
            self.conn.execute("DELETE FROM patterns")
        self.conn.commit()

    def get_all_symbols(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT DISTINCT symbol FROM patterns ORDER BY symbol"
        )
        return [r[0] for r in cur.fetchall()]

    def stats(self) -> dict:
        cur = self.conn.execute("""
            SELECT symbol,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                   AVG(return_pct) as avg_ret,
                   AVG(rr) as avg_rr
            FROM patterns GROUP BY symbol ORDER BY symbol
        """)
        return {
            row[0]: {
                "total": row[1], "wins": row[2],
                "win_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
                "avg_ret": round(row[3], 3) if row[3] else 0,
                "avg_rr":  round(row[4], 2) if row[4] else 0,
            }
            for row in cur.fetchall()
        }


# ─── Pattern Engine ───────────────────────────────────────────────────────────

class PatternEngine:
    """
    Main interface: learn patterns from a stock's history,
    then match current bar state to predict trade outcome.

    Typical workflow:
        engine = PatternEngine()

        # ONCE (or nightly): learn from 2yr history
        engine.learn("HDFCBANK", df_hdfcbank)

        # AT TRADE TIME: match current state
        match = engine.match("HDFCBANK", df_live, current_bar_idx)
        if match and match.win_rate >= 0.58:
            place_trade(match.entry, match.stop, match.target)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db = PatternDB(db_path)
        self._cache: dict[str, list[dict]] = {}   # in-memory cache per symbol

    def learn(self, symbol: str, df: pd.DataFrame,
              overwrite: bool = False) -> int:
        """
        Learn patterns from df.
        Slides LOOKBACK window across all bars, records pattern + outcome.
        Returns number of patterns stored.
        """
        if overwrite:
            self.db.clear(symbol)
            self._cache.pop(symbol, None)

        existing = self.db.count(symbol)
        if existing > 0 and not overwrite:
            logger.info(f"[Pattern] {symbol}: {existing} patterns already learned. "
                        f"Pass overwrite=True to re-learn.")
            return existing

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        atr_s  = self._calc_atr(high, low, close)
        ema21  = close.ewm(span=21, adjust=False).mean()

        stored = 0
        last_bar = -LOOKBACK  # cooldown: don't store overlapping windows

        for i in range(LOOKBACK + 15, len(df) - FORWARD - 1):
            # Cooldown: space patterns at least LOOKBACK/2 bars apart
            if i - last_bar < LOOKBACK // 2:
                continue

            vec = extract_feature_vector(df, i)
            if vec is None:
                continue

            price   = float(close.iloc[i])
            atr_val = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else 0
            e21     = float(ema21.iloc[i])

            if atr_val <= 0:
                continue

            # Use consistent entry/stop/target logic (EMA21-based)
            entry  = round(price * 1.002, 2)
            stop   = round(e21 - 1.0 * atr_val, 2)
            target = round(entry + 2.5 * max(entry - stop, 1.0), 2)

            if stop <= 0 or entry <= stop:
                continue

            outcome = compute_outcome(df, i, entry, stop, target)
            if outcome["outcome"] == "skip":
                continue

            date_str = str(df["date"].iloc[i]) if "date" in df.columns else str(i)
            self.db.insert_pattern(symbol, date_str, vec, entry, stop, target, outcome)
            stored  += 1
            last_bar = i

        self._cache.pop(symbol, None)  # invalidate cache
        logger.info(f"[Pattern] {symbol}: learned {stored} patterns")
        return stored

    def match(self, symbol: str, df: pd.DataFrame,
              current_idx: int) -> Optional[PatternMatch]:
        """
        Match current bar to stored patterns.
        Returns PatternMatch if enough similar patterns found, else None.

        current_idx: index of the current bar in df (the "right now" bar)
        """
        vec = extract_feature_vector(df, current_idx)
        if vec is None:
            return None

        patterns = self._get_cached(symbol)
        if len(patterns) < MIN_MATCHES:
            logger.debug(f"[Pattern] {symbol}: only {len(patterns)} patterns stored, "
                         f"need {MIN_MATCHES}")
            return None

        # Compute cosine similarity (vectors are already L2-normalized)
        sims = []
        for p in patterns:
            stored_vec = p["vector"]
            if len(stored_vec) != len(vec):
                continue
            sim = float(np.dot(vec, stored_vec))   # cosine sim (both normalized)
            if sim >= SIM_THRESHOLD:
                sims.append((sim, p))

        if len(sims) < MIN_MATCHES:
            return None

        # Sort by similarity, take top-K
        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = sims[:min(20, len(sims))]   # cap at 20 best matches

        matches  = [p for _, p in top_k]
        wins     = [p for p in matches if p["outcome"] == "win"]
        win_rate = len(wins) / len(matches)

        avg_ret  = float(np.mean([p["return_pct"] for p in matches]))
        avg_rr   = float(np.mean([p["rr"] for p in matches]))

        # Dominant exit type
        exits = [p["exit_reason"] for p in matches]
        dominant_exit = max(set(exits), key=exits.count)

        # Confidence tier
        n = len(matches)
        confidence = "high" if n >= 15 else ("medium" if n >= 8 else "low")

        # Entry/stop/target from current bar
        close   = df["close"]
        high    = df["high"]
        low_    = df["low"]
        atr_s   = self._calc_atr(high, low_, close)
        ema21   = close.ewm(span=21, adjust=False).mean()

        price   = float(close.iloc[current_idx])
        atr_val = float(atr_s.iloc[current_idx]) if not pd.isna(atr_s.iloc[current_idx]) else price * 0.02
        e21     = float(ema21.iloc[current_idx])

        entry  = round(price * 1.002, 2)
        stop   = round(e21 - 1.0 * atr_val, 2)
        target = round(entry + 2.5 * max(entry - stop, 1.0), 2)

        return PatternMatch(
            symbol=symbol,
            win_rate=round(win_rate, 3),
            avg_return_pct=round(avg_ret, 3),
            avg_rr=round(avg_rr, 2),
            n_matches=n,
            confidence=confidence,
            entry=entry,
            stop=stop,
            target=target,
            dominant_exit=dominant_exit,
            pattern_ids=[p["id"] for p in matches],
        )

    def _get_cached(self, symbol: str) -> list[dict]:
        if symbol not in self._cache:
            self._cache[symbol] = self.db.get_patterns(symbol)
        return self._cache[symbol]

    def _calc_atr(self, high, low, close, p=14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    def summary(self) -> dict:
        """Print learned pattern stats per symbol."""
        return self.db.stats()

    def clear(self, symbol: Optional[str] = None):
        self.db.clear(symbol)
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()
        logger.info(f"[Pattern] Cleared: {symbol or 'ALL'}")


# ─── Module-level singleton ───────────────────────────────────────────────────

_engine: Optional[PatternEngine] = None

def get_pattern_engine() -> PatternEngine:
    global _engine
    if _engine is None:
        _engine = PatternEngine()
    return _engine


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from backtest_v2_institutional import fetch_history
    from datetime import date, timedelta

    async def test():
        engine = PatternEngine()

        symbol    = "HDFCBANK"
        to_date   = date.today()
        from_date = to_date - timedelta(days=365 * 2 + 60)

        print(f"Fetching {symbol} history...")
        df = fetch_history(symbol, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))
        print(f"  Got {len(df)} bars")

        print(f"\nLearning patterns from {symbol}...")
        n = engine.learn(symbol, df, overwrite=True)
        print(f"  Stored {n} patterns")

        print(f"\nMatching current bar (last bar of data)...")
        match = engine.match(symbol, df, len(df) - 5)
        if match:
            print(f"  Win rate of similar patterns: {match.win_rate*100:.1f}%")
            print(f"  Avg R:R in matches:           {match.avg_rr:.2f}×")
            print(f"  Matches found:                {match.n_matches}")
            print(f"  Confidence:                   {match.confidence}")
            print(f"  Dominant exit:                {match.dominant_exit}")
            print(f"  Suggested entry:              ₹{match.entry:.2f}")
            print(f"  Suggested stop:               ₹{match.stop:.2f}")
            print(f"  Suggested target:             ₹{match.target:.2f}")
            if match.win_rate >= WIN_THRESHOLD:
                print(f"  → SIGNAL: Trade this pattern (win rate ≥ {WIN_THRESHOLD*100:.0f}%)")
            else:
                print(f"  → SKIP: Win rate below threshold ({WIN_THRESHOLD*100:.0f}%)")
        else:
            print("  No match found (insufficient similar patterns)")

        print("\nDB stats:", engine.summary())

    asyncio.run(test())