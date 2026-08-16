"""
screener_engine.py — NiftyLens Main FastAPI Server
===================================================
Scans Nifty 500 universe, applies swing + intraday filters, calls LLM scoring,
exposes REST API consumed by index.html dashboard.

Start:
    cd ai-trading-screener/backend
    uvicorn screener_engine:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                         → server status
    GET  /api/swing                → all swing setups with scores
    GET  /api/intraday             → all intraday setups with scores
    GET  /api/regime               → market regime + macro news
    GET  /api/news/{symbol}        → live news for one stock
    POST /api/analyze/stock/{sym}  → deep AI analysis for one stock
    POST /api/refresh/swing        → force re-scan swing
    POST /api/refresh/intraday     → force re-scan intraday
    POST /webhook/tradingview      → TradingView Pine Script webhook

Dependencies: pip install fastapi uvicorn kiteconnect pandas numpy httpx python-dotenv
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_client import score_trade_setup, classify_market_regime, deep_stock_analysis
from news_engine import get_news_engine
from order_execution import get_order_executor, SwingOrderRequest, IntradayOrderRequest
from sectors import VOLATILE_UNIVERSE, SECTORS, get_sector, get_all_sectors, SECTOR_MAP
from pattern_engine import get_pattern_engine

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("screener")

# ─── Config ───────────────────────────────────────────────────────────────────

KITE_API_KEY      = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET   = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TRADING_CAPITAL   = float(os.getenv("TRADING_CAPITAL", "1000000"))

# Volatile universe — 200 high-ATR mid/small-caps (no slow large-caps)
# Imported from sectors.py — curated for swing & intraday volatility
NIFTY_UNIVERSE    = VOLATILE_UNIVERSE       # ~200 stocks
INTRADAY_UNIVERSE = VOLATILE_UNIVERSE[:80]  # Top 80 for intraday speed

# Scan intervals (seconds)
SWING_SCAN_INTERVAL    = 300   # 5 min
INTRADAY_SCAN_INTERVAL = 60    # 1 min

# Technical thresholds
SWING_RSI_MAX      = 48
SWING_ADX_MIN      = 20
SWING_RR_MIN       = 2.0
SIDEWAYS_ADX_MAX   = 25       # per README: ADX < 25 = no trend
SIDEWAYS_BBW_MAX   = 0.05     # per README: BB width < 0.05
SIDEWAYS_SLOPE_MAX = 2.0      # EMA20 slope < 2° = flat
SIDEWAYS_VOL_RATIO = 0.85     # 5d vol < 85% of 20d vol = declining
INTRADAY_VOL_MIN   = 1.5      # 1.5x volume vs 20d avg
SCORE_MIN_SWING    = 65
SCORE_MIN_INTRADAY = 70

# ─── Kite Connect setup ───────────────────────────────────────────────────────

def _get_kite():
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)
    return kite


# ─── Technical indicator functions ────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    return round(float(rsi.iloc[-1]), 2)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr  = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up  = high.diff()
    dn  = -low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0)
    pdi = 100 * pd.Series(pdm).rolling(period).mean() / atr
    ndi = 100 * pd.Series(ndm).rolling(period).mean() / atr
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return round(float(adx.iloc[-1]), 2)


def compute_vwap(df: pd.DataFrame) -> float:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    return round(float(vwap.iloc[-1]), 2)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr  = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return round(float(atr.iloc[-1]), 2)


def compute_bb_width(close: pd.Series, period: int = 20) -> float:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    bbw = (2 * std) / mid
    return round(float(bbw.iloc[-1]), 4)


def compute_slope(close: pd.Series, period: int = 20) -> float:
    """Raw price slope of EMA(period) — used internally."""
    y = close.iloc[-period:].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return round(float(slope), 4)


def compute_ema_slope_degrees(close: pd.Series, period: int = 20) -> float:
    """
    Compute the EMA(20) slope in degrees using arctangent.
    Normalized by price level so it's comparable across stocks.
    Returns angle in degrees: <2° = flat, >5° = strong trend.
    """
    ema = close.ewm(span=period, adjust=False).mean()
    if len(ema) < 5:
        return 0.0
    # Slope over last 5 bars, normalized by price
    y = ema.iloc[-5:].values
    price = y[-1] if y[-1] > 0 else 1.0
    pct_change_per_bar = (y[-1] - y[0]) / price / 4  # per-bar % change
    angle = float(np.degrees(np.arctan(pct_change_per_bar * 100)))  # scale for readability
    return round(abs(angle), 2)


def is_volume_declining(volume: pd.Series) -> bool:
    """
    Layer 4: Volume declining — 5-day avg < 85% of 20-day avg.
    Indicates fading interest → sideways or pre-breakdown.
    """
    if len(volume) < 20:
        return False
    vol_5d  = float(volume.iloc[-5:].mean())
    vol_20d = float(volume.iloc[-20:].mean())
    if vol_20d <= 0:
        return False
    return (vol_5d / vol_20d) < SIDEWAYS_VOL_RATIO


def is_sideways_5layer(adx: float, bb_width: float, ema_slope_deg: float,
                       volume: pd.Series, regime: dict) -> dict:
    """
    Full 5-layer sideways market filter from README spec.
    Returns dict with result and which layers triggered.

    Layers:
      1. ADX(14) < 25           → no directional trend
      2. EMA(20) slope < 2°     → market flat
      3. BB Width < 0.05        → volatility squeeze
      4. Volume declining       → 5d avg < 85% of 20d avg
      5. Claude AI regime       → classified as 'sideways' or 'risk_off'

    If ANY 3 of 5 layers trigger → sideways (block signals)
    If layers 1+3 both trigger  → definite squeeze (block signals)
    """
    layers = {
        "adx_low":          adx < SIDEWAYS_ADX_MAX,
        "ema_flat":         ema_slope_deg < SIDEWAYS_SLOPE_MAX,
        "bb_squeeze":       bb_width < SIDEWAYS_BBW_MAX,
        "vol_declining":    is_volume_declining(volume),
        "ai_sideways":      regime.get("regime") in ("sideways", "risk_off"),
    }

    triggered = sum(1 for v in layers.values() if v)
    adx_and_bb = layers["adx_low"] and layers["bb_squeeze"]

    is_sw = triggered >= 3 or adx_and_bb

    return {
        "is_sideways": is_sw,
        "layers_triggered": triggered,
        "layers": layers,
    }


# ─── Fetch OHLCV from Kite ────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str = "day", days: int = 60) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Zerodha Kite Connect.
    Returns DataFrame with columns: date, open, high, low, close, volume
    """
    kite = _get_kite()
    from datetime import timedelta
    from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date   = date.today().strftime("%Y-%m-%d")

    instruments = kite.ltp([f"NSE:{symbol}"])
    token       = list(instruments.values())[0].get("instrument_token")
    if not token:
        raise ValueError(f"Instrument token not found for {symbol}")

    data = kite.historical_data(token, from_date, to_date, interval)
    df   = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_intraday_ohlcv(symbol: str, interval: str = "5minute") -> pd.DataFrame:
    """Fetch today's intraday bars from Kite."""
    kite = _get_kite()
    today = date.today().strftime("%Y-%m-%d")
    instruments = kite.ltp([f"NSE:{symbol}"])
    token = list(instruments.values())[0].get("instrument_token")
    data  = kite.historical_data(token, today, today, interval)
    df    = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_nifty_data() -> dict:
    """Fetch Nifty 50 index data for regime classification."""
    kite = _get_kite()
    try:
        df = fetch_ohlcv("NIFTY 50", "day", 60)
        price    = float(df["close"].iloc[-1])
        adx      = compute_adx(df["high"], df["low"], df["close"])
        bb_width = compute_bb_width(df["close"])
        slope    = compute_slope(df["close"])
        return {"price": price, "adx": adx, "bb_width": bb_width, "slope": slope}
    except Exception as e:
        logger.error(f"Nifty data fetch failed: {e}")
        return {"price": 0, "adx": 25, "bb_width": 0.05, "slope": 0.5}


# ─── Application state ────────────────────────────────────────────────────────

class AppState:
    swing_results: list[dict] = []
    intraday_results: list[dict] = []
    regime: dict = {}
    last_swing_scan: float = 0
    last_intraday_scan: float = 0
    scanning_swing: bool = False
    scanning_intraday: bool = False


state = AppState()
news  = get_news_engine()


# ─── Swing scan logic ─────────────────────────────────────────────────────────

async def scan_swing() -> list[dict]:
    """
    Scan volatile universe for swing low setups.
    Uses 5-layer sideways filter + pattern engine confidence + sector tagging.
    """
    results = []
    regime  = state.regime or {}

    if regime.get("trade_ok") is False:
        logger.info("[Swing] Market sideways — skipping scan")
        return []

    pattern_engine = get_pattern_engine()

    for symbol in NIFTY_UNIVERSE:
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda s=symbol: fetch_ohlcv(s, "day", 60)
            )
            if len(df) < 30:
                continue

            close  = df["close"]
            high   = df["high"]
            low    = df["low"]
            volume = df["volume"]

            price      = float(close.iloc[-1])
            swing_low  = float(low.rolling(20).min().iloc[-1])
            swing_high = float(high.rolling(20).max().iloc[-1])

            # Is price within 3% of 20-day low?
            pct_from_low = (price - swing_low) / swing_low
            if pct_from_low > 0.03:
                continue

            rsi      = compute_rsi(close)
            adx      = compute_adx(high, low, close)
            atr      = compute_atr(high, low, close)
            bb_width = compute_bb_width(close)
            ema_slope = compute_ema_slope_degrees(close)
            vol_avg  = float(volume.rolling(20).mean().iloc[-1])
            vol_now  = float(volume.iloc[-1])

            if rsi > SWING_RSI_MAX or adx < SWING_ADX_MIN:
                continue

            # R:R check
            entry  = price * 1.001
            stop   = swing_low * 0.985
            target = swing_high
            risk   = entry - stop
            reward = target - entry
            if risk <= 0 or reward / risk < SWING_RR_MIN:
                continue

            # 5-layer sideways check
            sw_result = is_sideways_5layer(adx, bb_width, ema_slope, volume, regime)
            if sw_result["is_sideways"]:
                logger.debug(f"[Swing] {symbol} blocked by sideways filter "
                             f"({sw_result['layers_triggered']}/5 layers)")
                continue

            # Pattern engine confidence (if patterns learned for this stock)
            pattern_confidence = None
            pattern_wr = None
            try:
                pmatch = pattern_engine.match(symbol, df, len(df) - 1)
                if pmatch:
                    pattern_confidence = pmatch.confidence
                    pattern_wr = pmatch.win_rate
            except Exception:
                pass

            technicals = {
                "rsi": rsi, "adx": adx, "price": price,
                "swing_low": swing_low, "vwap": price,
                "volume_ratio": round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0,
                "atr": atr, "trend": "up" if adx > 25 else "neutral",
                "bb_width": bb_width, "ema_slope_deg": ema_slope,
                "pattern_wr": pattern_wr,
            }

            # Get news
            news_result = await news.get(symbol)
            if news_result.block_trade:
                logger.info(f"[Swing] Blocked {symbol}: {news_result.block_reason}")
                continue

            # LLM Score
            score_data = await score_trade_setup(
                symbol=symbol,
                setup_type="swing",
                technicals=technicals,
                news_summary=news_result.summary,
                capital=TRADING_CAPITAL,
            )

            if score_data.get("score", 0) < SCORE_MIN_SWING:
                continue

            results.append({
                "symbol": symbol,
                "sector": get_sector(symbol),
                "setup_type": "swing",
                "price": price,
                "rsi": rsi,
                "adx": adx,
                "bb_width": bb_width,
                "ema_slope": ema_slope,
                "swing_low": swing_low,
                "swing_high": swing_high,
                "entry": score_data.get("entry", entry),
                "stop": score_data.get("stop", stop),
                "target": score_data.get("target", target),
                "qty": score_data.get("qty", 0),
                "risk_amount": score_data.get("risk_amount", 0),
                "score": score_data.get("score", 0),
                "grade": score_data.get("grade", "C"),
                "rationale": score_data.get("rationale", ""),
                "news_summary": news_result.summary,
                "news_sentiment": news_result.sentiment,
                "pattern_confidence": pattern_confidence,
                "pattern_wr": pattern_wr,
                "sideways_layers": sw_result["layers_triggered"],
                "scanned_at": datetime.now().isoformat(),
            })

            await asyncio.sleep(0.5)   # Rate-limit Kite + OpenRouter

        except Exception as e:
            logger.error(f"[Swing] Error scanning {symbol}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"[Swing] Scan complete: {len(results)} setups found")
    return results


# ─── Intraday scan logic ──────────────────────────────────────────────────────

async def scan_intraday() -> list[dict]:
    """
    Scan for intraday setups: ORB breakouts + VWAP reclaims + pullback-to-EMA20.
    Only runs during market hours (9:15–3:20 IST).
    """
    from datetime import datetime
    now = datetime.now()
    hour, minute = now.hour, now.minute

    # Only scan during market hours
    if not ((9 <= hour < 15) or (hour == 15 and minute <= 20)):
        return []

    regime = state.regime or {}
    if regime.get("trade_ok") is False:
        return []

    results = []

    for symbol in INTRADAY_UNIVERSE:
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda s=symbol: fetch_intraday_ohlcv(s, "5minute")
            )

            if len(df) < 6:
                continue

            price   = float(df["close"].iloc[-1])
            vwap    = compute_vwap(df)
            vol_now = float(df["volume"].iloc[-1])
            vol_avg = float(df["volume"].mean()) if len(df) > 1 else vol_now

            # ORB: Opening Range = first 30 min (6 x 5min bars)
            orb_high = float(df["high"].iloc[:6].max())
            orb_low  = float(df["low"].iloc[:6].min())

            # EMA20 on intraday bars for pullback detection
            ema20_intra = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1]) if len(df) >= 20 else price

            setup_type_intra = None

            # Setup 1: ORB Breakout
            if price > orb_high and vol_now > INTRADAY_VOL_MIN * vol_avg:
                setup_type_intra = "orb_breakout"

            # Setup 2: VWAP Reclaim (just crossed above VWAP with volume)
            elif len(df) >= 2 and price > vwap and float(df["close"].iloc[-2]) <= vwap and vol_now > 1.2 * vol_avg:
                setup_type_intra = "vwap_reclaim"

            # Setup 3: Pullback to EMA20 in uptrend + volume confirmation
            elif (len(df) >= 20 and
                  price > ema20_intra * 0.998 and     # Price at or near EMA20
                  price < ema20_intra * 1.005 and     # Not already bounced far
                  price > vwap and                      # Above VWAP = uptrend
                  vol_now > 1.3 * vol_avg):             # Volume confirmation
                setup_type_intra = "pullback_ema20"

            if not setup_type_intra:
                continue

            # Daily data for RSI/ADX
            df_daily = await asyncio.get_event_loop().run_in_executor(
                None, lambda s=symbol: fetch_ohlcv(s, "day", 30)
            )
            rsi  = compute_rsi(df_daily["close"])
            adx  = compute_adx(df_daily["high"], df_daily["low"], df_daily["close"])
            atr  = compute_atr(df_daily["high"], df_daily["low"], df_daily["close"])

            technicals = {
                "rsi": rsi, "adx": adx, "price": price,
                "swing_low": orb_low, "vwap": vwap,
                "volume_ratio": round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0,
                "atr": atr, "trend": "up",
                "bb_width": 0.03,
                "ema20_intra": ema20_intra,
            }

            news_result = await news.get(symbol)
            if news_result.block_trade:
                continue

            score_data = await score_trade_setup(
                symbol=symbol,
                setup_type="intraday",
                technicals=technicals,
                news_summary=news_result.summary,
                capital=TRADING_CAPITAL,
            )

            if score_data.get("score", 0) < SCORE_MIN_INTRADAY:
                continue

            results.append({
                "symbol": symbol,
                "sector": get_sector(symbol),
                "setup_type": setup_type_intra,
                "price": price,
                "vwap": vwap,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "ema20_intra": ema20_intra,
                "rsi": rsi,
                "adx": adx,
                "entry": score_data.get("entry", price),
                "stop": score_data.get("stop", price * 0.99),
                "target": score_data.get("target", price * 1.02),
                "qty": score_data.get("qty", 0),
                "risk_amount": score_data.get("risk_amount", 0),
                "score": score_data.get("score", 0),
                "grade": score_data.get("grade", "C"),
                "rationale": score_data.get("rationale", ""),
                "news_summary": news_result.summary,
                "news_sentiment": news_result.sentiment,
                "scanned_at": datetime.now().isoformat(),
            })

            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"[Intraday] Error scanning {symbol}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"[Intraday] Scan complete: {len(results)} setups found")
    return results


# ─── Background scan loops ───────────────────────────────────────────────────

async def swing_scan_loop():
    while True:
        try:
            if time.time() - state.last_swing_scan >= SWING_SCAN_INTERVAL and not state.scanning_swing:
                state.scanning_swing = True
                state.swing_results  = await scan_swing()
                state.last_swing_scan = time.time()
                state.scanning_swing  = False
        except Exception as e:
            logger.error(f"Swing scan loop error: {e}")
            state.scanning_swing = False
        await asyncio.sleep(30)


async def intraday_scan_loop():
    while True:
        try:
            if time.time() - state.last_intraday_scan >= INTRADAY_SCAN_INTERVAL and not state.scanning_intraday:
                state.scanning_intraday = True
                state.intraday_results  = await scan_intraday()
                state.last_intraday_scan = time.time()
                state.scanning_intraday  = False
        except Exception as e:
            logger.error(f"Intraday scan loop error: {e}")
            state.scanning_intraday = False
        await asyncio.sleep(15)


async def regime_update_loop():
    while True:
        try:
            nifty_data = await asyncio.get_event_loop().run_in_executor(None, fetch_nifty_data)
            macro      = await news.get_macro()
            regime     = await classify_market_regime(
                nifty_data=nifty_data,
                vix=macro.vix,
                fii_net=macro.fii_net_cr,
            )
            regime["macro_summary"] = macro.summary
            regime["market_tone"]   = macro.market_tone
            regime["fii_net_cr"]    = macro.fii_net_cr
            regime["vix"]           = macro.vix
            state.regime = regime
            logger.info(f"[Regime] Updated: {regime.get('regime')} trade_ok={regime.get('trade_ok')}")
        except Exception as e:
            logger.error(f"Regime update error: {e}")
        await asyncio.sleep(300)   # Update every 5 min


# ─── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background scan loops on startup."""
    logger.info("NiftyLens screener starting up...")
    asyncio.create_task(regime_update_loop())
    asyncio.create_task(swing_scan_loop())
    asyncio.create_task(intraday_scan_loop())
    yield
    logger.info("NiftyLens shutting down.")


app = FastAPI(title="NiftyLens AI Trading Screener", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class WebhookPayload(BaseModel):
    symbol: str
    alert_type: str   # "swing_low" | "orb_breakout" | "vwap_reclaim"
    price: float


class ConfirmedOrderRequest(BaseModel):
    symbol: str
    confirmed: bool   # Must be True — safety gate


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "AI Trading Screener v2",
        "regime": state.regime.get("regime", "unknown"),
        "trade_ok": state.regime.get("trade_ok", True),
        "swing_setups": len(state.swing_results),
        "intraday_setups": len(state.intraday_results),
        "last_swing_scan": datetime.fromtimestamp(state.last_swing_scan).isoformat() if state.last_swing_scan else None,
        "capital": TRADING_CAPITAL,
    }


@app.get("/api/swing")
async def get_swing():
    return {
        "setups": state.swing_results,
        "count": len(state.swing_results),
        "regime": state.regime,
        "scanning": state.scanning_swing,
    }


@app.get("/api/intraday")
async def get_intraday():
    return {
        "setups": state.intraday_results,
        "count": len(state.intraday_results),
        "regime": state.regime,
        "scanning": state.scanning_intraday,
    }


@app.get("/api/regime")
async def get_regime():
    return state.regime


@app.get("/api/news/{symbol}")
async def get_news(symbol: str):
    result = await news.get(symbol.upper())
    return {
        "symbol": result.symbol,
        "summary": result.summary,
        "sentiment": result.sentiment,
        "block_trade": result.block_trade,
        "block_reason": result.block_reason,
        "key_events": result.key_events,
        "age_seconds": int(result.age_seconds),
    }


@app.post("/api/analyze/stock/{symbol}")
async def analyze_stock(symbol: str):
    sym = symbol.upper()
    try:
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_ohlcv(sym, "day", 60)
        )
        technicals = {
            "rsi":          compute_rsi(df["close"]),
            "adx":          compute_adx(df["high"], df["low"], df["close"]),
            "price":        float(df["close"].iloc[-1]),
            "swing_low":    float(df["low"].rolling(20).min().iloc[-1]),
            "vwap":         float(df["close"].iloc[-1]),
            "volume_ratio": float(df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]),
            "atr":          compute_atr(df["high"], df["low"], df["close"]),
            "trend":        "up",
            "bb_width":     compute_bb_width(df["close"]),
        }
    except Exception as e:
        technicals = {"error": str(e)}

    news_result = await news.get(sym)
    executor    = get_order_executor()
    positions   = [p["tradingsymbol"] for p in (executor.get_positions() if executor else [])]

    analysis = await deep_stock_analysis(
        symbol=sym,
        technicals=technicals,
        news_summary=news_result.summary,
        positions=positions,
    )
    return {"symbol": sym, "analysis": analysis, "technicals": technicals, "news": news_result.summary}


@app.post("/api/refresh/swing")
async def refresh_swing(background_tasks: BackgroundTasks):
    if not state.scanning_swing:
        state.last_swing_scan = 0
    return {"message": "Swing re-scan queued"}


@app.post("/api/refresh/intraday")
async def refresh_intraday(background_tasks: BackgroundTasks):
    if not state.scanning_intraday:
        state.last_intraday_scan = 0
    return {"message": "Intraday re-scan queued"}


@app.post("/webhook/tradingview")
async def tradingview_webhook(payload: WebhookPayload):
    logger.info(f"[Webhook] TradingView alert: {payload.symbol} {payload.alert_type} @ ₹{payload.price}")
    news_result = await news.get(payload.symbol.upper())
    return {
        "received": True,
        "symbol": payload.symbol,
        "alert_type": payload.alert_type,
        "price": payload.price,
        "news_block": news_result.block_trade,
        "news_reason": news_result.block_reason,
    }


@app.post("/api/order/swing")
async def place_swing_order(req: SwingOrderRequest):
    executor = get_order_executor()
    if not executor:
        raise HTTPException(503, "Order executor not available — check Kite token")
    result = await executor.place_swing_order(req)
    return result


@app.post("/api/order/intraday")
async def place_intraday_order(req: IntradayOrderRequest):
    executor = get_order_executor()
    if not executor:
        raise HTTPException(503, "Order executor not available — check Kite token")
    result = await executor.place_intraday_order(req)
    return result


@app.get("/api/positions")
async def get_positions():
    executor = get_order_executor()
    if not executor:
        return {"positions": [], "error": "Kite not connected"}
    return {"positions": executor.get_positions()}


@app.get("/api/orders")
async def get_orders():
    executor = get_order_executor()
    if not executor:
        return {"orders": [], "error": "Kite not connected"}
    return {"orders": executor.get_orders()}


@app.get("/api/margin")
async def get_margin():
    executor = get_order_executor()
    if not executor:
        return {"margin": {}, "error": "Kite not connected"}
    return {"margin": executor.get_margin()}


@app.post("/api/killswitch")
async def kill_switch(body: dict):
    confirm = body.get("confirm_text", "")
    if confirm != "CONFIRM":
        raise HTTPException(400, "Must send {confirm_text: 'CONFIRM'} to activate kill switch")
    executor = get_order_executor()
    if not executor:
        raise HTTPException(503, "Order executor not available")
    result = await executor.kill_switch()
    return result


# ─── Sector & Filter APIs ─────────────────────────────────────────────────────

@app.get("/api/sectors")
async def list_sectors():
    """Return all sectors with their stock counts."""
    return {
        "sectors": {name: len(stocks) for name, stocks in SECTORS.items()},
        "total_universe": len(NIFTY_UNIVERSE),
    }


@app.get("/api/sectors/{sector}")
async def get_sector_setups(sector: str):
    """Return swing + intraday setups filtered by sector."""
    swing_filtered = [s for s in state.swing_results if s.get("sector") == sector]
    intra_filtered = [s for s in state.intraday_results if s.get("sector") == sector]
    return {
        "sector": sector,
        "swing": swing_filtered,
        "intraday": intra_filtered,
        "count": len(swing_filtered) + len(intra_filtered),
    }


@app.get("/api/sideways")
async def get_sideways_status():
    """Return current 5-layer sideways filter status for Nifty."""
    regime = state.regime or {}
    try:
        nifty_data = await asyncio.get_event_loop().run_in_executor(None, fetch_nifty_data)
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_ohlcv("NIFTY 50", "day", 60)
        )
        adx = compute_adx(df["high"], df["low"], df["close"])
        bb_width = compute_bb_width(df["close"])
        ema_slope = compute_ema_slope_degrees(df["close"])
        sw = is_sideways_5layer(adx, bb_width, ema_slope, df["volume"], regime)
        return {
            "is_sideways": sw["is_sideways"],
            "layers_triggered": sw["layers_triggered"],
            "layers": sw["layers"],
            "adx": adx,
            "bb_width": bb_width,
            "ema_slope_deg": ema_slope,
        }
    except Exception as e:
        return {"is_sideways": False, "error": str(e)}