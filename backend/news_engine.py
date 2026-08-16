"""
news_engine.py — Real-Time News Enrichment for NiftyLens
=========================================================
Fetches and caches live news for NSE stocks using Perplexity Sonar
via OpenRouter (free). Used by screener_engine.py before scoring.

CACHE: 15 minutes per symbol (avoids hammering free API rate limits)
BLOCK list: any stock with fraud/results/halt news is excluded from signals
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from llm_client import get_stock_news, get_macro_news

logger = logging.getLogger(__name__)

# ─── Cache config ─────────────────────────────────────────────────────────────

NEWS_CACHE_TTL_SECONDS = 900   # 15 minutes
MACRO_CACHE_TTL_SECONDS = 300  # 5 minutes

# Company name map — improves Perplexity search quality
COMPANY_NAMES: dict[str, str] = {
    "RELIANCE":  "Reliance Industries",
    "TCS":       "Tata Consultancy Services",
    "HDFCBANK":  "HDFC Bank",
    "INFY":      "Infosys",
    "ICICIBANK": "ICICI Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "BAJFINANCE":"Bajaj Finance",
    "SBIN":      "State Bank of India",
    "HINDUNILVR":"Hindustan Unilever",
    "AXISBANK":  "Axis Bank",
    "ITC":       "ITC Limited",
    "WIPRO":     "Wipro",
    "MARUTI":    "Maruti Suzuki",
    "TITAN":     "Titan Company",
    "NESTLEIND": "Nestle India",
    "POWERGRID": "Power Grid Corporation",
    "NTPC":      "NTPC",
    "ONGC":      "Oil and Natural Gas Corporation",
    "COALINDIA": "Coal India",
    "BHARTIARTL":"Bharti Airtel",
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class NewsResult:
    symbol: str
    summary: str
    sentiment: str               # "positive" | "negative" | "neutral"
    block_trade: bool            # True = do NOT trade today
    block_reason: str
    key_events: list[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > NEWS_CACHE_TTL_SECONDS


@dataclass
class MacroNews:
    summary: str
    fii_net_cr: float
    vix: float
    top_sectors: list[str] = field(default_factory=list)
    weak_sectors: list[str] = field(default_factory=list)
    market_tone: str = "neutral"
    fetched_at: float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.fetched_at) > MACRO_CACHE_TTL_SECONDS


# ─── News Engine class ────────────────────────────────────────────────────────

class NewsEngine:
    """
    Fetches, caches, and exposes news for screener_engine.py consumption.

    Usage:
        engine = NewsEngine()
        result = await engine.get(symbol="RELIANCE")
        if result.block_trade:
            skip_signal()
    """

    def __init__(self):
        self._cache: dict[str, NewsResult] = {}
        self._macro_cache: Optional[MacroNews] = None
        self._lock = asyncio.Lock()

    async def get(self, symbol: str) -> NewsResult:
        """
        Return cached news if fresh, otherwise fetch fresh from Perplexity Sonar.
        Thread-safe via asyncio lock.
        """
        async with self._lock:
            cached = self._cache.get(symbol)
            if cached and not cached.is_stale:
                logger.debug(f"[News] Cache hit: {symbol} ({cached.age_seconds:.0f}s old)")
                return cached

        logger.info(f"[News] Fetching live news: {symbol}")
        company_name = COMPANY_NAMES.get(symbol.upper(), "")

        try:
            raw = await get_stock_news(symbol, company_name)
            result = NewsResult(
                symbol=symbol,
                summary=raw.get("summary", "No news available."),
                sentiment=raw.get("sentiment", "neutral"),
                block_trade=bool(raw.get("block_trade", False)),
                block_reason=raw.get("block_reason", ""),
                key_events=raw.get("key_events", []),
            )
        except Exception as e:
            logger.error(f"[News] Failed to fetch news for {symbol}: {e}")
            result = NewsResult(
                symbol=symbol,
                summary="News fetch failed — treating as neutral.",
                sentiment="neutral",
                block_trade=False,
                block_reason="",
                key_events=[],
            )

        async with self._lock:
            self._cache[symbol] = result

        if result.block_trade:
            logger.warning(f"[News] BLOCKED: {symbol} — {result.block_reason}")

        return result

    async def get_macro(self) -> MacroNews:
        """
        Return current macro market news. Cached for 5 minutes.
        """
        async with self._lock:
            if self._macro_cache and not self._macro_cache.is_stale:
                return self._macro_cache

        logger.info("[News] Fetching macro news...")
        try:
            raw = await get_macro_news()
            macro = MacroNews(
                summary=raw.get("summary", "Macro news unavailable."),
                fii_net_cr=float(raw.get("fii_net_cr", 0)),
                vix=float(raw.get("vix", 15.0)),
                top_sectors=raw.get("top_sectors", []),
                weak_sectors=raw.get("weak_sectors", []),
                market_tone=raw.get("market_tone", "neutral"),
            )
        except Exception as e:
            logger.error(f"[News] Macro fetch failed: {e}")
            macro = MacroNews(
                summary="Macro news unavailable.",
                fii_net_cr=0.0,
                vix=15.0,
            )

        async with self._lock:
            self._macro_cache = macro

        return macro

    async def prefetch_batch(self, symbols: list[str], concurrency: int = 3) -> None:
        """
        Prefetch news for a batch of symbols concurrently (respects rate limits).
        Call this at startup or before the scan loop to warm the cache.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(sym: str):
            async with semaphore:
                await self.get(sym)
                await asyncio.sleep(1)  # 1s spacing between free API calls

        logger.info(f"[News] Prefetching news for {len(symbols)} symbols...")
        await asyncio.gather(*[fetch_one(s) for s in symbols])
        logger.info("[News] Prefetch complete.")

    def get_blocked_symbols(self) -> list[str]:
        """Return list of currently blocked symbols (cached, not re-fetched)."""
        return [sym for sym, n in self._cache.items() if n.block_trade and not n.is_stale]

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for one symbol or all symbols."""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()
            self._macro_cache = None
        logger.info(f"[News] Cache cleared: {symbol or 'ALL'}")

    def cache_stats(self) -> dict:
        """Debug helper: return cache hit counts and ages."""
        return {
            sym: {
                "age_s": int(n.age_seconds),
                "stale": n.is_stale,
                "blocked": n.block_trade,
                "sentiment": n.sentiment,
            }
            for sym, n in self._cache.items()
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

_engine: Optional[NewsEngine] = None


def get_news_engine() -> NewsEngine:
    """
    Return the shared NewsEngine singleton.
    Call this from screener_engine.py instead of creating a new instance.
    """
    global _engine
    if _engine is None:
        _engine = NewsEngine()
    return _engine


# ─── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _test():
        engine = NewsEngine()

        print("=== Testing single stock news ===")
        result = await engine.get("RELIANCE")
        print(f"Symbol:      {result.symbol}")
        print(f"Summary:     {result.summary}")
        print(f"Sentiment:   {result.sentiment}")
        print(f"Block Trade: {result.block_trade}")
        if result.block_reason:
            print(f"Block Reason:{result.block_reason}")
        if result.key_events:
            print(f"Key Events:  {result.key_events}")

        print("\n=== Testing macro news ===")
        macro = await engine.get_macro()
        print(f"Summary:     {macro.summary}")
        print(f"FII Net:     ₹{macro.fii_net_cr:,.0f}Cr")
        print(f"VIX:         {macro.vix}")
        print(f"Tone:        {macro.market_tone}")

        print("\n=== Testing batch prefetch (3 symbols) ===")
        await engine.prefetch_batch(["TCS", "INFY", "HDFCBANK"], concurrency=2)
        print("Cache stats:", engine.cache_stats())

    asyncio.run(_test())