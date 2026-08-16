"""
llm_client.py — OpenRouter API Client for NiftyLens
====================================================
Single module that replaces both Anthropic Claude SDK and Perplexity SDK.
Uses OpenRouter's free tier via standard HTTP (httpx).

FREE MODELS USED:
  - meta-llama/llama-3.3-70b-instruct:free   → AI scoring, regime analysis, trade plans
  - perplexity/llama-3.1-sonar-small-128k-online:free → Live news search (web-grounded)

SETUP:
  Add to your backend/.env file:
      OPENROUTER_API_KEY=sk-or-v1-your_key_here
  Get your free key at: https://openrouter.ai → Sign Up → Keys → Create Key
"""

import os
import httpx
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models — updated July 2026
# Qwen3-Coder free: strong reasoning + structured JSON output (confirmed available)
# Qwen3-Next-80B free: general-purpose fallback
# News: DISABLED — Perplexity free endpoint removed. News engine falls back to neutral.
#       To re-enable, set NEWS_MODEL = "perplexity/sonar" (paid ~$1/M tokens)
SCORING_MODEL   = "qwen/qwen3-coder:free"                              # Primary: strong reasoning + JSON
SCORING_MODEL_B = "qwen/qwen3-next-80b-a3b-instruct:free"              # Fallback: general-purpose
NEWS_MODEL      = ""                                                    # Disabled — no free news model available

# Timeouts (seconds)
SCORING_TIMEOUT = 30
NEWS_TIMEOUT    = 25

# ─── Core HTTP Client ─────────────────────────────────────────────────────────

def _get_headers() -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or not api_key.startswith("sk-or-"):
        raise ValueError(
            "OPENROUTER_API_KEY missing or invalid in .env. "
            "Get your free key at https://openrouter.ai → Keys → Create Key"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://niftylens.local",   # Required by OpenRouter
        "X-Title": "NiftyLens AI Trading Screener",
    }


async def _call_openrouter(
    messages: list[dict],
    model: str,
    max_tokens: int = 600,
    temperature: float = 0.2,
    timeout: int = 30,
) -> str:
    """
    Core async HTTP call to OpenRouter.
    Auto-falls back to SCORING_MODEL_B (Qwen3-Coder) if primary (DeepSeek-R1)
    is rate-limited or unavailable. Returns assistant text, raises on hard failure.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_BASE_URL,
                headers=_get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Rate limited on primary model → try fallback model once
            if model == SCORING_MODEL:
                logger.warning(f"DeepSeek-R1 rate limited — falling back to Qwen3-Coder")
                await asyncio.sleep(2)
                payload["model"] = SCORING_MODEL_B
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        OPENROUTER_BASE_URL,
                        headers=_get_headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"OpenRouter rate limited on {model} — waiting 5s")
                await asyncio.sleep(5)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        OPENROUTER_BASE_URL,
                        headers=_get_headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
        logger.error(f"OpenRouter HTTP error {e.response.status_code}: {e.response.text}")
        raise
    except httpx.TimeoutException:
        logger.error(f"OpenRouter timed out after {timeout}s (model={model})")
        raise
    except Exception as e:
        logger.error(f"OpenRouter unexpected error: {e}")
        raise


# ─── Public API: Scoring / Analysis ───────────────────────────────────────────

async def score_trade_setup(
    symbol: str,
    setup_type: str,          # "swing" or "intraday"
    technicals: dict,         # dict with rsi, adx, price, swing_low, etc.
    news_summary: str = "",   # output of get_stock_news()
    capital: float = 1_000_000,
) -> dict:
    """
    Score a trade setup using Llama 3.3 70B via OpenRouter.

    Returns dict:
        score        (int 0-100)
        grade        (str: A/B/C/D/F)
        entry        (float)
        stop         (float)
        target       (float)
        qty          (int)
        rationale    (str — plain English 2-3 sentences)
        risk_amount  (float — ₹)
        regime_ok    (bool)
    """
    news_block = f"\nLive News Summary:\n{news_summary}" if news_summary else ""

    system_prompt = (
        "You are a professional NSE equity trader. Score setups strictly. "
        "Reply ONLY with a JSON object — no markdown, no explanation outside the JSON."
    )

    user_prompt = f"""
Score this {setup_type} trade setup for {symbol} (NSE).

Technicals:
  RSI(14): {technicals.get('rsi', 'N/A')}
  ADX(14): {technicals.get('adx', 'N/A')}
  Price:   ₹{technicals.get('price', 'N/A')}
  20d Low: ₹{technicals.get('swing_low', 'N/A')}
  VWAP:    ₹{technicals.get('vwap', 'N/A')}
  Volume vs 20d avg: {technicals.get('volume_ratio', 'N/A')}x
  ATR(14): ₹{technicals.get('atr', 'N/A')}
  Trend:   {technicals.get('trend', 'N/A')}
  BB Width: {technicals.get('bb_width', 'N/A')}
{news_block}

Capital: ₹{capital:,.0f} | Risk per trade: 1% = ₹{capital * 0.01:,.0f}

Return JSON only:
{{
  "score": <0-100>,
  "grade": "<A|B|C|D|F>",
  "entry": <price float>,
  "stop": <stop loss float>,
  "target": <target float>,
  "qty": <shares int based on 1% risk>,
  "risk_amount": <₹ float>,
  "regime_ok": <true|false>,
  "rationale": "<2-3 sentence plain English analysis>"
}}
""".strip()

    raw = await _call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=SCORING_MODEL,
        max_tokens=400,
        temperature=0.1,
        timeout=SCORING_TIMEOUT,
    )

    # Parse JSON safely
    import json, re
    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Score parse failed for {symbol}, returning defaults. Raw: {raw[:200]}")
        result = {
            "score": 50, "grade": "C",
            "entry": technicals.get("price", 0),
            "stop": technicals.get("price", 0) * 0.97,
            "target": technicals.get("price", 0) * 1.05,
            "qty": 0, "risk_amount": 0,
            "regime_ok": True,
            "rationale": "Scoring unavailable — check OpenRouter key.",
        }
    return result


async def classify_market_regime(
    nifty_data: dict,
    vix: float,
    fii_net: float = 0,
) -> dict:
    """
    Classify the current Nifty market regime.

    Returns dict:
        regime     (str: "trending_up" | "trending_down" | "sideways" | "volatile")
        adx        (float)
        slope      (float)
        bb_width   (float)
        trade_ok   (bool — False = sideways, skip all trades)
        summary    (str — 1-2 sentences)
    """
    system_prompt = (
        "You are a market regime classifier for Indian equity markets. "
        "Reply ONLY with a JSON object."
    )
    user_prompt = f"""
Classify current Nifty 50 market regime.

Data:
  Nifty 50 Price: {nifty_data.get('price', 'N/A')}
  ADX(14):        {nifty_data.get('adx', 'N/A')}
  20d Slope:      {nifty_data.get('slope', 'N/A')}
  BB Width:       {nifty_data.get('bb_width', 'N/A')}
  India VIX:      {vix}
  FII Net (₹Cr):  {fii_net}

Rules:
  - ADX < 20 AND BB Width < 0.04  → sideways  (trade_ok = false)
  - ADX > 25 AND slope > 0        → trending_up
  - ADX > 25 AND slope < 0        → trending_down
  - VIX > 22                      → volatile

Return JSON only:
{{
  "regime": "<trending_up|trending_down|sideways|volatile>",
  "adx": <float>,
  "slope": <float>,
  "bb_width": <float>,
  "trade_ok": <true|false>,
  "summary": "<1-2 sentence market summary>"
}}
""".strip()

    raw = await _call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=SCORING_MODEL,
        max_tokens=250,
        temperature=0.1,
        timeout=SCORING_TIMEOUT,
    )

    import json, re
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "regime": "sideways", "adx": 0, "slope": 0, "bb_width": 0,
            "trade_ok": False,
            "summary": "Regime classification unavailable — defaulting to sideways (safe).",
        }


async def deep_stock_analysis(
    symbol: str,
    technicals: dict,
    news_summary: str,
    positions: list = None,
) -> str:
    """
    Full plain-English deep analysis for a single stock.
    Used when user clicks a row in the dashboard.
    Returns a multi-paragraph string (no JSON).
    """
    pos_block = ""
    if positions:
        pos_block = f"\nExisting Positions: {', '.join(positions)}"

    prompt = f"""
You are a senior NSE swing and intraday trader. Give a complete trade analysis for {symbol}.

Technicals:
{_format_technicals(technicals)}

Live News:
{news_summary or 'No recent news found.'}
{pos_block}

Write a 3-section analysis:
1. SETUP QUALITY (2-3 sentences — is this a valid setup, why/why not)
2. TRADE PLAN (entry trigger, exact stop loss, 2 targets, position size reasoning)
3. KEY RISKS (1-2 specific risks for this stock right now)

Be direct. No disclaimers. No generic statements. NSE/BSE context only.
""".strip()

    return await _call_openrouter(
        messages=[{"role": "user", "content": prompt}],
        model=SCORING_MODEL,
        max_tokens=700,
        temperature=0.3,
        timeout=SCORING_TIMEOUT,
    )


# ─── Public API: News via Perplexity Sonar ────────────────────────────────────

async def get_stock_news(symbol: str, company_name: str = "") -> dict:
    """
    Fetch live news for a stock using Perplexity Sonar (web-grounded) via OpenRouter.

    Returns dict:
        summary        (str — 2-3 sentences)
        sentiment      (str: "positive" | "negative" | "neutral")
        block_trade    (bool — True = do NOT trade this stock today)
        block_reason   (str — why blocked, if applicable)
        key_events     (list[str] — earnings, results, FII, regulatory)
    """
    # Skip API call if news model is disabled
    if not NEWS_MODEL:
        return {
            "summary": "News disabled — no free news model configured.",
            "sentiment": "neutral",
            "block_trade": False,
            "block_reason": "",
            "key_events": [],
        }

    name_str = f" ({company_name})" if company_name else ""
    system_prompt = (
        "You are a financial news analyst for Indian equity markets. "
        "Search for real-time news and reply ONLY with a JSON object."
    )
    user_prompt = f"""
Search for the latest news about {symbol}{name_str} on NSE India.

Look for:
1. Q4/quarterly results announcements (last 7 days)
2. Promoter buying or selling (bulk/block deals)
3. FII/DII significant activity
4. Regulatory actions, SEBI notices, fraud allegations
5. Major corporate announcements (mergers, splits, buybacks)
6. Sector-specific news (RBI policy, GST, sector bans)

Return JSON only:
{{
  "summary": "<2-3 sentence news summary>",
  "sentiment": "<positive|negative|neutral>",
  "block_trade": <true if any red flag found, else false>,
  "block_reason": "<reason for block if applicable, else empty string>",
  "key_events": ["<event 1>", "<event 2>"]
}}

If no significant news found, return sentiment=neutral, block_trade=false.
""".strip()

    raw = await _call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=NEWS_MODEL,
        max_tokens=400,
        temperature=0.1,
        timeout=NEWS_TIMEOUT,
    )

    import json, re
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"News parse failed for {symbol}. Raw: {raw[:200]}")
        result = {
            "summary": "News fetch failed — treating as neutral.",
            "sentiment": "neutral",
            "block_trade": False,
            "block_reason": "",
            "key_events": [],
        }
    return result


async def get_macro_news() -> dict:
    """
    Get macro market news for the regime dashboard header.
    Returns dict with summary, fii_activity, vix_note, top_sectors.
    """
    # Skip API call if news model is disabled
    if not NEWS_MODEL:
        return {
            "summary": "News disabled — no free news model configured.",
            "fii_net_cr": 0.0,
            "vix": 15.0,
            "top_sectors": [],
            "weak_sectors": [],
            "market_tone": "neutral",
        }

    system_prompt = "You are a macro analyst for Indian equity markets. Reply ONLY with JSON."
    user_prompt = """
Search for today's Indian equity market macro news:
1. Nifty 50 / Sensex direction and volume
2. FII net buy/sell today (₹ crore)
3. India VIX level
4. Top gaining and losing sectors

Return JSON only:
{
  "summary": "<2-3 sentence macro summary>",
  "fii_net_cr": <float — positive = buy, negative = sell>,
  "vix": <float>,
  "top_sectors": ["<sector>"],
  "weak_sectors": ["<sector>"],
  "market_tone": "<bullish|bearish|neutral>"
}
""".strip()

    raw = await _call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=NEWS_MODEL,
        max_tokens=350,
        temperature=0.1,
        timeout=NEWS_TIMEOUT,
    )

    import json, re
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "summary": "Macro news unavailable.",
            "fii_net_cr": 0.0,
            "vix": 15.0,
            "top_sectors": [],
            "weak_sectors": [],
            "market_tone": "neutral",
        }


# ─── Utilities ────────────────────────────────────────────────────────────────

def _format_technicals(t: dict) -> str:
    lines = []
    field_labels = {
        "rsi": "RSI(14)", "adx": "ADX(14)", "price": "Price ₹",
        "swing_low": "20d Low ₹", "vwap": "VWAP ₹",
        "volume_ratio": "Vol/Avg", "atr": "ATR(14) ₹",
        "trend": "Trend", "bb_width": "BB Width",
    }
    for key, label in field_labels.items():
        val = t.get(key)
        if val is not None:
            lines.append(f"  {label}: {val}")
    return "\n".join(lines)


async def health_check() -> dict:
    """Quick ping to verify OpenRouter connectivity. Tests primary model (DeepSeek-R1)."""
    try:
        resp = await _call_openrouter(
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            model=SCORING_MODEL,
            max_tokens=10,
            temperature=0,
            timeout=15,
        )
        return {
            "ok": "ok" in resp.lower(),
            "primary_model": SCORING_MODEL,
            "fallback_model": SCORING_MODEL_B,
            "response": resp,
        }
    except Exception as e:
        return {"ok": False, "primary_model": SCORING_MODEL, "error": str(e)}


# ─── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _test():
        print("Testing OpenRouter connectivity...")
        result = await health_check()
        print(f"Health check: {result}")

        print("\nTesting news fetch for RELIANCE...")
        news = await get_stock_news("RELIANCE", "Reliance Industries")
        print(f"News: {news}")

        print("\nTesting trade scoring...")
        score = await score_trade_setup(
            symbol="HDFCBANK",
            setup_type="swing",
            technicals={
                "rsi": 44, "adx": 28, "price": 1680.5,
                "swing_low": 1645.0, "vwap": 1675.0,
                "volume_ratio": 1.4, "atr": 22.5,
                "trend": "up", "bb_width": 0.035,
            },
            news_summary=news.get("summary", ""),
            capital=1_000_000,
        )
        print(f"Score: {score}")

    asyncio.run(_test())