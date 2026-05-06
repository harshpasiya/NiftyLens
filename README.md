# 🔭 NiftyLens

> **AI-powered NSE trading screener** — real-time swing & intraday signals across the Nifty 500 universe, enriched with live news intelligence, sideways market filtering, and Claude AI scoring.

---

## What It Does

NiftyLens connects your **Zerodha Kite API**, **TradingView Premium**, **Perplexity news**, and **Claude AI** into one live dashboard that:

- Scans all **Nifty 500 stocks** every 60 seconds during market hours
- Detects **swing lows** with 5%+ upside potential and minimum 2:1 R:R
- Detects **intraday setups** — ORB breakouts, VWAP reclaims, pullback-to-support
- **Blocks sideways markets automatically** using a 5-layer filter (ADX + EMA slope + Bollinger Bands + Volume + Claude AI regime classifier)
- Runs **live news checks** via Perplexity Sonar before allowing any trade — blocks on earnings, fraud, FII selling, regulatory events
- Scores every setup with **Claude AI** (0–99) combining technical + news signals
- Displays everything in a **live auto-refreshing dark dashboard** with per-stock AI analysis on click

---

## Project Name

**NiftyLens** — because it brings the Nifty 500 into sharp focus, cutting through noise with AI.

---

## Architecture

```
TradingView Premium          Zerodha Kite API          Perplexity Sonar
  (Pine Script alerts)    →   (live OHLCV data)    →   (real-time news)
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    ▼
                         screener_engine.py
                         ┌─────────────────────────────────────────┐
                         │  • Sideways detection (5 layers)        │
                         │  • RSI / ADX / BB / VWAP / ORB calc     │
                         │  • news_engine.py — stock + sector news │
                         │  • Claude AI scoring & regime classify   │
                         │  • FastAPI endpoints (/api/swing etc.)  │
                         └─────────────────────────────────────────┘
                                    ▼
                           dashboard/index.html
                    (live dark UI — auto-refresh — click for AI analysis)
```

---

## File Structure

```
niftylens/
├── backend/
│   ├── screener_engine.py      # Main FastAPI server — Kite + Claude + sideways filters
│   ├── news_engine.py          # Perplexity real-time news module
│   ├── backtest.py             # 2-year historical backtest runner
│   ├── get_token.py            # Zerodha daily token refresh (generate via Claude)
│   └── .env                    # API keys — NEVER commit this file
├── dashboard/
│   └── index.html              # Live dashboard — open directly in Chrome
├── tradingview/
│   └── alerts.pine             # Pine Script — paste into TradingView editor
├── backtest_results/           # Auto-created when backtest runs
│   ├── report.html             # Full visual backtest report
│   ├── swing_trades.csv        # Every swing trade with P&L
│   ├── intraday_trades.csv     # Every intraday trade with P&L
│   ├── equity_curve.csv        # Running capital over time
│   └── summary.json            # Stats: win rate, Sharpe, drawdown
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Zerodha Kite Connect API (₹2,000/month)
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- Perplexity API key — [perplexity.ai](https://perplexity.ai) → Settings → API
- TradingView Premium (optional but recommended)

### 1. Install dependencies

```bash
pip install fastapi uvicorn kiteconnect pandas numpy anthropic python-dotenv websockets httpx
```

### 2. Create your `.env` file inside `backend/`

```env
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_ACCESS_TOKEN=refreshed_every_morning
ANTHROPIC_API_KEY=your_anthropic_key
PERPLEXITY_API_KEY=your_perplexity_key
```

### 3. Run the backtest first (validates strategy on 2 years of data)

```bash
cd backend
python backtest.py --mode both --sideways --years 2
# Open backtest_results/report.html in Chrome when done
```

### 4. Daily startup

```bash
# Every morning at 9:00 AM IST
python get_token.py                              # Refresh Zerodha token
uvicorn screener_engine:app --host 0.0.0.0 --port 8000  # Start backend
# Then open dashboard/index.html in Chrome → type localhost:8000 → Connect
```

---

## Strategy Logic

### Swing Screener
| Signal | Condition |
|--------|-----------|
| Swing Low | Price ≤ 20-day low × 1.02, bouncing up |
| RSI Filter | RSI(14) < 48 |
| Volume | Last candle > 1.3× 10-day average |
| Upside | Resistance target ≥ 5% above entry |
| Risk:Reward | ≥ 2:1 (3:1 required in weakening market) |
| News | Cleared by Perplexity — no earnings/fraud/FII selling |

### Intraday Screener
| Setup | Trigger |
|-------|---------|
| ORB Breakout | Price > first-candle high, volume ≥ 1.8× avg |
| VWAP Reclaim | Close crosses above VWAP with vol ≥ 1.5× avg |
| Pullback | Pull to EMA20 in uptrend, volume confirmation |
| Target | 1–1.5% move minimum |
| Direction | Long only (positive momentum only) |

### Sideways Market Filters (5 layers)
1. **ADX(14) < 25** → no directional trend, block all signals
2. **EMA(20) slope < 2°** → market flat, block signals
3. **Bollinger Band Width < 0.05** → volatility squeeze, block signals
4. **Volume declining** → 5-day avg < 85% of 20-day avg, reduce size
5. **Claude AI regime classifier** → reads Perplexity macro news, classifies as Trending / Weakening / Sideways / Risk-Off

---

## Expected Accuracy

| Configuration | Win Rate |
|--------------|---------|
| Swing — no filters | 55–60% |
| Swing — sideways filter | 62–66% |
| Swing — all filters + news + score ≥ 75 | **68–74%** |
| Intraday — no filters | 50–55% |
| Intraday — all filters + score ≥ 80 | **63–68%** |
| Sideways market days | **0% — no trades taken** |

> At 70% win rate with 2:1 R:R, risking 1% of capital per trade → expected value of +1.1R per trade.

---

## API Endpoints

All available at `http://localhost:8000` when backend is running:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server status, regime, sector blocks |
| GET | `/api/swing` | All swing setups + Claude AI analysis |
| GET | `/api/intraday` | All intraday setups + Claude AI analysis |
| GET | `/api/regime` | Market regime + Perplexity macro news |
| GET | `/api/news/{SYMBOL}` | Live news analysis for one stock |
| POST | `/api/analyze/stock/{SYMBOL}` | Deep Claude analysis — technicals + news + trade plan |
| POST | `/api/refresh/swing` | Force re-scan swing setups |
| POST | `/api/refresh/intraday` | Force re-scan intraday setups |
| POST | `/webhook/tradingview` | Receives Pine Script alerts |

---

## TradingView Integration

1. Open `tradingview/alerts.pine` → copy contents → paste into TradingView Pine Editor → Add to chart
2. Run `ngrok http 8000` in a separate terminal to get a public URL
3. Create TradingView alerts with webhook URL: `https://your-ngrok-url.ngrok.io/webhook/tradingview`
4. Alert message format:
```json
{"symbol": "{{ticker}}", "alert_type": "swing_low", "price": {{close}}}
```

---

## Backtesting

```bash
# Full 2-year backtest with all filters
python backtest.py --mode both --sideways --news-filter --years 2

# Swing only
python backtest.py --mode swing --sideways --years 2

# Intraday only
python backtest.py --mode intraday --sideways --years 2

# Single stock
python backtest.py --symbol HDFCBANK --sideways --years 2

# Compare: unfiltered vs filtered
python backtest.py --mode swing --years 2
python backtest.py --mode swing --sideways --years 2
```

Output: `backtest_results/report.html` — full visual report with equity curve, monthly P&L, sector win rates, trade log.

---

## Security

- `.env` is listed in `.gitignore` — never committed
- API keys are loaded via `python-dotenv` — never hardcoded
- Kite access token rotates daily — `get_token.py` automates refresh
- No trade execution — NiftyLens is a **screener only**, all trade decisions are made by you

---

## Disclaimer

NiftyLens is a personal research and screening tool. It does **not** constitute financial advice. Past backtest performance does not guarantee future results. Always use strict position sizing, maintain stop losses, and never risk capital you cannot afford to lose. The authors are not SEBI-registered advisors.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Zerodha Kite Connect API |
| News | Perplexity Sonar API |
| AI Scoring | Anthropic Claude (claude-sonnet-4-20250514) |
| Alerts | TradingView Pine Script v5 |
| Backend | Python 3.10 · FastAPI · Uvicorn |
| Indicators | pandas · numpy (RSI, ADX, VWAP, BB, EMA) |
| Frontend | Vanilla HTML/CSS/JS · Chart.js |
| Backtest | Walk-forward · Zerodha historical data |

---

*Built for the Indian equity market. NSE stocks only. Market hours: Mon–Fri 9:15 AM – 3:30 PM IST.*
