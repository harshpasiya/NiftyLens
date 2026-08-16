"""
order_execution.py — Kite Connect Order Execution for NiftyLens
===============================================================
Places CNC (swing) and MIS (intraday) orders on Zerodha Kite.
ALL orders require explicit UI confirmation (confirmed=True).
NO autonomous trading. Every order is logged to orders.log.

Safety rules (hard-coded, cannot be bypassed):
  ✓ Max 5 concurrent positions
  ✓ 1% capital risk per trade
  ✓ Limit orders only (no market orders except kill switch)
  ✓ GTT stop loss for swing / SL-M for intraday
  ✓ 2× margin check before order
  ✓ No duplicate positions
  ✓ Kill switch: market exit ALL positions
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger("order_execution")

# ─── Config ───────────────────────────────────────────────────────────────────

KITE_API_KEY      = os.getenv("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TRADING_CAPITAL   = float(os.getenv("TRADING_CAPITAL", "1000000"))

MAX_POSITIONS    = 5
RISK_PCT         = 0.01          # 1% of capital per trade
MARGIN_SAFETY    = 2.0           # Need 2× order value in available margin
SLIPPAGE_BUFFER  = 0.001         # 0.1% above entry for limit order

LOG_FILE = os.path.join(os.path.dirname(__file__), "orders.log")


# ─── Pydantic request models (used by screener_engine routes) ────────────────

class SwingOrderRequest(BaseModel):
    symbol: str
    entry: float
    stop: float
    target: float
    confirmed: bool = False     # Must be True from dashboard confirm modal


class IntradayOrderRequest(BaseModel):
    symbol: str
    entry: float
    stop: float
    target: float
    confirmed: bool = False


# ─── Order logger ─────────────────────────────────────────────────────────────

def _log_order(event: str, data: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {event} | {data}\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception as e:
        logger.error(f"Log write failed: {e}")
    logger.info(f"ORDER LOG: {event} — {data}")


# ─── Order Executor ───────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Wraps Zerodha KiteConnect for order placement with full safety checks.
    Instantiated once as a module singleton.
    """

    def __init__(self):
        from kiteconnect import KiteConnect
        self.kite = KiteConnect(api_key=KITE_API_KEY)
        self.kite.set_access_token(KITE_ACCESS_TOKEN)
        logger.info("OrderExecutor initialized with Kite Connect")

    # ── Safety checks ─────────────────────────────────────────────────────────

    def _check_max_positions(self) -> bool:
        positions = self.kite.positions()
        net = positions.get("net", [])
        open_positions = [p for p in net if abs(p.get("quantity", 0)) > 0]
        return len(open_positions) < MAX_POSITIONS

    def _check_no_duplicate(self, symbol: str) -> bool:
        positions = self.kite.positions()
        net = positions.get("net", [])
        existing = {p["tradingsymbol"] for p in net if abs(p.get("quantity", 0)) > 0}
        return symbol.upper() not in existing

    def _check_margin(self, required_value: float) -> bool:
        margins = self.kite.margins(segment="equity")
        available = margins.get("net", 0)
        return available >= required_value * MARGIN_SAFETY

    def _check_market_hours(self) -> bool:
        now = datetime.now()
        h, m = now.hour, now.minute
        # 9:15 to 15:20
        return (h == 9 and m >= 15) or (10 <= h < 15) or (h == 15 and m <= 20)

    def _compute_qty(self, entry: float, stop: float) -> int:
        risk_per_trade = TRADING_CAPITAL * RISK_PCT
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        return max(1, int(risk_per_trade / risk_per_share))

    def _estimate_charges(self, qty: int, entry: float, exit_price: float) -> dict:
        """Estimate Zerodha brokerage + STT + exchange charges."""
        turnover   = qty * (entry + exit_price)
        brokerage  = min(20, 0.0003 * qty * entry) * 2   # ₹20 max per side
        stt        = 0.001 * qty * exit_price             # STT on sell side
        exchange   = 0.0000335 * turnover                 # NSE + BSE charges
        gst        = 0.18 * (brokerage + exchange)
        total      = round(brokerage + stt + exchange + gst, 2)
        net_pnl    = round((exit_price - entry) * qty - total, 2)
        return {"total_charges": total, "net_pnl_at_target": net_pnl}

    # ── Swing Order (CNC) ─────────────────────────────────────────────────────

    async def place_swing_order(self, req: SwingOrderRequest) -> dict:
        if not req.confirmed:
            return {"success": False, "error": "Order not confirmed by user"}

        sym   = req.symbol.upper()
        entry = round(req.entry * (1 + SLIPPAGE_BUFFER), 2)
        qty   = self._compute_qty(entry, req.stop)

        if qty == 0:
            return {"success": False, "error": "Quantity is 0 — stop too close to entry"}

        order_value = qty * entry

        # ── Safety gates ──────────────────────────────────────────────────────
        if not self._check_market_hours():
            return {"success": False, "error": "Outside market hours (9:15–15:20)"}
        if not self._check_max_positions():
            return {"success": False, "error": f"Max {MAX_POSITIONS} positions already open"}
        if not self._check_no_duplicate(sym):
            return {"success": False, "error": f"Already have an open position in {sym}"}
        if not self._check_margin(order_value):
            return {"success": False, "error": f"Insufficient margin (need 2× ₹{order_value:,.0f})"}

        charges = self._estimate_charges(qty, entry, req.target)

        try:
            order_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=sym,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty,
                    product=self.kite.PRODUCT_CNC,
                    order_type=self.kite.ORDER_TYPE_LIMIT,
                    price=entry,
                )
            )
        except Exception as e:
            _log_order("SWING_FAILED", {"symbol": sym, "error": str(e)})
            return {"success": False, "error": f"Kite order failed: {e}"}

        # Place GTT stop loss
        try:
            gtt_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.kite.place_gtt(
                    trigger_type=self.kite.GTT_TYPE_SINGLE,
                    tradingsymbol=sym,
                    exchange="NSE",
                    trigger_values=[req.stop],
                    last_price=entry,
                    orders=[{
                        "exchange": "NSE",
                        "tradingsymbol": sym,
                        "transaction_type": self.kite.TRANSACTION_TYPE_SELL,
                        "quantity": qty,
                        "order_type": self.kite.ORDER_TYPE_LIMIT,
                        "product": self.kite.PRODUCT_CNC,
                        "price": round(req.stop * 0.995, 2),
                    }]
                )
            )
        except Exception as e:
            logger.warning(f"GTT placement failed for {sym}: {e} — stop loss NOT active")
            gtt_id = None

        result = {
            "success": True,
            "order_id": order_id,
            "gtt_id": gtt_id,
            "symbol": sym,
            "qty": qty,
            "entry": entry,
            "stop": req.stop,
            "target": req.target,
            "order_value": order_value,
            "estimated_charges": charges["total_charges"],
            "net_pnl_at_target": charges["net_pnl_at_target"],
            "product": "CNC",
        }
        _log_order("SWING_PLACED", result)
        return result

    # ── Intraday Order (MIS) ──────────────────────────────────────────────────

    async def place_intraday_order(self, req: IntradayOrderRequest) -> dict:
        if not req.confirmed:
            return {"success": False, "error": "Order not confirmed by user"}

        sym   = req.symbol.upper()
        entry = round(req.entry * (1 + SLIPPAGE_BUFFER), 2)
        qty   = self._compute_qty(entry, req.stop)

        if qty == 0:
            return {"success": False, "error": "Quantity is 0 — stop too close to entry"}

        order_value = qty * entry

        if not self._check_market_hours():
            return {"success": False, "error": "Outside market hours"}
        if not self._check_max_positions():
            return {"success": False, "error": f"Max {MAX_POSITIONS} positions already open"}
        if not self._check_no_duplicate(sym):
            return {"success": False, "error": f"Already in {sym}"}
        if not self._check_margin(order_value):
            return {"success": False, "error": "Insufficient margin"}

        try:
            order_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=sym,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_LIMIT,
                    price=entry,
                )
            )
        except Exception as e:
            _log_order("INTRADAY_FAILED", {"symbol": sym, "error": str(e)})
            return {"success": False, "error": f"Kite order failed: {e}"}

        # SL-M stop loss order
        try:
            sl_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=sym,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_SLM,
                    trigger_price=req.stop,
                )
            )
        except Exception as e:
            logger.warning(f"SL-M failed for {sym}: {e}")
            sl_id = None

        result = {
            "success": True,
            "order_id": order_id,
            "sl_order_id": sl_id,
            "symbol": sym,
            "qty": qty,
            "entry": entry,
            "stop": req.stop,
            "target": req.target,
            "order_value": order_value,
            "product": "MIS",
            "auto_exit": "3:20 PM (Zerodha MIS)",
        }
        _log_order("INTRADAY_PLACED", result)
        return result

    # ── Portfolio getters ─────────────────────────────────────────────────────

    def get_positions(self) -> list:
        try:
            positions = self.kite.positions()
            net = positions.get("net", [])
            return [
                {
                    "tradingsymbol": p["tradingsymbol"],
                    "quantity": p["quantity"],
                    "average_price": p["average_price"],
                    "last_price": p["last_price"],
                    "pnl": p["pnl"],
                    "product": p["product"],
                }
                for p in net if abs(p.get("quantity", 0)) > 0
            ]
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return []

    def get_orders(self) -> list:
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"get_orders failed: {e}")
            return []

    def get_margin(self) -> dict:
        try:
            margins = self.kite.margins(segment="equity")
            return {
                "available": margins.get("net", 0),
                "used": margins.get("utilised", {}).get("debits", 0),
            }
        except Exception as e:
            logger.error(f"get_margin failed: {e}")
            return {"available": 0, "used": 0}

    # ── Kill Switch ───────────────────────────────────────────────────────────

    async def kill_switch(self) -> dict:
        """
        Emergency: square off ALL open positions at MARKET price.
        Requires double-confirmation from UI (handled in screener_engine).
        """
        positions = self.get_positions()
        if not positions:
            return {"success": True, "message": "No open positions to exit", "exits": []}

        exits = []
        for pos in positions:
            sym = pos["tradingsymbol"]
            qty = abs(pos["quantity"])
            if qty == 0:
                continue
            direction = "SELL" if pos["quantity"] > 0 else "BUY"
            try:
                order_id = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda s=sym, q=qty, d=direction: self.kite.place_order(
                        variety=self.kite.VARIETY_REGULAR,
                        exchange=self.kite.EXCHANGE_NSE,
                        tradingsymbol=s,
                        transaction_type=d,
                        quantity=q,
                        product=pos["product"],
                        order_type=self.kite.ORDER_TYPE_MARKET,
                    )
                )
                exits.append({"symbol": sym, "qty": qty, "order_id": order_id, "success": True})
                _log_order("KILL_SWITCH_EXIT", {"symbol": sym, "qty": qty, "order_id": order_id})
            except Exception as e:
                logger.error(f"Kill switch exit failed for {sym}: {e}")
                exits.append({"symbol": sym, "success": False, "error": str(e)})

        return {
            "success": True,
            "message": f"Kill switch executed — {len(exits)} exits placed",
            "exits": exits,
            "timestamp": datetime.now().isoformat(),
        }


# ─── Module singleton ─────────────────────────────────────────────────────────

_executor: Optional[OrderExecutor] = None


def get_order_executor() -> Optional[OrderExecutor]:
    """
    Return the shared OrderExecutor singleton.
    Returns None if Kite credentials are not set.
    """
    global _executor
    if _executor is None:
        if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
            logger.warning("Kite credentials not set — order execution disabled")
            return None
        try:
            _executor = OrderExecutor()
        except Exception as e:
            logger.error(f"OrderExecutor init failed: {e}")
            return None
    return _executor