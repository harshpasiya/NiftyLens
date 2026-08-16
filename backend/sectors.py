"""
sectors.py — NiftyLens Volatile Universe & Sector Mappings
===========================================================
200 high-volatility NSE mid/small-cap stocks curated for swing & intraday.

SELECTION CRITERIA:
  ✗ NO mega large-caps (RELIANCE, TCS, HDFCBANK, HINDUNILVR, etc.)
    — they have volume but 0.5% daily moves = useless for trading
  ✓ Mid-caps & small-caps with 2-5% avg daily range
  ✓ F&O stocks preferred (liquidity + tight spreads)
  ✓ Sectors with momentum: defense, PSU banks, power, railways,
    chemicals, pharma mid-caps, auto ancillaries, real estate, IT mid-caps

USAGE:
    from sectors import VOLATILE_UNIVERSE, SECTOR_MAP, get_sector
"""

# ─── Sector-to-stock mapping ─────────────────────────────────────────────────
# Each sector contains volatile, tradeable mid/small-cap stocks

SECTORS: dict[str, list[str]] = {
    # ── Defense & Aerospace ──────────────────────────────────────────────────
    # High govt spending, event-driven swings, strong momentum plays
    "Defense": [
        "BEL", "HAL", "BEML", "COCHINSHIP", "MAZDOCK", "GRSE",
        "BDL", "DATAPATTNS", "PARAS", "SOLARINDS", "MIDHANI",
    ],

    # ── Railways & Infra ─────────────────────────────────────────────────────
    # Capex-driven, budget plays, 3-8% daily swings common
    "Railways": [
        "RVNL", "IRCON", "TITAGARH", "RAILTEL", "IRFC",
        "RITES", "IRCTC", "JUPITERWAY", "TEXRAIL",
    ],

    # ── Power & Energy ───────────────────────────────────────────────────────
    # Renewable energy momentum, PSU re-rating plays
    "Power": [
        "TATAPOWER", "JSWENERGY", "ADANIPOWER", "ADANIGREEN", "SUZLON",
        "NHPC", "SJVN", "TORNTPOWER", "CESC", "RPOWER",
        "INOXWIND", "WAAREENER", "KPIGLOBAL",
    ],

    # ── PSU Banks ────────────────────────────────────────────────────────────
    # NPA recovery plays, high beta to Nifty, 2-4% daily moves
    "PSU Banks": [
        "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IOB",
        "INDIANB", "MAHABANK", "CENTRALBK", "UCOBANK", "PSB",
        "BANKINDIA",
    ],

    # ── Private Banks (Mid-cap) ──────────────────────────────────────────────
    # Higher volatility than HDFC/ICICI, still liquid
    "Pvt Banks": [
        "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK",
        "AUBANK", "KARURVYSYA", "CITYUNIONBK", "DCBBANK", "EQUITASBNK",
    ],

    # ── NBFCs & Financials ───────────────────────────────────────────────────
    # Rate-sensitive, high beta, earnings-driven swings
    "NBFC": [
        "CHOLAFIN", "MANAPPURAM", "MUTHOOTFIN", "POONAWALLA",
        "RECLTD", "PFC", "IREDA", "LICHSGFIN",
        "CANFINHOME", "IIFL", "JMFINANCIL",
    ],

    # ── Capital Goods & Engineering ──────────────────────────────────────────
    # Order book driven, capex cycle plays
    "Capital Goods": [
        "CUMMINSIND", "SIEMENS", "ABB", "CGPOWER", "KEI",
        "POLYCAB", "APARINDS", "THERMAX", "TRIVENI",
        "AIAENG", "GPIL", "TTML",
    ],

    # ── Metals & Mining ──────────────────────────────────────────────────────
    # Commodity-driven, global macro sensitive, high daily ATR%
    "Metals": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "SAIL",
        "NATIONALUM", "VEDL", "NMDC", "COALINDIA",
        "JINDALSTEL", "RATNAMANI", "WELCORP",
    ],

    # ── Auto & Auto Ancillaries ──────────────────────────────────────────────
    # Monthly sales data catalysts, supplier chain momentum
    "Auto": [
        "TATAMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSON",
        "TVSMOTOR", "EICHERMOT", "HEROMOTOCO", "EXIDEIND",
        "AMARAJABAT", "SUNDRMFAST", "ENDURANCE", "CRAFTSMAN",
    ],

    # ── Pharma (Mid/Small) ───────────────────────────────────────────────────
    # USFDA events, ANDA approvals, 3-7% event swings
    "Pharma": [
        "LUPIN", "AUROPHARMA", "LAURUSLABS", "GLENMARK",
        "GRANULES", "NATCOPHARMA", "LALPATHLAB", "IPCALAB",
        "AJANTPHARM", "ALKEM", "GLAND", "MANKIND",
    ],

    # ── Chemicals & Specialty ────────────────────────────────────────────────
    # China+1 plays, capacity expansion cycles
    "Chemicals": [
        "DEEPAKNTR", "AARTIIND", "SRF", "PIDILITIND",
        "CLEAN", "GALAXYSURF", "VINATIORGN", "FINEORG",
        "ANANTRAJ", "NOCIL", "TATACHEM",
    ],

    # ── IT (Mid/Small) ──────────────────────────────────────────────────────
    # Deal wins, margin surprise plays — NOT the boring large-cap IT
    "IT Mid": [
        "COFORGE", "MPHASIS", "PERSISTENT", "LTTS",
        "HAPPSTMNDS", "TATAELXSI", "CYIENT", "ROUTE",
        "MASTEK", "BSOFT", "SONATSOFTW",
    ],

    # ── Real Estate ──────────────────────────────────────────────────────────
    # Interest rate plays, booking data, 3-5% daily common
    "Realty": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE",
        "PHOENIXLTD", "BRIGADE", "SOBHA", "LODHA",
        "SUNTECK",
    ],

    # ── Consumer & Retail ────────────────────────────────────────────────────
    # New-age consumer, FMCG disruptors, earnings swings
    "Consumer": [
        "TRENT", "DMART", "ZOMATO", "NYKAA", "PAYTM",
        "POLICYBZR", "DELHIVERY", "MAPMYINDIA",
        "MEDANTA", "STARHEALTH",
    ],

    # ── Telecom & Media ──────────────────────────────────────────────────────
    "Telecom": [
        "BHARTIARTL", "IDEA", "TATACOMM",
        "ROUTE", "NAZARA",
    ],

    # ── Textiles & Misc ──────────────────────────────────────────────────────
    "Textiles": [
        "PAGEIND", "RAJESHEXPO", "TRIDENT",
        "WELSPUNLIV", "ARVIND", "KPRMILL",
    ],

    # ── Fertilizer & Agri ────────────────────────────────────────────────────
    # Seasonal plays, govt subsidy events, monsoon driven
    "Fertilizer": [
        "CHAMBLFERT", "GNFC", "GSFC", "COROMANDEL",
        "RCF", "FACT", "NFL", "PIIND",
        "UPL", "DHANUKA", "RALLIS",
    ],

    # ── Infrastructure & Construction ────────────────────────────────────────
    "Infra": [
        "LTIM", "NBCC", "NCC", "KEC",
        "PNCINFRA", "JKCEMENT", "RAMCOCEM",
        "JKLAKSHMI", "HEIDELBERG",
    ],

    # ── Oil & Gas (Mid) ──────────────────────────────────────────────────────
    "Oil & Gas": [
        "BPCL", "IOC", "GAIL", "PETRONET",
        "GSPL", "GUJGASLTD", "IGL", "MGL",
    ],

    # ── Insurance & AMC ──────────────────────────────────────────────────────
    "Insurance": [
        "SBILIFE", "HDFCLIFE", "ICICIPRULI",
        "NIACL", "GICRE",
    ],

    # ── Sugar & Ethanol ──────────────────────────────────────────────────────
    # Seasonal, govt policy driven, 4-8% daily swings
    "Sugar": [
        "BALRAMCHIN", "RENUKA", "TRIVENI", "DALMIASUGAR",
    ],
}


# ─── Flat universe (all stocks, deduplicated) ────────────────────────────────

def _build_universe() -> list[str]:
    """Build deduplicated list preserving sector order."""
    seen = set()
    universe = []
    for stocks in SECTORS.values():
        for s in stocks:
            if s not in seen:
                seen.add(s)
                universe.append(s)
    return universe


VOLATILE_UNIVERSE: list[str] = _build_universe()

# ─── Reverse lookup: stock → sector ──────────────────────────────────────────

SECTOR_MAP: dict[str, str] = {}
for sector_name, stocks in SECTORS.items():
    for s in stocks:
        if s not in SECTOR_MAP:  # first sector wins if duplicated
            SECTOR_MAP[s] = sector_name


def get_sector(symbol: str) -> str:
    """Return sector name for a stock, or 'Other' if not mapped."""
    return SECTOR_MAP.get(symbol.upper(), "Other")


def get_sector_stocks(sector: str) -> list[str]:
    """Return all stocks in a given sector."""
    return SECTORS.get(sector, [])


def get_all_sectors() -> list[str]:
    """Return sorted list of sector names."""
    return sorted(SECTORS.keys())


# ─── Sector-level news blockers ──────────────────────────────────────────────
# If sector-level bad news hits (e.g. RBI policy for banks, USFDA for pharma),
# block the entire sector instead of checking each stock individually

SECTOR_NEWS_KEYWORDS: dict[str, list[str]] = {
    "PSU Banks":  ["RBI policy", "NPA crisis", "bank fraud", "moratorium"],
    "Pvt Banks":  ["RBI policy", "NPA", "bank fraud"],
    "NBFC":       ["RBI regulation", "NBFC crisis", "liquidity crunch"],
    "Pharma":     ["USFDA warning", "FDA import alert", "drug recall"],
    "Metals":     ["China dumping", "export duty", "import duty metals"],
    "Power":      ["coal shortage", "power crisis", "tariff revision"],
    "Realty":     ["RERA violation", "real estate crisis", "housing bubble"],
    "IT Mid":     ["H1B visa", "tech layoffs India", "IT spending cut"],
    "Oil & Gas":  ["crude oil crash", "OPEC cut", "fuel subsidy"],
}


# ─── Quick stats ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"NiftyLens Volatile Universe: {len(VOLATILE_UNIVERSE)} stocks")
    print(f"Sectors: {len(SECTORS)}")
    print()
    for sector, stocks in SECTORS.items():
        print(f"  {sector:20s} → {len(stocks):3d} stocks  {', '.join(stocks[:5])}...")
    print(f"\n  Total unique stocks: {len(VOLATILE_UNIVERSE)}")
