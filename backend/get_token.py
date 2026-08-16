"""
get_token.py  —  Zerodha Kite Daily Access Token Refresher
===========================================================
Run this every morning before starting screener_engine.py.
Zerodha resets all access tokens at 6 AM daily.

Usage:
    cd NiftyLens/backend
    python get_token.py

What it does:
    1. Reads KITE_API_KEY and KITE_API_SECRET from .env
    2. Prints the Zerodha login URL — open it in your browser
    3. You log in — Zerodha redirects to a URL like:
           http://127.0.0.1:5000/?request_token=xxxxxxxx&action=login&status=success
    4. That page will show an error (connection refused) — that is NORMAL
    5. Copy the full URL from your browser address bar and paste it here
    6. Script extracts the request_token, generates an access_token, and saves to .env

Requirements:
    pip install kiteconnect python-dotenv
"""

import os
import sys
import webbrowser
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

# ── Config ────────────────────────────────────────────────────────────────────

ENV_FILE = Path(__file__).parent / ".env"

# ── Load .env ─────────────────────────────────────────────────────────────────

if not ENV_FILE.exists():
    print(f"\n❌  .env file not found at: {ENV_FILE}")
    print("    Create it from .env.example and fill in your keys.\n")
    sys.exit(1)

load_dotenv(ENV_FILE)

API_KEY    = os.getenv("KITE_API_KEY", "").strip()
API_SECRET = os.getenv("KITE_API_SECRET", "").strip()

if not API_KEY or API_KEY == "your_kite_api_key_here":
    print("\n❌  KITE_API_KEY is missing or not filled in your .env file.")
    print("    Get it from: developers.kite.trade → My Apps\n")
    sys.exit(1)

if not API_SECRET or API_SECRET == "your_kite_api_secret_here":
    print("\n❌  KITE_API_SECRET is missing or not filled in your .env file.")
    print("    Get it from: developers.kite.trade → My Apps\n")
    sys.exit(1)

# ── Helper — extract request_token from a full redirect URL ──────────────────

def extract_request_token(raw: str) -> str:
    """
    Accepts any of these formats and returns the raw request_token string:
      - Full URL : http://127.0.0.1:5000/?request_token=abc123&action=login&status=success
      - Just the query string : ?request_token=abc123&action=login&status=success
      - Just the token itself : abc123
    Returns empty string if nothing useful is found.
    """
    raw = raw.strip()
    if not raw:
        return ""

    # If it looks like a URL or query string, parse it properly
    if "request_token" in raw:
        # Ensure it can be parsed — prefix with a dummy scheme if bare query string
        if raw.startswith("?"):
            raw = "http://dummy" + raw
        elif not raw.startswith("http"):
            raw = "http://dummy/?" + raw.lstrip("&")
        parsed = urlparse(raw)
        params = parse_qs(parsed.query)
        tokens = params.get("request_token", [])
        return tokens[0].strip() if tokens else ""

    # Assume user pasted the raw token directly
    return raw


# ── Main flow ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 62)
    print("  🔑  Zerodha Kite — Daily Token Refresh")
    print("═" * 62)
    print(f"  API Key : {API_KEY[:8]}{'*' * max(0, len(API_KEY) - 8)}")
    print(f"  .env    : {ENV_FILE}")
    print("═" * 62)

    # Step 1 — Build login URL and open browser
    kite = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()

    print("\n  STEP 1 — Open this URL in your browser and log in:")
    print(f"\n  {login_url}\n")
    webbrowser.open(login_url)

    # Step 2 — Instruct user what to do after login
    print("  ─" * 31)
    print("  STEP 2 — After logging in, your browser will redirect to a URL")
    print("  that looks like this (the page itself may show a connection error):")
    print()
    print("  http://127.0.0.1:5000/?request_token=xxxxxxxxxxxxxxxx&action=login&status=success")
    print()
    print("  Copy that ENTIRE URL from the browser address bar.")
    print("  ─" * 31)

    # Step 3 — Accept the URL via terminal input
    request_token = ""
    attempts = 0
    while not request_token:
        attempts += 1
        if attempts > 3:
            print("\n❌  Too many invalid attempts. Please re-run the script.\n")
            sys.exit(1)

        print()
        raw_input = input("  Paste the full redirect URL here and press Enter:\n  > ").strip()

        if not raw_input:
            print("  ⚠️   Nothing was pasted. Please try again.")
            continue

        request_token = extract_request_token(raw_input)

        if not request_token:
            print("  ⚠️   Could not find a request_token in what you pasted.")
            print("       Make sure you copied the full URL from the browser address bar.")

    print(f"\n  ✓  request_token extracted : {request_token[:12]}{'*' * max(0, len(request_token) - 12)}")

    # Step 4 — Exchange request_token for access_token
    print("  ✓  Generating session with Zerodha...")
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
    except Exception as exc:
        print(f"\n❌  Failed to generate session: {exc}")
        print("    Possible causes:")
        print("    • Wrong KITE_API_SECRET in .env")
        print("    • request_token already used (each token is single-use)")
        print("    • Token expired (must be used within a few minutes of login)\n")
        sys.exit(1)

    access_token = data.get("access_token", "")
    user_name    = data.get("user_name", "Unknown")
    user_id      = data.get("user_id", "")

    if not access_token:
        print("\n❌  Session created but access_token was empty. Contact Zerodha support.\n")
        sys.exit(1)

    # Step 5 — Write the new token back to .env
    set_key(str(ENV_FILE), "KITE_ACCESS_TOKEN", access_token)
    print(f"  ✓  KITE_ACCESS_TOKEN saved to .env")

    # Step 6 — Quick verification ping
    kite.set_access_token(access_token)
    try:
        profile = kite.profile()
        verified_name = profile.get("user_name", user_name)
        verified_id   = profile.get("user_id", user_id)
    except Exception:
        verified_name = user_name   # Non-fatal — token is saved regardless
        verified_id   = user_id

    print("\n" + "═" * 62)
    print("  ✅  Token refreshed successfully!")
    print(f"  👤  Logged in as : {verified_name} ({verified_id})")
    print(f"  🔑  Token saved  : KITE_ACCESS_TOKEN in .env")
    print("═" * 62)
    print("\n  You can now start the backend:")
    print("  uvicorn screener_engine:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()