"""
brain.py - Weather bot trading brain for New Orleans Kalshi markets

Strategy:
  - Fetch Open-Meteo hourly forecast for New Orleans
  - Fetch open Kalshi markets closing within 12 hours (KXHIGHTNOLA, KXLOWTNOLA)
  - Parse each market's threshold from its title (e.g. "High temp above 88°F")
  - Compute confidence the market resolves Yes or No using forecast + uncertainty model
  - If confidence >= 85%, place a limit order scaled to confidence ($10-$50)
"""

import os
import re
import math
import uuid
import requests
import datetime
import base64
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

API_KEY_ID        = os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH  = os.getenv("KALSHI_PRIVATE_KEY_PATH")
BASE_URL          = "https://api.elections.kalshi.com/trade-api/v2"
CENTRAL           = ZoneInfo("America/Chicago")

CONFIDENCE_THRESHOLD = 0.85   # minimum confidence to trade
MAX_DOLLARS          = 50     # max position size in dollars
MIN_DOLLARS          = 10     # min position size at threshold confidence
WINDOW_HOURS         = 12     # only trade markets closing within this many hours

# Forecast uncertainty model:
# Open-Meteo daily forecasts have ~2-3°F typical error for same-day,
# slightly more for next-day. We use a normal distribution around the
# forecast to compute P(actual > threshold).
FORECAST_STD_DEV = 2.5        # °F — tunable based on observed accuracy

WEATHER_SERIES = [
    ("KXHIGHTNOLA", "high"),
    ("KXLOWTNOLA",  "low"),
]

DRY_RUN = False  # Set True to print orders without submitting


# ── Kalshi auth helpers ────────────────────────────────────────────────────────

def load_private_key(key_path):
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

def _timestamp():
    return str(int(datetime.datetime.now().timestamp() * 1000))

def _sign(private_key, timestamp, method, path):
    path_without_query = path.split("?")[0]
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
    sig = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(sig).decode("utf-8")

def _headers(private_key, method, path):
    ts = _timestamp()
    sign_path = urlparse(BASE_URL + path).path
    return {
        "KALSHI-ACCESS-KEY":       API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": _sign(private_key, ts, method, sign_path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type":            "application/json",
    }

def kalshi_get(private_key, path):
    return requests.get(BASE_URL + path, headers=_headers(private_key, "GET", path))

def kalshi_post(private_key, path, body):
    return requests.post(
        BASE_URL + path,
        headers=_headers(private_key, "POST", path),
        json=body
    )


# ── Weather forecast ───────────────────────────────────────────────────────────

def fetch_forecast():
    """
    Returns today's forecast high and low for New Orleans (°F)
    using Open-Meteo hourly data for a tighter same-day read.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=29.9511&longitude=-90.0715"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=America%2FChicago"
        "&forecast_days=2"
    )
    data = requests.get(url).json()
    times  = data["hourly"]["time"]
    temps  = data["hourly"]["temperature_2m"]

    now        = datetime.datetime.now(CENTRAL)
    today_str  = now.strftime("%Y-%m-%d")
    today_temps = [
        t for ts, t in zip(times, temps)
        if ts.startswith(today_str) and t is not None
    ]

    if not today_temps:
        raise RuntimeError("No forecast data for today.")

    # For hours already past, use actuals-in-forecast as best estimate.
    # For remaining hours, this is still the best we have.
    forecast_high = max(today_temps)
    forecast_low  = min(today_temps)

    print(f"[forecast] Today high={forecast_high:.1f}°F  low={forecast_low:.1f}°F")
    return forecast_high, forecast_low


# ── Market parsing ─────────────────────────────────────────────────────────────

def parse_threshold(title):
    """
    Extract the numeric threshold and direction from a market title.
    Examples:
      "New Orleans high temp above 90°F on May 25" → (90.0, "above")
      "Will New Orleans low be at or below 72°F?"  → (72.0, "at_or_below")
    Returns (threshold_float, direction_str) or None.
    """
    title_lower = title.lower()

    # Direction keywords
    if "at or above" in title_lower or "at or higher" in title_lower:
        direction = "at_or_above"
    elif "above" in title_lower or "higher than" in title_lower or "exceed" in title_lower:
        direction = "above"
    elif "at or below" in title_lower or "at or lower" in title_lower:
        direction = "at_or_below"
    elif "below" in title_lower or "lower than" in title_lower:
        direction = "below"
    else:
        return None

    # Find the temperature number (handles "90", "90°F", "90 °F", "90f")
    match = re.search(r"(\d+(?:\.\d+)?)\s*[°]?\s*f\b", title_lower)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)", title)
    if not match:
        return None

    return float(match.group(1)), direction


def market_closes_within(market, hours):
    """Return True if the market closes within `hours` from now."""
    close_str = market.get("close_time") or market.get("expiration_time")
    if not close_str:
        return False
    try:
        close_dt = datetime.datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        now_utc  = datetime.datetime.now(datetime.timezone.utc)
        delta    = (close_dt - now_utc).total_seconds() / 3600
        return 0 < delta <= hours
    except Exception:
        return False


# ── Confidence & sizing ────────────────────────────────────────────────────────

def normal_cdf(x, mu, sigma):
    """P(X <= x) for X ~ N(mu, sigma)."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def compute_confidence(forecast_temp, threshold, direction):
    """
    Use a normal distribution centred on the forecast to compute
    P(actual resolves Yes) for the given direction.
    """
    mu    = forecast_temp
    sigma = FORECAST_STD_DEV

    if direction in ("above",):
        # P(actual > threshold)
        confidence_yes = 1 - normal_cdf(threshold, mu, sigma)
    elif direction in ("at_or_above",):
        # P(actual >= threshold) ≈ P(actual > threshold - 0.5) for integer °F
        confidence_yes = 1 - normal_cdf(threshold - 0.5, mu, sigma)
    elif direction in ("below",):
        # P(actual < threshold)
        confidence_yes = normal_cdf(threshold, mu, sigma)
    elif direction in ("at_or_below",):
        confidence_yes = normal_cdf(threshold + 0.5, mu, sigma)
    else:
        return None, None

    confidence_no = 1 - confidence_yes

    if confidence_yes >= confidence_no:
        return "yes", confidence_yes
    else:
        return "no", confidence_no

def scale_dollars(confidence):
    """
    Linear scale: 85% confidence → $10, 99%+ confidence → $50.
    """
    low_conf  = CONFIDENCE_THRESHOLD       # 0.85
    high_conf = 0.99
    clamped   = min(max(confidence, low_conf), high_conf)
    frac      = (clamped - low_conf) / (high_conf - low_conf)
    dollars   = MIN_DOLLARS + frac * (MAX_DOLLARS - MIN_DOLLARS)
    return round(dollars, 2)

def dollars_to_contracts(dollars, price_cents):
    """
    On Kalshi, each contract costs price_cents / 100 dollars.
    Number of contracts = floor(budget / cost_per_contract).
    """
    if price_cents <= 0:
        return 0
    cost_per = price_cents / 100
    return max(1, int(dollars // cost_per))


# ── Order placement ────────────────────────────────────────────────────────────

def place_limit_order(private_key, ticker, side, price_cents, num_contracts):
    """
    Place a limit order on Kalshi.
    side: "yes" or "no"
    price_cents: limit price in cents (1-99)
    num_contracts: integer number of contracts
    """
    body = {
        "ticker":        ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":          "limit",
        "action":        "buy",
        "side":          side,
        "count":         num_contracts,
        "yes_price":     price_cents if side == "yes" else (100 - price_cents),
        "no_price":      price_cents if side == "no"  else (100 - price_cents),
    }

    if DRY_RUN:
        print(f"  [DRY RUN] Would place order: {body}")
        return {"dry_run": True, "order": body}

    resp = kalshi_post(private_key, "/portfolio/orders", body)
    return resp.json()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    private_key = load_private_key(PRIVATE_KEY_PATH)
    forecast_high, forecast_low = fetch_forecast()

    for series, temp_type in WEATHER_SERIES:
        forecast_temp = forecast_high if temp_type == "high" else forecast_low
        print(f"\n── {series} (forecast {temp_type}: {forecast_temp:.1f}°F) ──")

        resp    = kalshi_get(private_key, f"/markets?series_ticker={series}&status=open")
        markets = resp.json().get("markets", [])

        if not markets:
            print("  No open markets.")
            continue

        for m in markets:
            ticker = m.get("ticker", "")
            title  = m.get("title", "")

            # 1. Filter: must close within window
            if not market_closes_within(m, WINDOW_HOURS):
                print(f"  SKIP (outside {WINDOW_HOURS}h window): {title}")
                continue

            # 2. Parse threshold from title
            parsed = parse_threshold(title)
            if not parsed:
                print(f"  SKIP (can't parse threshold): {title}")
                continue
            threshold, direction = parsed

            # 3. Compute confidence
            side, confidence = compute_confidence(forecast_temp, threshold, direction)
            if confidence is None or confidence < CONFIDENCE_THRESHOLD:
                print(f"  SKIP (confidence {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}): {title}")
                continue

            # 4. Determine ask price for our chosen side
            yes_ask = m.get("yes_ask")  # cents
            no_ask  = m.get("no_ask")   # cents
            ask_price = yes_ask if side == "yes" else no_ask

            if ask_price is None or ask_price <= 0:
                print(f"  SKIP (no ask price for {side}): {title}")
                continue

            # 5. Scale dollars → contracts
            dollars   = scale_dollars(confidence)
            contracts = dollars_to_contracts(dollars, ask_price)

            print(f"  TRADE: {title}")
            print(f"    Threshold={threshold}°F  Direction={direction}")
            print(f"    Confidence={confidence:.1%}  Side={side.upper()}")
            print(f"    Ask={ask_price}¢  Budget=${dollars:.2f}  Contracts={contracts}")

            # 6. Place limit order (1¢ below ask to get a slightly better fill)
            limit_price = max(1, ask_price - 1)
            result = place_limit_order(private_key, ticker, side, limit_price, contracts)
            print(f"    Order result: {result}")


if __name__ == "__main__":
    run()
