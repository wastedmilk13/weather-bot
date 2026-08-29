"""
brain.py - Weather bot trading brain for New Orleans Kalshi markets

Strategy:
  - Fetch Open-Meteo daily forecast for New Orleans
  - Fetch open Kalshi markets closing within 18 hours (KXHIGHTNOLA, KXLOWTNOLA)
  - Parse each market's threshold from its title
  - Compute confidence the market resolves Yes or No using forecast + uncertainty model
  - If confidence >= 80%, place a limit order scaled to confidence ($10-$50)
"""

import logging
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

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

# ── Config ─────────────────────────────────────────────────────────────────────

API_KEY_ID        = os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH  = os.getenv("KALSHI_PRIVATE_KEY_PATH")
BASE_URL          = "https://api.elections.kalshi.com/trade-api/v2"
CENTRAL           = ZoneInfo("America/Chicago")

CONFIDENCE_THRESHOLD = 0.80   # minimum confidence to trade
MAX_DOLLARS          = 50     # max position size in dollars
MIN_DOLLARS          = 10     # min position size at threshold confidence
WINDOW_HOURS         = 18     # only trade markets closing within this many hours
FORECAST_STD_DEV     = 2.5    # degrees F uncertainty model standard deviation
MIN_ASK_CENTS        = 10     # skip trades where ask < 10 cents
MAX_ASK_CENTS        = 98     # skip trades where ask > 98 cents

WEATHER_SERIES = [
    ("KXHIGHTNOLA", "high"),
    ("KXLOWTNOLA",  "low"),
]

DRY_RUN = True  # Set True to simulate orders without submitting


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

def fetch_observed_high_low():
    """
    Fetch today's observed high and low from NWS station KMSY.
    Uses limit=48 to capture overnight lows even during afternoon runs.
    """
    try:
        url = "https://api.weather.gov/stations/KMSY/observations?limit=48"
        headers = {"User-Agent": "weather-bot/1.0"}
        resp = requests.get(url, headers=headers, timeout=10).json()
        features = resp.get("features", [])
        now = datetime.datetime.now(CENTRAL)
        today_str = now.strftime("%Y-%m-%d")
        temps = []
        for f in features:
            ts = f["properties"]["timestamp"]
            temp_c = f["properties"]["temperature"]["value"]
            if ts.startswith(today_str) and temp_c is not None:
                temp_f = temp_c * 9 / 5 + 32
                temps.append(temp_f)
        if not temps:
            return None, None
        observed_high = max(temps)
        observed_low  = min(temps)
        log.info(f"[observed] high={observed_high:.1f}F  low={observed_low:.1f}F  ({len(temps)} readings)")
        return observed_high, observed_low
    except Exception as e:
        log.info(f"[observed] FAILED: {e}")
        return None, None


def fetch_forecast():
    """
    Returns today's forecast high and low for New Orleans in degrees F.
    Sources: Open-Meteo daily + NWS hourly, blended with observed temps.
    Kalshi now uses The Weather Company for resolution.
    """
    # Source 1: Open-Meteo daily
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=29.9511&longitude=-90.0715"
            "&daily=temperature_2m_max,temperature_2m_min"
            "&temperature_unit=fahrenheit"
            "&timezone=America%2FChicago"
            "&forecast_days=2"
        )
        data = requests.get(url, timeout=10).json()
        today_str = datetime.datetime.now(CENTRAL).strftime("%Y-%m-%d")
        dates = data["daily"]["time"]
        highs = data["daily"]["temperature_2m_max"]
        lows  = data["daily"]["temperature_2m_min"]
        idx = dates.index(today_str) if today_str in dates else 0
        ensemble_high = highs[idx]
        ensemble_low  = lows[idx]
        log.info(f"[open-meteo] high={ensemble_high:.1f}F  low={ensemble_low:.1f}F")
    except Exception as e:
        log.info(f"[open-meteo] FAILED: {e}")
        ensemble_high = ensemble_low = None

    # Source 2: NWS Hourly Forecast
    try:
        points_url = "https://api.weather.gov/points/29.9511,-90.0715"
        headers = {"User-Agent": "weather-bot/1.0"}
        points = requests.get(points_url, headers=headers, timeout=10).json()
        forecast_url = points["properties"]["forecastHourly"]
        forecast = requests.get(forecast_url, headers=headers, timeout=10).json()
        periods = forecast["properties"]["periods"]
        now = datetime.datetime.now(CENTRAL)
        today_str = now.strftime("%Y-%m-%d")
        nws_temps = [
            p["temperature"] for p in periods
            if p["startTime"].startswith(today_str)
            and p["temperatureUnit"] == "F"
        ]
        nws_high = max(nws_temps) if nws_temps else None
        nws_low  = min(nws_temps) if nws_temps else None
        log.info(f"[nws forecast] high={nws_high:.1f}F  low={nws_low:.1f}F")
    except Exception as e:
        log.info(f"[nws forecast] FAILED: {e}")
        nws_high = nws_low = None

    # Average forecast sources
    highs = [h for h in [ensemble_high, nws_high] if h is not None]
    lows  = [l for l in [ensemble_low,  nws_low]  if l is not None]
    if not highs or not lows:
        raise RuntimeError("All forecast sources failed.")
    forecast_high = sum(highs) / len(highs)
    forecast_low  = sum(lows)  / len(lows)
    log.info(f"[forecast avg] high={forecast_high:.1f}F  low={forecast_low:.1f}F")

    # Blend with observed
    observed_high, observed_low = fetch_observed_high_low()
    if observed_high is not None:
        hour = datetime.datetime.now(CENTRAL).hour
        # High: trust observed more as afternoon progresses
        high_obs_weight = min(1.0, hour / 15)
        # Low: trust observed more in morning, less in afternoon
        low_obs_weight = min(1.0, hour / 8) if hour <= 8 else max(0.3, 1 - (hour - 8) / 16)
        blended_high = high_obs_weight * observed_high + (1 - high_obs_weight) * forecast_high
        blended_low  = low_obs_weight  * observed_low  + (1 - low_obs_weight)  * forecast_low
        log.info(f"[blended] high={blended_high:.1f}F (obs={high_obs_weight:.0%})  low={blended_low:.1f}F (obs={low_obs_weight:.0%})")
        return blended_high, blended_low

    return forecast_high, forecast_low


# ── Market parsing ─────────────────────────────────────────────────────────────

def parse_threshold(title):
    title_lower = title.lower()

    # Range markets like "83-84 degrees"
    range_match = re.search(r"(\d+)\s*[-]\s*(\d+)\s*", title)
    if range_match:
        low  = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low, high, "range")

    # Direction from symbols or words
    if ">" in title or "above" in title_lower or "higher" in title_lower:
        direction = "above"
    elif "<" in title or "below" in title_lower or "lower" in title_lower:
        direction = "below"
    else:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", title)
    if not match:
        return None
    return float(match.group(1)), direction


def market_closes_within(market, hours):
    """Return True if the market closes within hours from now."""
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


def compute_confidence(forecast_temp, threshold, direction=None):
    mu    = forecast_temp
    sigma = FORECAST_STD_DEV

    if direction == "range":
        low, high = threshold
        confidence_yes = normal_cdf(high + 0.5, mu, sigma) - normal_cdf(low - 0.5, mu, sigma)
        confidence_no  = 1 - confidence_yes
        return ("yes", confidence_yes) if confidence_yes >= confidence_no else ("no", confidence_no)

    if direction == "above":
        confidence_yes = 1 - normal_cdf(threshold, mu, sigma)
    elif direction == "at_or_above":
        confidence_yes = 1 - normal_cdf(threshold - 0.5, mu, sigma)
    elif direction == "below":
        confidence_yes = normal_cdf(threshold, mu, sigma)
    elif direction == "at_or_below":
        confidence_yes = normal_cdf(threshold + 0.5, mu, sigma)
    else:
        return None, None

    confidence_no = 1 - confidence_yes
    return ("yes", confidence_yes) if confidence_yes >= confidence_no else ("no", confidence_no)


def scale_dollars(confidence):
    """Linear scale: 80% confidence maps to $10, 99%+ maps to $50."""
    low_conf  = CONFIDENCE_THRESHOLD
    high_conf = 0.99
    clamped   = min(max(confidence, low_conf), high_conf)
    frac      = (clamped - low_conf) / (high_conf - low_conf)
    return round(MIN_DOLLARS + frac * (MAX_DOLLARS - MIN_DOLLARS), 2)


def dollars_to_contracts(dollars, price_cents):
    """Number of contracts = floor(budget / cost_per_contract)."""
    if price_cents <= 0:
        return 0
    cost_per = price_cents / 100
    return max(1, int(dollars // cost_per))


# ── Order placement ────────────────────────────────────────────────────────────

def place_limit_order(private_key, ticker, side, price_cents, num_contracts):
    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          "buy",
        "side":            side,
        "count":           num_contracts,
        "yes_price":       price_cents if side == "yes" else (100 - price_cents),
        "no_price":        price_cents if side == "no"  else (100 - price_cents),
    }

    if DRY_RUN:
        log.info(f"  [DRY RUN] Would place order: {body}")
        return {"dry_run": True, "order": body}

    resp = kalshi_post(private_key, "/portfolio/orders", body)
    return resp.json()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    log.info("===== Bot run started =====")
    private_key = load_private_key(PRIVATE_KEY_PATH)
    forecast_high, forecast_low = fetch_forecast()
    log.info(f"[using] high={forecast_high:.1f}F  low={forecast_low:.1f}F  std_dev={FORECAST_STD_DEV}F")

    for series, temp_type in WEATHER_SERIES:
        forecast_temp = forecast_high if temp_type == "high" else forecast_low
        log.info(f"\n-- {series} (forecast {temp_type}: {forecast_temp:.1f}F) --")

        resp    = kalshi_get(private_key, f"/markets?series_ticker={series}&status=open")
        markets = resp.json().get("markets", [])

        if not markets:
            log.info("  No open markets.")
            continue

        for m in markets:
            ticker = m.get("ticker", "")
            title  = m.get("title", "")

            # 1. Filter: must close within window
            if not market_closes_within(m, WINDOW_HOURS):
                log.info(f"  SKIP (outside {WINDOW_HOURS}h window): {title}")
                continue

            # 2. Parse threshold from title
            parsed = parse_threshold(title)
            if not parsed:
                log.info(f"  SKIP (cant parse threshold): {title}")
                continue
            if len(parsed) == 3:
                threshold = (parsed[0], parsed[1])
                direction = parsed[2]
            else:
                threshold, direction = parsed

            # 3. Compute confidence
            side, confidence = compute_confidence(forecast_temp, threshold, direction)
            if confidence is None or confidence < CONFIDENCE_THRESHOLD:
                log.info(f"  SKIP (confidence {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}): {title}")
                continue

            # 4. Determine ask price for our chosen side
            yes_ask = m.get("yes_ask_dollars")
            no_ask  = m.get("no_ask_dollars")
            raw = yes_ask if side == "yes" else no_ask
            ask_price = round(float(raw) * 100) if raw else None

            if ask_price is None or ask_price < MIN_ASK_CENTS or ask_price > MAX_ASK_CENTS:
                log.info(f"  SKIP (ask {ask_price}c out of range for {side}): {title}")
                continue

            # 5. Scale dollars to contracts
            dollars   = scale_dollars(confidence)
            contracts = dollars_to_contracts(dollars, ask_price)

            log.info(f"  TRADE: {title}")
            log.info(f"    Threshold={threshold}  Direction={direction}")
            log.info(f"    Confidence={confidence:.1%}  Side={side.upper()}")
            log.info(f"    Ask={ask_price}c  Budget=${dollars:.2f}  Contracts={contracts}")

            # 6. Place limit order 1 cent below ask for better fill
            limit_price = max(1, ask_price - 1)
            result = place_limit_order(private_key, ticker, side, limit_price, contracts)
            log.info(f"    Order result: {result}")

    log.info("===== Bot run complete =====")


if __name__ == "__main__":
    run()