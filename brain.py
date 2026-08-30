"""
brain.py - Weather bot trading brain for New Orleans Kalshi markets

Strategy:
  - Fetch Open-Meteo daily + NWS hourly + Tomorrow.io forecasts for New Orleans
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

CONFIDENCE_THRESHOLD = 0.80
MAX_DOLLARS          = 50
MIN_DOLLARS          = 10
WINDOW_HOURS         = 18
FORECAST_STD_DEV     = 2.5
MIN_ASK_CENTS        = 10
MAX_ASK_CENTS        = 99
TOMORROW_API_KEY     = "RQJDkNtidWYYhmo7GwQweWB38eEzTFGv"

WEATHER_SERIES = [
    ("KXHIGHTNOLA", "high"),
    ("KXLOWTNOLA",  "low"),
]

DRY_RUN = True


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
    Uses limit=200 to capture overnight readings.
    Low temp only uses readings from 12am-6am and 10pm-11:59pm.
    """
    try:
        url = "https://api.weather.gov/stations/KMSY/observations?limit=200"
        headers = {"User-Agent": "weather-bot/1.0"}
        resp = requests.get(url, headers=headers, timeout=10).json()
        features = resp.get("features", [])
        now = datetime.datetime.now(CENTRAL)
        today_str = now.strftime("%Y-%m-%d")

        all_temps = []
        low_temps = []

        for f in features:
            ts_raw = f["properties"]["timestamp"]
            temp_c = f["properties"]["temperature"]["value"]
            if temp_c is None:
                continue
            try:
                ts_dt = datetime.datetime.fromisoformat(
                    ts_raw.replace("Z", "+00:00")
                ).astimezone(CENTRAL)
            except Exception:
                continue
            if ts_dt.strftime("%Y-%m-%d") != today_str:
                continue
            temp_f = temp_c * 9 / 5 + 32
            all_temps.append(temp_f)
            hour = ts_dt.hour
            if hour <= 6 or hour >= 22:
                low_temps.append(temp_f)

        if not all_temps:
            return None, None

        observed_high = max(all_temps)
        observed_low  = min(low_temps) if low_temps else None

        if observed_low is not None:
            log.info(f"[observed] high={observed_high:.1f}F  low={observed_low:.1f}F  ({len(all_temps)} readings, {len(low_temps)} overnight)")
        else:
            log.info(f"[observed] high={observed_high:.1f}F  low=None (no overnight readings yet)  ({len(all_temps)} readings)")

        return observed_high, observed_low

    except Exception as e:
        log.info(f"[observed] FAILED: {e}")
        return None, None


def fetch_tomorrow_forecast():
    """
    Fetch today's forecast high and low from Tomorrow.io.
    """
    try:
        url = (
            "https://api.tomorrow.io/v4/weather/forecast"
            "?location=29.9511,-90.0715"
            "&timesteps=1d"
            "&units=imperial"
            f"&apikey={TOMORROW_API_KEY}"
        )
        resp = requests.get(url, timeout=10).json()
        today_str = datetime.datetime.now(CENTRAL).strftime("%Y-%m-%d")
        timelines = resp["timelines"]["daily"]
        for day in timelines:
            if day["time"].startswith(today_str):
                high = day["values"]["temperatureMax"]
                low  = day["values"]["temperatureMin"]
                log.info(f"[tomorrow.io] high={high:.1f}F  low={low:.1f}F")
                return high, low
        log.info("[tomorrow.io] No data for today")
        return None, None
    except Exception as e:
        log.info(f"[tomorrow.io] FAILED: {e}")
        return None, None


def fetch_forecast():
    """
    Returns today's forecast high and low for New Orleans in degrees F.
    Sources: Open-Meteo (tiebreaker) + NWS hourly + Tomorrow.io + Visual Crossing
    blended with observed temps.
    """
    # Source 1: Open-Meteo daily (tiebreaker)
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

    # Source 3: Tomorrow.io
    tomorrow_high, tomorrow_low = fetch_tomorrow_forecast()

    # Source 4: Visual Crossing
    vc_high, vc_low = fetch_visual_crossing_forecast()

    # Average sources — NWS, Tomorrow.io, Visual Crossing primary; Open-Meteo tiebreaker
    highs = [h for h in [nws_high, tomorrow_high, vc_high, ensemble_high] if h is not None]
    lows  = [l for l in [nws_low,  tomorrow_low,  vc_low,  ensemble_low]  if l is not None]
    if not highs or not lows:
        raise RuntimeError("All forecast sources failed.")
    forecast_high = sum(highs) / len(highs)
    forecast_low  = sum(lows)  / len(lows)
    log.info(f"[forecast avg] high={forecast_high:.1f}F  low={forecast_low:.1f}F")

    # Blend with observed
    observed_high, observed_low = fetch_observed_high_low()
    hour = datetime.datetime.now(CENTRAL).hour
    high_obs_weight = min(1.0, hour / 15)

    if observed_high is not None:
        forecast_high = high_obs_weight * observed_high + (1 - high_obs_weight) * forecast_high

    # For low: only fully trust observed after 10pm when overnight is complete
    if observed_low is not None:
        if hour >= 22:
            forecast_low = observed_low
            log.info(f"[using observed low] {forecast_low:.1f}F (overnight complete)")
        elif hour >= 18:
            forecast_low = 0.7 * observed_low + 0.3 * forecast_low
            log.info(f"[partial observed low] {forecast_low:.1f}F (70% observed)")
        else:
            forecast_low = 0.3 * observed_low + 0.7 * forecast_low
            log.info(f"[partial observed low] {forecast_low:.1f}F (30% observed)")

    log.info(f"[blended] high={forecast_high:.1f}F  low={forecast_low:.1f}F")
    return forecast_high, forecast_low
# ── Market parsing ─────────────────────────────────────────────────────────────

def parse_threshold(title):
    title_lower = title.lower()

    range_match = re.search(r"(\d+)\s*[-]\s*(\d+)\s*", title)
    if range_match:
        low  = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low, high, "range")

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
    low_conf  = CONFIDENCE_THRESHOLD
    high_conf = 0.99
    clamped   = min(max(confidence, low_conf), high_conf)
    frac      = (clamped - low_conf) / (high_conf - low_conf)
    return round(MIN_DOLLARS + frac * (MAX_DOLLARS - MIN_DOLLARS), 2)


def dollars_to_contracts(dollars, price_cents):
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
    forecast_high = round(forecast_high)
    forecast_low  = round(forecast_low)
    hour = datetime.datetime.now(CENTRAL).hour
    global FORECAST_STD_DEV
    if hour >= 22:
        FORECAST_STD_DEV = 0.5   # low is locked in
    elif hour >= 18:
        FORECAST_STD_DEV = 1.0   # high is settled, low nearly there
    elif hour >= 14:
        FORECAST_STD_DEV = 1.5   # high nearly settled
    else:
        FORECAST_STD_DEV = 2.5   # morning, wide uncertainty    log.info(f"[using] high={forecast_high}F  low={forecast_low}F  std_dev={FORECAST_STD_DEV}F")

    for series, temp_type in WEATHER_SERIES:
        forecast_temp = forecast_high if temp_type == "high" else forecast_low
        log.info(f"\n-- {series} (forecast {temp_type}: {forecast_temp}F) --")

        resp    = kalshi_get(private_key, f"/markets?series_ticker={series}&status=open")
        markets = resp.json().get("markets", [])

        if not markets:
            log.info("  No open markets.")
            continue

        for m in markets:
            ticker = m.get("ticker", "")
            title  = m.get("title", "")

            if not market_closes_within(m, WINDOW_HOURS):
                log.info(f"  SKIP (outside {WINDOW_HOURS}h window): {title}")
                continue

            parsed = parse_threshold(title)
            if not parsed:
                log.info(f"  SKIP (cant parse threshold): {title}")
                continue
            if len(parsed) == 3:
                threshold = (parsed[0], parsed[1])
                direction = parsed[2]
            else:
                threshold, direction = parsed

            side, confidence = compute_confidence(forecast_temp, threshold, direction)
            if confidence is None or confidence < CONFIDENCE_THRESHOLD:
                log.info(f"  SKIP (confidence {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}): {title}")
                continue

            yes_ask = m.get("yes_ask_dollars")
            no_ask  = m.get("no_ask_dollars")
            raw = yes_ask if side == "yes" else no_ask
            ask_price = round(float(raw) * 100) if raw else None

            if ask_price is None or ask_price < MIN_ASK_CENTS or ask_price > MAX_ASK_CENTS:
                log.info(f"  SKIP (ask {ask_price}c out of range for {side}): {title}")
                continue

            dollars   = scale_dollars(confidence)
            contracts = dollars_to_contracts(dollars, ask_price)

            log.info(f"  TRADE: {title}")
            log.info(f"    Threshold={threshold}  Direction={direction}")
            log.info(f"    Confidence={confidence:.1%}  Side={side.upper()}")
            log.info(f"    Ask={ask_price}c  Budget=${dollars:.2f}  Contracts={contracts}")

            limit_price = max(1, ask_price - 1)
            result = place_limit_order(private_key, ticker, side, limit_price, contracts)
            log.info(f"    Order result: {result}")

    log.info("===== Bot run complete =====")


if __name__ == "__main__":
    run()