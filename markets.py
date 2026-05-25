import os
import requests
import datetime
import base64
from urllib.parse import urlparse
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()

API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def load_private_key(key_path):
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

def create_signature(private_key, timestamp, method, path):
    path_without_query = path.split("?")[0]
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode("utf-8")

def get(private_key, path):
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    sign_path = urlparse(BASE_URL + path).path
    signature = create_signature(private_key, timestamp, "GET", sign_path)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp
    }
    return requests.get(BASE_URL + path, headers=headers)

WEATHER_SERIES = [
    ("KXHIGHTNOLA", "New Orleans High Temp"),
    ("KXLOWTNOLA",  "New Orleans Low Temp"),
]

private_key = load_private_key(PRIVATE_KEY_PATH)

for series, label in WEATHER_SERIES:
    print(f"\n=== {label} ({series}) ===")
    response = get(private_key, f"/markets?series_ticker={series}&status=open")
    markets = response.json().get("markets", [])
    if not markets:
        print("  No open markets found.")
    for m in markets:
        print(f"  {m.get('title')}")
        print(f"  Yes: {m.get('yes_ask')}¢  |  No: {m.get('no_ask')}¢")
        print(f"  Ticker: {m.get('ticker')}")
        print()
