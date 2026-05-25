import requests

# New Orleans coordinates
LAT = 29.9511
LON = -90.0715

# Open-Meteo free API - no key needed
url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    f"&daily=temperature_2m_max,temperature_2m_min"
    f"&temperature_unit=fahrenheit"
    f"&timezone=America%2FChicago"
    f"&forecast_days=3"
)

response = requests.get(url)
data = response.json()

dates = data["daily"]["time"]
highs = data["daily"]["temperature_2m_max"]
lows  = data["daily"]["temperature_2m_min"]

print("New Orleans Weather Forecast\n")
for date, high, low in zip(dates, highs, lows):
    print(f"Date: {date}")
    print(f"  Forecast High: {high}°F")
    print(f"  Forecast Low:  {low}°F")
    print()
