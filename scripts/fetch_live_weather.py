import requests
import pandas as pd


def fetch_openweathermap(api_key, city="Astana, KZ"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    rainfall = data.get("rain", {}).get("1h", 0)
    temperature = data["main"]["temp"]
    return {"city": city, "rainfall": rainfall, "temperature": temperature, "timestamp": pd.Timestamp.now()}


if __name__ == "__main__":
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
    print(fetch_openweathermap(API_KEY))
