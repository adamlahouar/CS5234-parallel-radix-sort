import requests
latitude = 60.4720
longitude = 8.4689

url = "https://api.met.no/weatherapi/subseasonal/1.0/complete"
params = {
    'lat': latitude,
    'lon': longitude
}

headers = {
    "User-Agent": "MyWeatherApp/1.0 (your.email@example.com)"  
}

response = requests.get(url, params=params, headers=headers)
j = response.json()
print(j["properties"]["timeSeries"][0]["data"]["instant"]["details"]["air_temperature_max"])
print(response.json())