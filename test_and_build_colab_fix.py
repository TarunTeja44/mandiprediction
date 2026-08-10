import os
import sys
import json
import urllib.request
import urllib.parse
import ssl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Market Coordinates Mapping for Open-Meteo
MARKET_COORDS = {
    'Kuchinapudi': {'lat': 15.90, 'lon': 80.47},
    'Tiruvuru': {'lat': 16.50, 'lon': 80.64},
    'Nandigama': {'lat': 16.50, 'lon': 80.64},
    'Rapur': {'lat': 14.44, 'lon': 79.98},
    'Kovvur': {'lat': 17.00, 'lon': 81.78},
    'Polavaram': {'lat': 16.71, 'lon': 81.10},
    'Jaggampet': {'lat': 17.00, 'lon': 81.78},
    'Mylavaram': {'lat': 16.50, 'lon': 80.64},
    'Rampachodvaram': {'lat': 17.83, 'lon': 81.88}
}

def fetch_open_meteo_3day_forecast(market_name):
    coords = MARKET_COORDS.get(market_name, {'lat': 16.50, 'lon': 80.64})
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&daily=precipitation_sum&forecast_days=3&timezone=Asia%2FKolkata"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, context=ctx, timeout=5)
        data = json.loads(res.read().decode('utf-8'))
        precip = data.get('daily', {}).get('precipitation_sum', [0.0, 0.0, 0.0])
        return [float(p) for p in precip[:3]]
    except Exception as e:
        return [0.0, 0.0, 0.0]

print("✓ Weather forecast API helper ready!")
