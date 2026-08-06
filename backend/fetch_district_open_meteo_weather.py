import urllib.request
import urllib.parse
import json
import ssl
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

DISTRICT_COORDS = {
    'Nandyal': {'lat': 15.48, 'lon': 78.48},
    'Kurnool': {'lat': 15.83, 'lon': 78.03},
    'East Godavari': {'lat': 17.00, 'lon': 81.78},
    'Kakinada': {'lat': 16.98, 'lon': 82.24},
    'NTR': {'lat': 16.50, 'lon': 80.64},
    'West Godavari': {'lat': 16.75, 'lon': 81.68},
    'Eluru': {'lat': 16.71, 'lon': 81.10},
    'SPSR Nellore': {'lat': 14.44, 'lon': 79.98},
    'Nellore': {'lat': 14.44, 'lon': 79.98},
    'Bapatla': {'lat': 15.90, 'lon': 80.47},
    'Alluri Sitharama Raju': {'lat': 17.83, 'lon': 81.88},
    'Visakhapatnam': {'lat': 17.68, 'lon': 83.21},
    'Krishna': {'lat': 16.18, 'lon': 81.13},
    'Guntur': {'lat': 16.30, 'lon': 80.43},
    'Prakasam': {'lat': 15.50, 'lon': 80.05}
}

def fetch_district_weather():
    print("="*85)
    print("FETCHING REAL OPEN-METEO DAILY RAINFALL & WEATHER FOR AP DISTRICTS (2021-2026)")
    print("="*85)
    
    all_dfs = []
    
    for dist_name, coords in DISTRICT_COORDS.items():
        lat = coords['lat']
        lon = coords['lon']
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date=2021-01-01&end_date=2026-08-05&"
            f"daily=rain_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean&"
            f"timezone=Asia%2FKolkata"
        )
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=20)
            data = json.loads(res.read().decode('utf-8'))
            daily = data.get('daily', {})
            
            d_df = pd.DataFrame({
                'date': daily.get('time', []),
                'District': dist_name,
                'rainfall_mm': daily.get('rain_sum', []),
                'temp_max': daily.get('temperature_2m_max', []),
                'temp_min': daily.get('temperature_2m_min', []),
                'humidity': daily.get('relative_humidity_2m_mean', [])
            })
            
            d_df['date'] = pd.to_datetime(d_df['date'])
            d_df['rainfall_mm'] = pd.to_numeric(d_df['rainfall_mm'], errors='coerce').fillna(0.0)
            d_df['temp_max'] = pd.to_numeric(d_df['temp_max'], errors='coerce').ffill().bfill()
            d_df['temp_min'] = pd.to_numeric(d_df['temp_min'], errors='coerce').ffill().bfill()
            d_df['humidity'] = pd.to_numeric(d_df['humidity'], errors='coerce').ffill().bfill()
            
            # Compute 7-day rolling rainfall and 7-day rainfall anomaly per district
            d_df['week_of_year'] = d_df['date'].dt.isocalendar().week.astype(int)
            d_df['rainfall_7d'] = d_df['rainfall_mm'].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
            
            weekly_hist_rain = d_df.groupby('week_of_year')['rainfall_7d'].transform('mean')
            d_df['rainfall_anomaly_7d'] = d_df['rainfall_7d'] - weekly_hist_rain
            
            d_df['heavy_rain_flag'] = np.where(d_df['rainfall_7d'] > 40.0, 1, 0)
            d_df['dry_spell_flag'] = np.where((d_df['rainfall_7d'] < 1.0) & (d_df['date'].dt.month.isin([6, 7, 8, 9])), 1, 0)
            
            all_dfs.append(d_df)
            print(f"  Fetched Open-Meteo weather for district: {dist_name:<22} | Rows: {len(d_df)} | Total Rain: {d_df['rainfall_mm'].sum():.1f} mm")
            time.sleep(0.4)
            
        except Exception as e:
            print(f"  Error fetching weather for {dist_name}: {e}")
            
    if all_dfs:
        final_weather_df = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'district_open_meteo_weather_real.csv')
        final_weather_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved Real Open-Meteo District Weather Dataset to: {out_path}")
        print(f"Total Rows: {len(final_weather_df)}")
        return final_weather_df

if __name__ == '__main__':
    fetch_district_weather()
