import urllib.request
import urllib.parse
import json
import ssl
import os
import pandas as pd
import datetime

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
HISTORICAL_RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
LIVE_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_historical(state="Andhra Pradesh", commodity="Rice", market="Tadepalligudem"):
    print(f"Fetching HISTORICAL data for '{commodity}' in '{market}'...")
    limit = 500
    offset = 0
    records = []
    
    while True:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': str(limit),
            'offset': str(offset),
            'filters[State]': state,
            'filters[Commodity]': commodity,
            'filters[Market]': market
        }
        url = f"https://api.data.gov.in/resource/{HISTORICAL_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            data = json.loads(res.read().decode('utf-8'))
            recs = data.get('records', [])
            if not recs:
                break
            records.extend(recs)
            offset += limit
            if len(recs) < limit:
                break
        except Exception as e:
            print(f"Historical error: {e}")
            break
            
    print(f"Got {len(records)} historical records.")
    return records

def fetch_live_current(state="Andhra Pradesh", commodity="Rice"):
    print(f"Fetching LIVE CURRENT (2026) data for '{commodity}' in '{state}'...")
    limit = 500
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': str(limit),
        'filters[state]': state,
        'filters[commodity]': commodity
    }
    url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    records = []
    try:
        res = urllib.request.urlopen(req, context=ctx, timeout=15)
        data = json.loads(res.read().decode('utf-8'))
        recs = data.get('records', [])
        records.extend(recs)
        print(f"Got {len(records)} live records for today/recent.")
    except Exception as e:
        print(f"Live error: {e}")
        
    return records

def merge_and_clean():
    hist_recs = fetch_historical(state="Andhra Pradesh", commodity="Rice", market="Tadepalligudem")
    live_recs = fetch_live_current(state="Andhra Pradesh", commodity="Rice")
    
    # Save raw files separately
    with open(os.path.join(RAW_DIR, 'hist_raw.json'), 'w', encoding='utf-8') as f:
        json.dump(hist_recs, f, indent=2)
    with open(os.path.join(RAW_DIR, 'live_raw.json'), 'w', encoding='utf-8') as f:
        json.dump(live_recs, f, indent=2)
        
    hist_df = pd.DataFrame(hist_recs)
    hist_df['date'] = pd.to_datetime(hist_df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
    hist_df['modal_price'] = pd.to_numeric(hist_df['Modal_Price'], errors='coerce')
    hist_df['min_price'] = pd.to_numeric(hist_df['Min_Price'], errors='coerce')
    hist_df['max_price'] = pd.to_numeric(hist_df['Max_Price'], errors='coerce')
    
    live_df = pd.DataFrame(live_recs)
    if not live_df.empty:
        live_df['date'] = pd.to_datetime(live_df['arrival_date'], format='%d/%m/%Y', errors='coerce')
        live_df['modal_price'] = pd.to_numeric(live_df['modal_price'], errors='coerce')
        live_df['min_price'] = pd.to_numeric(live_df['min_price'], errors='coerce')
        live_df['max_price'] = pd.to_numeric(live_df['max_price'], errors='coerce')
        
    combined = pd.concat([
        hist_df[['date', 'modal_price', 'min_price', 'max_price']],
        live_df[['date', 'modal_price', 'min_price', 'max_price']]
    ], ignore_index=True)
    
    combined = combined.dropna(subset=['date', 'modal_price'])
    combined = combined[combined['modal_price'] > 0]
    
    daily_df = combined.groupby('date').agg({
        'modal_price': 'mean',
        'min_price': 'min',
        'max_price': 'max'
    }).reset_index()
    
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    print(f"\nFinal Combined Dataset Date Range: {daily_df['date'].min().strftime('%Y-%m-%d')} to {daily_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Unique Days: {len(daily_df)}")
    print(f"Latest Recorded Modal Price as of {daily_df['date'].max().strftime('%Y-%m-%d')}: Rs. {daily_df['modal_price'].iloc[-1]:.2f} / Quintal")
    
    processed_path = os.path.join(PROCESSED_DIR, 'tadepalligudem_rice_cleaned.csv')
    daily_df.to_csv(processed_path, index=False)
    print(f"Saved merged dataset to: {processed_path}")

    # Fetch weather up to today (August 2026)
    start_s = daily_df['date'].min().strftime('%Y-%m-%d')
    end_s = datetime.date.today().strftime('%Y-%m-%d')
    
    print(f"\nFetching Open-Meteo weather from {start_s} to {end_s}...")
    w_params = {
        'latitude': '16.83',
        'longitude': '81.53',
        'start_date': start_s,
        'end_date': end_s,
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,relative_humidity_2m_mean,windspeed_10m_max',
        'timezone': 'Asia/Kolkata'
    }
    w_url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(w_params)}"
    req = urllib.request.Request(w_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req, timeout=20)
        w_data = json.loads(res.read().decode('utf-8'))
        w_daily = w_data.get('daily', {})
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(w_daily.get('time', [])),
            'temp_max': w_daily.get('temperature_2m_max', []),
            'temp_min': w_daily.get('temperature_2m_min', []),
            'precipitation': w_daily.get('precipitation_sum', []),
            'rainfall': w_daily.get('rain_sum', []),
            'humidity': w_daily.get('relative_humidity_2m_mean', []),
            'wind_max': w_daily.get('windspeed_10m_max', [])
        })
        weather_path = os.path.join(PROCESSED_DIR, 'weather_cleaned.csv')
        weather_df.to_csv(weather_path, index=False)
        print(f"Saved updated weather dataset ({len(weather_df)} days) to: {weather_path}")
    except Exception as e:
        print(f"Weather error: {e}")

if __name__ == '__main__':
    merge_and_clean()
