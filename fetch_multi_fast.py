import os
import urllib.request
import urllib.parse
import json
import ssl
import pandas as pd
import numpy as np

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
HIST_RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
LIVE_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'processed')

MARKETS = ['Tadepalligudem', 'Machilipatnam', 'Narasaraopet', 'Ongole']

def run_fast_multi_market():
    print("Fetching multi-market data for AP Rice...")
    all_records = []
    
    for m in MARKETS:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': '1000',
            'filters[State]': 'Andhra Pradesh',
            'filters[Commodity]': 'Rice',
            'filters[Market]': m
        }
        url = f"https://api.data.gov.in/resource/{HIST_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=10)
            data = json.loads(res.read().decode('utf-8'))
            recs = data.get('records', [])
            all_records.extend(recs)
            print(f"  * {m}: {len(recs)} historical records")
        except Exception as e:
            print(f"Error {m}: {e}")
            
    # Live data
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': '500',
        'filters[state]': 'Andhra Pradesh',
        'filters[commodity]': 'Rice'
    }
    url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req, context=ctx, timeout=10)
        data = json.loads(res.read().decode('utf-8'))
        recs = data.get('records', [])
        for r in recs:
            all_records.append({
                'State': r.get('state'),
                'District': r.get('district'),
                'Market': r.get('market'),
                'Commodity': r.get('commodity'),
                'Arrival_Date': r.get('arrival_date'),
                'Min_Price': r.get('min_price'),
                'Max_Price': r.get('max_price'),
                'Modal_Price': r.get('modal_price')
            })
        print(f"  * Live AP records: {len(recs)}")
    except Exception as e:
        print(f"Live error: {e}")
        
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
    df['modal_price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
    df = df.dropna(subset=['date', 'modal_price', 'Market'])
    df = df[df['modal_price'] > 0]
    
    clean_df = df.groupby(['Market', 'date'])['modal_price'].mean().reset_index()
    clean_df = clean_df.sort_values(['Market', 'date']).reset_index(drop=True)
    
    out_path = os.path.join(PROCESSED_DIR, 'ap_multi_market_rice_cleaned.csv')
    clean_df.to_csv(out_path, index=False)
    print(f"\nSaved multi-market CSV: {len(clean_df)} records across markets: {clean_df['Market'].unique()}")
    return clean_df

if __name__ == '__main__':
    run_fast_multi_market()
