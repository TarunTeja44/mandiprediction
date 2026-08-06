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
RAW_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

TOP_AP_MARKETS = [
    'Tadepalligudem', 'Machilipatnam', 'Narasaraopet', 
    'Ongole', 'Chirala', 'Visakhapatnam', 'Divi', 'Tiruvuru', 'Addanki'
]

def fetch_multi_market_data():
    print("==================================================")
    print("STEP 1: Fetching Multi-Market AP Rice Data")
    print("==================================================")
    
    all_records = []
    
    # 1. Fetch Historical Archive Records for AP Markets
    for m in TOP_AP_MARKETS:
        limit = 500
        offset = 0
        m_recs = 0
        while True:
            params = {
                'api-key': API_KEY,
                'format': 'json',
                'limit': str(limit),
                'offset': str(offset),
                'filters[State]': 'Andhra Pradesh',
                'filters[Commodity]': 'Rice',
                'filters[Market]': m
            }
            url = f"https://api.data.gov.in/resource/{HIST_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                res = urllib.request.urlopen(req, context=ctx, timeout=15)
                data = json.loads(res.read().decode('utf-8'))
                recs = data.get('records', [])
                if not recs:
                    break
                for r in recs:
                    r['_source'] = 'historical'
                all_records.extend(recs)
                m_recs += len(recs)
                offset += limit
                if len(recs) < limit:
                    break
            except Exception as e:
                print(f"Error fetching historical {m}: {e}")
                break
        print(f"  * Historical {m}: {m_recs} records")

    # 2. Fetch Live Current Records for AP
    limit = 500
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': str(limit),
        'filters[state]': 'Andhra Pradesh',
        'filters[commodity]': 'Rice'
    }
    url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req, context=ctx, timeout=15)
        data = json.loads(res.read().decode('utf-8'))
        recs = data.get('records', [])
        for r in recs:
            # normalize keys
            norm_r = {
                'State': r.get('state'),
                'District': r.get('district'),
                'Market': r.get('market'),
                'Commodity': r.get('commodity'),
                'Variety': r.get('variety'),
                'Grade': r.get('grade'),
                'Arrival_Date': r.get('arrival_date'),
                'Min_Price': r.get('min_price'),
                'Max_Price': r.get('max_price'),
                'Modal_Price': r.get('modal_price'),
                '_source': 'live'
            }
            all_records.append(norm_r)
        print(f"  * Live Current AP Records: {len(recs)} records")
    except Exception as e:
        print(f"Live error: {e}")

    # Save multi-market raw JSON
    with open(os.path.join(RAW_DIR, 'multi_market_raw.json'), 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)
        
    df = pd.DataFrame(all_records)
    print(f"\nTotal multi-market records gathered: {len(df)}")
    
    df['date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
    df['modal_price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
    df['min_price'] = pd.to_numeric(df['Min_Price'], errors='coerce')
    df['max_price'] = pd.to_numeric(df['Max_Price'], errors='coerce')
    
    df = df.dropna(subset=['date', 'modal_price', 'Market'])
    df = df[df['modal_price'] > 0]
    
    # Clean per market & date
    clean_df = df.groupby(['Market', 'date']).agg({
        'modal_price': 'mean',
        'min_price': 'min',
        'max_price': 'max'
    }).reset_index()
    
    clean_df = clean_df.sort_values(['Market', 'date']).reset_index(drop=True)
    
    csv_path = os.path.join(PROCESSED_DIR, 'ap_multi_market_rice_cleaned.csv')
    clean_df.to_csv(csv_path, index=False)
    print(f"Saved cleaned multi-market dataset to: {csv_path}")
    print(f"Markets included: {clean_df['Market'].unique()}")
    print(f"Date range: {clean_df['date'].min().strftime('%Y-%m-%d')} to {clean_df['date'].max().strftime('%Y-%m-%d')}")
    
    return clean_df

if __name__ == '__main__':
    fetch_multi_market_data()
