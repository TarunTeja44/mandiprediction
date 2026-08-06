import urllib.request
import urllib.parse
import json
import ssl
import time
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
HIST_RESOURCE = "35985678-0d79-46b4-9ed6-6f13308a1d24"
LIVE_RESOURCE = "9ef84268-d588-465a-a308-a864a43d0070"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# Known AP districts (to filter out UP/MP leakage from API)
AP_DISTRICTS = {
    'Alluri Sitharama Raju', 'Anakapally', 'Ananthapuramu', 'Annamayya',
    'Bapatla', 'Dr.B.R.A.Konaseema', 'East Godavari', 'Eluru',
    'Guntur', 'Kakinada', 'Krishna', 'Kurnool', 'Nandyal', 'NTR',
    'Palnadu', 'Polavaram', 'Prakasam', 'SPSR Nellore', 'Sri Potti Sriramulu Nellore',
    'Sri Sathya Sai', 'Srikakulam', 'Tirupathi', 'Vijayanagaram',
    'Visakhapatnam', 'West Godavari', 'YSR Kadapa', 'Chittoor',
    'Vizianagaram', 'Nellore', 'Kadapa'
}

def fetch_all_hist():
    """Fetch ALL historical Paddy(Common) records for AP"""
    print("="*70)
    print("PHASE 1: FETCHING ALL HISTORICAL PADDY(COMMON) DATA FOR AP")
    print("="*70)
    
    all_records = []
    limit = 1000
    offset = 0
    
    while True:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': str(limit),
            'offset': str(offset),
            'filters[state]': 'Andhra Pradesh',
            'filters[commodity]': 'Paddy(Common)'
        }
        url = f"https://api.data.gov.in/resource/{HIST_RESOURCE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            data = json.loads(res.read().decode('utf-8'))
            records = data.get('records', [])
            
            if not records:
                break
                
            for r in records:
                district = r.get('District', '')
                # Filter: only keep if district is a known AP district
                if any(d.lower() in district.lower() for d in AP_DISTRICTS) or district in AP_DISTRICTS:
                    all_records.append({
                        'date': r.get('Arrival_Date', ''),
                        'state': 'Andhra Pradesh',
                        'district': district,
                        'market': r.get('Market', ''),
                        'commodity': 'Paddy(Common)',
                        'variety': r.get('Variety', ''),
                        'grade': r.get('Grade', ''),
                        'min_price': r.get('Min_Price', ''),
                        'max_price': r.get('Max_Price', ''),
                        'modal_price': r.get('Modal_Price', '')
                    })
            
            offset += limit
            print(f"  Fetched offset {offset}, total AP records so far: {len(all_records)}")
            
            if len(records) < limit:
                break
                
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  Error at offset {offset}: {e}")
            time.sleep(2)
            continue
    
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df['min_price'] = pd.to_numeric(df['min_price'], errors='coerce')
    df['max_price'] = pd.to_numeric(df['max_price'], errors='coerce')
    df = df.dropna(subset=['date', 'modal_price', 'market'])
    
    raw_path = os.path.join(RAW_DIR, 'paddy_common_ap_historical_raw.csv')
    df.to_csv(raw_path, index=False, encoding='utf-8-sig')
    print(f"\n  Saved raw historical: {raw_path}")
    print(f"  Total AP records: {len(df)}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Unique markets: {df['market'].nunique()}")
    return df

def fetch_live():
    """Fetch live/current day Paddy(Common) records for AP"""
    print("\n" + "="*70)
    print("PHASE 2: FETCHING LIVE PADDY(COMMON) DATA FOR AP")
    print("="*70)
    
    all_records = []
    limit = 1000
    offset = 0
    
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': str(limit),
        'offset': str(offset),
        'filters[state]': 'Andhra Pradesh',
        'filters[commodity]': 'Paddy(Common)'
    }
    url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        res = urllib.request.urlopen(req, context=ctx, timeout=15)
        data = json.loads(res.read().decode('utf-8'))
        records = data.get('records', [])
        
        for r in records:
            all_records.append({
                'date': r.get('arrival_date', ''),
                'state': 'Andhra Pradesh',
                'district': r.get('district', ''),
                'market': r.get('market', ''),
                'commodity': 'Paddy(Common)',
                'variety': r.get('variety', ''),
                'grade': r.get('grade', ''),
                'min_price': r.get('min_price', ''),
                'max_price': r.get('max_price', ''),
                'modal_price': r.get('modal_price', '')
            })
    except Exception as e:
        print(f"  Error: {e}")
    
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df['min_price'] = pd.to_numeric(df['min_price'], errors='coerce')
    df['max_price'] = pd.to_numeric(df['max_price'], errors='coerce')
    df = df.dropna(subset=['date', 'modal_price', 'market'])
    
    raw_path = os.path.join(RAW_DIR, 'paddy_common_ap_live_raw.csv')
    df.to_csv(raw_path, index=False, encoding='utf-8-sig')
    print(f"  Saved raw live: {raw_path}")
    print(f"  Total live AP records: {len(df)}")
    return df

def select_top_10_and_clean(hist_df, live_df):
    """Select top 10 AP markets by record count and create clean CSV"""
    print("\n" + "="*70)
    print("PHASE 3: SELECTING TOP 10 AP PADDY(COMMON) MARKETS")
    print("="*70)
    
    combined = pd.concat([hist_df, live_df], ignore_index=True)
    combined = combined.sort_values(['market', 'date']).reset_index(drop=True)
    
    # Clean market names
    combined['market_clean'] = combined['market'].str.replace(' APMC', '', regex=False).str.strip()
    
    # Count records per market
    market_counts = combined.groupby(['market_clean', 'district']).size().reset_index(name='record_count')
    market_counts = market_counts.sort_values('record_count', ascending=False).reset_index(drop=True)
    
    print("\nALL AP PADDY(COMMON) MARKETS BY RECORD COUNT:")
    print(f"{'Rank':<6} {'Market':<35} {'District':<30} {'Records':<10}")
    print("-"*85)
    for i, row in market_counts.head(20).iterrows():
        print(f"{i+1:<6} {row['market_clean']:<35} {row['district']:<30} {row['record_count']:<10}")
    
    # Select top 10
    top10 = market_counts.head(10)
    top10_markets = top10['market_clean'].tolist()
    
    print(f"\n>> SELECTED TOP 10 MARKETS: {top10_markets}")
    
    # Filter combined to top 10 only
    cleaned = combined[combined['market_clean'].isin(top10_markets)].copy()
    cleaned = cleaned.sort_values(['market_clean', 'date']).reset_index(drop=True)
    
    # Rename for clarity
    cleaned = cleaned.rename(columns={'market_clean': 'Market', 'district': 'District'})
    cleaned = cleaned[['date', 'state', 'District', 'Market', 'commodity', 'variety', 'grade', 'min_price', 'max_price', 'modal_price']]
    
    clean_path = os.path.join(PROC_DIR, 'paddy_common_top10_ap_cleaned.csv')
    cleaned.to_csv(clean_path, index=False, encoding='utf-8-sig')
    
    print(f"\n  Saved cleaned top 10 CSV: {clean_path}")
    print(f"  Total cleaned records: {len(cleaned)}")
    print(f"  Date range: {cleaned['date'].min().strftime('%Y-%m-%d')} to {cleaned['date'].max().strftime('%Y-%m-%d')}")
    print(f"\n  Per-market breakdown:")
    for mkt in top10_markets:
        m_df = cleaned[cleaned['Market'] == mkt]
        print(f"    {mkt:<35}: {len(m_df):>5} records | {m_df['date'].min().strftime('%Y-%m-%d')} to {m_df['date'].max().strftime('%Y-%m-%d')} | Avg Price: Rs. {m_df['modal_price'].mean():.0f}")
    
    return cleaned, top10_markets

if __name__ == '__main__':
    hist_df = fetch_all_hist()
    live_df = fetch_live()
    cleaned_df, top10 = select_top_10_and_clean(hist_df, live_df)
