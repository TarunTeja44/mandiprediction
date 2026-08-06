import urllib.request
import urllib.parse
import json
import ssl
import pandas as pd

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
LIVE_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_recent_ap():
    print("Fetching 2021-2026 Mandi Prices for AP...")
    all_recs = []
    
    for comm in ['Paddy(Common)', 'Rice', 'Paddy(Dhan)']:
        limit = 1000
        offset = 0
        while True:
            params = {
                'api-key': API_KEY,
                'format': 'json',
                'limit': str(limit),
                'offset': str(offset),
                'filters[state]': 'Andhra Pradesh',
                'filters[commodity]': comm
            }
            url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                res = urllib.request.urlopen(req, context=ctx, timeout=15)
                data = json.loads(res.read().decode('utf-8'))
                recs = data.get('records', [])
                if not recs:
                    break
                for r in recs:
                    all_recs.append({
                        'Market': r.get('market'),
                        'Commodity': r.get('commodity'),
                        'date': r.get('arrival_date'),
                        'modal_price': r.get('modal_price'),
                        'min_price': r.get('min_price'),
                        'max_price': r.get('max_price')
                    })
                offset += limit
                if len(recs) < limit:
                    break
            except Exception as e:
                print(f"Error fetching {comm}: {e}")
                break
                
    print(f"Total live records fetched: {len(all_recs)}")
    if all_recs:
        df = pd.DataFrame(all_recs)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df = df.dropna(subset=['date', 'modal_price', 'Market'])
        print("Date range:", df['date'].min(), "to", df['date'].max())
        print("Markets:", df['Market'].value_counts().head(10).to_dict())
        
    return all_recs

if __name__ == '__main__':
    fetch_recent_ap()
