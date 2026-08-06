import urllib.request
import urllib.parse
import json
import ssl
import pandas as pd

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
HIST_RESOURCE = "35985678-0d79-46b4-9ed6-6f13308a1d24"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def check_2021_data():
    print("Checking 2021-2026 Paddy records in data.gov.in...")
    all_recs = []
    
    # Try Paddy(Common), Paddy(Dhan), Paddy
    for comm in ['Paddy(Common)', 'Paddy(Dhan)', 'Paddy']:
        limit = 1000
        offset = 0
        while offset < 50000:
            params = {
                'api-key': API_KEY,
                'format': 'json',
                'limit': str(limit),
                'offset': str(offset),
                'filters[state]': 'Andhra Pradesh',
                'filters[commodity]': comm
            }
            url = f"https://api.data.gov.in/resource/{HIST_RESOURCE}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                res = urllib.request.urlopen(req, context=ctx, timeout=15)
                data = json.loads(res.read().decode('utf-8'))
                recs = data.get('records', [])
                if not recs:
                    break
                for r in recs:
                    all_recs.append({
                        'date': r.get('Arrival_Date'),
                        'state': r.get('State'),
                        'district': r.get('District'),
                        'market': r.get('Market'),
                        'commodity': r.get('Commodity'),
                        'variety': r.get('Variety'),
                        'modal_price': r.get('Modal_Price'),
                        'min_price': r.get('Min_Price'),
                        'max_price': r.get('Max_Price')
                    })
                offset += limit
                if len(recs) < limit:
                    break
            except Exception as e:
                print(f"Error at offset {offset}: {e}")
                break
                
    print(f"Total raw Paddy records fetched: {len(all_recs)}")
    if all_recs:
        df = pd.DataFrame(all_recs)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['date', 'modal_price', 'market'])
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("Market record counts (Top 15):")
        print(df.groupby(['market', 'district']).size().sort_values(ascending=False).head(15))
        
if __name__ == '__main__':
    check_2021_data()
