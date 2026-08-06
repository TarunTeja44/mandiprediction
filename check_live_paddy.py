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

def check_live():
    print("=== FETCHING RECENT 2024-2026 MANDI DATA FROM DATA.GOV.IN ===")
    
    for comm in ['Paddy(Common)', 'Paddy(Dhan)', 'Rice']:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': '500',
            'filters[state]': 'Andhra Pradesh',
            'filters[commodity]': comm
        }
        url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            data = json.loads(res.read().decode('utf-8'))
            records = data.get('records', [])
            print(f"\nCommodity: '{comm}' in AP | Live total records: {data.get('total')}")
            if records:
                df = pd.DataFrame(records)
                print("  Markets:", df['market'].value_counts().head(10).to_dict())
                print("  Dates found:", df['arrival_date'].unique()[:10])
                print("  Sample row:", records[0])
        except Exception as e:
            print(f"Error for {comm}: {e}")

if __name__ == '__main__':
    check_live()
