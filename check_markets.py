import urllib.request
import urllib.parse
import json
import ssl
import os
import pandas as pd

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def check_commodities_and_markets():
    print("--- Searching for Paddy/Rice markets in Andhra Pradesh ---")
    limit = 500
    offset = 0
    all_records = []
    
    # Try Paddy (Dhan) or Rice
    for commodity in ['Paddy(Dhan)', 'Rice', 'Paddy']:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': str(limit),
            'offset': str(offset),
            'filters[State]': 'Andhra Pradesh',
            'filters[Commodity]': commodity
        }
        url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            data = json.loads(res.read().decode('utf-8'))
            total = data.get('total', 0)
            records = data.get('records', [])
            print(f"Commodity '{commodity}': total records = {total}")
            if records:
                df = pd.DataFrame(records)
                print("Markets found:", df['Market'].value_counts().head(10).to_dict())
        except Exception as e:
            print(f"Error for '{commodity}': {e}")

if __name__ == '__main__':
    check_commodities_and_markets()
