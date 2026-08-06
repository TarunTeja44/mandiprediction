import urllib.request
import urllib.parse
import json
import ssl
import pandas as pd
from collections import Counter

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def find_top_market():
    market_counts = Counter()
    limit = 1000
    total_to_check = 10000
    
    print(f"Sampling up to {total_to_check} records for 'Rice' in Andhra Pradesh...")
    for offset in range(0, total_to_check, limit):
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': str(limit),
            'offset': str(offset),
            'filters[State]': 'Andhra Pradesh',
            'filters[Commodity]': 'Rice'
        }
        url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            data = json.loads(res.read().decode('utf-8'))
            records = data.get('records', [])
            if not records:
                break
            for r in records:
                m = r.get('Market')
                if m:
                    market_counts[m] += 1
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    print("\nTop 15 markets by record count in AP for Rice (from sample):")
    for m, c in market_counts.most_common(15):
        print(f"  {m}: {c} records")

if __name__ == '__main__':
    find_top_market()
