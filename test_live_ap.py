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

def test_live_ap():
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': '500',
        'filters[state]': 'Andhra Pradesh'
    }
    url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req, context=ctx, timeout=15)
        data = json.loads(res.read().decode('utf-8'))
        print(f"Total live records for AP: {data.get('total')}")
        records = data.get('records', [])
        if records:
            df = pd.DataFrame(records)
            print("AP Live Commodities:", df['commodity'].value_counts().to_dict())
            print("AP Live Markets:", df['market'].value_counts().to_dict())
            print("Live arrival date sample:", df['arrival_date'].unique())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_live_ap()
