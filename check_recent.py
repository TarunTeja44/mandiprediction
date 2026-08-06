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

def check_live_resource():
    print("Checking live/current resource 9ef84268-d588-465a-a308-a864a43d0070...")
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': '500',
        'filters[State]': 'Andhra Pradesh',
        'filters[Commodity]': 'Rice'
    }
    url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req, context=ctx, timeout=15)
        data = json.loads(res.read().decode('utf-8'))
        print(f"Total live records available for Rice in AP: {data.get('total')}")
        records = data.get('records', [])
        if records:
            df = pd.DataFrame(records)
            print("Sample live record date:", df['Arrival_Date'].unique()[:5])
            print("Live markets:", df['Market'].value_counts().head(10).to_dict())
    except Exception as e:
        print(f"Live resource error: {e}")

if __name__ == '__main__':
    check_live_resource()
