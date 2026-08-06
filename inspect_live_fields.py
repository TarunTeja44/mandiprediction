import urllib.request
import urllib.parse
import json
import ssl

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
LIVE_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = f"https://api.data.gov.in/resource/{LIVE_RESOURCE_ID}?api-key={API_KEY}&format=json&limit=2"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
res = urllib.request.urlopen(req, context=ctx, timeout=15)
data = json.loads(res.read().decode('utf-8'))
print("Field definitions:", [f.get('name') or f.get('id') for f in data.get('field', [])])
print("Sample record:", data.get('records', [])[0] if data.get('records') else "None")
