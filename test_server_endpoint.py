import urllib.request
import json

url = "http://127.0.0.1:8000/predict"
req = urllib.request.Request(url, data=json.dumps({"commodity": "Rice", "market": "Tadepalligudem", "state": "Andhra Pradesh"}).encode('utf-8'), headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
print(json.loads(res.read().decode('utf-8')))
