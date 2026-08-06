import urllib.request
import json

url = "http://127.0.0.1:8000/predict"
payload = {"commodity": "Rice", "market": "Machilipatnam", "state": "Andhra Pradesh", "model_preference": "XGBoost"}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
print(json.loads(res.read().decode('utf-8')))
