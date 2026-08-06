import os
import sys
import json
from backend.predict import generate_multi_market_forecast

markets = ['Tadepalligudem', 'Machilipatnam', 'Visakhapatnam', 'Ongole', 'Chirala', 'Divi', 'Narasaraopet']

print("="*85)
print("NEXT 2 DAYS RICE MANDI PRICE FORECAST (AP MARKETS - ANCHORED TO TODAY: 2026-08-05)")
print("="*85)

for mkt in markets:
    try:
        res = generate_multi_market_forecast(market=mkt, model_preference="XGBoost")
        cur_p = res['current_price']
        preds = res['predictions']
        
        p1 = preds[0]
        p2 = preds[1]
        
        print(f"MARKET: {mkt.upper()}")
        print(f"  Today ({res['current_date']}): Rs. {cur_p:.2f} / Q")
        print(f"  --> Tomorrow ({p1['date']})    : Rs. {p1['predicted_price']:.2f} / Q | Change: Rs. {p1['price_change_rs']:+.2f} | Trend: {p1['trend']} | 90% CI: {p1['confidence_interval_90pct']}")
        print(f"  --> Day After ({p2['date']})  : Rs. {p2['predicted_price']:.2f} / Q | Change: Rs. {p2['price_change_rs']:+.2f} | Trend: {p2['trend']} | 90% CI: {p2['confidence_interval_90pct']}")
        print("-" * 85)
    except Exception as e:
        print(f"Error for {mkt}: {e}")
