import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from multi_market_feature_engine import create_multi_market_features
from multi_market_train import train_and_evaluate_multi_market
from predict import generate_multi_market_forecast

app = FastAPI(
    title="Farmer Mandi Price Prediction API",
    description="Real-Time Multi-Market Crop Price Prediction Engine (Andhra Pradesh)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'ap_multi_market_rice_cleaned.csv')

class PredictRequest(BaseModel):
    commodity: str = "Rice"
    market: str = "Machilipatnam"
    state: str = "Andhra Pradesh"
    model_preference: str = "XGBoost"

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Farmer Mandi Price Prediction API",
        "state": "Andhra Pradesh",
        "commodity": "Rice"
    }

@app.get("/crops")
def get_supported_crops():
    return {
        "supported_crops": ["Rice"],
        "default": "Rice"
    }

@app.get("/markets")
def get_supported_markets():
    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV)
        markets = list(df['Market'].unique())
    else:
        markets = ["Tadepalligudem", "Machilipatnam", "Narasaraopet", "Ongole", "Chirala", "Visakhapatnam", "Divi"]
        
    return {
        "state": "Andhra Pradesh",
        "supported_markets": markets,
        "default": "Machilipatnam"
    }

@app.post("/predict")
def predict_price(req: PredictRequest):
    try:
        forecast = generate_multi_market_forecast(market=req.market, model_preference=req.model_preference)
        return {
            "success": True,
            "data": forecast
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_price_history(market: str = "Machilipatnam", limit: int = 100):
    if not os.path.exists(PROCESSED_CSV):
        raise HTTPException(status_code=404, detail="Cleaned price data not found.")
    
    df = pd.read_csv(PROCESSED_CSV)
    m_df = df[df['Market'] == market]
    if m_df.empty:
        m_df = df[df['Market'] == 'Machilipatnam']
        
    m_df['date'] = pd.to_datetime(m_df['date'])
    m_df = m_df.sort_values('date').tail(limit)
    
    records = []
    for _, row in m_df.iterrows():
        records.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "modal_price": float(row['modal_price']),
            "min_price": float(row['min_price']),
            "max_price": float(row['max_price'])
        })
        
    return {
        "market": market,
        "commodity": "Rice",
        "record_count": len(records),
        "history": records
    }

@app.post("/retrain")
def retrain_model():
    try:
        create_multi_market_features()
        artifact = train_and_evaluate_multi_market()
        return {
            "success": True,
            "message": "Multi-market models retrained successfully",
            "best_model": artifact['model_name'],
            "metrics": artifact['metrics']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
