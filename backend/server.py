import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import generate_multi_market_forecast
from paddy_feature_engine import build_paddy_features
from train_paddy_model import train_paddy_model

app = FastAPI(
    title="Farmer Mandi Price Prediction API",
    description="Real-Time Multi-Market Crop Price Prediction Engine (Andhra Pradesh) — Prophet + ARIMA + ML",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
FALLBACK_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'paddy_common_top10_ap_cleaned.csv')

class PredictRequest(BaseModel):
    commodity: str = "Rice"
    market: str = "Jaggampet"
    state: str = "Andhra Pradesh"
    model_preference: str = "Auto"  # Auto, Prophet, ARIMA, XGBoost, GradientBoosting, Naive

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Farmer Mandi Price Prediction API v3.0",
        "state": "Andhra Pradesh",
        "commodity": "Rice",
        "models_available": ["Auto", "Prophet", "ARIMA", "XGBoost", "GradientBoosting", "Naive"],
        "forecast_horizon": "7 days"
    }

@app.get("/crops")
def get_supported_crops():
    return {
        "supported_crops": ["Rice"],
        "default": "Rice"
    }

@app.get("/markets")
def get_supported_markets():
    csv_path = PROCESSED_CSV if os.path.exists(PROCESSED_CSV) else FALLBACK_CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        markets = list(df['Market'].unique())
    else:
        markets = ["Jaggampet", "Rapur", "Tiruvuru", "Rampachodvaram", "Mylavaram", "Polavaram", "Kovvur", "Nandigama", "Kuchinapudi"]

    return {
        "state": "Andhra Pradesh",
        "supported_markets": markets,
        "default": "Jaggampet"
    }

@app.get("/models")
def get_available_models():
    """Return available model types and their descriptions."""
    return {
        "models": [
            {"name": "Auto", "description": "Regime-based automatic selection (recommended). Uses Prophet for active markets, ARIMA for low-volatility, Naive for flat."},
            {"name": "Prophet", "description": "Facebook Prophet time-series model with weather & MSP regressors. Best for markets with real price swings."},
            {"name": "ARIMA", "description": "Auto-ARIMA with exogenous regressors. Best for low-to-medium volatility markets."},
            {"name": "XGBoost", "description": "XGBoost gradient boosting on engineered features. Legacy ML model."},
            {"name": "GradientBoosting", "description": "Scikit-learn GradientBoosting per market. Legacy ML model."},
            {"name": "Naive", "description": "Last-price-forward baseline. Best for completely flat/sticky markets."},
        ],
        "default": "Auto"
    }

@app.post("/predict")
def predict_price(req: PredictRequest):
    try:
        forecast = generate_multi_market_forecast(market=req.market, model_preference=req.model_preference)
        return {
            "success": True,
            "market": forecast.get('market', req.market),
            "district": forecast.get('district', ''),
            "current_price": forecast.get('current_weighted_avg_price', 0.0),
            "model_used": forecast.get('model_used', req.model_preference),
            "market_regime": forecast.get('market_regime', ''),
            "market_regime_type": forecast.get('market_regime_type', ''),
            "regime_std": forecast.get('regime_std', 0),
            "market_note": forecast.get('market_note', ''),
            "forecast_horizon_days": forecast.get('forecast_horizon_days', 7),
            "predictions": forecast.get('predictions', []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_price_history(market: str = "Jaggampet", limit: int = 100):
    if not os.path.exists(PROCESSED_CSV):
        raise HTTPException(status_code=404, detail="Cleaned price data not found.")

    df = pd.read_csv(PROCESSED_CSV)
    m_df = df[df['Market'] == market]
    if m_df.empty:
        m_df = df[df['Market'] == 'Jaggampet']

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
        build_paddy_features()
        train_paddy_model()
        return {
            "success": True,
            "message": "Paddy forecasting model retrained successfully (Prophet + ARIMA + ML)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
