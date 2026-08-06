import os
import sys
import io
import json
import ssl
import urllib.request
import urllib.parse
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

TARGET_DIR = r"C:\Users\Praveen\OneDrive\Desktop\aa"
os.makedirs(TARGET_DIR, exist_ok=True)

CSV_FILE_PATH = r"C:\Users\Praveen\OneDrive\Desktop\All_Type_of_Report_(All_Grades)_06-08-2026_02-38-20_PM.csv"
DISTRICT_WEATHER_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'district_open_meteo_weather_real.csv')
PROCUREMENT_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'procurement_events.csv')

API_KEY = "579b464db66ec23bdd000001a0a99e04a75a40666201931688acb738"
HISTORICAL_RESOURCE = "35985678-0d79-46b4-9ed6-6f13308a1d24"

# ---------------------------------------------------------
# FEATURE ENGINE HELPER
# ---------------------------------------------------------
def build_featured_dataset(df_raw, is_api=False):
    df = df_raw.copy()
    if is_api:
        df['Market'] = df['market'].str.replace(' APMC', '', regex=False).str.strip()
        df['District'] = df['district']
    else:
        arrival_col = [c for c in df.columns if c.startswith('Arrival Quantity')][0]
        modal_col = [c for c in df.columns if c.startswith('Modal Price')][0]
        df.rename(columns={
            'District': 'District',
            'Market': 'Market_raw',
            'Date': 'date_str',
            arrival_col: 'arrival_qty_mt',
            modal_col: 'modal_price'
        }, inplace=True)
        df = df[df['Commodity'] == 'Paddy(Common)'].copy()
        df['date'] = pd.to_datetime(df['date_str'], format='%d-%m-%Y', errors='coerce')
        df['arrival_qty_mt'] = pd.to_numeric(df['arrival_qty_mt'], errors='coerce').fillna(0.0)
        df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df['min_price'] = df['modal_price'] * 0.985
        df['max_price'] = df['modal_price'] * 1.015
        df['Market'] = df['Market_raw'].str.replace(' APMC', '', regex=False).str.strip()

    df = df.dropna(subset=['date', 'modal_price', 'Market']).sort_values(['Market', 'date']).reset_index(drop=True)
    df['weighted_avg_modal_price'] = 0.60 * df['modal_price'] + 0.20 * df['min_price'] + 0.20 * df['max_price']
    
    top_markets = df.groupby('Market').size().sort_values(ascending=False).head(10).index.tolist()
    
    weather_df = pd.read_csv(DISTRICT_WEATHER_CSV) if os.path.exists(DISTRICT_WEATHER_CSV) else None
    if weather_df is not None:
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
    procurement_df = pd.read_csv(PROCUREMENT_CSV)
    procurement_df['date'] = pd.to_datetime(procurement_df['date'])
    
    processed = []
    for mkt in top_markets:
        m_df = df[df['Market'] == mkt].sort_values('date').reset_index(drop=True).drop_duplicates(subset=['date']).reset_index(drop=True)
        dist = m_df['District'].iloc[0]
        
        min_d, max_d = m_df['date'].min(), m_df['date'].max()
        full_dates = pd.date_range(min_d, max_d, freq='D')
        
        res = m_df.set_index('date').reindex(full_dates).reset_index()
        res.rename(columns={'index': 'date'}, inplace=True)
        res['Market'] = mkt
        res['District'] = dist
        res['commodity'] = 'Paddy(Common)'
        
        res['modal_price'] = res['modal_price'].ffill().bfill()
        res['min_price'] = res['min_price'].ffill().bfill()
        res['max_price'] = res['max_price'].ffill().bfill()
        res['weighted_avg_modal_price'] = res['weighted_avg_modal_price'].ffill().bfill()
        
        if 'arrival_qty_mt' in res.columns:
            res['arrival_qty_mt'] = pd.to_numeric(res['arrival_qty_mt'], errors='coerce').fillna(0.0)
        else:
            res['arrival_qty_mt'] = 0.0
            
        res['arrival_lag_1'] = res['arrival_qty_mt'].shift(1).fillna(0.0)
        res['arrival_7d_mean'] = res['arrival_qty_mt'].shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
        res['arrival_change_pct'] = res['arrival_qty_mt'].shift(1).pct_change(1, fill_method=None).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        dow = res['date'].dt.dayofweek
        month = res['date'].dt.month
        day = res['date'].dt.day
        res['day_of_week'] = dow
        res['month'] = month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        
        is_sunday = np.where(dow == 6, 1, 0)
        is_public_holiday = np.where(((month == 1) & (day.isin([14, 15, 26]))) | ((month == 8) & (day == 15)) | ((month == 10) & (day == 2)), 1, 0)
        res['is_likely_non_trading_day'] = np.where((is_sunday == 1) | (is_public_holiday == 1), 1, 0)
        
        if weather_df is not None:
            dist_w = weather_df[weather_df['District'].str.lower() == dist.lower()]
            if dist_w.empty:
                dist_w = weather_df[weather_df['District'] == 'East Godavari']
            res = pd.merge(res, dist_w[['date', 'rainfall_mm', 'rainfall_7d', 'rainfall_anomaly_7d', 'heavy_rain_flag', 'dry_spell_flag']], on='date', how='left')
            res['rainfall_7d'] = res['rainfall_7d'].fillna(0.0)
            res['rainfall_anomaly_7d'] = res['rainfall_anomaly_7d'].fillna(0.0)
            res['heavy_rain_flag'] = res['heavy_rain_flag'].fillna(0)
            res['dry_spell_flag'] = res['dry_spell_flag'].fillna(0)
        else:
            res['rainfall_7d'] = 0.0
            res['rainfall_anomaly_7d'] = 0.0
            res['heavy_rain_flag'] = 0
            res['dry_spell_flag'] = 0
            
        res = pd.merge(res, procurement_df, on='date', how='left')
        res['is_procurement_active'] = res['is_procurement_active'].fillna(0)
        res['msp_value'] = res['msp_value'].ffill().bfill()
        res['msp_announced'] = res['msp_announced'].fillna(0)
        res['msp_changed'] = res['msp_changed'].fillna(0)
        res['procurement_started'] = res['procurement_started'].fillna(0)
        res['procurement_ended'] = res['procurement_ended'].fillna(0)
        
        p = res['weighted_avg_modal_price']
        res['target_modal_price'] = p.shift(-1)
        res['target_return'] = (res['target_modal_price'] - p) / (p + 1e-5)
        
        res['ret_1'] = (p - p.shift(1)) / (p.shift(1) + 1e-5)
        res['ret_3'] = (p - p.shift(3)) / (p.shift(3) + 1e-5)
        res['ret_7'] = (p - p.shift(7)) / (p.shift(7) + 1e-5)
        
        res['rolling_mean_7'] = p.shift(1).rolling(7, min_periods=2).mean()
        res['rolling_mean_14'] = p.shift(1).rolling(14, min_periods=3).mean()
        res['rolling_std_7'] = p.shift(1).rolling(7, min_periods=2).std().fillna(0.0)
        res['rolling_std_14'] = p.shift(1).rolling(14, min_periods=3).std().fillna(0.0)
        
        res['ratio_ma7'] = p / (res['rolling_mean_7'] + 1e-5)
        res['ratio_ma14'] = p / (res['rolling_mean_14'] + 1e-5)
        res['is_harvest_season'] = np.where(month.isin([10, 11, 12, 4, 5]), 1, 0)
        
        processed.append(res.dropna(subset=['rolling_mean_7', 'target_modal_price']))
        
    final_df = pd.concat(processed, ignore_index=True)
    market_dummies = pd.get_dummies(final_df['Market'], prefix='mkt')
    district_dummies = pd.get_dummies(final_df['District'], prefix='dist')
    return pd.concat([final_df, market_dummies, district_dummies], axis=1)

# ---------------------------------------------------------
# MODEL TRAINER HELPER
# ---------------------------------------------------------
def train_and_save_model(featured_df, model_name, save_filename):
    print(f"\nTraining {model_name}...")
    mkt_cols = [c for c in featured_df.columns if c.startswith('mkt_')]
    dist_cols = [c for c in featured_df.columns if c.startswith('dist_')]
    feature_cols = [
        'ret_1', 'ret_3', 'ret_7', 'ratio_ma7', 'ratio_ma14', 'rolling_std_7', 'rolling_std_14',
        'arrival_lag_1', 'arrival_7d_mean', 'arrival_change_pct', 'is_likely_non_trading_day',
        'rainfall_7d', 'rainfall_anomaly_7d', 'heavy_rain_flag', 'dry_spell_flag',
        'is_procurement_active', 'msp_value', 'msp_announced', 'msp_changed', 'procurement_started', 'procurement_ended',
        'day_of_week', 'month', 'week_of_year', 'is_harvest_season'
    ] + mkt_cols + dist_cols
    
    n = len(featured_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = featured_df.iloc[:train_end][feature_cols].fillna(0.0), featured_df.iloc[:train_end]['target_return'].fillna(0.0)
    X_val, y_val = featured_df.iloc[train_end:val_end][feature_cols].fillna(0.0), featured_df.iloc[train_end:val_end]['target_return'].fillna(0.0)
    X_test, y_test = featured_df.iloc[val_end:][feature_cols].fillna(0.0), featured_df.iloc[val_end:]['target_return'].fillna(0.0)
    
    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_test_true = featured_df.iloc[val_end:]['target_modal_price'].values
    today_test_price = featured_df.iloc[val_end:]['weighted_avg_modal_price'].values
    pred_test_price = today_test_price * (1.0 + xgb_model.predict(X_test))
    
    mape = mean_absolute_percentage_error(y_test_true, pred_test_price) * 100.0
    mae = mean_absolute_error(y_test_true, pred_test_price)
    print(f"  [{model_name}] Test MAPE: {mape:.2f}% | Test MAE: Rs. {mae:.2f} / Quintal")
    
    artifact = {
        'model_name': f'Paddy(Common) AP - {model_name}',
        'model_object': xgb_model,
        'feature_cols': feature_cols,
        'target_col': 'weighted_avg_modal_price',
        'metrics': {
            'XGBoost': {
                'model': xgb_model,
                'test_mape': mape,
                'test_mae': mae
            }
        }
    }
    
    save_path = os.path.join(TARGET_DIR, save_filename)
    joblib.dump(artifact, save_path)
    print(f"  Saved model artifact to: {save_path}")
    return save_path

# ---------------------------------------------------------
# MAIN BUILD PIPELINE
# ---------------------------------------------------------
def build_both_models():
    print("="*85)
    print("BUILDING 2 EXACT DISTINCT MODEL ARTIFACTS IN C:\\Users\\Praveen\\OneDrive\\Desktop\\aa")
    print("="*85)
    
    # 1. BUILD MODEL 1: CSV INGESTION MODEL
    print("\n[MODEL 1/2] Ingesting CSV File: All_Type_of_Report_06-08-2026_02-38-20_PM.csv...")
    df_csv_raw = pd.read_csv(CSV_FILE_PATH, header=1)
    featured_csv = build_featured_dataset(df_csv_raw, is_api=False)
    path_csv = train_and_save_model(featured_csv, "CSV Ingestion Model", "paddy_common_ap_model_csv.joblib")
    
    # 2. BUILD MODEL 2: API INGESTION MODEL
    print("\n[MODEL 2/2] Ingesting data.gov.in API Records...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    api_records = []
    limit = 1000
    offset = 0
    while offset < 20000:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': str(limit),
            'offset': str(offset),
            'filters[state]': 'Andhra Pradesh',
            'filters[commodity]': 'Paddy(Common)'
        }
        url = f"https://api.data.gov.in/resource/{HISTORICAL_RESOURCE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            data = json.loads(res.read().decode('utf-8'))
            recs = data.get('records', [])
            if not recs:
                break
            for r in recs:
                api_records.append({
                    'date': r.get('Arrival_Date'),
                    'state': 'Andhra Pradesh',
                    'district': r.get('District'),
                    'market': r.get('Market'),
                    'commodity': 'Paddy(Common)',
                    'modal_price': r.get('Modal_Price'),
                    'min_price': r.get('Min_Price'),
                    'max_price': r.get('Max_Price')
                })
            offset += limit
            if len(recs) < limit:
                break
        except Exception as e:
            break
            
    df_api_raw = pd.DataFrame(api_records)
    df_api_raw['date'] = pd.to_datetime(df_api_raw['date'], format='%d/%m/%Y', errors='coerce')
    df_api_raw['modal_price'] = pd.to_numeric(df_api_raw['modal_price'], errors='coerce')
    df_api_raw['min_price'] = pd.to_numeric(df_api_raw['min_price'], errors='coerce')
    df_api_raw['max_price'] = pd.to_numeric(df_api_raw['max_price'], errors='coerce')
    
    featured_api = build_featured_dataset(df_api_raw, is_api=True)
    path_api = train_and_save_model(featured_api, "API Ingestion Model", "paddy_common_ap_model_api.joblib")
    
    print("\n="*85)
    print("SUCCESSFULLY CREATED BOTH DISTINCT MODEL ARTIFACTS IN C:\\Users\\Praveen\\OneDrive\\Desktop\\aa")
    print(f"1. CSV Model Artifact : {path_csv}")
    print(f"2. API Model Artifact : {path_api}")
    print("="*85)

if __name__ == '__main__':
    build_both_models()
