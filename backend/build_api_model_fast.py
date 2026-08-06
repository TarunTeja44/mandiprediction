import os
import sys
import io
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

TARGET_DIR = r"C:\Users\Praveen\OneDrive\Desktop\aa"
os.makedirs(TARGET_DIR, exist_ok=True)

CLEANED_API_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_top10_ap_cleaned.csv')
DISTRICT_WEATHER_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'district_open_meteo_weather_real.csv')
PROCUREMENT_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'procurement_events.csv')

def build_api_model():
    print("="*85)
    print("BUILDING API INGESTION MODEL ARTIFACT (paddy_common_ap_model_api.joblib)")
    print("="*85)
    
    df = pd.read_csv(CLEANED_API_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df['Market'] = df['Market'].str.replace(' APMC', '', regex=False).str.strip()
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
        
        dow = res['date'].dt.dayofweek
        month = res['date'].dt.month
        day = res['date'].dt.day
        res['day_of_week'] = dow
        res['month'] = month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        
        res['arrival_qty_mt'] = 0.0
        res['arrival_lag_1'] = 0.0
        res['arrival_7d_mean'] = 0.0
        res['arrival_change_pct'] = 0.0
        
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
    featured_df = pd.concat([final_df, market_dummies, district_dummies], axis=1)
    
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
    
    artifact = {
        'model_name': 'Paddy(Common) AP - data.gov.in API Ingestion Model',
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
    
    save_path = os.path.join(TARGET_DIR, "paddy_common_ap_model_api.joblib")
    joblib.dump(artifact, save_path)
    print(f"Saved API model artifact to: {save_path}")

if __name__ == '__main__':
    build_api_model()
