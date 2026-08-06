import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import datetime
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

RAW_ARRIVAL_CSV = r"C:\Users\Praveen\OneDrive\Desktop\All_Type_of_Report_(All_Grades)_05-08-2026_09-32-20_PM.csv"
PROCUREMENT_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'procurement_events.csv')
CLEANED_PRICE_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'ap_multi_market_rice_cleaned.csv')
WEATHER_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'weather_cleaned.csv')
FINAL_FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'enhanced_featured_dataset_real.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'enhanced_rice_model.joblib')

os.makedirs(os.path.dirname(PROCUREMENT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ============================================================
# STEP 1: CREATE PROCUREMENT & MSP EVENTS CSV
# ============================================================
def create_procurement_events():
    print("[1/5] Creating procurement_events.csv for Rice/Paddy in AP (2003 - 2026)...")
    
    dates = pd.date_range('2003-01-01', '2026-08-31', freq='D')
    p_df = pd.DataFrame({'date': dates})
    
    p_df['year'] = p_df['date'].dt.year
    p_df['month'] = p_df['date'].dt.month
    
    msp_map = {
        2003: 550.0, 2004: 560.0, 2005: 570.0, 2006: 620.0, 2007: 745.0, 2008: 850.0,
        2009: 950.0, 2010: 1000.0, 2011: 1080.0, 2012: 1250.0, 2013: 1310.0, 2014: 1360.0,
        2015: 1410.0, 2016: 1470.0, 2017: 1550.0, 2018: 1750.0, 2019: 1815.0, 2020: 1868.0,
        2021: 1940.0, 2022: 2040.0, 2023: 2183.0, 2024: 2300.0, 2025: 2320.0, 2026: 2320.0
    }
    p_df['msp_value'] = p_df['year'].map(msp_map).fillna(2320.0)
    
    p_df['is_procurement_active'] = np.where(p_df['month'].isin([10, 11, 12, 1, 4, 5, 6]), 1, 0)
    p_df['msp_announced'] = np.where(p_df['month'] == 6, 1, 0)
    p_df['msp_changed'] = np.where((p_df['month'] == 10) & (p_df['date'].dt.day == 1), 1, 0)
    p_df['procurement_started'] = np.where(((p_df['month'] == 10) | (p_df['month'] == 4)) & (p_df['date'].dt.day == 1), 1, 0)
    p_df['procurement_ended'] = np.where(((p_df['month'] == 1) | (p_df['month'] == 6)) & (p_df['date'].dt.day == 31), 1, 0)
    
    p_df = p_df[['date', 'is_procurement_active', 'msp_value', 'msp_announced', 'msp_changed', 'procurement_started', 'procurement_ended']]
    p_df.to_csv(PROCUREMENT_CSV, index=False)
    print(f"  Saved procurement events to: {PROCUREMENT_CSV}")
    return p_df

# ============================================================
# STEP 2: LOAD 100% REAL ARRIVALS CSV
# ============================================================
def load_arrival_data():
    print("[2/5] Processing 100% REAL user arrival quantity CSV (2021-2026)...")
    df = pd.read_csv(RAW_ARRIVAL_CSV, header=1)
    
    df.rename(columns={
        'Date': 'date_str',
        'Arrival Quantity 01-01-2021 to 03-08-2026': 'arrival_qty_mt'
    }, inplace=True)
    
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%m-%Y', errors='coerce')
    df['arrival_qty_mt'] = pd.to_numeric(df['arrival_qty_mt'], errors='coerce').fillna(0.0)
    
    daily_arrival = df.groupby('date')['arrival_qty_mt'].sum().reset_index()
    daily_arrival = daily_arrival.sort_values('date').reset_index(drop=True)
    
    daily_arrival['week_of_year'] = daily_arrival['date'].dt.isocalendar().week.astype(int)
    weekly_profile = daily_arrival.groupby('week_of_year')['arrival_qty_mt'].mean().to_dict()
    
    print(f"  Processed {len(daily_arrival)} daily arrival records from {daily_arrival['date'].min().strftime('%Y-%m-%d')} to {daily_arrival['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Mean Daily Arrival Quantity: {daily_arrival['arrival_qty_mt'].mean():.2f} Metric Tonnes")
    return daily_arrival, weekly_profile

# ============================================================
# STEP 3: ENHANCED FEATURE ENGINEERING ON REAL MANDI DATA
# ============================================================
def build_enhanced_features():
    print("[3/5] Building Enhanced Features on REAL Mandi Data...")
    
    procurement_df = create_procurement_events()
    arrival_df, weekly_profile = load_arrival_data()
    
    mandi_df = pd.read_csv(CLEANED_PRICE_CSV)
    weather_df = pd.read_csv(WEATHER_CSV)
    
    mandi_df['date'] = pd.to_datetime(mandi_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    mandi_df = mandi_df.sort_values(['Market', 'date']).reset_index(drop=True)
    
    processed_markets = []
    
    for market, m_group in mandi_df.groupby('Market'):
        m_df = m_group.sort_values('date').reset_index(drop=True)
        
        min_d, max_d = m_df['date'].min(), m_df['date'].max()
        full_dates = pd.date_range(min_d, max_d, freq='D')
        
        res = m_df.set_index('date').reindex(full_dates).reset_index()
        res.rename(columns={'index': 'date'}, inplace=True)
        res['Market'] = market
        
        res['modal_price'] = res['modal_price'].ffill().bfill()
        res['min_price'] = res['min_price'].ffill().bfill()
        res['max_price'] = res['max_price'].ffill().bfill()
        
        # Calendar units
        dow = res['date'].dt.dayofweek
        month = res['date'].dt.month
        day = res['date'].dt.day
        res['day_of_week'] = dow
        res['month'] = month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        
        # 1. MERGE REAL ARRIVALS
        res = pd.merge(res, arrival_df[['date', 'arrival_qty_mt']], on='date', how='left')
        res['arrival_qty_mt'] = res['arrival_qty_mt'].fillna(res['week_of_year'].map(weekly_profile)).fillna(350.0)
        
        res['arrival_lag_1'] = res['arrival_qty_mt'].shift(1).fillna(350.0)
        res['arrival_7d_mean'] = res['arrival_qty_mt'].shift(1).rolling(7, min_periods=1).mean().fillna(350.0)
        res['arrival_change_pct'] = res['arrival_qty_mt'].shift(1).pct_change(1, fill_method=None).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # 2. HOLIDAY & MARKET CLOSURE CALENDAR
        is_sunday = np.where(dow == 6, 1, 0)
        is_public_holiday = np.where(
            ((month == 1) & (day.isin([14, 15, 26]))) |
            ((month == 8) & (day == 15)) |
            ((month == 10) & (day == 2)) |
            ((month == 11) & (day == 1)), 1, 0
        )
        no_arrivals = np.where(res['arrival_qty_mt'] < 10.0, 1, 0)
        res['is_likely_non_trading_day'] = np.where((is_sunday == 1) | (is_public_holiday == 1) | (no_arrivals == 1), 1, 0)
        
        # 3. MERGE REAL WEATHER & RAINFALL ANOMALY
        res = pd.merge(res, weather_df[['date', 'rainfall', 'temp_max', 'temp_min', 'humidity']], on='date', how='left')
        res['temp_avg'] = (res['temp_max'] + res['temp_min']) / 2.0
        res['rainfall'] = res['rainfall'].fillna(0.0)
        res['humidity'] = res['humidity'].ffill().bfill()
        res['temp_avg'] = res['temp_avg'].ffill().bfill()
        
        res['rainfall_7d'] = res['rainfall'].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
        weekly_hist_rain = res.groupby('week_of_year')['rainfall_7d'].transform('mean')
        res['rainfall_anomaly_7d'] = res['rainfall_7d'] - weekly_hist_rain
        res['heavy_rain_flag'] = np.where(res['rainfall_7d'] > 40.0, 1, 0)
        res['dry_spell_flag'] = np.where((res['rainfall_7d'] < 1.0) & (res['month'].isin([6, 7, 8, 9])), 1, 0)
        
        # 4. MERGE PROCUREMENT / MSP FLAGS
        res = pd.merge(res, procurement_df, on='date', how='left')
        res['is_procurement_active'] = res['is_procurement_active'].fillna(0)
        res['msp_value'] = res['msp_value'].ffill().bfill()
        res['msp_announced'] = res['msp_announced'].fillna(0)
        res['msp_changed'] = res['msp_changed'].fillna(0)
        res['procurement_started'] = res['procurement_started'].fillna(0)
        res['procurement_ended'] = res['procurement_ended'].fillna(0)
        
        # Price Lags & Targets (STRICT PAST-ONLY)
        p = res['modal_price']
        res['target_modal_price'] = p.shift(-1)
        res['target_return'] = (res['target_modal_price'] - p) / (p + 1e-5)
        
        res['lag_1'] = p.shift(1)
        res['lag_3'] = p.shift(3)
        res['lag_7'] = p.shift(7)
        
        res['ret_1'] = (p - res['lag_1']) / (res['lag_1'] + 1e-5)
        res['ret_3'] = (p - res['lag_3']) / (res['lag_3'] + 1e-5)
        res['ret_7'] = (p - res['lag_7']) / (res['lag_7'] + 1e-5)
        
        res['rolling_mean_7'] = p.shift(1).rolling(7, min_periods=3).mean()
        res['rolling_mean_14'] = p.shift(1).rolling(14, min_periods=7).mean()
        res['rolling_std_7'] = p.shift(1).rolling(7, min_periods=3).std().fillna(0.0)
        res['rolling_std_14'] = p.shift(1).rolling(14, min_periods=7).std().fillna(0.0)
        
        res['ratio_ma7'] = p / (res['rolling_mean_7'] + 1e-5)
        res['ratio_ma14'] = p / (res['rolling_mean_14'] + 1e-5)
        
        res['is_harvest_season'] = np.where(res['month'].isin([10, 11, 12, 4, 5]), 1, 0)
        
        clean_m = res.dropna(subset=['lag_7', 'rolling_mean_14', 'target_modal_price']).reset_index(drop=True)
        processed_markets.append(clean_m)
        
    final_df = pd.concat(processed_markets, ignore_index=True)
    final_df = final_df.sort_values(['date', 'Market']).reset_index(drop=True)
    
    market_dummies = pd.get_dummies(final_df['Market'], prefix='mkt')
    final_df = pd.concat([final_df, market_dummies], axis=1)
    
    final_df.to_csv(FINAL_FEATURED_CSV, index=False)
    print(f"  Saved real featured dataset to: {FINAL_FEATURED_CSV}")
    print(f"  Dataset Shape: {final_df.shape} (from {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')})")
    return final_df

# ============================================================
# STEP 4: MODEL TRAINING & METRICS COMPARISON
# ============================================================
def train_enhanced_model():
    print("[4/5] Retraining Rice Model on REAL Data with Enhanced Features...")
    
    df = pd.read_csv(FINAL_FEATURED_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    market_cols = [c for c in df.columns if c.startswith('mkt_')]
    
    feature_cols = [
        # Price Lags & Ratios
        'ret_1', 'ret_3', 'ret_7', 'ratio_ma7', 'ratio_ma14', 'rolling_std_7', 'rolling_std_14',
        # 100% Real Arrivals
        'arrival_lag_1', 'arrival_7d_mean', 'arrival_change_pct',
        # Holiday / Closure Calendar
        'is_likely_non_trading_day',
        # Weather & Rainfall Anomaly
        'rainfall_7d', 'rainfall_anomaly_7d', 'heavy_rain_flag', 'dry_spell_flag',
        # Procurement & MSP
        'is_procurement_active', 'msp_value', 'msp_announced', 'msp_changed', 'procurement_started', 'procurement_ended',
        # Calendar & Seasonality
        'day_of_week', 'month', 'week_of_year', 'is_harvest_season'
    ] + market_cols
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train, y_train_ret = train_df[feature_cols], train_df['target_return']
    X_val, y_val_ret = val_df[feature_cols], val_df['target_return']
    X_test, y_test_ret = test_df[feature_cols], test_df['target_return']
    
    y_val_true = val_df['target_modal_price'].values
    y_test_true = test_df['target_modal_price'].values
    today_val_price = val_df['modal_price'].values
    today_test_price = test_df['modal_price'].values
    
    def evaluate(y_true, y_pred, y_today):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
        
        actual_diff = y_true - y_today
        pred_diff = y_pred - y_today
        dir_acc = np.mean((actual_diff >= 0) == (pred_diff >= 0)) * 100.0
        pred_std = np.std(pred_diff)
        return {'MAPE': round(mape, 2), 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'DirAcc': round(dir_acc, 2), 'PredStd': round(pred_std, 2)}

    results = {}
    
    # 1. Naive
    results['Naive'] = {
        'val': evaluate(y_val_true, today_val_price, today_val_price),
        'test': evaluate(y_test_true, today_test_price, today_test_price),
        'model': None
    }
    
    # 2. Ridge
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_train_ret)
    val_pred_ridge = today_val_price * (1.0 + ridge.predict(X_val))
    test_pred_ridge = today_test_price * (1.0 + ridge.predict(X_test))
    results['Ridge'] = {
        'val': evaluate(y_val_true, val_pred_ridge, today_val_price),
        'test': evaluate(y_test_true, test_pred_ridge, today_test_price),
        'model': ridge
    }
    
    # 3. XGBoost
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.03,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        xgb_model.fit(X_train, y_train_ret, eval_set=[(X_val, y_val_ret)], verbose=False)
        val_pred_xgb = today_val_price * (1.0 + xgb_model.predict(X_val))
        test_pred_xgb = today_test_price * (1.0 + xgb_model.predict(X_test))
        results['XGBoost'] = {
            'val': evaluate(y_val_true, val_pred_xgb, today_val_price),
            'test': evaluate(y_test_true, test_pred_xgb, today_test_price),
            'model': xgb_model
        }
        
    print("\n" + "="*95)
    print(f"{'REAL DATA ENHANCED MODEL RETRAINING SUMMARY':^95}")
    print("="*95)
    print(f"{'Model':<12} | {'Val MAPE':<10} | {'Val MAE':<10} | {'Val DirAcc':<10} | {'Val PredStd':<11} || {'Test MAPE':<10} | {'Test MAE':<10} | {'Test DirAcc':<10} | {'Test PredStd':<11}")
    print("-" * 95)
    
    best_model_name = None
    best_val_mape = float('inf')
    
    for name, res in results.items():
        v = res['val']
        t = res['test']
        print(f"{name:<12} | {v['MAPE']:>8.2f}% | Rs.{v['MAE']:>6.1f} | {v['DirAcc']:>8.1f}% | Rs.{v['PredStd']:>8.2f} || {t['MAPE']:>8.2f}% | Rs.{t['MAE']:>6.1f} | {t['DirAcc']:>8.1f}% | Rs.{t['PredStd']:>8.2f}")
        
        if v['MAPE'] < best_val_mape:
            best_val_mape = v['MAPE']
            best_model_name = name

    print("-" * 95)
    print(f">> BEST ENHANCED MODEL: {best_model_name} (Val MAPE: {best_val_mape:.2f}%)\n")
    
    if HAS_XGB and 'XGBoost' in results:
        xgb_m = results['XGBoost']['model']
        fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': xgb_m.feature_importances_})
        fi_df = fi_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        print("TOP 15 FEATURE IMPORTANCES (XGBoost):")
        for _, r in fi_df.head(15).iterrows():
            bar = '#' * int(r['Importance'] * 150)
            print(f"  {r['Feature']:25s}: {r['Importance']*100:5.2f}%  {bar}")
            
    val_preds = today_val_price * (1.0 + results['XGBoost']['model'].predict(X_val)) if HAS_XGB else today_val_price
    val_residuals = y_val_true - val_preds
    q10 = float(np.percentile(val_residuals, 10))
    q90 = float(np.percentile(val_residuals, 90))
    
    model_artifact = {
        'model_name': 'XGBoost' if HAS_XGB else best_model_name,
        'model_object': results['XGBoost']['model'] if HAS_XGB else results[best_model_name]['model'],
        'feature_cols': feature_cols,
        'metrics': results,
        'q10_residual': q10,
        'q90_residual': q90,
        'markets': list(df['Market'].unique()),
        'last_trained_date': df['date'].max().strftime('%Y-%m-%d')
    }
    
    joblib.dump(model_artifact, os.path.join(PROJECT_ROOT, 'backend', 'models', 'rice_ap_multi_market_model.joblib'))
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\n[5/5] Saved trained model artifact to: {MODEL_PATH}")

if __name__ == '__main__':
    df_feat = build_enhanced_features()
    train_enhanced_model()
