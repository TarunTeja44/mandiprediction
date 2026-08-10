import os
import sys
import json
import urllib.request
import urllib.parse
import ssl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

MARKET_COORDS = {
    'Kuchinapudi': {'lat': 15.90, 'lon': 80.47},
    'Tiruvuru': {'lat': 16.50, 'lon': 80.64},
    'Nandigama': {'lat': 16.50, 'lon': 80.64},
    'Rapur': {'lat': 14.44, 'lon': 79.98},
    'Kovvur': {'lat': 17.00, 'lon': 81.78},
    'Polavaram': {'lat': 16.71, 'lon': 81.10},
    'Jaggampet': {'lat': 17.00, 'lon': 81.78},
    'Mylavaram': {'lat': 16.50, 'lon': 80.64},
    'Rampachodvaram': {'lat': 17.83, 'lon': 81.88}
}

df = pd.read_csv('paddy_ap_master_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df['Market'] = df['Market'].astype(str).str.replace(' APMC', '').str.strip()
df = df.sort_values(['Market', 'date']).reset_index(drop=True)

# Feature engineering
processed = []
for mkt, m_df in df.groupby('Market'):
    res = m_df.sort_values('date').reset_index(drop=True)
    p = res['weighted_avg_modal_price']
    res['arrival_3d_mean'] = res['arrival_qty_mt'].shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
    res['rainfall_3d'] = res['rainfall'].shift(1).rolling(3, min_periods=1).sum().fillna(0.0) if 'rainfall' in res.columns else 0.0
    res['rainfall_7d'] = res['rainfall'].shift(1).rolling(7, min_periods=1).sum().fillna(0.0) if 'rainfall' in res.columns else 0.0
    res['rolling_mean_3'] = p.shift(1).rolling(3, min_periods=1).mean()
    res['rolling_std_3'] = p.shift(1).rolling(3, min_periods=1).std().fillna(0.0)
    res['heavy_rain_flag'] = np.where(res['rainfall_7d'] > 40.0, 1, 0)
    res['is_likely_non_trading_day'] = np.where((res['date'].dt.dayofweek == 6) | (res['arrival_qty_mt'] == 0.0), 1, 0)
    
    if 'temp_max' in res.columns and 'temp_min' in res.columns:
        res['heat_stress_days_7d'] = (res['temp_max'].shift(1) > 35.0).astype(float).rolling(7, min_periods=1).sum()
        res['temp_diurnal'] = res['temp_max'] - res['temp_min']
    else:
        res['heat_stress_days_7d'] = 0.0
        res['temp_diurnal'] = 0.0
        
    dry = (res['rainfall'].shift(1) < 1.0).astype(int) if 'rainfall' in res.columns else pd.Series(0, index=res.index)
    res['consecutive_dry_days'] = dry.groupby((dry != dry.shift()).cumsum()).cumsum()
    res['dry_spell_5d'] = (res['consecutive_dry_days'] >= 5).astype(int)
    
    res['month'] = res['date'].dt.month
    res['is_harvest_season'] = np.where(res['month'].isin([10, 11, 12, 4, 5]), 1, 0)
    res['is_monsoon_season'] = np.where(res['month'].isin([6, 7, 8, 9]), 1, 0)
    res['rain_harvest_interaction'] = res['rainfall_7d'] * res['is_harvest_season']
    res['non_trading_lag1_interaction'] = res['is_likely_non_trading_day'] * res['lag_1'] if 'lag_1' in res.columns else 0.0
    res['rain_arrival_interaction'] = res['heavy_rain_flag'] * res['arrival_3d_mean']
    res['msp_value'] = 2300.0
    processed.append(res)

featured_df = pd.concat(processed, ignore_index=True)

# Feature column list for XGBoost (numeric only)
EXCLUDE_COLS = ['date', 'Market', 'District', 'State', 'Commodity', 'weighted_avg_modal_price', 'min_price', 'max_price', 'spread', 'log_spread',
                'target_modal_price', 'target_min_price', 'target_max_price', 'target_spread', 'target_log_spread', 'target_return', 'target_min_return', 'target_max_return', 'msp_value']
FEATURE_COLS = [c for c in featured_df.columns if c not in EXCLUDE_COLS and not c.startswith('target_') and featured_df[c].dtype != 'object']

print(f"✓ Feature matrix built with {len(FEATURE_COLS)} engineered columns.")

# Compute regime classification on 80% train split (FIX 1)
market_regimes_new = {}
market_regimes_old = {}

for mkt, m_df in featured_df.groupby('Market'):
    m_df = m_df.sort_values('date').reset_index(drop=True)
    n = len(m_df)
    train_len = int(n * 0.80)
    
    # Old (full series leak)
    std_old = float(m_df['weighted_avg_modal_price'].std())
    reg_old = 'flat' if std_old < 5.0 else ('low_volatility' if std_old < 30.0 else 'active')
    market_regimes_old[mkt] = reg_old
    
    # New (80% train only - FIX 1)
    std_new = float(m_df['weighted_avg_modal_price'].iloc[:train_len].std())
    reg_new = 'flat' if std_new < 5.0 else ('low_volatility' if std_new < 30.0 else 'active')
    market_regimes_new[mkt] = reg_new

print("\n--- REGIME ASSIGNMENT COMPARISON (FIX 1) ---")
for mkt in sorted(market_regimes_new.keys()):
    print(f"Market: {mkt:20s} | Old (Full Leak): {market_regimes_old[mkt]:15s} | New (80% Train): {market_regimes_new[mkt]:15s}")

# ======================== RUN BACKTEST COMPARISON ========================
print("\n" + "="*90)
print("COMPUTING OUT-OF-SAMPLE 3-DAY MAPE COMPARISON (OLD PROPHET/ARIMA vs NEW PROPHET+XGBOOST ENSEMBLE)")
print("="*90)

old_results = {}
new_results = {}

for mkt, m_df in featured_df.groupby('Market'):
    m_df = m_df.sort_values('date').reset_index(drop=True)
    n = len(m_df)
    if n < 20:
        continue
    split_idx = int(n * 0.80)
    test_df = m_df.iloc[split_idx:].reset_index(drop=True)
    
    reg_new = market_regimes_new[mkt]
    reg_old = market_regimes_old[mkt]
    
    old_preds_3d = []
    new_preds_3d = []
    actuals_3d = []
    
    for i in range(len(test_df) - 3):
        hist = m_df.iloc[:split_idx + i].copy()
        target_w = test_df.iloc[i:i+3]
        cur_p = float(hist['weighted_avg_modal_price'].iloc[-1])
        act_3d = target_w['weighted_avg_modal_price'].values
        actuals_3d.append(act_3d)
        
        # 1. OLD PREDICTION (Prophet / ARIMA alone, with msp_value)
        if reg_old == 'active':
            f_dates = pd.date_range(pd.Timestamp(hist['date'].iloc[-1]) + pd.Timedelta(days=1), periods=3, freq='D')
            f_df_old = pd.DataFrame({'ds': f_dates, 'msp_value': 2300.0, 'rainfall_3d': 0.0, 'arrival_3d_mean': float(hist['arrival_3d_mean'].iloc[-1])})
            pm_m = Prophet(changepoint_prior_scale=0.1, weekly_seasonality=True, yearly_seasonality=False)
            pm_m.add_regressor('msp_value')
            pm_m.add_regressor('rainfall_3d')
            pm_m.add_regressor('arrival_3d_mean')
            pm_m.fit(hist[['date', 'weighted_avg_modal_price', 'msp_value', 'rainfall_3d', 'arrival_3d_mean']].rename(columns={'date': 'ds', 'weighted_avg_modal_price': 'y'}))
            p_old = pm_m.predict(f_df_old)['yhat'].values
        elif reg_old == 'low_volatility':
            ex_old = np.tile([2300.0, 0.0, float(hist['arrival_3d_mean'].iloc[-1])], (3, 1))
            ar_m = pm.auto_arima(hist['weighted_avg_modal_price'].values, X=hist[['msp_value', 'rainfall_3d', 'arrival_3d_mean']].values, seasonal=False, suppress_warnings=True)
            p_old = ar_m.predict(n_periods=3, X=ex_old)
        else:
            p_old = np.full(3, cur_p)
        old_preds_3d.append(p_old)
        
        # 2. NEW PREDICTION (Fix 2: No msp_value, Fix 4: Prophet + XGBoost Ensemble for Active)
        if reg_new == 'active':
            # Prophet (No msp_value - FIX 2)
            f_dates = pd.date_range(pd.Timestamp(hist['date'].iloc[-1]) + pd.Timedelta(days=1), periods=3, freq='D')
            f_df_new = pd.DataFrame({'ds': f_dates, 'rainfall_3d': float(hist['rainfall_3d'].iloc[-1]), 'arrival_3d_mean': float(hist['arrival_3d_mean'].iloc[-1])})
            pm_m2 = Prophet(changepoint_prior_scale=0.1, weekly_seasonality=True, yearly_seasonality=False)
            pm_m2.add_regressor('rainfall_3d')
            pm_m2.add_regressor('arrival_3d_mean')
            pm_m2.fit(hist[['date', 'weighted_avg_modal_price', 'rainfall_3d', 'arrival_3d_mean']].rename(columns={'date': 'ds', 'weighted_avg_modal_price': 'y'}))
            p_proph = pm_m2.predict(f_df_new)['yhat'].values
            
            # Direct Multi-Horizon XGBoost (FIX 4)
            p_xgb = []
            for h in [1, 2, 3]:
                hist_copy = hist.copy()
                hist_copy['target_h'] = hist_copy['weighted_avg_modal_price'].shift(-h)
                clean_h = hist_copy.dropna(subset=['target_h'] + FEATURE_COLS)
                if len(clean_h) >= 15:
                    X_tr = clean_h[FEATURE_COLS].fillna(0.0)
                    y_tr = clean_h['target_h']
                    xgb_h = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
                    xgb_h.fit(X_tr, y_tr)
                    
                    latest_feat = hist[FEATURE_COLS].iloc[[-1]].fillna(0.0)
                    pred_val = float(xgb_h.predict(latest_feat)[0])
                else:
                    pred_val = cur_p
                p_xgb.append(pred_val)
                
            p_xgb = np.array(p_xgb)
            # Ensemble (0.5 Prophet + 0.5 XGBoost - FIX 4)
            p_new = 0.5 * p_proph + 0.5 * p_xgb
            
        elif reg_new == 'low_volatility':
            # Auto-ARIMA (No msp_value - FIX 2)
            ex_new = np.tile([float(hist['rainfall_3d'].iloc[-1]), float(hist['arrival_3d_mean'].iloc[-1])], (3, 1))
            ar_m2 = pm.auto_arima(hist['weighted_avg_modal_price'].values, X=hist[['rainfall_3d', 'arrival_3d_mean']].values, seasonal=False, suppress_warnings=True)
            p_new = ar_m2.predict(n_periods=3, X=ex_new)
        else:
            p_new = np.full(3, cur_p)
            
        new_preds_3d.append(p_new)
        
    old_arr = np.array(old_preds_3d)
    new_arr = np.array(new_preds_3d)
    act_arr = np.array(actuals_3d)
    
    # Compute 3-Day aggregate MAPE per market
    mape_old_m = mean_absolute_percentage_error(act_arr, old_arr) * 100.0
    mape_new_m = mean_absolute_percentage_error(act_arr, new_arr) * 100.0
    mae_old_m = mean_absolute_error(act_arr, old_arr)
    mae_new_m = mean_absolute_error(act_arr, new_arr)
    
    old_results[mkt] = {'mape': mape_old_m, 'mae': mae_old_m}
    new_results[mkt] = {'mape': mape_new_m, 'mae': mae_new_m}

print(f"\n{'Market':<20} | {'Regime':<15} | {'Old 3-Day MAPE (%)':<20} | {'New Ensemble 3-Day MAPE (%)':<25} | {'Accuracy Impact':<20}")
print("-" * 105)

for mkt in sorted(old_results.keys()):
    reg = market_regimes_new[mkt]
    o_m = old_results[mkt]['mape']
    n_m = new_results[mkt]['mape']
    diff = n_m - o_m
    impact = f"🟢 Improved ({abs(diff):.2f}%)" if diff < -0.01 else (f"🔴 Slight (+{diff:.2f}%)" if diff > 0.01 else "🟡 Equal (0.00%)")
    print(f"{mkt:<20} | {reg:<15} | {o_m:>18.2f}% | {n_m:>23.2f}% | {impact:<20}")
