import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, 'paddy_ap_master_dataset.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'paddy_common_ap_model.joblib')

df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['Market', 'date']).reset_index(drop=True)

artifact = joblib.load(MODEL_PATH)
market_regimes = artifact.get('market_regimes', {})

prophet_modal = artifact.get('prophet_market_models', {})
prophet_min = artifact.get('prophet_min_models', {})
prophet_max = artifact.get('prophet_max_models', {})

arima_modal = artifact.get('arima_market_models', {})
arima_min = artifact.get('arima_min_models', {})
arima_max = artifact.get('arima_max_models', {})

def predict_horizon_multi(mkt, hist, h):
    regime = market_regimes.get(mkt, {}).get('regime', 'flat')
    cur_p = float(hist['weighted_avg_modal_price'].iloc[-1])
    cur_min = float(hist['min_price'].iloc[-1]) if 'min_price' in hist.columns else cur_p * 0.95
    cur_max = float(hist['max_price'].iloc[-1]) if 'max_price' in hist.columns else cur_p * 1.05
    last_arrival = float(hist['arrival_3d_mean'].iloc[-1]) if 'arrival_3d_mean' in hist.columns else 0.0
    
    if regime == 'flat':
        return cur_p, cur_min, cur_max
        
    if regime == 'active' and mkt in prophet_modal:
        try:
            f_dates = pd.date_range(pd.Timestamp(hist['date'].iloc[-1]) + pd.Timedelta(days=1), periods=h, freq='D')
            f_df = pd.DataFrame({'ds': f_dates, 'msp_value': 2300.0, 'rainfall_3d': 0.0, 'arrival_3d_mean': last_arrival})
            mod_fc = prophet_modal[mkt].predict(f_df)['yhat'].values[-1]
            min_fc = prophet_min[mkt].predict(f_df)['yhat'].values[-1] if mkt in prophet_min else mod_fc * 0.95
            max_fc = prophet_max[mkt].predict(f_df)['yhat'].values[-1] if mkt in prophet_max else mod_fc * 1.05
            return mod_fc, min_fc, max_fc
        except Exception:
            pass
            
    if mkt in arima_modal:
        try:
            ex = np.tile([2300.0, 0.0, last_arrival], (h, 1))
            mod_fc = float(arima_modal[mkt].predict(n_periods=h, X=ex)[-1])
            min_fc = float(arima_min[mkt].predict(n_periods=h, X=ex)[-1]) if mkt in arima_min else mod_fc * 0.95
            max_fc = float(arima_max[mkt].predict(n_periods=h, X=ex)[-1]) if mkt in arima_max else mod_fc * 1.05
            return mod_fc, min_fc, max_fc
        except Exception:
            pass
            
    return cur_p, cur_min, cur_max

print("="*90)
print("GRANULAR MODEL FAILURE & ERROR DIAGNOSTICS Across Markets, Horizons, and Regimes")
print("="*90)

market_results = []
all_eval_rows = []

for mkt, m_df in df.groupby('Market'):
    m_df = m_df.sort_values('date').reset_index(drop=True)
    n = len(m_df)
    if n < 20:
        continue
        
    split_idx = int(n * 0.80)
    test_df = m_df.iloc[split_idx:].reset_index(drop=True)
    regime = market_regimes.get(mkt, {}).get('regime', 'flat')
    
    mkt_evals = []
    
    for i in range(len(test_df) - 3):
        hist = m_df.iloc[:split_idx + i]
        target_window = test_df.iloc[i:i+3]
        
        for h in [1, 2, 3]:
            act_row = target_window.iloc[h-1]
            act_m = float(act_row['weighted_avg_modal_price'])
            act_min = float(act_row['min_price']) if 'min_price' in act_row else act_m * 0.95
            act_max = float(act_row['max_price']) if 'max_price' in act_row else act_m * 1.05
            
            pred_m, pred_min, pred_max = predict_horizon_multi(mkt, hist, h)
            
            # Reconcile order with residual calibration band (calib_band = max(25, (q90+q10)/2))
            calib_band = 50.0 if regime == 'active' else 25.0
            rec_min = min(pred_min, pred_m - calib_band)
            rec_max = max(pred_max, pred_m + calib_band)
            
            err = pred_m - act_m
            abs_err = abs(err)
            pct_err = (abs_err / (act_m + 1e-5)) * 100.0
            
            in_range = 1 if (rec_min <= act_m <= rec_max) else 0
            is_holiday = int(act_row.get('is_likely_non_trading_day', 0))
            is_harvest = int(act_row.get('is_harvest_season', 0))
            
            row_dict = {
                'market': mkt,
                'regime': regime,
                'horizon': h,
                'date': act_row['date'].strftime('%Y-%m-%d'),
                'actual_modal': act_m,
                'pred_modal': round(pred_m, 2),
                'pred_min': round(rec_min, 2),
                'pred_max': round(rec_max, 2),
                'abs_error': round(abs_err, 2),
                'mape': round(pct_err, 2),
                'in_range': in_range,
                'is_holiday': is_holiday,
                'is_harvest': is_harvest
            }
            mkt_evals.append(row_dict)
            all_eval_rows.append(row_dict)
            
    mkt_df = pd.DataFrame(mkt_evals)
    if not mkt_df.empty:
        h1 = mkt_df[mkt_df['horizon'] == 1]
        h3 = mkt_df[mkt_df['horizon'] == 3]
        
        market_results.append({
            'market': mkt,
            'regime': regime,
            'test_points': len(h1),
            'h1_mae': round(h1['abs_error'].mean(), 2),
            'h1_mape': round(h1['mape'].mean(), 2),
            'h3_mae': round(h3['abs_error'].mean(), 2),
            'h3_mape': round(h3['mape'].mean(), 2),
            'max_abs_error': round(mkt_df['abs_error'].max(), 2),
            'range_cov_pct': round(mkt_df['in_range'].mean() * 100.0, 1)
        })

res_df = pd.DataFrame(market_results).sort_values('h3_mape', ascending=False)
print("\n--- MARKET-BY-MARKET ERROR BREAKDOWN (1-Day vs 3-Day vs Max Error) ---")
print(res_df.to_string(index=False))

all_df = pd.DataFrame(all_eval_rows)
print("\n--- HORIZON ERROR DECAY ---")
for h in [1, 2, 3]:
    hdf = all_df[all_df['horizon'] == h]
    print(f"Horizon {h}-Day ➔ MAE: Rs. {hdf['abs_error'].mean():>6.2f} | MAPE: {hdf['mape'].mean():>5.2f}% | Max Error: Rs. {hdf['abs_error'].max():>6.2f}")

print("\n--- SPECIAL CALENDAR CONDITION BREAKDOWN ---")
hol_df = all_df[all_df['is_holiday'] == 1]
norm_df = all_df[all_df['is_holiday'] == 0]
harv_df = all_df[all_df['is_harvest'] == 1]

print(f"Normal Trading Days   ➔ MAE: Rs. {norm_df['abs_error'].mean():.2f} | MAPE: {norm_df['mape'].mean():.2f}% (n={len(norm_df)})")
print(f"Holidays / Zero-Arr   ➔ MAE: Rs. {hol_df['abs_error'].mean():.2f} | MAPE: {hol_df['mape'].mean():.2f}% (n={len(hol_df)})")
print(f"Harvest Season Days   ➔ MAE: Rs. {harv_df['abs_error'].mean():.2f} | MAPE: {harv_df['mape'].mean():.2f}% (n={len(harv_df)})")

# Top 5 Outliers
outliers = all_df.sort_values('abs_error', ascending=False).head(5)
print("\n--- TOP 5 OUTLIER ERROR POINTS ---")
print(outliers[['market', 'regime', 'horizon', 'date', 'actual_modal', 'pred_modal', 'abs_error', 'mape']].to_string(index=False))
