import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
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

FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'paddy_common_ap_model.joblib')

def train_paddy_model():
    print("="*85)
    print("TRAINING MODEL FOR WEIGHTED AVERAGE MODAL PRICE PREDICTION")
    print("="*85)
    
    df = pd.read_csv(FEATURED_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    mkt_cols = [c for c in df.columns if c.startswith('mkt_')]
    dist_cols = [c for c in df.columns if c.startswith('dist_')]
    
    feature_cols = [
        # Price Lags & Ratios
        'ret_1', 'ret_3', 'ret_7', 'ratio_ma7', 'ratio_ma14', 'rolling_std_7', 'rolling_std_14',
        # Real Arrivals
        'arrival_lag_1', 'arrival_7d_mean', 'arrival_change_pct',
        # Holiday / Closure Calendar
        'is_likely_non_trading_day',
        # Real Weather & Rainfall Anomaly
        'rainfall_7d', 'rainfall_anomaly_7d', 'heavy_rain_flag', 'dry_spell_flag',
        # Procurement & MSP
        'is_procurement_active', 'msp_value', 'msp_announced', 'msp_changed', 'procurement_started', 'procurement_ended',
        # Calendar & Seasonality
        'day_of_week', 'month', 'week_of_year', 'is_harvest_season'
    ] + mkt_cols + dist_cols
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train = train_df[feature_cols].fillna(0.0)
    y_train_ret = train_df['target_return'].fillna(0.0)
    
    X_val = val_df[feature_cols].fillna(0.0)
    y_val_ret = val_df['target_return'].fillna(0.0)
    
    X_test = test_df[feature_cols].fillna(0.0)
    y_test_ret = test_df['target_return'].fillna(0.0)
    
    y_val_true = val_df['target_modal_price'].values
    y_test_true = test_df['target_modal_price'].values
    today_val_price = val_df['weighted_avg_modal_price'].values
    today_test_price = test_df['weighted_avg_modal_price'].values
    
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
            n_estimators=300,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.03,
            gamma=0.1,
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
    print(f"{'WEIGHTED AVERAGE MODAL PRICE MODEL TRAINING SUMMARY':^95}")
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
    print(f">> BEST MODEL: {best_model_name} (Val MAPE: {best_val_mape:.2f}%)\n")
    
    if HAS_XGB and 'XGBoost' in results:
        xgb_m = results['XGBoost']['model']
        fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': xgb_m.feature_importances_})
        fi_df = fi_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        print("TOP 15 FEATURE IMPORTANCES (Weighted Average Modal Price XGBoost):")
        for _, r in fi_df.head(15).iterrows():
            bar = '#' * int(r['Importance'] * 150)
            print(f"  {r['Feature']:25s}: {r['Importance']*100:5.2f}%  {bar}")
            
    val_preds = today_val_price * (1.0 + results['XGBoost']['model'].predict(X_val)) if HAS_XGB else today_val_price
    val_residuals = y_val_true - val_preds
    q10 = float(np.percentile(val_residuals, 10))
    q90 = float(np.percentile(val_residuals, 90))
    
    model_artifact = {
        'target_type': 'Weighted Average Modal Price (60% Modal + 20% Min + 20% Max)',
        'commodity': 'Paddy(Common)',
        'model_name': 'XGBoost' if HAS_XGB else best_model_name,
        'model_object': results['XGBoost']['model'] if HAS_XGB else results[best_model_name]['model'],
        'feature_cols': feature_cols,
        'metrics': results,
        'q10_residual': q10,
        'q90_residual': q90,
        'markets': list(df['Market'].unique()),
        'districts': list(df['District'].unique()),
        'last_trained_date': df['date'].max().strftime('%Y-%m-%d')
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_artifact, MODEL_PATH)
    joblib.dump(model_artifact, os.path.join(PROJECT_ROOT, 'backend', 'models', 'rice_ap_multi_market_model.joblib'))
    joblib.dump(model_artifact, os.path.join(PROJECT_ROOT, 'backend', 'models', 'enhanced_rice_model.joblib'))
    
    print(f"\nSaved Weighted Average Modal Price model artifact to: {MODEL_PATH}")

if __name__ == '__main__':
    train_paddy_model()
