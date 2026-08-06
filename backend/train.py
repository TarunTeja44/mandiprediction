import os
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

def train_and_evaluate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'featured_dataset.csv')
    models_dir = os.path.join(project_root, 'backend', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("[Model Trainer] Loading featured dataset...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate price return targets for stationary model training
    df['today_price'] = df['modal_price']
    df['target_diff'] = df['target_modal_price'] - df['today_price']
    df['target_return'] = df['target_diff'] / df['today_price']
    
    # Stationary features
    df['ret_1'] = (df['modal_price'] - df['lag_1']) / (df['lag_1'] + 1e-5)
    df['ret_3'] = (df['modal_price'] - df['lag_3']) / (df['lag_3'] + 1e-5)
    df['ret_7'] = (df['modal_price'] - df['lag_7']) / (df['lag_7'] + 1e-5)
    df['ratio_ma7'] = df['modal_price'] / (df['rolling_mean_7'] + 1e-5)
    df['ratio_ma14'] = df['modal_price'] / (df['rolling_mean_14'] + 1e-5)
    
    feature_cols = [
        'ret_1', 'ret_3', 'ret_7',
        'ratio_ma7', 'ratio_ma14',
        'rolling_std_7', 'rolling_std_14',
        'day_of_week', 'month', 'week_of_year',
        'is_harvest_season',
        'rainfall_7d', 'temp_avg_7d', 'humidity_avg_7d'
    ]
    
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
    
    today_val_price = val_df['today_price'].values
    today_test_price = test_df['today_price'].values
    
    print(f"\n[Model Trainer] Dataset Split (Chronological):")
    print(f"  Train Set : {len(train_df)} days ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Val Set   : {len(val_df)} days ({val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Test Set  : {len(test_df)} days ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")
    
    def evaluate_predictions(y_true, y_pred, y_today):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
        
        actual_diff = y_true - y_today
        pred_diff = y_pred - y_today
        dir_acc = np.mean((actual_diff >= 0) == (pred_diff >= 0)) * 100.0
        return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2), 'DirAcc': round(dir_acc, 2)}

    results = {}
    
    # 1. NAIVE BASELINE (Tomorrow = Today)
    val_pred_naive = today_val_price
    test_pred_naive = today_test_price
    results['Naive'] = {
        'val': evaluate_predictions(y_val_true, val_pred_naive, today_val_price),
        'test': evaluate_predictions(y_test_true, test_pred_naive, today_test_price),
        'model': None
    }
    
    # 2. RIDGE REGRESSION (Predicting return)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train_ret)
    val_pred_ridge = today_val_price * (1.0 + ridge.predict(X_val))
    test_pred_ridge = today_test_price * (1.0 + ridge.predict(X_test))
    results['Ridge'] = {
        'val': evaluate_predictions(y_val_true, val_pred_ridge, today_val_price),
        'test': evaluate_predictions(y_test_true, test_pred_ridge, today_test_price),
        'model': ridge
    }
    
    # 3. CONSERVATIVE XGBOOST (Predicting return)
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            learning_rate=0.03,
            random_state=42
        )
        xgb_model.fit(
            X_train, y_train_ret,
            eval_set=[(X_val, y_val_ret)],
            verbose=False
        )
        val_pred_xgb = today_val_price * (1.0 + xgb_model.predict(X_val))
        test_pred_xgb = today_test_price * (1.0 + xgb_model.predict(X_test))
        results['XGBoost'] = {
            'val': evaluate_predictions(y_val_true, val_pred_xgb, today_val_price),
            'test': evaluate_predictions(y_test_true, test_pred_xgb, today_test_price),
            'model': xgb_model
        }

    # Print Evaluation Table
    print("\n" + "="*80)
    print(f"{'MODEL PERFORMANCE EVALUATION SUMMARY (Stationary Return Prediction)':^80}")
    print("="*80)
    print(f"{'Model':<12} | {'Val MAPE':<10} | {'Val MAE':<10} | {'Val DirAcc':<10} || {'Test MAPE':<10} | {'Test MAE':<10} | {'Test DirAcc':<10}")
    print("-" * 80)
    
    best_model_name = None
    best_val_mape = float('inf')
    
    for name, res in results.items():
        v = res['val']
        t = res['test']
        print(f"{name:<12} | {v['MAPE']:>8.2f}% | Rs.{v['MAE']:>6.1f} | {v['DirAcc']:>8.1f}% || {t['MAPE']:>8.2f}% | Rs.{t['MAE']:>6.1f} | {t['DirAcc']:>8.1f}%")
        
        if v['MAPE'] < best_val_mape:
            best_val_mape = v['MAPE']
            best_model_name = name

    print("-" * 80)
    print(f">> BEST SELECTED MODEL BASED ON VAL MAPE: {best_model_name} (MAPE: {best_val_mape:.2f}%)\n")
    
    best_model_obj = results[best_model_name]['model']
    
    # Calculate residual error quantiles on validation set
    if best_model_name == 'XGBoost':
        val_preds = today_val_price * (1.0 + best_model_obj.predict(X_val))
    elif best_model_name == 'Ridge':
        val_preds = today_val_price * (1.0 + best_model_obj.predict(X_val))
    else:
        val_preds = today_val_price
        
    val_residuals = y_val_true - val_preds
    q10 = float(np.percentile(val_residuals, 10))
    q90 = float(np.percentile(val_residuals, 90))
    
    # Save model artifact
    model_artifact = {
        'model_name': best_model_name,
        'model_object': best_model_obj,
        'feature_cols': feature_cols,
        'metrics': results,
        'q10_residual': q10,
        'q90_residual': q90,
        'last_trained_date': df['date'].max().strftime('%Y-%m-%d')
    }
    
    model_path = os.path.join(models_dir, 'rice_tadepalligudem_model.joblib')
    joblib.dump(model_artifact, model_path)
    print(f"[Model Trainer] Saved best model artifact to: {model_path}")
    
    return model_artifact

if __name__ == '__main__':
    train_and_evaluate()
