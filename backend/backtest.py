import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib

def run_backtest(market="Machilipatnam", backtest_days=90):
    """
    Walk-forward backtest: For each of the last N days in the dataset,
    use only past data to predict next-day price, then compare with actual.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    
    model_path = os.path.join(project_root, 'backend', 'models', 'rice_ap_multi_market_model.joblib')
    csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'multi_market_featured.csv')
    
    artifact = joblib.load(model_path)
    feature_cols = artifact['feature_cols']
    
    # Load all models
    models = {}
    for name in ['Naive', 'Ridge', 'XGBoost']:
        if name in artifact['metrics'] and artifact['metrics'][name]['model'] is not None:
            models[name] = artifact['metrics'][name]['model']
        elif name == 'Naive':
            models[name] = None
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    m_df = df[df['Market'] == market].sort_values('date').reset_index(drop=True)
    
    if len(m_df) < backtest_days + 1:
        backtest_days = len(m_df) - 1
    
    backtest_start_idx = len(m_df) - backtest_days
    
    print(f"{'='*80}")
    print(f"WALK-FORWARD BACKTEST: {market}, AP | Rice")
    print(f"{'='*80}")
    print(f"Backtest Window : Last {backtest_days} days")
    print(f"Date Range      : {m_df['date'].iloc[backtest_start_idx].strftime('%Y-%m-%d')} to {m_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Models Evaluated: {list(models.keys())}")
    print()
    
    # Storage for backtest results per model
    bt_results = {name: {'actual': [], 'predicted': [], 'dates': []} for name in models}
    
    all_markets = [c.replace('mkt_', '') for c in feature_cols if c.startswith('mkt_')]
    
    for i in range(backtest_start_idx, len(m_df) - 1):
        row = m_df.iloc[i]
        next_row = m_df.iloc[i + 1]
        
        today_price = float(row['modal_price'])
        actual_tomorrow = float(next_row['modal_price'])
        dt = row['date']
        
        # Build feature vector from this row (all features already computed from past data)
        feat_dict = {}
        for col in feature_cols:
            if col.startswith('mkt_'):
                mkt_name = col.replace('mkt_', '')
                feat_dict[col] = 1 if mkt_name == market else 0
            else:
                feat_dict[col] = float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
        
        X_curr = pd.DataFrame([feat_dict])[feature_cols]
        
        for name, model_obj in models.items():
            if name == 'Naive' or model_obj is None:
                pred_price = today_price
            else:
                pred_ret = float(model_obj.predict(X_curr)[0])
                pred_price = today_price * (1.0 + pred_ret)
            
            bt_results[name]['actual'].append(actual_tomorrow)
            bt_results[name]['predicted'].append(pred_price)
            bt_results[name]['dates'].append(dt)
    
    # Compute metrics per model
    print(f"{'='*80}")
    print(f"{'BACKTEST RESULTS SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Model':<12} | {'MAPE':<8} | {'MAE':<12} | {'RMSE':<12} | {'Dir Acc':<10} | {'Pred Std':<10}")
    print(f"{'-'*80}")
    
    best_name = None
    best_mape = float('inf')
    
    for name, res in bt_results.items():
        actual = np.array(res['actual'])
        predicted = np.array(res['predicted'])
        
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100.0
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Direction accuracy: did the price move match prediction direction?
        today_prices = np.array([float(m_df.iloc[backtest_start_idx + j]['modal_price']) for j in range(len(actual))])
        actual_dir = actual - today_prices
        pred_dir = predicted - today_prices
        dir_acc = np.mean((actual_dir >= 0) == (pred_dir >= 0)) * 100.0
        
        pred_diff_std = np.std(predicted - today_prices)
        
        print(f"{name:<12} | {mape:>5.2f}% | Rs. {mae:>7.2f} | Rs. {rmse:>7.2f} | {dir_acc:>7.1f}% | Rs. {pred_diff_std:>6.2f}")
        
        if mape < best_mape:
            best_mape = mape
            best_name = name
    
    print(f"{'-'*80}")
    print(f">> Best Backtest Model: {best_name} (MAPE: {best_mape:.2f}%)\n")
    
    # Print day-by-day backtest log for XGBoost (or best non-naive)
    log_model = 'XGBoost' if 'XGBoost' in bt_results else best_name
    print(f"{'='*80}")
    print(f"DAY-BY-DAY BACKTEST LOG ({log_model} Model - {market})")
    print(f"{'='*80}")
    print(f"{'Date':<12} | {'Today Price':<14} | {'Actual Tmrw':<14} | {'Predicted Tmrw':<15} | {'Error':<12} | {'Dir?':<6}")
    print(f"{'-'*80}")
    
    actual_arr = np.array(bt_results[log_model]['actual'])
    pred_arr = np.array(bt_results[log_model]['predicted'])
    dates_arr = bt_results[log_model]['dates']
    
    correct_count = 0
    total_count = 0
    
    for j in range(len(actual_arr)):
        today_p = float(m_df.iloc[backtest_start_idx + j]['modal_price'])
        actual_t = actual_arr[j]
        pred_t = pred_arr[j]
        error = pred_t - actual_t
        
        actual_moved_up = actual_t >= today_p
        pred_moved_up = pred_t >= today_p
        dir_correct = actual_moved_up == pred_moved_up
        if dir_correct:
            correct_count += 1
        total_count += 1
        
        dir_symbol = "OK" if dir_correct else "MISS"
        
        print(f"{dates_arr[j].strftime('%Y-%m-%d'):<12} | Rs. {today_p:>8.1f}   | Rs. {actual_t:>8.1f}   | Rs. {pred_t:>8.1f}     | Rs. {error:>+8.1f} | {dir_symbol}")
    
    print(f"{'-'*80}")
    print(f"Direction Accuracy over {total_count} days: {correct_count}/{total_count} = {correct_count/total_count*100:.1f}%")
    print(f"{'='*80}")

if __name__ == '__main__':
    run_backtest(market="Machilipatnam", backtest_days=90)
