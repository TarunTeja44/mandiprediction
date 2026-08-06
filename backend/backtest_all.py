import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib

def run_paddy_backtest(market="Tiruvuru", backtest_days=90):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    
    model_path = os.path.join(project_root, 'backend', 'models', 'paddy_common_ap_model.joblib')
    csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'paddy_common_2021_2026_featured.csv')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, 'backend', 'models', 'enhanced_rice_model.joblib')
        csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'enhanced_featured_dataset_real.csv')
        
    artifact = joblib.load(model_path)
    feature_cols = artifact['feature_cols']
    
    models = {}
    for name in ['Naive', 'Ridge', 'XGBoost']:
        if name in artifact['metrics'] and artifact['metrics'][name]['model'] is not None:
            models[name] = artifact['metrics'][name]['model']
        elif name == 'Naive':
            models[name] = None
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    m_df = df[df['Market'].str.lower() == market.lower()].sort_values('date').reset_index(drop=True)
    if m_df.empty:
        return
    
    district_name = str(m_df['District'].iloc[-1]) if 'District' in m_df.columns else "AP"
    
    if len(m_df) < backtest_days + 1:
        backtest_days = len(m_df) - 1
    
    backtest_start_idx = len(m_df) - backtest_days
    
    print(f"\n{'='*85}")
    print(f"PADDY(COMMON) 90-DAY BACKTEST: {market} (District: {district_name}), AP")
    print(f"{'='*85}")
    print(f"Backtest Window : Last {backtest_days} days")
    print(f"Date Range      : {m_df['date'].iloc[backtest_start_idx].strftime('%Y-%m-%d')} to {m_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    bt_results = {name: {'actual': [], 'predicted': [], 'today_prices': []} for name in models}
    
    for i in range(backtest_start_idx, len(m_df) - 1):
        row = m_df.iloc[i]
        next_row = m_df.iloc[i + 1]
        
        today_price = float(row['modal_price'])
        actual_tomorrow = float(next_row['modal_price'])
        
        feat_dict = {}
        for col in feature_cols:
            if col.startswith('mkt_'):
                mkt_name = col.replace('mkt_', '')
                feat_dict[col] = 1 if mkt_name.lower() == market.lower() else 0
            elif col.startswith('dist_'):
                dist_col_name = col.replace('dist_', '')
                feat_dict[col] = 1 if dist_col_name.lower() == district_name.lower() else 0
            elif col in row.index and pd.notna(row[col]):
                feat_dict[col] = float(row[col])
            else:
                feat_dict[col] = 0.0
        
        X_curr = pd.DataFrame([feat_dict])[feature_cols].fillna(0.0)
        
        for name, model_obj in models.items():
            if name == 'Naive' or model_obj is None:
                pred_price = today_price
            else:
                pred_ret = float(model_obj.predict(X_curr)[0])
                pred_price = today_price * (1.0 + pred_ret)
            
            bt_results[name]['actual'].append(actual_tomorrow)
            bt_results[name]['predicted'].append(pred_price)
            bt_results[name]['today_prices'].append(today_price)
    
    print(f"\n{'Model':<12} | {'MAPE':<8} | {'MAE':<12} | {'RMSE':<12} | {'Dir Acc':<10} | {'Pred Std':<10}")
    print(f"{'-'*75}")
    
    for name, res in bt_results.items():
        actual = np.array(res['actual'])
        predicted = np.array(res['predicted'])
        today_p = np.array(res['today_prices'])
        
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100.0
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        actual_dir = actual - today_p
        pred_dir = predicted - today_p
        dir_acc = np.mean((actual_dir >= 0) == (pred_dir >= 0)) * 100.0
        pred_std = np.std(predicted - today_p)
        
        print(f"{name:<12} | {mape:>5.2f}% | Rs. {mae:>7.2f} | Rs. {rmse:>7.2f} | {dir_acc:>7.1f}% | Rs. {pred_std:>6.2f}")
    
    print(f"{'='*75}\n")

if __name__ == '__main__':
    for mkt in ['Jaggampet', 'Tiruvuru', 'Rampur', 'Rampachodvaram', 'Mylavaram', 'Polavaram', 'Nandigama', 'Kuchinapudi', 'Kovvur']:
        run_paddy_backtest(market=mkt, backtest_days=90)
