import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib

def evaluate_aug4_prediction():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    
    model_path = os.path.join(project_root, 'backend', 'models', 'paddy_common_ap_model.joblib')
    csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'paddy_common_2021_2026_featured.csv')
    
    artifact = joblib.load(model_path)
    feature_cols = artifact['feature_cols']
    model_obj = artifact['model_object']
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    aug3_date = pd.to_datetime('2026-08-03')
    aug4_date = pd.to_datetime('2026-08-04')
    
    print("="*85)
    print("MODEL PREDICTION EVALUATION FOR AUGUST 4, 2026 (USING DATA UP TO AUGUST 3, 2026)")
    print("="*85)
    print(f"{'Market Name':<20} | {'District':<18} | {'Aug 03 Price':<14} | {'Aug 04 Predicted':<16} | {'Aug 04 Actual':<14} | {'Error (Rs)':<12} | {'Error (%)':<10}")
    print("-" * 110)
    
    markets = df['Market'].unique()
    
    for market in markets:
        m_df = df[df['Market'] == market].sort_values('date').reset_index(drop=True)
        
        # Get August 3 row (used as input feature baseline)
        aug3_rows = m_df[m_df['date'] == aug3_date]
        aug4_rows = m_df[m_df['date'] == aug4_date]
        
        if aug3_rows.empty or aug4_rows.empty:
            continue
            
        row_aug3 = aug3_rows.iloc[0]
        row_aug4 = aug4_rows.iloc[0]
        
        aug3_price = float(row_aug3['modal_price'])
        actual_aug4 = float(row_aug4['modal_price'])
        district_name = str(row_aug3['District']) if 'District' in row_aug3 else "AP"
        
        feat_dict = {}
        for col in feature_cols:
            if col.startswith('mkt_'):
                mkt_name = col.replace('mkt_', '')
                feat_dict[col] = 1 if mkt_name.lower() == market.lower() else 0
            elif col.startswith('dist_'):
                dist_col_name = col.replace('dist_', '')
                feat_dict[col] = 1 if dist_col_name.lower() == district_name.lower() else 0
            elif col in row_aug3.index and pd.notna(row_aug3[col]):
                feat_dict[col] = float(row_aug3[col])
            else:
                feat_dict[col] = 0.0
                
        X_curr = pd.DataFrame([feat_dict])[feature_cols].fillna(0.0)
        pred_ret = float(model_obj.predict(X_curr)[0])
        predicted_aug4 = aug3_price * (1.0 + pred_ret)
        
        error_rs = predicted_aug4 - actual_aug4
        error_pct = (abs(error_rs) / actual_aug4) * 100.0
        
        print(f"{market:<20} | {district_name:<18} | Rs. {aug3_price:>8.2f}   | Rs. {predicted_aug4:>8.2f}       | Rs. {actual_aug4:>8.2f}   | Rs. {error_rs:>+6.2f}    | {error_pct:>6.2f}%")
        
    print("="*110)

if __name__ == '__main__':
    evaluate_aug4_prediction()
