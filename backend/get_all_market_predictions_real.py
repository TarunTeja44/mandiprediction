import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

def run_all_predictions():
    model_path = os.path.join(PROJECT_ROOT, 'backend', 'models', 'paddy_common_ap_model.joblib')
    csv_path = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
    
    artifact = joblib.load(model_path)
    feature_cols = artifact['feature_cols']
    xgb_model = artifact['metrics']['XGBoost']['model'] if 'XGBoost' in artifact['metrics'] else artifact['model_object']
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    markets = df['Market'].unique().tolist()
    
    results = []
    
    for mkt in markets:
        m_df = df[df['Market'] == mkt].sort_values('date').reset_index(drop=True)
        if m_df.empty:
            continue
            
        last_row = m_df.iloc[-1]
        today_date = last_row['date']
        curr_price = float(last_row['weighted_avg_modal_price'])
        district = str(last_row['District'])
        
        # Day +1
        feat_dict = {}
        for col in feature_cols:
            if col.startswith('mkt_'):
                feat_dict[col] = 1 if col.replace('mkt_', '').lower() == mkt.lower() else 0
            elif col.startswith('dist_'):
                feat_dict[col] = 1 if col.replace('dist_', '').lower() == district.lower() else 0
            elif col in last_row.index and pd.notna(last_row[col]):
                feat_dict[col] = float(last_row[col])
            else:
                feat_dict[col] = 0.0
                
        X_1 = pd.DataFrame([feat_dict])[feature_cols].fillna(0.0)
        ret_1 = float(xgb_model.predict(X_1)[0])
        p_day1 = curr_price * (1.0 + ret_1)
        
        # Day +2
        p_day2 = p_day1 * (1.0 + ret_1)
        
        d1_date = today_date + pd.Timedelta(days=1)
        d2_date = today_date + pd.Timedelta(days=2)
        
        band = 25.0
        
        results.append({
            'Market': mkt,
            'District': district,
            'Today_Date': today_date.strftime('%Y-%m-%d'),
            'Current_Price': round(curr_price, 2),
            'Day1_Date': d1_date.strftime('%Y-%m-%d'),
            'Day1_Pred': round(p_day1, 2),
            'Day1_Lower': round(p_day1 - band, 1),
            'Day1_Upper': round(p_day1 + band, 1),
            'Day2_Date': d2_date.strftime('%Y-%m-%d'),
            'Day2_Pred': round(p_day2, 2),
            'Day2_Lower': round(p_day2 - band, 1),
            'Day2_Upper': round(p_day2 + band, 1),
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df

if __name__ == '__main__':
    run_all_predictions()
