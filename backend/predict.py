import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib
import datetime

def generate_multi_market_forecast(market="Jaggampet", model_preference="XGBoost"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    
    model_path = os.path.join(project_root, 'backend', 'models', 'paddy_common_ap_model.joblib')
    csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
    
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'paddy_common_2021_2026_featured.csv')
        
    artifact = joblib.load(model_path)
    metrics = artifact['metrics']
    feature_cols = artifact['feature_cols']
    commodity_name = artifact.get('commodity', 'Paddy(Common)')
    target_type = artifact.get('target_type', 'Weighted Average Modal Price (60% Modal + 20% Min + 20% Max)')
    
    if model_preference in metrics and metrics[model_preference]['model'] is not None:
        model_obj = metrics[model_preference]['model']
        active_model_name = model_preference
    else:
        model_obj = metrics['XGBoost']['model'] if 'XGBoost' in metrics and metrics['XGBoost']['model'] is not None else artifact['model_object']
        active_model_name = 'XGBoost' if model_obj is not None else artifact['model_name']
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    m_df = df[df['Market'].str.lower() == market.lower()]
    if m_df.empty:
        available_markets = df['Market'].unique()
        market = available_markets[0]
        m_df = df[df['Market'] == market]
        
    m_df = m_df.sort_values('date').reset_index(drop=True)
    district_name = str(m_df['District'].iloc[-1]) if 'District' in m_df.columns else "Andhra Pradesh"
    
    # Check if weighted_avg_modal_price exists
    if 'weighted_avg_modal_price' not in m_df.columns:
        m_df['weighted_avg_modal_price'] = 0.60 * m_df['modal_price'] + 0.20 * m_df['min_price'] + 0.20 * m_df['max_price']
        
    price_diffs = m_df['weighted_avg_modal_price'].diff().dropna()
    daily_vol = float(price_diffs.std()) if len(price_diffs) > 5 else 25.0
    typical_band = max(20.0, round(daily_vol * 1.645, 1))
    
    volatile_markets = ['machilipatnam', 'visakhapatnam', 'jaggampet', 'kuchinapudi']
    if market.lower() in volatile_markets:
        market_regime = "Active Trading Hub (Higher Volatility)"
        market_note = f"Active market with daily price moves. Expected daily trading band for Weighted Average Modal Price: ±Rs. {typical_band:.0f} / Quintal."
    else:
        market_regime = "Baseline Government MSP Hub (Sticky Price)"
        market_note = f"Prices in this APMC are sticky around government MSP support. Daily moves are gradual; typical band is ±Rs. {typical_band:.0f} / Quintal."
        
    today_real = datetime.date(2026, 8, 5)
    current_date = today_real
    
    last_row = m_df.iloc[-1]
    current_price = float(last_row['weighted_avg_modal_price'])
    
    print(f"\n[Weighted Average Modal Predict Engine] Market: {market}, District: {district_name}, AP")
    print(f"  Commodity          : {commodity_name}")
    print(f"  Target Price Metric: {target_type}")
    print(f"  Current Date       : {current_date.strftime('%Y-%m-%d')} (TODAY)")
    print(f"  Weighted Avg Price : Rs. {current_price:.2f} / Quintal")
    print(f"  Active Model       : {active_model_name}")
    
    predictions = []
    running_price = current_price
    recent_history = m_df.copy()
    
    for step in range(1, 3):
        forecast_date = current_date + datetime.timedelta(days=step)
        last_h_row = recent_history.iloc[-1]
        
        feat_dict = {}
        for col in feature_cols:
            if col.startswith('mkt_'):
                mkt_name = col.replace('mkt_', '')
                feat_dict[col] = 1 if mkt_name.lower() == market.lower() else 0
            elif col.startswith('dist_'):
                dist_col_name = col.replace('dist_', '')
                feat_dict[col] = 1 if dist_col_name.lower() == district_name.lower() else 0
            elif col in last_h_row.index and pd.notna(last_h_row[col]):
                feat_dict[col] = float(last_h_row[col])
            else:
                feat_dict[col] = 0.0
                
        p = recent_history['weighted_avg_modal_price']
        lag_1 = p.iloc[-1]
        lag_3 = p.iloc[-3] if len(p) >= 3 else p.iloc[-1]
        lag_7 = p.iloc[-7] if len(p) >= 7 else p.iloc[-1]
        
        rolling_mean_7 = p.iloc[-7:].mean()
        rolling_mean_14 = p.iloc[-14:].mean() if len(p) >= 14 else p.iloc[-7:].mean()
        
        feat_dict['ret_1'] = (running_price - lag_1) / (lag_1 + 1e-5)
        feat_dict['ret_3'] = (running_price - lag_3) / (lag_3 + 1e-5)
        feat_dict['ret_7'] = (running_price - lag_7) / (lag_7 + 1e-5)
        feat_dict['ratio_ma7'] = running_price / (rolling_mean_7 + 1e-5)
        feat_dict['ratio_ma14'] = running_price / (rolling_mean_14 + 1e-5)
        
        X_curr = pd.DataFrame([feat_dict])[feature_cols].fillna(0.0)
        
        if active_model_name == 'Naive' or model_obj is None:
            pred_price = running_price
        else:
            pred_ret = float(model_obj.predict(X_curr)[0])
            pred_price = running_price * (1.0 + pred_ret)
            
        band_scale = typical_band * (1.0 if step == 1 else 1.3)
        lower_bound = round(max(0.0, pred_price - band_scale), 1)
        upper_bound = round(pred_price + band_scale, 1)
        
        price_change = pred_price - running_price
        if abs(price_change) < 5.0:
            trend = "STABLE"
        elif price_change > 0:
            trend = "BULLISH / UPWARD"
        else:
            trend = "BEARISH / DOWNWARD"
            
        horizon_label = "Tomorrow (Day +1)" if step == 1 else "Day After Tomorrow (Day +2)"
        
        pred_item = {
            'horizon_day': horizon_label,
            'date': forecast_date.strftime('%Y-%m-%d'),
            'expected_weighted_avg_price': round(pred_price, 2),
            'trend': trend,
            'expected_trading_range': [lower_bound, upper_bound],
            'typical_daily_range_rs': f"±Rs. {band_scale:.0f}",
            'price_change_rs': round(price_change, 2)
        }
        predictions.append(pred_item)
        
        new_row = recent_history.iloc[-1].copy()
        new_row['date'] = forecast_date
        new_row['weighted_avg_modal_price'] = pred_price
        recent_history = pd.concat([recent_history, pd.DataFrame([new_row])], ignore_index=True)
        running_price = pred_price

    output_payload = {
        'market': market,
        'district': district_name,
        'state': 'Andhra Pradesh',
        'commodity': commodity_name,
        'target_metric': target_type,
        'current_date': current_date.strftime('%Y-%m-%d'),
        'current_weighted_avg_price': round(current_price, 2),
        'market_regime': market_regime,
        'market_note': market_note,
        'model_used': active_model_name,
        'predictions': predictions
    }
    
    print("="*75)
    print(f"WEIGHTED AVERAGE MODAL PRICE FORECAST ({active_model_name} Model - {market}, {district_name})")
    print("="*75)
    print(f"Today's Weighted Avg Price ({current_date.strftime('%Y-%m-%d')}): Rs. {current_price:.2f} / Quintal")
    print(f"Target Metric: {target_type}")
    print(f"Market Regime: {market_regime}")
    print(f"Context Note : {market_note}\n")
    for p in predictions:
        print(f"  * {p['horizon_day']} ({p['date']}):")
        print(f"      Trend Direction   : {p['trend']}")
        print(f"      Expected Wt Avg   : Rs. {p['expected_weighted_avg_price']:.2f} / Quintal")
        print(f"      Expected Trading Band: [Rs. {p['expected_trading_range'][0]:.1f} – Rs. {p['expected_trading_range'][1]:.1f}]")
        print(f"      Calibrated Scale  : {p['typical_daily_range_rs']}\n")
    print("="*75)
    
    return output_payload

if __name__ == '__main__':
    generate_multi_market_forecast(market="Jaggampet", model_preference="XGBoost")
