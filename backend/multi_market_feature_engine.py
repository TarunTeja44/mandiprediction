import os
import pandas as pd
import numpy as np

def create_multi_market_features():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    
    mandi_csv = os.path.join(project_root, 'backend', 'data', 'processed', 'ap_multi_market_rice_cleaned.csv')
    weather_csv = os.path.join(project_root, 'backend', 'data', 'processed', 'weather_cleaned.csv')
    
    print("[Multi-Market Feature Engine] Reading multi-market data...")
    mandi_df = pd.read_csv(mandi_csv)
    weather_df = pd.read_csv(weather_csv)
    
    mandi_df['date'] = pd.to_datetime(mandi_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    mandi_df = mandi_df.sort_values(['Market', 'date']).reset_index(drop=True)
    weather_df = weather_df.sort_values('date').reset_index(drop=True)
    
    # Merge weather strictly on date
    mandi_df = pd.merge(mandi_df, weather_df, on='date', how='left')
    mandi_df['temp_avg'] = (mandi_df['temp_max'] + mandi_df['temp_min']) / 2.0
    mandi_df['rainfall'] = mandi_df['rainfall'].fillna(0.0)
    mandi_df['humidity'] = mandi_df['humidity'].ffill().bfill()
    mandi_df['temp_avg'] = mandi_df['temp_avg'].ffill().bfill()
    
    processed_market_dfs = []
    
    print("[Multi-Market Feature Engine] Engineering per-market lag & rolling features...")
    
    for market, m_group in mandi_df.groupby('Market'):
        m_df = m_group.sort_values('date').reset_index(drop=True)
        
        # Resample to daily frequency per market
        min_d, max_d = m_df['date'].min(), m_df['date'].max()
        full_dates = pd.date_range(min_d, max_d, freq='D')
        
        res = m_df.set_index('date').reindex(full_dates).reset_index()
        res.rename(columns={'index': 'date'}, inplace=True)
        res['Market'] = market
        
        res['modal_price'] = res['modal_price'].ffill()
        res['min_price'] = res['min_price'].ffill()
        res['max_price'] = res['max_price'].ffill()
        res['temp_avg'] = res['temp_avg'].ffill().bfill()
        res['rainfall'] = res['rainfall'].fillna(0.0)
        res['humidity'] = res['humidity'].ffill().bfill()
        
        p = res['modal_price']
        
        # --- DIAGNOSTIC 1: VERIFY TARGET SHIFT ---
        # Target = tomorrow's price (t+1) EXACTLY
        res['target_modal_price'] = p.shift(-1)
        res['target_return'] = (res['target_modal_price'] - p) / (p + 1e-5)
        
        # --- DIAGNOSTIC 2: INSPECT FEATURE VALUES ---
        res['lag_1'] = p.shift(1)
        res['lag_3'] = p.shift(3)
        res['lag_7'] = p.shift(7)
        
        res['ret_1'] = (p - res['lag_1']) / (res['lag_1'] + 1e-5)
        res['ret_3'] = (p - res['lag_3']) / (res['lag_3'] + 1e-5)
        res['ret_7'] = (p - res['lag_7']) / (res['lag_7'] + 1e-5)
        
        res['rolling_mean_7'] = p.shift(1).rolling(window=7, min_periods=3).mean()
        res['rolling_mean_14'] = p.shift(1).rolling(window=14, min_periods=7).mean()
        res['rolling_std_7'] = p.shift(1).rolling(window=7, min_periods=3).std().fillna(0)
        res['rolling_std_14'] = p.shift(1).rolling(window=14, min_periods=7).std().fillna(0)
        
        res['ratio_ma7'] = p / (res['rolling_mean_7'] + 1e-5)
        res['ratio_ma14'] = p / (res['rolling_mean_14'] + 1e-5)
        
        res['rainfall_7d'] = res['rainfall'].shift(1).rolling(window=7, min_periods=1).sum()
        res['temp_avg_7d'] = res['temp_avg'].shift(1).rolling(window=7, min_periods=1).mean()
        res['humidity_avg_7d'] = res['humidity'].shift(1).rolling(window=7, min_periods=1).mean()
        
        res['day_of_week'] = res['date'].dt.dayofweek
        res['month'] = res['date'].dt.month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        res['is_harvest_season'] = np.where(res['month'].isin([10, 11, 12, 4, 5]), 1, 0)
        
        # Drop rows with NaN targets/lags
        clean_m = res.dropna(subset=['lag_7', 'rolling_mean_14', 'target_modal_price']).reset_index(drop=True)
        processed_market_dfs.append(clean_m)
        
    final_df = pd.concat(processed_market_dfs, ignore_index=True)
    final_df = final_df.sort_values(['date', 'Market']).reset_index(drop=True)
    
    # One-hot encode Market
    market_dummies = pd.get_dummies(final_df['Market'], prefix='mkt')
    final_df = pd.concat([final_df, market_dummies], axis=1)
    
    print(f"[Multi-Market Feature Engine] Combined panel dataset: {len(final_df)} rows, {len(final_df.columns)} columns.")
    
    # Print Diagnostic inspection of feature value variation
    print("\n--- DIAGNOSTIC FEATURE VALUE INSPECTION ---")
    print(final_df[['date', 'Market', 'modal_price', 'lag_1', 'rolling_mean_7', 'rainfall_7d', 'temp_avg_7d', 'target_modal_price']].head(10))
    print("\nFeature Standard Deviations (Checking non-zero variance):")
    inspect_cols = ['modal_price', 'lag_1', 'ret_1', 'rolling_mean_7', 'rolling_std_7', 'rainfall_7d', 'temp_avg_7d', 'target_return']
    print(final_df[inspect_cols].std())
    
    out_csv = os.path.join(project_root, 'backend', 'data', 'processed', 'multi_market_featured.csv')
    final_df.to_csv(out_csv, index=False)
    print(f"[Multi-Market Feature Engine] Saved featured panel dataset to: {out_csv}")
    return final_df

if __name__ == '__main__':
    create_multi_market_features()
