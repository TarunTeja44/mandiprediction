import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import datetime
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

NEW_CSV_PATH = r"C:\Users\Praveen\OneDrive\Desktop\All_Type_of_Report_(All_Grades)_06-08-2026_02-38-20_PM.csv"
WEATHER_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'weather_cleaned.csv')
PROCUREMENT_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'procurement_events.csv')
FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_2021_2026_new_csv_featured.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'paddy_common_ap_model.joblib')

def process_new_csv():
    print("="*85)
    print("PROCESSING NEW 2021-2026 AP PADDY(COMMON) ARRIVAL & PRICE CSV DATASET")
    print("="*85)
    
    df = pd.read_csv(NEW_CSV_PATH, header=1)
    
    arrival_col = [c for c in df.columns if c.startswith('Arrival Quantity')][0]
    modal_col = [c for c in df.columns if c.startswith('Modal Price')][0]
    
    df.rename(columns={
        'State/UT': 'state',
        'District': 'District',
        'Market': 'Market_raw',
        'Commodity': 'commodity',
        'Date': 'date_str',
        arrival_col: 'arrival_qty_mt',
        modal_col: 'modal_price'
    }, inplace=True)
    
    df = df[df['commodity'] == 'Paddy(Common)'].copy()
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%m-%Y', errors='coerce')
    df['arrival_qty_mt'] = pd.to_numeric(df['arrival_qty_mt'], errors='coerce').fillna(0.0)
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    
    df['Market'] = df['Market_raw'].str.replace(' APMC', '', regex=False).str.strip()
    df = df.dropna(subset=['date', 'modal_price', 'Market']).sort_values(['Market', 'date']).reset_index(drop=True)
    
    print(f"Loaded Paddy(Common) records: {len(df)} across {df['Market'].nunique()} markets.")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    top_markets = df.groupby('Market').size().sort_values(ascending=False).head(10).index.tolist()
    print(f"Top 10 markets by record volume: {top_markets}")
    
    weather_df = pd.read_csv(WEATHER_CSV)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    procurement_df = pd.read_csv(PROCUREMENT_CSV)
    procurement_df['date'] = pd.to_datetime(procurement_df['date'])
    
    processed_markets = []
    
    for mkt in top_markets:
        m_df = df[df['Market'] == mkt].sort_values('date').reset_index(drop=True)
        m_df = m_df.drop_duplicates(subset=['date']).reset_index(drop=True)
        dist = m_df['District'].iloc[0]
        
        min_d, max_d = m_df['date'].min(), m_df['date'].max()
        full_dates = pd.date_range(min_d, max_d, freq='D')
        
        res = m_df.set_index('date').reindex(full_dates).reset_index()
        res.rename(columns={'index': 'date'}, inplace=True)
        res['Market'] = mkt
        res['District'] = dist
        res['commodity'] = 'Paddy(Common)'
        res['state'] = 'Andhra Pradesh'
        
        # Forward fill recorded market price across weekend gaps
        res['modal_price'] = res['modal_price'].ffill().bfill()
        res['min_price'] = res['modal_price'] * 0.985
        res['max_price'] = res['modal_price'] * 1.015
        res['weighted_avg_modal_price'] = 0.60 * res['modal_price'] + 0.20 * res['min_price'] + 0.20 * res['max_price']
        
        # Arrival quantities from NEW CSV (zero mock data)
        res['arrival_qty_mt'] = res['arrival_qty_mt'].fillna(0.0)
        res['arrival_lag_1'] = res['arrival_qty_mt'].shift(1).fillna(0.0)
        res['arrival_7d_mean'] = res['arrival_qty_mt'].shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
        res['arrival_change_pct'] = res['arrival_qty_mt'].shift(1).pct_change(1, fill_method=None).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        dow = res['date'].dt.dayofweek
        month = res['date'].dt.month
        day = res['date'].dt.day
        res['day_of_week'] = dow
        res['month'] = month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        
        is_sunday = np.where(dow == 6, 1, 0)
        is_public_holiday = np.where(((month == 1) & (day.isin([14, 15, 26]))) | ((month == 8) & (day == 15)) | ((month == 10) & (day == 2)), 1, 0)
        no_arrivals = np.where(res['arrival_qty_mt'] < 10.0, 1, 0)
        res['is_likely_non_trading_day'] = np.where((is_sunday == 1) | (is_public_holiday == 1) | (no_arrivals == 1), 1, 0)
        
        # Open-Meteo Weather
        res = pd.merge(res, weather_df[['date', 'rainfall', 'temp_max', 'temp_min', 'humidity']], on='date', how='left')
        res['rainfall'] = res['rainfall'].fillna(0.0)
        res['humidity'] = res['humidity'].ffill().bfill()
        res['temp_avg'] = ((res['temp_max'] + res['temp_min']) / 2.0).ffill().bfill()
        
        res['rainfall_7d'] = res['rainfall'].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
        weekly_hist_rain = res.groupby('week_of_year')['rainfall_7d'].transform('mean')
        res['rainfall_anomaly_7d'] = res['rainfall_7d'] - weekly_hist_rain
        res['heavy_rain_flag'] = np.where(res['rainfall_7d'] > 40.0, 1, 0)
        res['dry_spell_flag'] = np.where((res['rainfall_7d'] < 1.0) & (res['month'].isin([6, 7, 8, 9])), 1, 0)
        
        # Government MSP Floor CSV
        res = pd.merge(res, procurement_df, on='date', how='left')
        res['is_procurement_active'] = res['is_procurement_active'].fillna(0)
        res['msp_value'] = res['msp_value'].ffill().bfill()
        res['msp_announced'] = res['msp_announced'].fillna(0)
        res['msp_changed'] = res['msp_changed'].fillna(0)
        res['procurement_started'] = res['procurement_started'].fillna(0)
        res['procurement_ended'] = res['procurement_ended'].fillna(0)
        
        # Target & Lags
        p = res['weighted_avg_modal_price']
        res['target_modal_price'] = p.shift(-1)
        res['target_return'] = (res['target_modal_price'] - p) / (p + 1e-5)
        
        res['lag_1'] = p.shift(1)
        res['lag_3'] = p.shift(3)
        res['lag_7'] = p.shift(7)
        
        res['ret_1'] = (p - res['lag_1']) / (res['lag_1'] + 1e-5)
        res['ret_3'] = (p - res['lag_3']) / (res['lag_3'] + 1e-5)
        res['ret_7'] = (p - res['lag_7']) / (res['lag_7'] + 1e-5)
        
        res['rolling_mean_7'] = p.shift(1).rolling(7, min_periods=2).mean()
        res['rolling_mean_14'] = p.shift(1).rolling(14, min_periods=3).mean()
        res['rolling_std_7'] = p.shift(1).rolling(7, min_periods=2).std().fillna(0.0)
        res['rolling_std_14'] = p.shift(1).rolling(14, min_periods=3).std().fillna(0.0)
        
        res['ratio_ma7'] = p / (res['rolling_mean_7'] + 1e-5)
        res['ratio_ma14'] = p / (res['rolling_mean_14'] + 1e-5)
        res['is_harvest_season'] = np.where(res['month'].isin([10, 11, 12, 4, 5]), 1, 0)
        
        clean_m = res.dropna(subset=['lag_3', 'rolling_mean_7', 'target_modal_price']).reset_index(drop=True)
        processed_markets.append(clean_m)
        
    final_df = pd.concat(processed_markets, ignore_index=True)
    final_df = final_df.sort_values(['date', 'Market']).reset_index(drop=True)
    
    market_dummies = pd.get_dummies(final_df['Market'], prefix='mkt')
    district_dummies = pd.get_dummies(final_df['District'], prefix='dist')
    
    final_df = pd.concat([final_df, market_dummies, district_dummies], axis=1)
    
    final_df.to_csv(FEATURED_CSV, index=False, encoding='utf-8-sig')
    # Also save to main dataset file
    final_df.to_csv(os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\nSaved Featured Dataset to: {FEATURED_CSV}")
    print(f"Shape: {final_df.shape}")
    print(f"Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Markets included ({final_df['Market'].nunique()}): {list(final_df['Market'].unique())}")
    return final_df

if __name__ == '__main__':
    process_new_csv()
