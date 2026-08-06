import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

RAW_ARRIVAL_CSV = r"C:\Users\Praveen\OneDrive\Desktop\All_Type_of_Report_(All_Grades)_05-08-2026_09-32-20_PM.csv"
CLEANED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_top10_ap_cleaned.csv')
WEATHER_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'weather_cleaned.csv')
PROCUREMENT_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'procurement_events.csv')
FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')

def load_real_arrivals():
    df = pd.read_csv(RAW_ARRIVAL_CSV, header=1)
    df.rename(columns={'Date': 'date_str', 'Arrival Quantity 01-01-2021 to 03-08-2026': 'arrival_qty_mt'}, inplace=True)
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%m-%Y', errors='coerce')
    df['arrival_qty_mt'] = pd.to_numeric(df['arrival_qty_mt'], errors='coerce').fillna(0.0)
    
    daily_arrival = df.groupby('date')['arrival_qty_mt'].sum().reset_index()
    daily_arrival = daily_arrival.sort_values('date').reset_index(drop=True)
    return daily_arrival

def build_pure_real_paddy_dataset():
    print("="*85)
    print("BUILDING PURE REAL PADDY DATASET (ZERO SYNTHETIC/MOCK GENERATED DATA)")
    print("="*85)
    
    # 1. Load Real Mandi Prices (from data.gov.in API response)
    df_clean = pd.read_csv(CLEANED_CSV)
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Calculate Weighted Average Modal Price: 60% Modal + 20% Min + 20% Max
    df_clean['weighted_avg_modal_price'] = (
        0.60 * df_clean['modal_price'] +
        0.20 * df_clean['min_price'] +
        0.20 * df_clean['max_price']
    )
    
    # 2. Load Real Arrival Quantities CSV
    arrival_df = load_real_arrivals()
    
    # 3. Load Real Open-Meteo Weather CSV
    weather_df = pd.read_csv(WEATHER_CSV)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    # 4. Load Real Government MSP & Procurement CSV
    procurement_df = pd.read_csv(PROCUREMENT_CSV)
    procurement_df['date'] = pd.to_datetime(procurement_df['date'])
    
    processed_markets = []
    
    # Process each market on REAL observed market dates (NO synthetic trend curve fill)
    for (mkt, dist), m_group in df_clean.groupby(['Market', 'District']):
        m_df = m_group.sort_values('date').reset_index(drop=True)
        m_df = m_df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        # Build continuous date grid ONLY over the market's active date range
        min_d, max_d = m_df['date'].min(), m_df['date'].max()
        full_dates = pd.date_range(min_d, max_d, freq='D')
        
        res = m_df.set_index('date').reindex(full_dates).reset_index()
        res.rename(columns={'index': 'date'}, inplace=True)
        res['Market'] = mkt
        res['District'] = dist
        res['commodity'] = 'Paddy(Common)'
        res['state'] = 'Andhra Pradesh'
        
        # Forward fill real price across non-trading weekend gaps (NO synthetic trend curve)
        res['modal_price'] = res['modal_price'].ffill().bfill()
        res['min_price'] = res['min_price'].ffill().bfill()
        res['max_price'] = res['max_price'].ffill().bfill()
        res['weighted_avg_modal_price'] = res['weighted_avg_modal_price'].ffill().bfill()
        
        dow = res['date'].dt.dayofweek
        month = res['date'].dt.month
        day = res['date'].dt.day
        res['day_of_week'] = dow
        res['month'] = month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        
        # MERGE REAL ARRIVALS (Zero random numbers)
        res = pd.merge(res, arrival_df[['date', 'arrival_qty_mt']], on='date', how='left')
        res['arrival_qty_mt'] = res['arrival_qty_mt'].fillna(0.0)
        res['arrival_lag_1'] = res['arrival_qty_mt'].shift(1).fillna(0.0)
        res['arrival_7d_mean'] = res['arrival_qty_mt'].shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
        res['arrival_change_pct'] = res['arrival_qty_mt'].shift(1).pct_change(1, fill_method=None).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # HOLIDAY & CLOSURE CALENDAR
        is_sunday = np.where(dow == 6, 1, 0)
        is_public_holiday = np.where(
            ((month == 1) & (day.isin([14, 15, 26]))) |
            ((month == 8) & (day == 15)) |
            ((month == 10) & (day == 2)) |
            ((month == 11) & (day == 1)), 1, 0
        )
        no_arrivals = np.where(res['arrival_qty_mt'] < 10.0, 1, 0)
        res['is_likely_non_trading_day'] = np.where((is_sunday == 1) | (is_public_holiday == 1) | (no_arrivals == 1), 1, 0)
        
        # MERGE REAL OPEN-METEO WEATHER (Zero random numbers)
        res = pd.merge(res, weather_df[['date', 'rainfall', 'temp_max', 'temp_min', 'humidity']], on='date', how='left')
        res['rainfall'] = res['rainfall'].fillna(0.0)
        res['humidity'] = res['humidity'].ffill().bfill()
        res['temp_avg'] = ((res['temp_max'] + res['temp_min']) / 2.0).ffill().bfill()
        
        res['rainfall_7d'] = res['rainfall'].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
        weekly_hist_rain = res.groupby('week_of_year')['rainfall_7d'].transform('mean')
        res['rainfall_anomaly_7d'] = res['rainfall_7d'] - weekly_hist_rain
        res['heavy_rain_flag'] = np.where(res['rainfall_7d'] > 40.0, 1, 0)
        res['dry_spell_flag'] = np.where((res['rainfall_7d'] < 1.0) & (res['month'].isin([6, 7, 8, 9])), 1, 0)
        
        # MERGE REAL PROCUREMENT / MSP CSV (Zero placeholders)
        res = pd.merge(res, procurement_df, on='date', how='left')
        res['is_procurement_active'] = res['is_procurement_active'].fillna(0)
        res['msp_value'] = res['msp_value'].ffill().bfill()
        res['msp_announced'] = res['msp_announced'].fillna(0)
        res['msp_changed'] = res['msp_changed'].fillna(0)
        res['procurement_started'] = res['procurement_started'].fillna(0)
        res['procurement_ended'] = res['procurement_ended'].fillna(0)
        
        # Targets & Lags on REAL Weighted Average Modal Price
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
    print(f"  Saved 100% Pure Real Paddy featured dataset to: {FEATURED_CSV}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Markets included ({final_df['Market'].nunique()}): {list(final_df['Market'].unique())}")
    return final_df

if __name__ == '__main__':
    build_pure_real_paddy_dataset()
