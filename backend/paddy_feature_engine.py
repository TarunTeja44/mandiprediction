import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import datetime
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

CLEANED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_top10_ap_cleaned.csv')
# Arrival CSV path handling (uses local arrival CSV if present)
DESKTOP_DIR = os.path.dirname(PROJECT_ROOT)
LOCAL_ARRIVAL_CSV = os.path.join(PROJECT_ROOT, "All_Type_of_Report_(All_Grades)_09-08-2026_07-22-58_PM.csv")
LEGACY_ARRIVAL_CSV = r"C:\Users\Praveen\OneDrive\Desktop\All_Type_of_Report_(All_Grades)_05-08-2026_09-32-20_PM.csv"

PROCUREMENT_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'procurement_events.csv')
WEATHER_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'weather_cleaned.csv')
FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
LEGACY_FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_top10_ap_featured.csv')

def load_arrival_data():
    raw_path = LOCAL_ARRIVAL_CSV if os.path.exists(LOCAL_ARRIVAL_CSV) else LEGACY_ARRIVAL_CSV
    if not os.path.exists(raw_path):
        # Find any matching arrival CSV in PROJECT_ROOT
        import glob
        matches = glob.glob(os.path.join(PROJECT_ROOT, "All_Type_of_Report_*.csv")) + glob.glob(os.path.join(DESKTOP_DIR, "All_Type_of_Report_*.csv"))
        raw_path = matches[0] if matches else raw_path

    print(f"Loading arrival data from: {os.path.basename(raw_path)}")
    try:
        df = pd.read_csv(raw_path, header=1, encoding='utf-8')
    except Exception:
        df = pd.read_csv(raw_path, header=1, encoding='latin1')
    
    # Identify arrival column dynamically
    arr_col = [c for c in df.columns if 'Arrival' in c and 'Quantity' in c]
    date_col = 'Date' if 'Date' in df.columns else df.columns[5]
    arr_name = arr_col[0] if arr_col else df.columns[6]

    df.rename(columns={date_col: 'date_str', arr_name: 'arrival_qty_mt'}, inplace=True)
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%m-%Y', errors='coerce')
    df['arrival_qty_mt'] = pd.to_numeric(df['arrival_qty_mt'], errors='coerce').fillna(0.0)
    
    daily_arrival = df.groupby('date')['arrival_qty_mt'].sum().reset_index()
    daily_arrival = daily_arrival.sort_values('date').reset_index(drop=True)
    return daily_arrival

def build_paddy_features():
    print("="*75)
    print("BUILDING ENHANCED FEATURE DATASET FOR PADDY(COMMON) - TOP 10 AP MARKETS")
    print("="*75)
    
    df_clean = pd.read_csv(CLEANED_CSV)
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    arrival_df = load_arrival_data()
    weather_df = pd.read_csv(WEATHER_CSV)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    procurement_df = pd.read_csv(PROCUREMENT_CSV)
    procurement_df['date'] = pd.to_datetime(procurement_df['date'])
    
    processed_markets = []
    
    # Process each market independently
    for (mkt, dist), m_group in df_clean.groupby(['Market', 'District']):
        m_df = m_group.sort_values('date').reset_index(drop=True)
        m_df = m_df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        min_d, max_d = m_df['date'].min(), m_df['date'].max()
        full_dates = pd.date_range(min_d, max_d, freq='D')
        
        res = m_df.set_index('date').reindex(full_dates).reset_index()
        res.rename(columns={'index': 'date'}, inplace=True)
        res['Market'] = mkt
        res['District'] = dist
        res['commodity'] = 'Paddy(Common)'
        res['state'] = 'Andhra Pradesh'
        
        res['modal_price'] = res['modal_price'].ffill().bfill()
        res['min_price'] = res['min_price'].ffill().bfill()
        res['max_price'] = res['max_price'].ffill().bfill()
        res['weighted_avg_modal_price'] = (
            0.60 * res['modal_price'] +
            0.20 * res['min_price'] +
            0.20 * res['max_price']
        ).ffill().bfill()
        
        # Calendar units
        dow = res['date'].dt.dayofweek
        month = res['date'].dt.month
        day = res['date'].dt.day
        res['day_of_week'] = dow
        res['month'] = month
        res['week_of_year'] = res['date'].dt.isocalendar().week.astype(int)
        
        # 1. MERGE ARRIVALS
        res = pd.merge(res, arrival_df[['date', 'arrival_qty_mt']], on='date', how='left')
        res['arrival_qty_mt'] = res['arrival_qty_mt'].fillna(0.0)
        res['arrival_lag_1'] = res['arrival_qty_mt'].shift(1).fillna(0.0)
        res['arrival_3d_mean'] = res['arrival_qty_mt'].shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
        res['arrival_7d_mean'] = res['arrival_qty_mt'].shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
        res['arrival_30d_mean'] = res['arrival_qty_mt'].shift(1).rolling(30, min_periods=1).mean().fillna(0.0)
        res['arrival_change_pct'] = res['arrival_qty_mt'].shift(1).pct_change(1, fill_method=None).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # 2. HOLIDAY & MARKET CLOSURE CALENDAR
        is_sunday = np.where(dow == 6, 1, 0)
        is_public_holiday = np.where(
            ((month == 1) & (day.isin([14, 15, 26]))) |
            ((month == 8) & (day == 15)) |
            ((month == 10) & (day == 2)) |
            ((month == 11) & (day == 1)), 1, 0
        )
        no_arrivals = np.where(res['arrival_qty_mt'] == 0.0, 1, 0)
        res['is_likely_non_trading_day'] = np.where((is_sunday == 1) | (is_public_holiday == 1) | (no_arrivals == 1), 1, 0)
        
        # 3. MERGE WEATHER & RAINFALL ANOMALY
        res = pd.merge(res, weather_df[['date', 'rainfall', 'temp_max', 'temp_min', 'humidity']], on='date', how='left')
        res['temp_avg'] = (res['temp_max'] + res['temp_min']) / 2.0
        res['rainfall'] = res['rainfall'].fillna(0.0)
        res['humidity'] = res['humidity'].ffill().bfill()
        res['temp_avg'] = res['temp_avg'].ffill().bfill()
        
        res['rainfall_3d'] = res['rainfall'].shift(1).rolling(3, min_periods=1).sum().fillna(0.0)
        res['rainfall_7d'] = res['rainfall'].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
        res['rainfall_30d'] = res['rainfall'].shift(1).rolling(30, min_periods=1).sum().fillna(0.0)
        weekly_hist_rain = res.groupby('week_of_year')['rainfall_7d'].transform('mean')
        res['rainfall_anomaly_7d'] = res['rainfall_7d'] - weekly_hist_rain
        res['heavy_rain_flag'] = np.where(res['rainfall_7d'] > 40.0, 1, 0)
        res['dry_spell_flag'] = np.where((res['rainfall_7d'] < 1.0) & (res['month'].isin([6, 7, 8, 9])), 1, 0)
        
        # 4. MERGE PROCUREMENT / MSP FLAGS
        res = pd.merge(res, procurement_df, on='date', how='left')
        res['is_procurement_active'] = res['is_procurement_active'].fillna(0)
        res['msp_value'] = res['msp_value'].ffill().bfill()
        res['msp_announced'] = res['msp_announced'].fillna(0)
        res['msp_changed'] = res['msp_changed'].fillna(0)
        res['procurement_started'] = res['procurement_started'].fillna(0)
        res['procurement_ended'] = res['procurement_ended'].fillna(0)
        
        # 5. Price Level & Spread Features (Multi-Target Engineering)
        p = res['weighted_avg_modal_price']
        mn = res['min_price']
        mx = res['max_price']

        res['spread'] = (mx - mn).clip(lower=0.0)
        res['log_spread'] = np.log(res['spread'] + 1.0)

        # Multi-Targets (Shift -1)
        res['target_modal_price'] = p.shift(-1)
        res['target_min_price'] = mn.shift(-1)
        res['target_max_price'] = mx.shift(-1)
        res['target_spread'] = res['spread'].shift(-1)
        res['target_log_spread'] = res['log_spread'].shift(-1)

        res['target_return'] = (res['target_modal_price'] - p) / (p + 1e-5)
        res['target_min_return'] = (res['target_min_price'] - mn) / (mn + 1e-5)
        res['target_max_return'] = (res['target_max_price'] - mx) / (mx + 1e-5)

        # Price Lags & Moving Averages (Weighted Avg)
        res['lag_1'] = p.shift(1)
        res['lag_3'] = p.shift(3)
        res['lag_7'] = p.shift(7)
        res['lag_14'] = p.shift(14)

        # Min, Max & Spread Specific Lags
        res['min_price_lag_1'] = mn.shift(1)
        res['min_price_rolling_mean_7'] = mn.shift(1).rolling(7, min_periods=2).mean()

        res['max_price_lag_1'] = mx.shift(1)
        res['max_price_rolling_mean_7'] = mx.shift(1).rolling(7, min_periods=2).mean()

        res['spread_lag_1'] = res['spread'].shift(1)
        res['spread_rolling_mean_7'] = res['spread'].shift(1).rolling(7, min_periods=2).mean()
        res['spread_rolling_std_7'] = res['spread'].shift(1).rolling(7, min_periods=2).std().fillna(0.0)
        res['log_spread_lag_1'] = res['log_spread'].shift(1)

        res['ret_1'] = (p - res['lag_1']) / (res['lag_1'] + 1e-5)
        res['ret_3'] = (p - res['lag_3']) / (res['lag_3'] + 1e-5)
        res['ret_7'] = (p - res['lag_7']) / (res['lag_7'] + 1e-5)
        res['ret_14'] = (p - res['lag_14']) / (res['lag_14'] + 1e-5)

        res['rolling_mean_3'] = p.shift(1).rolling(3, min_periods=1).mean()
        res['rolling_mean_7'] = p.shift(1).rolling(7, min_periods=2).mean()
        res['rolling_mean_14'] = p.shift(1).rolling(14, min_periods=3).mean()
        res['rolling_mean_30'] = p.shift(1).rolling(30, min_periods=5).mean()
        res['rolling_std_3'] = p.shift(1).rolling(3, min_periods=1).std().fillna(0.0)
        res['rolling_std_7'] = p.shift(1).rolling(7, min_periods=2).std().fillna(0.0)
        res['rolling_std_14'] = p.shift(1).rolling(14, min_periods=3).std().fillna(0.0)
        res['rolling_std_30'] = p.shift(1).rolling(30, min_periods=5).std().fillna(0.0)

        res['ratio_ma7'] = p / (res['rolling_mean_7'] + 1e-5)
        res['ratio_ma14'] = p / (res['rolling_mean_14'] + 1e-5)
        res['ratio_ma30'] = p / (res['rolling_mean_30'] + 1e-5)

        res['seasonal_ma_7'] = p.shift(1).rolling(7, min_periods=2).mean().fillna(0.0)
        res['seasonal_ma_30'] = p.shift(1).rolling(30, min_periods=5).mean().fillna(0.0)
        res['seasonal_trend'] = res['rolling_mean_14'] - res['rolling_mean_30']

        res['is_harvest_season'] = np.where(res['month'].isin([10, 11, 12, 4, 5]), 1, 0)
        res['is_monsoon_season'] = np.where(res['month'].isin([6, 7, 8, 9]), 1, 0)

        # 6. Interaction Terms for Special Days & Transport Disruption
        res['non_trading_lag1_interaction'] = res['is_likely_non_trading_day'] * res['lag_1']
        res['rain_arrival_interaction'] = res['heavy_rain_flag'] * res['arrival_3d_mean']

        clean_m = res.dropna(subset=['lag_1', 'target_modal_price', 'target_min_price', 'target_max_price']).reset_index(drop=True)
        processed_markets.append(clean_m)
        
    final_df = pd.concat(processed_markets, ignore_index=True)
    final_df = final_df.sort_values(['date', 'Market']).reset_index(drop=True)
    
    # Save master dataset to PROJECT_ROOT and backend directory
    master_csv_root = os.path.join(PROJECT_ROOT, "paddy_ap_master_dataset.csv")
    final_df.to_csv(master_csv_root, index=False, encoding='utf-8-sig')
    final_df.to_csv(FEATURED_CSV, index=False, encoding='utf-8-sig')
    if LEGACY_FEATURED_CSV != FEATURED_CSV:
        final_df.to_csv(LEGACY_FEATURED_CSV, index=False, encoding='utf-8-sig')

    print(f"  ✓ Saved master dataset CSV to: {master_csv_root}")
    print(f"  ✓ Saved featured CSV to: {FEATURED_CSV}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Markets included ({final_df['Market'].nunique()}): {list(final_df['Market'].unique())}")
    return final_df

if __name__ == '__main__':
    build_paddy_features()

if __name__ == '__main__':
    build_paddy_features()
