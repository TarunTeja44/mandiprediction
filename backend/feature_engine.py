import os
import pandas as pd
import numpy as np

def create_features(mandi_csv_path, weather_csv_path):
    print("[Feature Engine] Reading cleaned mandi and weather data...")
    mandi_df = pd.read_csv(mandi_csv_path)
    weather_df = pd.read_csv(weather_csv_path)
    
    mandi_df['date'] = pd.to_datetime(mandi_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    # Sort chronologically
    mandi_df = mandi_df.sort_values('date').reset_index(drop=True)
    weather_df = weather_df.sort_values('date').reset_index(drop=True)
    
    # Resample mandi_df to full calendar days to handle weekend gaps cleanly
    min_date = mandi_df['date'].min()
    max_date = mandi_df['date'].max()
    full_dates = pd.date_range(min_date, max_date, freq='D')
    
    df = mandi_df.set_index('date').reindex(full_dates).reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)
    
    # Forward fill missing price days (e.g., Sunday/holidays keep last trading day price)
    df['modal_price'] = df['modal_price'].ffill()
    df['min_price'] = df['min_price'].ffill()
    df['max_price'] = df['max_price'].ffill()
    
    # Merge weather data strictly on date
    df = pd.merge(df, weather_df, on='date', how='left')
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2.0
    df['rainfall'] = df['rainfall'].fillna(0.0)
    df['humidity'] = df['humidity'].ffill().bfill()
    df['temp_avg'] = df['temp_avg'].ffill().bfill()
    
    # --- 100% PAST DATA LAG & ROLLING FEATURES (ZERO LEAKAGE) ---
    p = df['modal_price']
    
    # Price Lags
    df['lag_1'] = p.shift(1)
    df['lag_3'] = p.shift(3)
    df['lag_7'] = p.shift(7)
    
    # Rolling Price Means & Std (shift(1) to exclude current day price)
    df['rolling_mean_7'] = p.shift(1).rolling(window=7, min_periods=3).mean()
    df['rolling_mean_14'] = p.shift(1).rolling(window=14, min_periods=7).mean()
    df['rolling_std_7'] = p.shift(1).rolling(window=7, min_periods=3).std()
    df['rolling_std_14'] = p.shift(1).rolling(window=14, min_periods=7).std()
    
    # Weather Rolling Aggregates (shift(1) so today's weather isn't leaked for tomorrow's prediction)
    df['rainfall_7d'] = df['rainfall'].shift(1).rolling(window=7, min_periods=1).sum()
    df['temp_avg_7d'] = df['temp_avg'].shift(1).rolling(window=7, min_periods=1).mean()
    df['humidity_avg_7d'] = df['humidity'].shift(1).rolling(window=7, min_periods=1).mean()
    
    # Calendar & Harvest Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Kharif harvest: Oct-Dec, Rabi harvest: Apr-May
    df['is_harvest_season'] = np.where(df['month'].isin([10, 11, 12, 4, 5]), 1, 0)
    
    # Target: Next day's price (t+1)
    df['target_modal_price'] = p.shift(-1)
    # Target 2-day ahead (t+2)
    df['target_modal_price_2d'] = p.shift(-2)
    
    # Drop initial NaN rows created by rolling windows
    featured_df = df.dropna(subset=['lag_7', 'rolling_mean_14', 'target_modal_price']).reset_index(drop=True)
    
    print(f"[Feature Engine] Generated dataset with {len(featured_df)} rows and {len(featured_df.columns)} columns.")
    return featured_df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir
    mandi_csv = os.path.join(project_root, 'backend', 'data', 'processed', 'tadepalligudem_rice_cleaned.csv')
    weather_csv = os.path.join(project_root, 'backend', 'data', 'processed', 'weather_cleaned.csv')
    
    df = create_features(mandi_csv, weather_csv)
    out_csv = os.path.join(project_root, 'backend', 'data', 'processed', 'featured_dataset.csv')
    df.to_csv(out_csv, index=False)
    print(f"[Feature Engine] Saved featured dataset to: {out_csv}")
