import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

ARTIFACT_DIR = r"C:\Users\Praveen\.gemini\antigravity-ide\brain\4f253fb3-1742-4c0a-a976-cbc3d75ebf99"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_graphs(backtest_days=90):
    model_path = os.path.join(PROJECT_ROOT, 'backend', 'models', 'paddy_common_ap_model.joblib')
    csv_path = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
    
    artifact = joblib.load(model_path)
    feature_cols = artifact['feature_cols']
    xgb_model = artifact['metrics']['XGBoost']['model'] if 'XGBoost' in artifact['metrics'] else artifact['model_object']
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    markets = ['Banaganapalli', 'Tiruvuru', 'Rajahmundry', 'Tanuku', 'Nandyal', 'Peddapuram']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False)
    axes = axes.flatten()
    
    saved_plots = []
    
    for idx, mkt in enumerate(markets):
        m_df = df[df['Market'].str.lower() == mkt.lower()].sort_values('date').reset_index(drop=True)
        if m_df.empty:
            continue
            
        district_name = str(m_df['District'].iloc[-1]) if 'District' in m_df.columns else "AP"
        if len(m_df) < backtest_days + 1:
            n_days = len(m_df) - 1
        else:
            n_days = backtest_days
            
        start_idx = len(m_df) - n_days
        
        dates = []
        actuals = []
        preds = []
        lower_bands = []
        upper_bands = []
        
        for i in range(start_idx, len(m_df) - 1):
            row = m_df.iloc[i]
            next_row = m_df.iloc[i + 1]
            
            today_price = float(row['weighted_avg_modal_price'])
            actual_tomorrow = float(next_row['weighted_avg_modal_price'])
            f_date = next_row['date']
            
            feat_dict = {}
            for col in feature_cols:
                if col.startswith('mkt_'):
                    mkt_name = col.replace('mkt_', '')
                    feat_dict[col] = 1 if mkt_name.lower() == mkt.lower() else 0
                elif col.startswith('dist_'):
                    dist_col_name = col.replace('dist_', '')
                    feat_dict[col] = 1 if dist_col_name.lower() == district_name.lower() else 0
                elif col in row.index and pd.notna(row[col]):
                    feat_dict[col] = float(row[col])
                else:
                    feat_dict[col] = 0.0
                    
            X_curr = pd.DataFrame([feat_dict])[feature_cols].fillna(0.0)
            pred_ret = float(xgb_model.predict(X_curr)[0])
            pred_price = today_price * (1.0 + pred_ret)
            
            dates.append(f_date)
            actuals.append(actual_tomorrow)
            preds.append(pred_price)
            
            vol_band = 25.0
            lower_bands.append(pred_price - vol_band)
            upper_bands.append(pred_price + vol_band)
            
        ax = axes[idx]
        ax.plot(dates, actuals, label='Actual Price (Rs/Q)', color='#1f77b4', linewidth=2.2)
        ax.plot(dates, preds, label='XGBoost Forecast (Rs/Q)', color='#ff7f0e', linestyle='--', linewidth=2.0)
        ax.fill_between(dates, lower_bands, upper_bands, color='#ff7f0e', alpha=0.15, label='90% Calibrated Band')
        
        ax.set_title(f"Market: {mkt} ({district_name} District)", fontsize=13, fontweight='bold', pad=8)
        ax.set_ylabel("Price (Rs. / Quintal)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(loc='upper left', fontsize=9)
        
        # Single market chart
        fig_single, ax_single = plt.subplots(figsize=(10, 5))
        ax_single.plot(dates, actuals, label='Actual Recorded Price (Rs/Q)', color='#1f77b4', linewidth=2.5)
        ax_single.plot(dates, preds, label='XGBoost Predicted Price (Rs/Q)', color='#d62728', linestyle='--', linewidth=2.2)
        ax_single.fill_between(dates, lower_bands, upper_bands, color='#d62728', alpha=0.15, label='Calibrated Trading Range (±Rs. 25/Q)')
        ax_single.set_title(f"Paddy(Common) 90-Day Backtest: Actual vs Predicted Price — {mkt} Mandi ({district_name})", fontsize=12, fontweight='bold')
        ax_single.set_xlabel("Date", fontsize=11)
        ax_single.set_ylabel("Price (Rs. / Quintal)", fontsize=11)
        ax_single.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        ax_single.tick_params(axis='x', rotation=25)
        ax_single.legend(loc='best', fontsize=10)
        fig_single.tight_layout()
        
        m_slug = mkt.lower().replace(' ', '_').replace('(', '').replace(')', '')
        single_path = os.path.join(ARTIFACT_DIR, f"actual_vs_predicted_{m_slug}.png")
        fig_single.savefig(single_path, dpi=150)
        plt.close(fig_single)
        saved_plots.append(single_path)
        print(f"Saved plot for {mkt} to: {single_path}")

    fig.suptitle("Paddy(Common) 90-Day Walk-Forward Backtest: Actual vs Predicted Price Across AP Mandis", fontsize=16, fontweight='bold', y=0.99)
    fig.tight_layout()
    grid_path = os.path.join(ARTIFACT_DIR, "actual_vs_predicted_grid.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"Saved grid plot to: {grid_path}")
    return saved_plots, grid_path

if __name__ == '__main__':
    generate_graphs()
