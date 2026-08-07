import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import pmdarima as pm
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'backend' else BASE_DIR

FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')
LEGACY_FEATURED_CSV = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'paddy_common_top10_ap_featured.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'paddy_common_ap_model.joblib')


def detect_market_regime(price_series):
    """Classify a market's price behavior into a regime for model selection."""
    std = float(price_series.std())
    price_range = float(price_series.max() - price_series.min())
    cv = std / (float(price_series.mean()) + 1e-5) * 100.0  # coefficient of variation %

    if std < 5.0 or price_range < 20.0:
        return 'flat', std, cv
    elif std < 30.0:
        return 'low_volatility', std, cv
    else:
        return 'active', std, cv


def evaluate(y_true, y_pred, y_today=None):
    """Compute standard forecast metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'MAPE': 999.0, 'MAE': 999.0, 'RMSE': 999.0, 'DirAcc': 0.0, 'PredStd': 0.0}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0

    dir_acc = 0.0
    pred_std = 0.0
    if y_today is not None:
        y_today = np.asarray(y_today, dtype=float)[mask] if len(np.asarray(y_today)) > len(y_true) else np.asarray(y_today, dtype=float)
        if len(y_today) == len(y_true):
            actual_diff = y_true - y_today
            pred_diff = y_pred - y_today
            dir_acc = np.mean((actual_diff >= 0) == (pred_diff >= 0)) * 100.0
            pred_std = np.std(pred_diff)

    return {
        'MAPE': round(mape, 2),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'DirAcc': round(dir_acc, 2),
        'PredStd': round(pred_std, 2)
    }


def train_prophet_for_market(m_df, feature_cols_available, train_end_idx, val_end_idx):
    """Train a Prophet model for a single market with exogenous regressors."""
    if not HAS_PROPHET:
        return None, None, None

    # Prepare Prophet dataframe
    prophet_df = m_df[['date', 'weighted_avg_modal_price']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Add exogenous regressors if available
    exog_cols = []
    candidate_exog = ['msp_value', 'is_procurement_active', 'rainfall_7d',
                      'is_harvest_season', 'is_monsoon_season', 'arrival_7d_mean']
    for col in candidate_exog:
        if col in m_df.columns:
            prophet_df[col] = m_df[col].fillna(0.0).values
            exog_cols.append(col)

    train_prophet = prophet_df.iloc[:train_end_idx].copy()
    val_prophet = prophet_df.iloc[train_end_idx:val_end_idx].copy()
    test_prophet = prophet_df.iloc[val_end_idx:].copy()

    if len(train_prophet) < 30:
        return None, None, None

    try:
        model = Prophet(
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=5.0,
            yearly_seasonality=False,  # Not enough data for yearly
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80
        )
        for col in exog_cols:
            model.add_regressor(col)

        model.fit(train_prophet)

        # Validate
        val_forecast = model.predict(val_prophet[['ds'] + exog_cols])
        test_forecast = model.predict(test_prophet[['ds'] + exog_cols])

        val_metrics = evaluate(
            val_prophet['y'].values,
            val_forecast['yhat'].values,
            m_df.iloc[train_end_idx:val_end_idx]['weighted_avg_modal_price'].shift(1).fillna(method='bfill').values
        )
        test_metrics = evaluate(
            test_prophet['y'].values,
            test_forecast['yhat'].values,
            m_df.iloc[val_end_idx:]['weighted_avg_modal_price'].shift(1).fillna(method='bfill').values
        )

        return model, val_metrics, test_metrics

    except Exception as e:
        print(f"    Prophet training failed: {e}")
        return None, None, None


def train_arima_for_market(m_df, train_end_idx, val_end_idx):
    """Train an auto-ARIMA model for a single market with exogenous regressors."""
    if not HAS_ARIMA:
        return None, None, None

    price_series = m_df['weighted_avg_modal_price'].values

    # Prepare exogenous matrix
    exog_cols = []
    candidate_exog = ['msp_value', 'is_procurement_active', 'rainfall_7d',
                      'is_harvest_season', 'is_monsoon_season']
    for col in candidate_exog:
        if col in m_df.columns:
            exog_cols.append(col)

    train_y = price_series[:train_end_idx]
    val_y = price_series[train_end_idx:val_end_idx]
    test_y = price_series[val_end_idx:]

    train_exog = m_df[exog_cols].fillna(0.0).values[:train_end_idx] if exog_cols else None
    val_exog = m_df[exog_cols].fillna(0.0).values[train_end_idx:val_end_idx] if exog_cols else None
    test_exog = m_df[exog_cols].fillna(0.0).values[val_end_idx:] if exog_cols else None

    if len(train_y) < 30:
        return None, None, None

    try:
        model = pm.auto_arima(
            train_y,
            X=train_exog,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_q=5, max_d=2,
            trace=False
        )

        # Walk-forward validation
        val_preds = []
        for i in range(len(val_y)):
            pred = model.predict(n_periods=1, X=val_exog[i:i+1] if val_exog is not None else None)
            val_preds.append(float(pred[0]))
            model.update(val_y[i:i+1], X=val_exog[i:i+1] if val_exog is not None else None)

        # Walk-forward test
        test_preds = []
        for i in range(len(test_y)):
            pred = model.predict(n_periods=1, X=test_exog[i:i+1] if test_exog is not None else None)
            test_preds.append(float(pred[0]))
            model.update(test_y[i:i+1], X=test_exog[i:i+1] if test_exog is not None else None)

        # Re-fit on full data for deployment
        full_y = price_series
        full_exog = m_df[exog_cols].fillna(0.0).values if exog_cols else None
        final_model = pm.auto_arima(
            full_y,
            X=full_exog,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_q=5, max_d=2,
            trace=False
        )

        base_val = np.concatenate([[price_series[train_end_idx - 1]], np.array(val_preds[:-1])])
        base_test = np.concatenate([[price_series[val_end_idx - 1]], np.array(test_preds[:-1])])

        val_metrics = evaluate(val_y, np.array(val_preds), base_val)
        test_metrics = evaluate(test_y, np.array(test_preds), base_test)

        return final_model, val_metrics, test_metrics

    except Exception as e:
        print(f"    ARIMA training failed: {e}")
        return None, None, None


def train_paddy_model():
    print("=" * 85)
    print("TRAINING PADDY(COMMON) PREDICTION MODEL — PROPHET + ARIMA + ML ENSEMBLE")
    print("=" * 85)

    feature_csv_path = FEATURED_CSV if os.path.exists(FEATURED_CSV) else LEGACY_FEATURED_CSV

    df = pd.read_csv(feature_csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    required_cols = ['weighted_avg_modal_price', 'target_return', 'target_modal_price']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Training features are missing required columns: {missing}")

    mkt_cols = [c for c in df.columns if c.startswith('mkt_')]
    dist_cols = [c for c in df.columns if c.startswith('dist_')]

    feature_cols = [
        # Price Lags & Ratios
        'ret_1', 'ret_3', 'ret_7', 'ret_14', 'ratio_ma7', 'ratio_ma14', 'ratio_ma30', 'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
        # Rolling / Seasonal Drivers
        'lag_1', 'lag_3', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30', 'seasonal_ma_7', 'seasonal_ma_30', 'seasonal_trend',
        # Real Arrivals
        'arrival_lag_1', 'arrival_7d_mean', 'arrival_change_pct',
        # Holiday / Closure Calendar
        'is_likely_non_trading_day',
        # Real Weather & Rainfall Anomaly
        'rainfall_7d', 'rainfall_anomaly_7d', 'heavy_rain_flag', 'dry_spell_flag',
        # Procurement & MSP
        'is_procurement_active', 'msp_value', 'msp_announced', 'msp_changed', 'procurement_started', 'procurement_ended',
        # Calendar & Seasonality
        'day_of_week', 'month', 'week_of_year', 'is_harvest_season', 'is_monsoon_season'
    ] + mkt_cols + dist_cols

    # ======================== REGIME DETECTION ========================
    print("\n--- MARKET REGIME DETECTION ---")
    market_regimes = {}
    for market in sorted(df['Market'].dropna().unique()):
        m_df = df[df['Market'] == market]
        regime, std_val, cv_val = detect_market_regime(m_df['weighted_avg_modal_price'])
        market_regimes[market] = {
            'regime': regime,
            'std': round(std_val, 2),
            'cv_pct': round(cv_val, 2),
            'n_rows': len(m_df)
        }
        emoji = {'flat': '🟢', 'low_volatility': '🟡', 'active': '🔴'}
        print(f"  {emoji.get(regime, '?')} {market:20s}: {regime:15s} (std=Rs.{std_val:.1f}, CV={cv_val:.2f}%, n={len(m_df)})")

    # ======================== GLOBAL TRAIN/VAL/TEST SPLIT ========================
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # ML target
    train_df['target_change'] = (train_df['target_modal_price'] - train_df['weighted_avg_modal_price']) / (train_df['weighted_avg_modal_price'] + 1e-5)
    val_df['target_change'] = (val_df['target_modal_price'] - val_df['weighted_avg_modal_price']) / (val_df['weighted_avg_modal_price'] + 1e-5)
    test_df['target_change'] = (test_df['target_modal_price'] - test_df['weighted_avg_modal_price']) / (test_df['weighted_avg_modal_price'] + 1e-5)

    y_train_change = train_df['target_change'].fillna(0.0)
    y_val_change = val_df['target_change'].fillna(0.0)
    y_test_change = test_df['target_change'].fillna(0.0)

    y_val_true = val_df['target_modal_price'].values
    y_test_true = test_df['target_modal_price'].values
    today_val_price = val_df['weighted_avg_modal_price'].values
    today_test_price = test_df['weighted_avg_modal_price'].values

    X_train = train_df[feature_cols].fillna(0.0)
    X_val = val_df[feature_cols].fillna(0.0)
    X_test = test_df[feature_cols].fillna(0.0)

    results = {}

    # ======================== 1. NAIVE BASELINE ========================
    print("\n--- Training Naive Baseline ---")
    results['Naive'] = {
        'val': evaluate(y_val_true, today_val_price, today_val_price),
        'test': evaluate(y_test_true, today_test_price, today_test_price),
        'model': None
    }

    # ======================== 2. RIDGE ========================
    print("--- Training Ridge ---")
    ridge = Ridge(alpha=3.0)
    ridge.fit(X_train, y_train_change)
    val_pred_ridge = today_val_price * (1.0 + ridge.predict(X_val))
    test_pred_ridge = today_test_price * (1.0 + ridge.predict(X_test))
    results['Ridge'] = {
        'val': evaluate(y_val_true, val_pred_ridge, today_val_price),
        'test': evaluate(y_test_true, test_pred_ridge, today_test_price),
        'model': ridge
    }

    # ======================== 3. GRADIENT BOOSTING (per market) ========================
    print("--- Training GradientBoosting (per market) ---")
    gb_market_models = {}
    gb_val_preds, gb_test_preds = [], []
    gb_val_true, gb_test_true = [], []
    gb_val_base, gb_test_base = [], []

    for market in sorted(df['Market'].dropna().unique()):
        tr = train_df[train_df['Market'] == market]
        va = val_df[val_df['Market'] == market]
        te = test_df[test_df['Market'] == market]
        if len(tr) < 20 or len(va) < 5 or len(te) < 5:
            continue
        X_tr = tr[feature_cols].fillna(0.0)
        X_va = va[feature_cols].fillna(0.0)
        X_te = te[feature_cols].fillna(0.0)
        y_tr = tr['target_modal_price'].fillna(0.0)
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_tr, y_tr)
        gb_market_models[market] = model
        gb_val_preds.append(model.predict(X_va))
        gb_test_preds.append(model.predict(X_te))
        gb_val_true.append(va['target_modal_price'].values)
        gb_test_true.append(te['target_modal_price'].values)
        gb_val_base.append(va['weighted_avg_modal_price'].values)
        gb_test_base.append(te['weighted_avg_modal_price'].values)
        print(f"    Trained GB for {market} ({len(tr)} train rows)")

    if gb_val_preds:
        results['GradientBoosting'] = {
            'val': evaluate(np.concatenate(gb_val_true), np.concatenate(gb_val_preds), np.concatenate(gb_val_base)),
            'test': evaluate(np.concatenate(gb_test_true), np.concatenate(gb_test_preds), np.concatenate(gb_test_base)),
            'model': gb_market_models
        }
    else:
        results['GradientBoosting'] = {
            'val': evaluate(y_val_true, today_val_price, today_val_price),
            'test': evaluate(y_test_true, today_test_price, today_test_price),
            'model': {}
        }

    # ======================== 4. XGBOOST ========================
    if HAS_XGB:
        print("--- Training XGBoost ---")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, min_child_weight=3,
            subsample=0.9, colsample_bytree=0.9, learning_rate=0.05,
            gamma=0.2, reg_alpha=0.3, reg_lambda=1.0, random_state=42
        )
        xgb_model.fit(X_train, y_train_change, eval_set=[(X_val, y_val_change)], verbose=False)
        val_pred_xgb = today_val_price * (1.0 + xgb_model.predict(X_val))
        test_pred_xgb = today_test_price * (1.0 + xgb_model.predict(X_test))
        results['XGBoost'] = {
            'val': evaluate(y_val_true, val_pred_xgb, today_val_price),
            'test': evaluate(y_test_true, test_pred_xgb, today_test_price),
            'model': xgb_model
        }

    # ======================== 5. PROPHET (per market) ========================
    prophet_market_models = {}
    prophet_exog_cols = {}
    if HAS_PROPHET:
        print("--- Training Prophet (per market) ---")
        prophet_val_preds, prophet_test_preds = [], []
        prophet_val_true, prophet_test_true = [], []
        prophet_val_base, prophet_test_base = [], []

        for market in sorted(df['Market'].dropna().unique()):
            m_df = df[df['Market'] == market].sort_values('date').reset_index(drop=True)
            regime = market_regimes[market]['regime']

            if regime == 'flat' or len(m_df) < 40:
                print(f"    Skipping Prophet for {market} (regime={regime}, n={len(m_df)})")
                continue

            m_train_end = int(len(m_df) * 0.70)
            m_val_end = int(len(m_df) * 0.85)

            prophet_model, val_m, test_m = train_prophet_for_market(
                m_df, feature_cols, m_train_end, m_val_end
            )

            if prophet_model is not None:
                prophet_market_models[market] = prophet_model
                # Track which exog cols this market's Prophet uses
                used_exog = []
                candidate_exog = ['msp_value', 'is_procurement_active', 'rainfall_7d',
                                  'is_harvest_season', 'is_monsoon_season', 'arrival_7d_mean']
                for col in candidate_exog:
                    if col in m_df.columns:
                        used_exog.append(col)
                prophet_exog_cols[market] = used_exog

                print(f"    ✓ Prophet trained for {market}: Val MAPE={val_m['MAPE']:.2f}%, Test MAPE={test_m['MAPE']:.2f}%")

                # Collect for aggregate metrics
                m_val = m_df.iloc[m_train_end:m_val_end]
                m_test = m_df.iloc[m_val_end:]
                if len(m_val) > 0 and len(m_test) > 0:
                    val_fc = prophet_model.predict(
                        m_val[['date']].rename(columns={'date': 'ds'}).assign(
                            **{c: m_val[c].fillna(0.0).values for c in used_exog}
                        )
                    )
                    test_fc = prophet_model.predict(
                        m_test[['date']].rename(columns={'date': 'ds'}).assign(
                            **{c: m_test[c].fillna(0.0).values for c in used_exog}
                        )
                    )
                    prophet_val_preds.append(val_fc['yhat'].values)
                    prophet_test_preds.append(test_fc['yhat'].values)
                    prophet_val_true.append(m_val['weighted_avg_modal_price'].values)
                    prophet_test_true.append(m_test['weighted_avg_modal_price'].values)
                    prophet_val_base.append(m_val['weighted_avg_modal_price'].shift(1).bfill().values)
                    prophet_test_base.append(m_test['weighted_avg_modal_price'].shift(1).bfill().values)

        if prophet_val_preds:
            results['Prophet'] = {
                'val': evaluate(np.concatenate(prophet_val_true), np.concatenate(prophet_val_preds), np.concatenate(prophet_val_base)),
                'test': evaluate(np.concatenate(prophet_test_true), np.concatenate(prophet_test_preds), np.concatenate(prophet_test_base)),
                'model': prophet_market_models
            }
            print(f"    Prophet aggregate: Val MAPE={results['Prophet']['val']['MAPE']:.2f}%, Test MAPE={results['Prophet']['test']['MAPE']:.2f}%")
    else:
        print("--- Prophet not available (pip install prophet) ---")

    # ======================== 6. ARIMA (per market) ========================
    arima_market_models = {}
    arima_exog_cols = {}
    if HAS_ARIMA:
        print("--- Training ARIMA (per market) ---")
        arima_val_preds, arima_test_preds = [], []
        arima_val_true, arima_test_true = [], []
        arima_val_base, arima_test_base = [], []

        for market in sorted(df['Market'].dropna().unique()):
            m_df = df[df['Market'] == market].sort_values('date').reset_index(drop=True)
            regime = market_regimes[market]['regime']

            if regime == 'flat' or len(m_df) < 40:
                print(f"    Skipping ARIMA for {market} (regime={regime}, n={len(m_df)})")
                continue

            m_train_end = int(len(m_df) * 0.70)
            m_val_end = int(len(m_df) * 0.85)

            arima_model, val_m, test_m = train_arima_for_market(
                m_df, m_train_end, m_val_end
            )

            if arima_model is not None:
                arima_market_models[market] = arima_model
                used_exog = []
                candidate_exog = ['msp_value', 'is_procurement_active', 'rainfall_7d',
                                  'is_harvest_season', 'is_monsoon_season']
                for col in candidate_exog:
                    if col in m_df.columns:
                        used_exog.append(col)
                arima_exog_cols[market] = used_exog
                print(f"    ✓ ARIMA trained for {market}: Val MAPE={val_m['MAPE']:.2f}%, Test MAPE={test_m['MAPE']:.2f}%, Order={arima_model.order}")

                if val_m is not None and test_m is not None:
                    arima_val_true.append(np.array([val_m['MAE']]))  # placeholder for aggregate
                    arima_test_true.append(np.array([test_m['MAE']]))

        # For aggregate ARIMA metrics, re-do walk-forward on the two largest markets
        arima_agg_val_preds, arima_agg_test_preds = [], []
        arima_agg_val_true, arima_agg_test_true = [], []
        arima_agg_val_base, arima_agg_test_base = [], []

        for market, model in arima_market_models.items():
            m_df = df[df['Market'] == market].sort_values('date').reset_index(drop=True)
            m_train_end = int(len(m_df) * 0.70)
            m_val_end = int(len(m_df) * 0.85)
            prices = m_df['weighted_avg_modal_price'].values
            used_exog = arima_exog_cols.get(market, [])

            # Re-fit on train for clean validation
            train_y = prices[:m_train_end]
            train_exog = m_df[used_exog].fillna(0.0).values[:m_train_end] if used_exog else None
            eval_model = pm.auto_arima(
                train_y, X=train_exog, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore', trace=False
            )

            val_y = prices[m_train_end:m_val_end]
            val_exog = m_df[used_exog].fillna(0.0).values[m_train_end:m_val_end] if used_exog else None
            val_preds_list = []
            for i in range(len(val_y)):
                p = eval_model.predict(n_periods=1, X=val_exog[i:i+1] if val_exog is not None else None)
                val_preds_list.append(float(p[0]))
                eval_model.update(val_y[i:i+1], X=val_exog[i:i+1] if val_exog is not None else None)

            test_y = prices[m_val_end:]
            test_exog = m_df[used_exog].fillna(0.0).values[m_val_end:] if used_exog else None
            test_preds_list = []
            for i in range(len(test_y)):
                p = eval_model.predict(n_periods=1, X=test_exog[i:i+1] if test_exog is not None else None)
                test_preds_list.append(float(p[0]))
                eval_model.update(test_y[i:i+1], X=test_exog[i:i+1] if test_exog is not None else None)

            if val_preds_list:
                arima_agg_val_preds.append(np.array(val_preds_list))
                arima_agg_val_true.append(val_y)
                arima_agg_val_base.append(np.concatenate([[prices[m_train_end-1]], np.array(val_preds_list[:-1])]))
            if test_preds_list:
                arima_agg_test_preds.append(np.array(test_preds_list))
                arima_agg_test_true.append(test_y)
                arima_agg_test_base.append(np.concatenate([[prices[m_val_end-1]], np.array(test_preds_list[:-1])]))

        if arima_agg_val_preds:
            results['ARIMA'] = {
                'val': evaluate(np.concatenate(arima_agg_val_true), np.concatenate(arima_agg_val_preds), np.concatenate(arima_agg_val_base)),
                'test': evaluate(np.concatenate(arima_agg_test_true), np.concatenate(arima_agg_test_preds), np.concatenate(arima_agg_test_base)),
                'model': arima_market_models
            }
            print(f"    ARIMA aggregate: Val MAPE={results['ARIMA']['val']['MAPE']:.2f}%, Test MAPE={results['ARIMA']['test']['MAPE']:.2f}%")
    else:
        print("--- ARIMA not available (pip install pmdarima) ---")

    # ======================== RESULTS SUMMARY ========================
    print("\n" + "=" * 110)
    print(f"{'PADDY(COMMON) MODEL TRAINING SUMMARY — ALL MODELS':^110}")
    print("=" * 110)
    print(f"{'Model':<18} | {'Val MAPE':<10} | {'Val MAE':<10} | {'Val DirAcc':<10} | {'Val PredStd':<11} || {'Test MAPE':<10} | {'Test MAE':<10} | {'Test DirAcc':<10} | {'Test PredStd':<11}")
    print("-" * 110)

    best_model_name = None
    best_test_mape = float('inf')

    for name, res in results.items():
        v = res['val']
        t = res['test']
        print(f"{name:<18} | {v['MAPE']:>8.2f}% | Rs.{v['MAE']:>6.1f} | {v['DirAcc']:>8.1f}% | Rs.{v['PredStd']:>8.2f} || {t['MAPE']:>8.2f}% | Rs.{t['MAE']:>6.1f} | {t['DirAcc']:>8.1f}% | Rs.{t['PredStd']:>8.2f}")

        # Pick best on test MAPE (excluding Naive for non-flat markets)
        if t['MAPE'] < best_test_mape and name != 'Naive':
            best_test_mape = t['MAPE']
            best_model_name = name

    print("-" * 110)
    print(f">> BEST NON-NAIVE MODEL: {best_model_name} (Test MAPE: {best_test_mape:.2f}%)")

    # ======================== REGIME-BASED BEST MODEL PER MARKET ========================
    print("\n--- REGIME-BASED MODEL ASSIGNMENT ---")
    market_model_assignment = {}
    for market, info in market_regimes.items():
        regime = info['regime']
        if regime == 'flat':
            assigned = 'Naive'
        elif regime == 'low_volatility':
            if market in arima_market_models:
                assigned = 'ARIMA'
            elif market in prophet_market_models:
                assigned = 'Prophet'
            else:
                assigned = 'Naive'
        else:  # active
            if market in prophet_market_models:
                assigned = 'Prophet'
            elif market in arima_market_models:
                assigned = 'ARIMA'
            else:
                assigned = 'Naive'

        market_model_assignment[market] = assigned
        print(f"  {market:20s}: regime={regime:15s} → model={assigned}")

    # ======================== COMPUTE RESIDUAL BOUNDS ========================
    # Use the best available model's residuals for confidence intervals
    val_residuals = y_val_true - today_val_price  # default fallback
    if 'ARIMA' in results and results['ARIMA']['val']['MAPE'] < 999:
        pass  # ARIMA residuals are computed per-market at predict time
    q10 = float(np.percentile(val_residuals, 10))
    q90 = float(np.percentile(val_residuals, 90))

    # ======================== SAVE CONSOLIDATED MODEL ARTIFACT ========================
    model_artifact = {
        'target_type': 'Weighted Average Modal Price (60% Modal + 20% Min + 20% Max)',
        'commodity': 'Paddy(Common)',
        'model_name': best_model_name,
        'forecast_horizon': 7,
        # ML models (global)
        'ridge_model': ridge,
        'xgb_model': results.get('XGBoost', {}).get('model'),
        'gb_market_models': gb_market_models,
        # Time-series models (per market)
        'prophet_market_models': prophet_market_models,
        'prophet_exog_cols': prophet_exog_cols,
        'arima_market_models': arima_market_models,
        'arima_exog_cols': arima_exog_cols,
        # Regime info
        'market_regimes': market_regimes,
        'market_model_assignment': market_model_assignment,
        # Feature metadata
        'feature_cols': feature_cols,
        'metrics': results,
        'q10_residual': q10,
        'q90_residual': q90,
        'markets': list(df['Market'].unique()),
        'districts': list(df['District'].unique()),
        'last_trained_date': df['date'].max().strftime('%Y-%m-%d')
    }

    # Keep backward compat field
    model_artifact['model_object'] = gb_market_models if gb_market_models else ridge

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\n✓ Saved consolidated model artifact to: {MODEL_PATH}")
    print(f"  Contains: {len(prophet_market_models)} Prophet models, {len(arima_market_models)} ARIMA models, {len(gb_market_models)} GB models")

    # Clean up old duplicate files
    old_files = [
        os.path.join(PROJECT_ROOT, 'backend', 'models', 'rice_ap_multi_market_model.joblib'),
        os.path.join(PROJECT_ROOT, 'backend', 'models', 'enhanced_rice_model.joblib'),
    ]
    for f in old_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed old duplicate: {os.path.basename(f)}")


if __name__ == '__main__':
    train_paddy_model()
