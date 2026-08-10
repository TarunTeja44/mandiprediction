import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

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


def _resolve_model_for_market(artifact, market, model_preference="Auto"):
    """
    Resolve the best model for a given market using regime-based assignment.
    model_preference can be: 'Auto', 'Prophet', 'ARIMA', 'XGBoost', 'GradientBoosting', 'Naive'
    """
    assignment = artifact.get('market_model_assignment', {})
    regimes = artifact.get('market_regimes', {})

    if model_preference == "Auto":
        # Use regime-based assignment
        assigned = assignment.get(market, 'Naive')
    else:
        assigned = model_preference

    # Try to find the requested model
    if assigned == 'Prophet':
        prophet_models = artifact.get('prophet_market_models', {})
        if market in prophet_models:
            return prophet_models[market], 'Prophet', artifact.get('prophet_exog_cols', {}).get(market, [])
        # Fallback
        assigned = 'ARIMA'

    if assigned == 'ARIMA':
        arima_models = artifact.get('arima_market_models', {})
        if market in arima_models:
            return arima_models[market], 'ARIMA', artifact.get('arima_exog_cols', {}).get(market, [])
        # Fallback
        assigned = 'GradientBoosting'

    if assigned == 'GradientBoosting':
        gb_models = artifact.get('gb_market_models', {})
        if market in gb_models:
            return gb_models[market], 'GradientBoosting', []
        # Fallback
        assigned = 'XGBoost'

    if assigned == 'XGBoost':
        xgb_model = artifact.get('xgb_model')
        if xgb_model is not None:
            return xgb_model, 'XGBoost', []

    # Final fallback: Naive
    return None, 'Naive', []


def _generate_prophet_forecast(model, last_date, periods, exog_cols, m_df):
    """Generate multi-step Prophet forecast."""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    # Fill exogenous regressors with last known values
    for col in exog_cols:
        if col in m_df.columns:
            last_val = float(m_df[col].iloc[-1]) if not m_df[col].isna().all() else 0.0
            future_df[col] = last_val

    forecast = model.predict(future_df)

    preds = []
    for i in range(periods):
        preds.append({
            'yhat': float(forecast['yhat'].iloc[i]),
            'yhat_lower': float(forecast['yhat_lower'].iloc[i]),
            'yhat_upper': float(forecast['yhat_upper'].iloc[i]),
            'date': future_dates[i]
        })
    return preds


def _generate_arima_forecast(model, periods, exog_cols, m_df):
    """Generate multi-step ARIMA forecast."""
    # Build future exogenous matrix from last known values
    future_exog = None
    if exog_cols:
        last_vals = {}
        for col in exog_cols:
            if col in m_df.columns:
                last_vals[col] = float(m_df[col].iloc[-1]) if not m_df[col].isna().all() else 0.0
            else:
                last_vals[col] = 0.0
        exog_arr = np.array(list(last_vals.values()))
        
        # Check statsmodels inner k_exog requirement
        req_k = len(exog_arr)
        if hasattr(model, 'arima_res_') and hasattr(model.arima_res_, 'model') and hasattr(model.arima_res_.model, 'k_exog'):
            req_k = model.arima_res_.model.k_exog
        elif hasattr(model, 'n_features_in_'):
            req_k = model.n_features_in_
            
        if req_k is None or req_k == 0:
            future_exog = None
        else:
            if len(exog_arr) > req_k:
                exog_arr = exog_arr[:req_k]
            elif len(exog_arr) < req_k:
                exog_arr = np.pad(exog_arr, (0, req_k - len(exog_arr)), 'constant')
            future_exog = np.tile(exog_arr, (periods, 1))

    forecast, conf_int = model.predict(n_periods=periods, X=future_exog, return_conf_int=True, alpha=0.20)
    return forecast, conf_int


def _build_model_input_frame(base_row, feature_cols, current_price, history_prices):
    """Build feature input for ML models (GB/XGBoost)."""
    if isinstance(base_row, pd.Series):
        model_inputs = base_row.reindex(feature_cols, fill_value=0.0)
    else:
        model_inputs = base_row.reindex(columns=feature_cols, fill_value=0.0)
    if 'weighted_avg_modal_price' in base_row.index:
        model_inputs['weighted_avg_modal_price'] = float(current_price)
    if len(history_prices) >= 1:
        model_inputs['lag_1'] = float(history_prices[-1])
    if len(history_prices) >= 3:
        model_inputs['lag_3'] = float(history_prices[-3])
    if len(history_prices) >= 7:
        model_inputs['lag_7'] = float(history_prices[-7])
    if len(history_prices) >= 14:
        model_inputs['lag_14'] = float(history_prices[-14])

    recent_window = np.asarray(history_prices[-7:], dtype=float)
    if recent_window.size:
        recent_mean = float(np.mean(recent_window))
        model_inputs['rolling_mean_7'] = recent_mean
        model_inputs['rolling_mean_14'] = float(np.mean(recent_window[-min(7, len(recent_window)):]))
        model_inputs['rolling_mean_30'] = float(np.mean(recent_window[-min(7, len(recent_window)):]))
        model_inputs['rolling_std_7'] = float(np.std(recent_window))
        model_inputs['rolling_std_14'] = float(np.std(recent_window[-min(7, len(recent_window)):]))
        model_inputs['rolling_std_30'] = float(np.std(recent_window[-min(7, len(recent_window)):]))
    return model_inputs.fillna(0.0)


def generate_multi_market_forecast(market="Jaggampet", model_preference="Auto", forecast_days=3):
    """
    Generate 3-day price forecast for a given market using regime-based model selection.
    Supports: Auto, Prophet, ARIMA, XGBoost, GradientBoosting, Naive
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'backend' else script_dir

    model_path = os.path.join(project_root, 'backend', 'models', 'paddy_common_ap_model.joblib')
    csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'paddy_common_weighted_avg_featured.csv')

    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, 'backend', 'data', 'processed', 'paddy_common_top10_ap_featured.csv')

    artifact = joblib.load(model_path)
    feature_cols = artifact.get('feature_cols', [])
    commodity_name = artifact.get('commodity', 'Paddy(Common)')
    target_type = artifact.get('target_type', 'Weighted Average Modal Price (60% Modal + 20% Min + 20% Max)')
    market_regimes = artifact.get('market_regimes', {})
    forecast_horizon = forecast_days

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Match market
    m_df = df[df['Market'].str.lower() == market.lower()]
    if m_df.empty:
        available_markets = df['Market'].unique()
        market = available_markets[0]
        m_df = df[df['Market'] == market]

    m_df = m_df.sort_values('date').reset_index(drop=True)
    district_name = str(m_df['District'].iloc[-1]) if 'District' in m_df.columns else "Andhra Pradesh"

    if 'weighted_avg_modal_price' not in m_df.columns:
        m_df['weighted_avg_modal_price'] = 0.60 * m_df['modal_price'] + 0.20 * m_df['min_price'] + 0.20 * m_df['max_price']

    # Regime info
    regime_info = market_regimes.get(market, {'regime': 'unknown', 'std': 0.0, 'cv_pct': 0.0})
    regime = regime_info.get('regime', 'unknown')

    price_diffs = m_df['weighted_avg_modal_price'].diff().dropna()
    daily_vol = float(price_diffs.std()) if len(price_diffs) > 5 else 25.0
    typical_band = max(20.0, round(daily_vol * 1.645, 1))

    if regime == 'active':
        market_regime_label = "Active Trading Hub (Higher Volatility)"
        market_note = f"Active market with daily price moves. Expected daily trading band: ±Rs. {typical_band:.0f} / Quintal."
    elif regime == 'low_volatility':
        market_regime_label = "Low Volatility Market (Moderate Trading)"
        market_note = f"Moderate price movement market. Typical daily band: ±Rs. {typical_band:.0f} / Quintal."
    else:
        market_regime_label = "Flat / MSP-Sticky Market"
        market_note = f"Prices in this APMC are sticky around government MSP support. Typical band: ±Rs. {typical_band:.0f} / Quintal."

    # Resolve model
    model_obj, active_model_name, exog_cols = _resolve_model_for_market(artifact, market, model_preference)

    today_real = datetime.date.today()
    current_date = today_real
    last_row = m_df.iloc[-1]
    current_price = float(last_row['weighted_avg_modal_price'])
    last_date = pd.to_datetime(m_df['date'].iloc[-1])

    print(f"\n[Predict Engine] Market: {market}, District: {district_name}, AP")
    print(f"  Commodity          : {commodity_name}")
    print(f"  Target Metric      : {target_type}")
    print(f"  Current Date       : {current_date.strftime('%Y-%m-%d')}")
    print(f"  Weighted Avg Price : Rs. {current_price:.2f} / Quintal")
    print(f"  Regime             : {regime} (std=Rs.{regime_info.get('std', 0):.1f})")
    print(f"  Active Model       : {active_model_name}")
    print(f"  Forecast Horizon   : {forecast_horizon} days")

    predictions = []

    # ======================== PROPHET FORECAST ========================
    if active_model_name == 'Prophet' and model_obj is not None and HAS_PROPHET:
        # Forecast starting from current_date (Today)
        forecast_start = pd.Timestamp(current_date)
        prophet_preds = _generate_prophet_forecast(model_obj, forecast_start, forecast_horizon, exog_cols, m_df)

        for step, pred_info in enumerate(prophet_preds, 1):
            pred_price = pred_info['yhat']
            lower_bound = pred_info['yhat_lower']
            upper_bound = pred_info['yhat_upper']
            forecast_date = pred_info['date']

            price_change = pred_price - current_price
            if abs(price_change) < 5.0:
                trend = "STABLE"
            elif price_change > 0:
                trend = "BULLISH / UPWARD"
            else:
                trend = "BEARISH / DOWNWARD"

            horizon_labels = {
                1: f"Today ({current_date.strftime('%Y-%m-%d')})",
                2: f"Tomorrow ({(current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')})",
                3: f"Day +2 ({(current_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')})"
            }

            pred_item = {
                'horizon_day': horizon_labels.get(step, f"Day +{step}"),
                'date': forecast_date.strftime('%Y-%m-%d'),
                'expected_weighted_avg_price': round(pred_price, 2),
                'trend': trend,
                'expected_trading_range': [round(max(0, lower_bound), 1), round(upper_bound, 1)],
                'typical_daily_range_rs': f"±Rs. {round((upper_bound - lower_bound) / 2, 0):.0f}",
                'price_change_rs': round(price_change, 2),
                'confidence_interval': [round(lower_bound, 2), round(upper_bound, 2)]
            }
            predictions.append(pred_item)

    # ======================== ARIMA FORECAST ========================
    elif active_model_name == 'ARIMA' and model_obj is not None and HAS_ARIMA:
        arima_forecast, arima_conf = _generate_arima_forecast(model_obj, forecast_horizon, exog_cols, m_df)

        for step in range(forecast_horizon):
            forecast_date = current_date + datetime.timedelta(days=step + 1)
            pred_price = float(arima_forecast[step])
            lower_bound = float(arima_conf[step, 0])
            upper_bound = float(arima_conf[step, 1])

            price_change = pred_price - current_price
            if abs(price_change) < 5.0:
                trend = "STABLE"
            elif price_change > 0:
                trend = "BULLISH / UPWARD"
            else:
                trend = "BEARISH / DOWNWARD"

            horizon_labels = {
                1: f"Today ({current_date.strftime('%Y-%m-%d')})",
                2: f"Tomorrow ({(current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')})",
                3: f"Day +2 ({(current_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')})"
            }

            pred_item = {
                'horizon_day': horizon_labels.get(step + 1, f"Day +{step + 1}"),
                'date': forecast_date.strftime('%Y-%m-%d'),
                'expected_weighted_avg_price': round(pred_price, 2),
                'trend': trend,
                'expected_trading_range': [round(max(0, lower_bound), 1), round(upper_bound, 1)],
                'typical_daily_range_rs': f"±Rs. {round((upper_bound - lower_bound) / 2, 0):.0f}",
                'price_change_rs': round(price_change, 2),
                'confidence_interval': [round(lower_bound, 2), round(upper_bound, 2)]
            }
            predictions.append(pred_item)

    # ======================== ML MODEL FORECAST (XGBoost/GB) ========================
    elif active_model_name in ('XGBoost', 'GradientBoosting') and model_obj is not None:
        history_prices = m_df['weighted_avg_modal_price'].astype(float).tolist()
        feature_history = m_df[feature_cols].fillna(0.0).copy() if feature_cols else pd.DataFrame()
        step_price = current_price

        for step in range(1, forecast_horizon + 1):
            forecast_date = current_date + datetime.timedelta(days=step)

            try:
                base_row = feature_history.iloc[[-1]].copy()
                model_input = _build_model_input_frame(base_row.iloc[0], feature_cols, step_price, history_prices)

                if active_model_name == 'XGBoost':
                    # XGBoost predicts change ratio
                    change_pred = float(np.asarray(model_obj.predict(model_input.to_frame().T))[0])
                    pred_price = step_price * (1.0 + change_pred)
                else:
                    # GradientBoosting predicts absolute price
                    pred_price = float(np.asarray(model_obj.predict(model_input.to_frame().T))[0])

                if not np.isfinite(pred_price):
                    pred_price = step_price
            except Exception:
                pred_price = step_price

            # Clip to reasonable range
            pred_price = float(np.clip(pred_price, current_price - 300.0, current_price + 300.0))

            band_scale = typical_band * (1.0 + 0.15 * (step - 1))
            lower_bound = round(max(0.0, pred_price - band_scale), 1)
            upper_bound = round(pred_price + band_scale, 1)

            price_change = pred_price - current_price
            if abs(price_change) < 5.0:
                trend = "STABLE"
            elif price_change > 0:
                trend = "BULLISH / UPWARD"
            else:
                trend = "BEARISH / DOWNWARD"

            horizon_labels = {
                1: f"Today ({current_date.strftime('%Y-%m-%d')})",
                2: f"Tomorrow ({(current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')})",
                3: f"Day +2 ({(current_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')})"
            }

            pred_item = {
                'horizon_day': horizon_labels.get(step, f"Day +{step}"),
                'date': forecast_date.strftime('%Y-%m-%d'),
                'expected_weighted_avg_price': round(pred_price, 2),
                'trend': trend,
                'expected_trading_range': [lower_bound, upper_bound],
                'typical_daily_range_rs': f"±Rs. {band_scale:.0f}",
                'price_change_rs': round(price_change, 2),
                'confidence_interval': [lower_bound, upper_bound]
            }
            predictions.append(pred_item)

            # Step forward
            history_prices.append(pred_price)
            step_price = pred_price
            if not feature_history.empty:
                next_row = feature_history.iloc[[-1]].copy()
                next_row = next_row.reindex(columns=feature_cols, fill_value=0.0)
                next_row = _build_model_input_frame(next_row.iloc[0], feature_cols, step_price, history_prices)
                feature_history = pd.concat([feature_history, next_row.to_frame().T], ignore_index=True)

    # ======================== MULTI-TARGET PREDICTIONS & RECONCILIATION ========================
    prophet_min_models = artifact.get('prophet_min_models', {})
    arima_min_models = artifact.get('arima_min_models', {})
    prophet_max_models = artifact.get('prophet_max_models', {})
    arima_max_models = artifact.get('arima_max_models', {})
    arima_spread_models = artifact.get('arima_spread_models', {})

    # Predict min series
    min_preds = []
    if active_model_name == 'Prophet' and market in prophet_min_models and HAS_PROPHET:
        pm_preds = _generate_prophet_forecast(prophet_min_models[market], pd.Timestamp(current_date), forecast_horizon, exog_cols, m_df)
        min_preds = [p['yhat'] for p in pm_preds]
    elif market in arima_min_models and HAS_ARIMA:
        ar_preds, _ = _generate_arima_forecast(arima_min_models[market], forecast_horizon, exog_cols, m_df)
        min_preds = [float(p) for p in ar_preds]

    # Predict max series
    max_preds = []
    if active_model_name == 'Prophet' and market in prophet_max_models and HAS_PROPHET:
        pm_preds = _generate_prophet_forecast(prophet_max_models[market], pd.Timestamp(current_date), forecast_horizon, exog_cols, m_df)
        max_preds = [p['yhat'] for p in pm_preds]
    elif market in arima_max_models and HAS_ARIMA:
        ar_preds, _ = _generate_arima_forecast(arima_max_models[market], forecast_horizon, exog_cols, m_df)
        max_preds = [float(p) for p in ar_preds]

    # Predict spread series (log-space exponentiated)
    spread_preds = []
    if market in arima_spread_models and HAS_ARIMA:
        ar_log_preds, _ = _generate_arima_forecast(arima_spread_models[market], forecast_horizon, exog_cols, m_df)
        spread_preds = [max(0.0, float(np.exp(p) - 1.0)) for p in ar_log_preds]

    # Apply residual-based error band calibration to all forecast steps
    q10_res = float(abs(artifact.get('q10_residual', 25.0)))
    q90_res = float(abs(artifact.get('q90_residual', 25.0)))
    calib_band = max(25.0, (q90_res + q10_res) / 2.0)

    for idx, p in enumerate(predictions):
        modal_val = p['expected_weighted_avg_price']
        min_val = min_preds[idx] if idx < len(min_preds) else p['expected_trading_range'][0]
        max_val = max_preds[idx] if idx < len(max_preds) else p['expected_trading_range'][1]
        spread_val = spread_preds[idx] if idx < len(spread_preds) else (max_val - min_val)

        # Post-processing reconciliation with residual calibration: min <= modal <= max
        min_candidate = min(min_val, modal_val - calib_band)
        max_candidate = max(max_val, modal_val + calib_band)
        
        min_rec = round(min(min_candidate, modal_val - 1.0), 2)
        max_rec = round(max(max_candidate, modal_val + 1.0), 2)
        spread_rec = round(max(0.0, max_rec - min_rec), 2)

        p['expected_min_price'] = min_rec
        p['expected_max_price'] = max_rec
        p['expected_spread'] = spread_rec
        p['expected_trading_range'] = [min_rec, max_rec]

    # ======================== OUTPUT ========================
    output_payload = {
        'market': market,
        'district': district_name,
        'state': 'Andhra Pradesh',
        'commodity': commodity_name,
        'target_metric': target_type,
        'current_date': current_date.strftime('%Y-%m-%d'),
        'current_weighted_avg_price': round(current_price, 2),
        'market_regime': market_regime_label,
        'market_regime_type': regime,
        'regime_std': regime_info.get('std', 0),
        'market_note': market_note,
        'model_used': active_model_name,
        'forecast_horizon_days': forecast_horizon,
        'predictions': predictions
    }

    print("=" * 85)
    print(f"7-DAY FORECAST ({active_model_name} — {market}, {district_name})")
    print("=" * 85)
    print(f"Today's Weighted Avg Price ({current_date.strftime('%Y-%m-%d')}): Rs. {current_price:.2f} / Quintal")
    print(f"Regime: {regime} | Model: {active_model_name}\n")
    for p in predictions:
        print(f"  * {p['horizon_day']} ({p['date']}):")
        print(f"      Trend       : {p['trend']}")
        print(f"      Expected    : Rs. {p['expected_weighted_avg_price']:.2f} / Quintal")
        print(f"      Range       : [Rs. {p['expected_trading_range'][0]:.1f} – Rs. {p['expected_trading_range'][1]:.1f}]")
        print(f"      Change      : Rs. {p['price_change_rs']:+.2f}\n")
    print("=" * 85)

    return output_payload


if __name__ == '__main__':
    generate_multi_market_forecast(market="Jaggampet", model_preference="Auto")
