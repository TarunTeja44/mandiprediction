import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import predict


def test_generate_multi_market_forecast_uses_saved_model(monkeypatch):
    class DummyModel:
        def predict(self, X):
            return np.array([0.10])  # 10% return change

    artifact = {
        'xgb_model': DummyModel(),
        'metrics': {
            'XGBoost': {'model': DummyModel()},
            'GradientBoosting': {'model': {}}
        },
        'feature_cols': ['lag_1', 'rolling_mean_7'],
        'commodity': 'Paddy(Common)',
        'target_type': 'Weighted Average Modal Price (60% Modal + 20% Min + 20% Max)'
    }

    rows = []
    for idx in range(10):
        rows.append({
            'date': f'2026-08-{idx + 1:02d}',
            'Market': 'Jaggampet',
            'District': 'Kakinada',
            'modal_price': 2500.0 + idx,
            'min_price': 2400.0 + idx,
            'max_price': 2600.0 + idx,
            'weighted_avg_modal_price': 2450.0 + idx,
            'lag_1': 2450.0 + idx,
            'rolling_mean_7': 2450.0 + idx,
        })
    df = pd.DataFrame(rows)

    monkeypatch.setattr(predict.joblib, 'load', lambda path: artifact)
    monkeypatch.setattr(predict.pd, 'read_csv', lambda path: df)

    payload = predict.generate_multi_market_forecast(market='Jaggampet', model_preference='XGBoost')

    assert len(payload['predictions']) == 30
    assert payload['predictions'][0]['expected_weighted_avg_price'] > 2000.0
    assert 'expected_min_price' in payload['predictions'][0]
    assert 'expected_max_price' in payload['predictions'][0]
    assert 'expected_spread' in payload['predictions'][0]
    assert payload['predictions'][0]['expected_min_price'] <= payload['predictions'][0]['expected_weighted_avg_price'] <= payload['predictions'][0]['expected_max_price']
