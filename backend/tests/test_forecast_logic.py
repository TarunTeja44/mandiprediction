import numpy as np

from predict import compute_holt_linear_forecast


def test_holt_linear_forecast_is_finite_and_reasonable():
    series = [100.0, 101.0, 103.0, 104.0, 106.0, 108.0, 109.0]

    preds = compute_holt_linear_forecast(series, horizon=2)

    assert len(preds) == 2
    assert np.isfinite(preds).all()
    assert preds[0] > 0
    assert abs(preds[0] - series[-1]) < 50
    assert abs(preds[1] - preds[0]) < 50
