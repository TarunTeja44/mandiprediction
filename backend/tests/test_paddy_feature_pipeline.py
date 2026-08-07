import os
import unittest

from paddy_feature_engine import build_paddy_features, FEATURED_CSV


class PaddyFeaturePipelineTests(unittest.TestCase):
    def test_feature_engine_writes_weighted_average_features(self):
        df = build_paddy_features()

        self.assertIn('weighted_avg_modal_price', df.columns)
        self.assertIn('target_return', df.columns)
        self.assertIn('lag_14', df.columns)
        self.assertIn('seasonal_ma_7', df.columns)
        self.assertIn('seasonal_ma_30', df.columns)
        self.assertIn('rolling_mean_30', df.columns)
        self.assertTrue(os.path.exists(FEATURED_CSV))
        self.assertIn('weighted_avg', os.path.basename(FEATURED_CSV))


if __name__ == '__main__':
    unittest.main()
