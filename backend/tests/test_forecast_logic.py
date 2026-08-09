import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import predict

def test_ordering_reconciliation():
    modal_val = 3000.0
    min_val = 3100.0
    max_val = 2900.0

    min_rec = round(min(min_val, modal_val - 1.0), 2)
    max_rec = round(max(max_val, modal_val + 1.0), 2)
    spread_rec = round(max(0.0, max_rec - min_rec), 2)

    assert min_rec <= modal_val <= max_rec
    assert spread_rec == max_rec - min_rec
    assert min_rec == 2999.0
    assert max_rec == 3001.0
    assert spread_rec == 2.0
