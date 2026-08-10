import json
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load notebook and test cells
nb = json.load(open('AP_Paddy_Price_Prediction_Colab.ipynb', encoding='utf-8'))
print(f"Loaded notebook with {len(nb['cells'])} cells.")

# Execute key code blocks
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if src.startswith('!pip'):
            continue
        print(f"\n--- Executing Code Cell {idx} ---")
        try:
            exec(src, globals())
            print(f"✓ Cell {idx} executed cleanly!")
        except Exception as e:
            print(f"❌ Error in Cell {idx}: {e}")
            raise e
