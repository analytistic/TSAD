# tests/test_visualize_clustering.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.visualize_clustering import load_yahoo_data


def test_load_yahoo_data():
    """Test loading Yahoo CSV file."""
    # Create a temporary CSV file for testing
    test_data = pd.DataFrame({
        'Data': [1.0, 2.0, 3.0, 4.0, 5.0],
        'Label': [0, 0, 1, 0, 0]
    })
    test_path = Path('test_yahoo.csv')
    test_data.to_csv(test_path, index=False)

    try:
        timeseries, labels = load_yahoo_data(test_path)
        assert len(timeseries) == 5
        assert len(labels) == 5
        assert np.array_equal(timeseries, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(labels, [0, 0, 1, 0, 0])
    finally:
        test_path.unlink()
