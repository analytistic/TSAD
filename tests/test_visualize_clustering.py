# tests/test_visualize_clustering.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.visualize_clustering import load_yahoo_data, run_rqtad_clustering


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


def test_run_rqtad_clustering():
    """Test RQTAD clustering execution."""
    # Create synthetic time series
    np.random.seed(42)
    timeseries = np.sin(np.linspace(0, 10 * np.pi, 500)) + np.random.normal(0, 0.1, 500)

    result = run_rqtad_clustering(timeseries)

    assert 'idx_list' in result
    assert 'codebook_list' in result
    assert 'config' in result
    assert len(result['idx_list']) == 3  # 3 levels
    assert len(result['codebook_list']) == 3
