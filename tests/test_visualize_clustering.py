# tests/test_visualize_clustering.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
from scripts.visualize_clustering import load_yahoo_data, run_rqtad_clustering, extract_clustering_results, create_3d_centroid_plot


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


def test_extract_clustering_results():
    """Test extraction of clustering results."""
    # Create synthetic time series
    np.random.seed(42)
    timeseries = np.sin(np.linspace(0, 10 * np.pi, 500)) + np.random.normal(0, 0.1, 500)

    # Run clustering
    clustering_results = run_rqtad_clustering(timeseries)

    # Extract results
    results = extract_clustering_results(clustering_results, window_size=40)

    assert 'centroids' in results
    assert 'cluster_assignments' in results
    assert 'cluster_representatives' in results
    assert len(results['centroids']) == 35  # 5 + 10 + 20 centroids
    assert len(results['cluster_assignments']) > 0
    assert len(results['cluster_representatives']) == 35


def test_create_3d_centroid_plot():
    """Test creation of 3D centroid plot."""
    # Create sample centroids
    centroids = [
        {
            'centroid_id': 'L0_C0',
            'level': 0,
            'centroid_idx': 0,
            'time': [0, 1, 2],
            'values': [1.0, 2.0, 3.0]
        },
        {
            'centroid_id': 'L0_C1',
            'level': 0,
            'centroid_idx': 1,
            'time': [0, 1, 2],
            'values': [4.0, 5.0, 6.0]
        }
    ]

    fig = create_3d_centroid_plot(centroids)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two traces
