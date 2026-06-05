# tests/test_visualize_clustering.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.visualize_clustering import load_yahoo_data, run_rqtad_clustering, extract_clustering_results, create_3d_centroid_plot, create_clustered_timeseries_plot, create_cluster_fragments_plot, generate_html_report


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


def test_create_clustered_timeseries_plot():
    """Test creation of clustered time series plot."""
    # Create sample data
    timeseries = np.sin(np.linspace(0, 10 * np.pi, 100))
    cluster_assignments = [
        {'window_idx': 0, 'start_time': 0, 'end_time': 10, 'cluster_id': 0, 'level': 0},
        {'window_idx': 1, 'start_time': 1, 'end_time': 11, 'cluster_id': 1, 'level': 0}
    ]

    fig = create_clustered_timeseries_plot(timeseries, cluster_assignments, window_size=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # At least the original time series


def test_create_cluster_fragments_plot():
    """Test creation of cluster fragments plot."""
    # Create sample cluster representatives
    cluster_representatives = [
        {
            'level': 0,
            'cluster_id': 0,
            'centroid': [1.0, 2.0, 3.0],
            'sample_indices': [0, 1, 2],
            'window_count': 10
        },
        {
            'level': 0,
            'cluster_id': 1,
            'centroid': [4.0, 5.0, 6.0],
            'sample_indices': [3, 4, 5],
            'window_count': 15
        }
    ]

    fig = create_cluster_fragments_plot(cluster_representatives)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two traces


def test_generate_html_report():
    """Test HTML report generation."""
    import torch
    # Create sample data
    timeseries = np.sin(np.linspace(0, 10 * np.pi, 100))

    clustering_results = {
        'idx_list': [np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([0, 1, 2, 0])],
        'codebook_list': [
            type('Codebook', (), {'num_embeddings': 2, 'weight': torch.randn(2, 10)})(),
            type('Codebook', (), {'num_embeddings': 2, 'weight': torch.randn(2, 10)})(),
            type('Codebook', (), {'num_embeddings': 3, 'weight': torch.randn(3, 10)})()
        ],
        'config': type('Config', (), {'k_list': [2, 2, 3], 'window_size': [10, 10, 10]})()
    }

    output_path = Path('test_report.html')
    try:
        result_path = generate_html_report(timeseries, clustering_results, output_path)
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        output_path.unlink()
