import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


def load_yahoo_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Yahoo CSV file and return time series and labels."""
    df = pd.read_csv(file_path)
    timeseries = df['Data'].values
    labels = df['Label'].values
    return timeseries, labels


def run_rqtad_clustering(timeseries: np.ndarray) -> dict:
    """Run RQTAD clustering and return results.

    Args:
        timeseries: 1D numpy array of time series data.

    Returns:
        Dictionary with keys:
            - idx_list: list of cluster assignment tensors (one per level)
            - codebook_list: list of codebook embeddings (one per level)
            - config: RQTADConfig used for clustering
    """
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from src.model.RQTAD.modeling_RQTAD import RQTAD
    from src.model.RQTAD.configuration_RQTAD import RQTADConfig
    from src.model.RQTAD.processing_RQTAD import RQTADProcessor

    # Create config
    config = RQTADConfig(
        k_list=[5, 10, 20],
        window_size=[40, 40, 40],
        stride=1,
        n_iter=300,
        tol=1e-4,
        codebook_num=3,
    )

    # Create processor and model
    processor = RQTADProcessor(window_size=40, stride=1)
    model = RQTAD(config)

    # Prepare data as HuggingFace Dataset
    train_data = Dataset.from_dict(
        {"timeseries": timeseries.tolist(), "timestamp": list(range(len(timeseries)))}
    )

    # Fit model
    training_args = TrainingArguments(output_dir="./tmp", report_to="none")
    model.fit(train_data, train_data, training_args, processor)

    # Run clustering
    inputs = processor(timeseries=timeseries)
    timeslide = inputs.data["timeslide"]
    timestamp = inputs.data["timestamp"]

    with torch.no_grad():
        model.eval()
        outputs = model(timeslide=timeslide, timestamp=timestamp)

    return {
        "idx_list": outputs.idx,
        "codebook_list": model.model.codebook_list,
        "config": config,
    }


def extract_clustering_results(clustering_results: dict, window_size: int) -> dict:
    """Extract detailed clustering results for visualization."""
    idx_list = clustering_results['idx_list']
    codebook_list = clustering_results['codebook_list']

    results = {
        'centroids': [],
        'cluster_assignments': [],
        'cluster_representatives': []
    }

    for level, (idx, codebook) in enumerate(zip(idx_list, codebook_list)):
        # Extract centroids
        for centroid_idx in range(codebook.num_embeddings):
            centroid_values = codebook.weight[centroid_idx].numpy()
            results['centroids'].append({
                'centroid_id': f"L{level}_C{centroid_idx}",
                'level': level,
                'centroid_idx': centroid_idx,
                'time': list(range(len(centroid_values))),
                'values': centroid_values.tolist()
            })

        # Extract cluster assignments
        for window_idx, cluster_id in enumerate(idx.numpy()):
            results['cluster_assignments'].append({
                'window_idx': window_idx,
                'start_time': window_idx,
                'end_time': window_idx + window_size,
                'cluster_id': int(cluster_id),
                'level': level
            })

        # Extract cluster representatives
        for cluster_id in range(codebook.num_embeddings):
            centroid_values = codebook.weight[cluster_id].numpy()
            # Find sample windows in this cluster
            sample_mask = idx.numpy() == cluster_id
            sample_indices = np.where(sample_mask)[0][:5]  # Up to 5 samples

            results['cluster_representatives'].append({
                'level': level,
                'cluster_id': cluster_id,
                'centroid': centroid_values.tolist(),
                'sample_indices': sample_indices.tolist(),
                'window_count': int(sample_mask.sum())
            })

    return results


def create_3d_centroid_plot(centroids: list) -> go.Figure:
    """Create 3D scatter plot of centroids."""
    fig = go.Figure()

    # Group by level
    levels = set(c['level'] for c in centroids)
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for level in sorted(levels):
        level_centroids = [c for c in centroids if c['level'] == level]

        # Create traces for each centroid
        for centroid in level_centroids:
            fig.add_trace(go.Scatter3d(
                x=[centroid['centroid_id']] * len(centroid['time']),
                y=centroid['time'],
                z=centroid['values'],
                mode='lines+markers',
                name=f"Level {level}, Centroid {centroid['centroid_idx']}",
                line=dict(color=colors[level], width=2),
                marker=dict(size=3),
                legendgroup=f"Level {level}"
            ))

    fig.update_layout(
        title="RQTAD Centroids in 3D Space",
        scene=dict(
            xaxis_title="Centroid ID",
            yaxis_title="Time",
            zaxis_title="Value"
        ),
        legend=dict(
            groupclick="toggleitem"
        )
    )

    return fig
