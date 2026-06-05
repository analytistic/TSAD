import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            w = codebook.weight[centroid_idx]
            centroid_values = w.numpy() if hasattr(w, 'numpy') else np.asarray(w)
            results['centroids'].append({
                'centroid_id': f"L{level}_C{centroid_idx}",
                'level': level,
                'centroid_idx': centroid_idx,
                'time': list(range(len(centroid_values))),
                'values': centroid_values.tolist()
            })

        # Extract cluster assignments
        idx_array = idx.numpy() if hasattr(idx, 'numpy') else np.asarray(idx)
        for window_idx, cluster_id in enumerate(idx_array):
            results['cluster_assignments'].append({
                'window_idx': window_idx,
                'start_time': window_idx,
                'end_time': window_idx + window_size,
                'cluster_id': int(cluster_id),
                'level': level
            })

        # Extract cluster representatives
        for cluster_id in range(codebook.num_embeddings):
            w = codebook.weight[cluster_id]
            centroid_values = w.numpy() if hasattr(w, 'numpy') else np.asarray(w)
            # Find sample windows in this cluster
            sample_mask = idx_array == cluster_id
            sample_indices = np.where(sample_mask)[0][:5]  # Up to 5 samples

            results['cluster_representatives'].append({
                'level': level,
                'cluster_id': cluster_id,
                'centroid': centroid_values.tolist(),
                'sample_indices': sample_indices.tolist(),
                'window_count': int(sample_mask.sum())
            })

    return results


def create_clustered_timeseries_plot(timeseries: np.ndarray, cluster_assignments: list, window_size: int) -> go.Figure:
    """Create time series plot with cluster coloring."""
    fig = go.Figure()

    # Plot original time series
    fig.add_trace(go.Scatter(
        x=list(range(len(timeseries))),
        y=timeseries,
        mode='lines',
        name='Original Time Series',
        line=dict(color='gray', width=1)
    ))

    # Add colored windows for each level
    levels = set(a['level'] for a in cluster_assignments)
    colors = ['blue', 'red', 'green']

    for level in sorted(levels):
        level_assignments = [a for a in cluster_assignments if a['level'] == level]

        for assignment in level_assignments:
            start = assignment['start_time']
            end = assignment['end_time']
            cluster_id = assignment['cluster_id']

            # Add colored rectangle
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=colors[level],
                opacity=0.3,
                layer="below",
                line_width=0,
                name=f"Level {level}, Cluster {cluster_id}"
            )

    fig.update_layout(
        title="Time Series with Cluster Coloring",
        xaxis_title="Time",
        yaxis_title="Value"
    )

    return fig


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


def create_cluster_fragments_plot(cluster_representatives: list) -> go.Figure:
    """Create cluster representative fragments plot."""
    levels = sorted(set(r['level'] for r in cluster_representatives))

    fig = make_subplots(
        rows=len(levels),
        cols=1,
        subplot_titles=[f"Level {level}" for level in levels]
    )

    for row, level in enumerate(levels, 1):
        level_reps = [r for r in cluster_representatives if r['level'] == level]

        for rep in level_reps:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(rep['centroid']))),
                    y=rep['centroid'],
                    mode='lines',
                    name=f"Cluster {rep['cluster_id']} (n={rep['window_count']})",
                    legendgroup=f"Level {level}"
                ),
                row=row, col=1
            )

    fig.update_layout(
        title="Cluster Representative Fragments",
        height=300 * len(levels)
    )

    return fig


def generate_html_report(timeseries: np.ndarray, clustering_results: dict, output_path: Path) -> Path:
    """Generate complete HTML report."""
    # Extract results
    results = extract_clustering_results(clustering_results, window_size=clustering_results['config'].window_size[0])

    # Create visualizations
    centroid_plot = create_3d_centroid_plot(results['centroids'])
    ts_plot = create_clustered_timeseries_plot(timeseries, results['cluster_assignments'], window_size=clustering_results['config'].window_size[0])
    fragments_plot = create_cluster_fragments_plot(results['cluster_representatives'])

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RQTAD Clustering Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .plot {{ width: 100%; height: 600px; }}
        </style>
    </head>
    <body>
        <h1>RQTAD Multi-Layer Residual Clustering Analysis</h1>

        <div class="section">
            <h2>Summary</h2>
            <p><strong>Quantization Levels:</strong> {len(clustering_results['config'].k_list)}</p>
            <p><strong>Codewords per Level:</strong> {clustering_results['config'].k_list}</p>
            <p><strong>Window Sizes:</strong> {clustering_results['config'].window_size}</p>
        </div>

        <div class="section">
            <h2>3D Centroid Visualization</h2>
            <div id="centroid-plot" class="plot"></div>
            <script>
                var centroidData = {centroid_plot.to_json()};
                Plotly.newPlot('centroid-plot', centroidData.data, centroidData.layout);
            </script>
        </div>

        <div class="section">
            <h2>Time Series with Cluster Coloring</h2>
            <div id="ts-plot" class="plot"></div>
            <script>
                var tsData = {ts_plot.to_json()};
                Plotly.newPlot('ts-plot', tsData.data, tsData.layout);
            </script>
        </div>

        <div class="section">
            <h2>Cluster Representative Fragments</h2>
            <div id="fragments-plot" class="plot"></div>
            <script>
                var fragmentsData = {fragments_plot.to_json()};
                Plotly.newPlot('fragments-plot', fragmentsData.data, fragmentsData.layout);
            </script>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path
