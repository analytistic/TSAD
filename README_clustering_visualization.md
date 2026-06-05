# RQTAD Clustering Visualization

This tool analyzes RQTAD's multi-layer residual quantization results and generates interactive HTML reports.

## Features

- 3D centroid visualization
- Time series with cluster coloring
- Cluster representative fragments

## Usage

```bash
python scripts/visualize_clustering.py \
    --data_path data/TSB-AD-U/YAHOO/551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv \
    --output_path outputs/clustering_analysis.html
```

## Output

The tool generates an HTML file with:
1. Summary of model configuration
2. Interactive 3D scatter plot of centroids
3. Time series plot with cluster coloring
4. Cluster representative fragments

## Dependencies

- torch
- numpy
- pandas
- plotly
- transformers (from existing project)
