import pandas as pd
import numpy as np
from pathlib import Path


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
