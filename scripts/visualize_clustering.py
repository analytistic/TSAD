import pandas as pd
import numpy as np
from pathlib import Path


def load_yahoo_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Yahoo CSV file and return time series and labels."""
    df = pd.read_csv(file_path)
    timeseries = df['Data'].values
    labels = df['Label'].values
    return timeseries, labels
