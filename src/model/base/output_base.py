from transformers.utils.generic import ModelOutput
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
@dataclass
class BaseTASDModelOutput(ModelOutput):
    """
    Base class for model outputs in time series anomaly detection.
    """
    sorce: torch.Tensor
    