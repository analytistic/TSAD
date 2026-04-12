from ..model import MODEL_REGISTRY, CONFIG_REGISTRY, PROCESSOR_REGISTRY
from transformers import PreTrainedModel, PreTrainedConfig
from transformers import set_seed
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import datetime
import numpy as np
from ..model.base.processing_base import BaseProcessor
from ..model.base.output_base import BaseTASDModelOutput
from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from ..evaluation.metrics import get_metrics
from ..dataset import DatasetFeature
import json
from typing import Dict, Any

class AnomalyStreamEvaluation:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        # TODO: eval on stream data
        pass