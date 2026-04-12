"""
This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
Original source: [https://github.com/TimeEval/TimeEval-algorithms]
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
# from ..utils.utility import zscore
from ..base.modeling_base import BaseTASDModel
from ..base.output_base import BaseTASDModelOutput
from .configuration_KMeansAD import KMeansADConfig
from torch import nn
import torch
from transformers.training_args import TrainingArguments
from .. import register_model
from ..base.processing_base import BaseProcessor
from datasets import concatenate_datasets
from ...dataset import DatasetFeature
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class KMeansADModelOutput(BaseTASDModelOutput):
    """
    Model output for KMeansAD.
    """
    idx: torch.Tensor | None = None




class KMeans(nn.Module):
    def __init__(self, config: KMeansADConfig):
        super().__init__()
        self.k = config.k
        self.window_size = config.window_size
        self.codebook = nn.Embedding(self.k, self.window_size)
        self.n_iter = config.n_iter
        self.tol = config.tol

        nn.init.uniform_(self.codebook.weight, a=-1.0, b=1.0)
        self.codebook.weight.requires_grad = False

    def forward(self, X, return_dist=False):
        # X shape: (n_samples, window_size)
        distance =  torch.cdist(X, self.codebook.weight)
        idx = torch.argmin(distance, dim=-1)
        min_dists = distance.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        if return_dist:
            return idx, min_dists
        else:
            return idx, None
    

    




@register_model("KMeansAD")
class KMeansAD(BaseTASDModel):
    def __init__(self, config: KMeansADConfig):
        super().__init__(config)
        self.model = KMeans(config)

    def partial_fit(self):
        return 

    def fit(self, train_data, test_data, train_args, processor, **kwargs):
        all_data = concatenate_datasets([train_data, test_data])
        all_data = processor.prepare_dataset(all_data)
        timeslide = processor(timeslide=all_data[DatasetFeature.TIMESLIDE.value])[DatasetFeature.TIMESLIDE.value].to(self.model.codebook.weight.device, dtype=self.model.codebook.weight.dtype)
        for _ in range(self.model.n_iter):
            outputs = self.forward(timeslide, return_dist=True)
            if outputs.idx is None:
                raise ValueError("Model forward did not return cluster indices.")
            new_centers = torch.zeros_like(self.model.codebook.weight, requires_grad=False, device=self.model.codebook.weight.device)
            new_centers = new_centers.scatter_add_(0, outputs.idx.unsqueeze(-1).expand(-1, timeslide.shape[-1]), timeslide)

            counts = torch.bincount(outputs.idx, minlength=self.model.k)# k,1
            new_centers[counts==0] = self.model.codebook.weight[counts==0]
            new_centers[counts!=0] /= counts[counts!=0].unsqueeze(-1)

            shift = torch.norm(self.model.codebook.weight - new_centers, dim=1).max()
            self.model.codebook.weight.data.copy_(new_centers)
            if shift < self.model.tol:
                break

        self.save_pretrained(train_args.output_dir)
        processor.save_pretrained(train_args.output_dir)
        return self
    
    
    def forward(self, timeslide, **kwargs):
        timeslide = timeslide.to(self.model.codebook.weight.device, dtype=self.model.codebook.weight.dtype)
        idx, dists = self.model(timeslide, return_dist=True)
        return KMeansADModelOutput(
            sorce=dists,
            idx=idx
        )


  