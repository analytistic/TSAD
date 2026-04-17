"""
This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
Original source: [https://github.com/TimeEval/TimeEval-algorithms]
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
# from ..utils.utility import zscore
from ..base.modeling_base import BaseTASDModel
from ..base.output_base import BaseTASDModelOutput
from .configuration_RQTAD import RQTADConfig
from torch import nn
import torch
from transformers.training_args import TrainingArguments
from .. import register_model
from ..base.processing_base import BaseProcessor
from datasets import concatenate_datasets
from ...dataset import DatasetFeature
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import argrelextrema

@dataclass
class RQTADModelOutput(BaseTASDModelOutput):
    """
    Model output for RQTAD.
    """
    idx: torch.Tensor | None = None







class RQKMeans(nn.Module):
    def __init__(self, config: RQTADConfig):
        super().__init__()
        self.k_list = config.k_list
        self.window_size = config.window_size
        self.codebook_list = nn.ModuleList([
            nn.Embedding(config.k_list[i], self.window_size) for i in range(config.codebook_num)
        ])
        self.n_iter = config.n_iter
        self.tol = config.tol

        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            nn.init.uniform_(codebook.weight, a=-1.0, b=1.0)
            codebook.weight.requires_grad = False


    def forward(self, X, return_dist=False):
        idx_list, dists_list = [], []
        residual = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            distance = torch.cdist(residual, codebook.weight)          
            idx = torch.argmin(distance, dim=-1)                      
            min_dists = distance.gather(-1, idx.unsqueeze(-1)).squeeze(-1) # (batch_size,)
            idx_list.append(idx)
            dists_list.append(min_dists)
            residual = residual - codebook(idx)  

        if return_dist:
            return idx_list, torch.stack(dists_list, dim=1)  
        else:
            return idx_list, None
        
    
    def fit(self, X):
        X = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        res = X
        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            for _ in range(self.n_iter):
                distance = torch.cdist(res, codebook.weight)          
                idx = torch.argmin(distance, dim=-1)                      
                new_centers = torch.zeros_like(codebook.weight, requires_grad=False, device=codebook.weight.device)
                new_centers = new_centers.scatter_add_(0, idx.unsqueeze(-1).expand(-1, res.shape[-1]), res)

                counts = torch.bincount(idx, minlength=codebook.num_embeddings)
                new_centers[counts==0] = codebook.weight[counts==0]
                new_centers[counts!=0] /= counts[counts!=0].unsqueeze(-1)

                shift = torch.norm(codebook.weight - new_centers, dim=1).max()
                codebook.weight.data.copy_(new_centers)
                if shift < self.tol:
                    break
            res = res - codebook(idx)
    

    




@register_model("RQTAD")
class RQTAD(BaseTASDModel):
    def __init__(self, config: RQTADConfig):
        super().__init__(config)
        self.model = RQKMeans(config)

    def _detect_period(self, data, rank=1)-> int:
        """
        calculate the period window of the data by acf
        """
        assert data.ndim == 1, "Only support univariate data for period detection"
        data = data[:min(2000, len(data))]

        base = 3
        auto_corr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[len(data)-1:]
        auto_corr /= auto_corr[0]
        auto_corr = auto_corr[base:]

        local_max = argrelextrema(auto_corr, np.greater)[0]

        try:
            sorted_local_max = np.argsort(auto_corr[local_max])[::-1]    # Ascending order
            max_local_max = sorted_local_max[0]     # Default
            if rank == 1: max_local_max = sorted_local_max[0]
            if rank == 2: 
                for i in sorted_local_max[1:]: 
                    if i > sorted_local_max[0]: 
                        max_local_max = i 
                        break
            if rank == 3:
                for i in sorted_local_max[1:]: 
                    if i > sorted_local_max[0]: 
                        id_tmp = i
                        break
                for i in sorted_local_max[id_tmp:]:
                    if i > id_tmp: 
                        max_local_max = i           
                        break

            final_period = local_max[max_local_max]+base
            if final_period>500:
                return 125
            return int(final_period)
        except:
            return 125

    def partial_fit(self):
        return 

    def fit(self, train_data, test_data, train_args, processor, **kwargs):
        all_data = concatenate_datasets([train_data, test_data])
        window_size = self._detect_period(np.array(all_data[DatasetFeature.TIMESERIES.value]))
        # window_size = self.config.window_size

        processor.window_size = window_size
        self.config.window_size = window_size
        self.model = RQKMeans(self.config)
        all_data = processor.prepare_dataset(all_data)
        timeslide = processor(timeslide=all_data[DatasetFeature.TIMESLIDE.value])[DatasetFeature.TIMESLIDE.value]
        self.model.fit(timeslide)
        self.save_pretrained(train_args.output_dir)
        processor.save_pretrained(train_args.output_dir)
        return self
    
    
    def forward(self, timeslide, **kwargs):
        idx, dists = self.model(timeslide, return_dist=True)
        return RQTADModelOutput(
            sorce=dists[:, -1],
            idx=idx
        )


  