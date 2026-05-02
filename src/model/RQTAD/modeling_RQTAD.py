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
from .. import register_model
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


class PyramidRQKMeans(nn.Module):
    def __init__(self, config: RQTADConfig):
        super().__init__()
        self.k_list = config.k_list
        self.window_size = config.window_size
        self.codebook_num = config.codebook_num
        self.codebook_list = nn.ModuleList([
            nn.Embedding(config.k_list[i], self.window_size[i]) for i in range(config.codebook_num)
        ])
        self.encoder_list = nn.ModuleList([
            nn.Linear(self.window_size[0], self.window_size[i+1], bias=False) for i in range(config.codebook_num-1)
        ])
        self.decoder_list = nn.ModuleList([
            nn.Linear(self.window_size[i+1], self.window_size[0], bias=False) for i in range(config.codebook_num-1)
        ])
        self.n_iter = config.n_iter
        self.tol = config.tol

        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            nn.init.uniform_(codebook.weight, a=-1.0, b=1.0)
            codebook.weight.requires_grad = False
        
        for encoder in self.encoder_list:
            assert isinstance(encoder, nn.Linear)
            nn.init.xavier_uniform_(encoder.weight)
            encoder.weight.requires_grad = False
        
        for decoder in self.decoder_list:
            assert isinstance(decoder, nn.Linear)
            nn.init.xavier_uniform_(decoder.weight)
            decoder.weight.requires_grad = False

    def fit(self, X):
        X = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        res = X
        for l, codebook in enumerate(self.codebook_list):
            res_down = res

            if l > 0:
                # cluster in the svd space
                encoder = self.encoder_list[l-1]
                decoder = self.decoder_list[l-1]
                U, S, Vh = torch.linalg.svd(res)
                assert isinstance(encoder, nn.Linear) and isinstance(decoder, nn.Linear)
                encoder.weight.data.copy_(Vh[:self.window_size[l], :].clone())          
                decoder.weight.data.copy_(Vh[:self.window_size[l], :].clone().t())     
                res_down = encoder(res)

            n_sample = res_down.shape[0]    
            assert isinstance(codebook, nn.Embedding)
            k = codebook.num_embeddings
            rand_idx = torch.randperm(n_sample)[:k]
            codebook.weight.data.copy_(res_down[rand_idx].clone())
            codebook.weight.requires_grad = False
            

            for _ in range(self.n_iter):
                distance = torch.cdist(res_down, codebook.weight)          
                idx = torch.argmin(distance, dim=-1)    

                new_centers = torch.zeros_like(codebook.weight, requires_grad=False, device=codebook.weight.device)
                new_centers = new_centers.scatter_add_(0, idx.unsqueeze(-1).expand(-1, res_down.shape[-1]), res_down)

                counts = torch.bincount(idx, minlength=codebook.num_embeddings)
                new_centers[counts==0] = codebook.weight[counts==0]
                new_centers[counts!=0] /= counts[counts!=0].unsqueeze(-1)

                shift = torch.norm(codebook.weight - new_centers, dim=1).max()
                codebook.weight.data.copy_(new_centers)
                if shift < self.tol:
                    break
                
            res_up = codebook(idx) if l == 0 else self.decoder_list[l-1](codebook(idx))
            res = res - res_up
    
        return self

    def forward(self, X, return_dist=False):
        idx_list, dists_list, score_list = [], [], []
        res = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        for l, codebook in enumerate(self.codebook_list):
            res_down = res
            assert isinstance(codebook, nn.Embedding)
            if l > 0:
                res_down = self.encoder_list[l-1](res)
            distance = torch.cdist(res_down, codebook.weight)          
            idx = torch.argmin(distance, dim=-1)                      
            min_dists = distance.gather(-1, idx.unsqueeze(-1)).squeeze(-1) # (batch_size,)
            idx_list.append(idx)
            dists_list.append(min_dists)
            res_up = codebook(idx) if l == 0 else self.decoder_list[l-1](codebook(idx))
            res = res - res_up  
            score = torch.abs(res - res.median(dim=-1, keepdim=True)[0]).max()
            score_list.append(score)

        if return_dist:
            return idx_list, torch.stack(dists_list, dim=1), torch.tensor(score_list)
        else:
            return idx_list, None
        
    
            

class PruneRQKMeans(nn.Module):
    def __init__(self, config: RQTADConfig):
        super().__init__()
        self.k_list = config.k_list
        self.window_size = config.window_size
        self.codebook_num = config.codebook_num
        self.codebook_list = nn.ModuleList([
            nn.Embedding(config.k_list[i], self.window_size[0]) for i in range(self.codebook_num)
        ])
        self.n_iter = config.n_iter
        self.tol = config.tol
        
        self.gamma = getattr(config, 'gamma', 1.5) 
        self.epsilon = 1e-8                  

        for i in range(self.codebook_num):
            self.register_buffer(f'mainstream_mask_{i}', torch.ones(config.k_list[i], dtype=torch.bool))

        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            nn.init.uniform_(codebook.weight, a=-1.0, b=1.0)
            codebook.weight.requires_grad = False

    def forward(self, X, return_dist=False):
        idx_list, dists_list = [], []
        residual = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        for l, codebook in enumerate(self.codebook_list):
            assert isinstance(codebook, nn.Embedding)        

            distance = torch.cdist(residual, codebook.weight)          
            idx = torch.argmin(distance, dim=-1)                      
            min_dists = distance.gather(-1, idx.unsqueeze(-1)).squeeze(-1) # (batch_size,)
            idx_list.append(idx)
            dists_list.append(min_dists)
            
            mainstream_mask = getattr(self, f'mainstream_mask_{l}')
            is_mainstream = mainstream_mask[idx] # bs
            reconstruction = codebook(idx) * is_mainstream.unsqueeze(-1).to(residual.dtype)
            residual = residual - reconstruction  

        if return_dist:
            return idx_list, torch.stack(dists_list, dim=1)  
        else:
            return idx_list, None
        
    def fit(self, X):
        X = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        res = X
        for l, codebook in enumerate(self.codebook_list):
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
                  
            distance = torch.cdist(res, codebook.weight)
            idx_final = torch.argmin(distance, dim=-1)
            
            x_norm = torch.sum(res**2, dim=-1)                
            reconstructed = codebook(idx_final)              
            res_next_temp = res - reconstructed
            x_res_norm = torch.sum(res_next_temp**2, dim=-1)    
            delta = x_norm - x_res_norm                         
            
            delta_k = torch.zeros(codebook.num_embeddings, device=res.device, dtype=res.dtype)
            delta_k.scatter_add_(0, idx_final, delta)            
            c_norm = torch.sum(codebook.weight**2, dim=-1) 
            rho_k = delta_k / (c_norm + self.epsilon)      
            rho_med = torch.median(rho_k)
            mad = torch.median(torch.abs(rho_k - rho_med))
            
            anomalous_mask = torch.abs(rho_k - rho_med) > (self.gamma * mad)
            mainstream_mask = ~anomalous_mask
            
            getattr(self, f'mainstream_mask_{l}').copy_(mainstream_mask)
 
            is_mainstream = mainstream_mask[idx_final]
            res = res - reconstructed * is_mainstream.unsqueeze(-1).to(res.dtype)


class RQKMeans(nn.Module):
    def __init__(self, config: RQTADConfig):
        super().__init__()
        self.k_list = config.k_list
        self.window_size = config.window_size
        self.codebook_list = nn.ModuleList([
            nn.Embedding(config.k_list[i], self.window_size[0]) for i in range(config.codebook_num)
        ])
        self.n_iter = config.n_iter
        self.tol = config.tol

        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            nn.init.uniform_(codebook.weight, a=-1.0, b=1.0)
            codebook.weight.requires_grad = False


    def forward(self, X, return_dist=False):
        idx_list, dists_list, score_list = [], [], []
        residual = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        for codebook in self.codebook_list:
            assert isinstance(codebook, nn.Embedding)
            distance = torch.cdist(residual, codebook.weight)
            idx = torch.argmin(distance, dim=-1)
            min_dists = distance.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            idx_list.append(idx)
            dists_list.append(min_dists)
            residual = residual - codebook(idx)
            score_list.append(min_dists)

        if return_dist:
            return idx_list, torch.stack(dists_list, dim=1), torch.tensor(score_list)
        else:
            return idx_list, None
        
    
    def fit(self, X):
        X = X.to(self.codebook_list[0].weight.device, dtype=self.codebook_list[0].weight.dtype)
        res = X
    
        for codebook_idx, codebook in enumerate(self.codebook_list):
            n_sample = res.shape[0]    
            assert isinstance(codebook, nn.Embedding)
            k = codebook.num_embeddings
            rand_idx = torch.randperm(n_sample)[:k]
            codebook.weight.data.copy_(res[rand_idx].clone())
            codebook.weight.requires_grad = False
         
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
            distance = torch.cdist(res, codebook.weight)
            idx = torch.argmin(distance, dim=-1)
            res = res - codebook(idx)
        return self
    

    

@register_model("RQTAD")
class RQTAD(BaseTASDModel):
    def __init__(self, config: RQTADConfig):
        super().__init__(config)
        self.model = RQKMeans(config)

    def _detect_period(self, data, rank=1)-> list:
        """
        calculate the period window of the data by acf
        """
        assert data.ndim == 1, "Only support univariate data for period detection"
        data = data[:min(2000, len(data))]

        base = 5
        auto_corr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[len(data)-1:]
        auto_corr /= auto_corr[0]
        auto_corr = auto_corr[base:]

        local_max = argrelextrema(auto_corr, np.greater)[0]
        window = []

        try:
            sorted_local_max = np.argsort(auto_corr[local_max])[::-1]    # Ascending order
            max_local_max = sorted_local_max[0]     # Default
            if rank >= 1: 
                max_local_max = sorted_local_max[0]
                window.append(local_max[max_local_max]+base)
            if rank >= 2: 
                for i in sorted_local_max[1:]: 
                    if i > sorted_local_max[0]: 
                        max_local_max = i
                        window.append(local_max[max_local_max]+base) 
                        break
            if rank >= 3:
                for i in sorted_local_max[1:]: 
                    if i > sorted_local_max[0]: 
                        id_tmp = i
                        break
                for i in sorted_local_max[id_tmp:]:
                    if i > id_tmp: 
                        max_local_max = i     
                        window.append(local_max[max_local_max]+base)      
                        break

            for idx, w in enumerate(window):
                if w > 500:
                    window[idx] = 125
                else:
                    window[idx] = int(w)
            return window
        except:
            return [125]

    def partial_fit(self):
        return

    @staticmethod
    def _window_zscore(x):
        """Per-window z-score normalization. Each row is normalized independently."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def fit(self, train_data, test_data, train_args, processor, **kwargs):
        all_data = concatenate_datasets([train_data, test_data])
        # all_data = train_data
        window_size = self._detect_period(np.array(all_data[DatasetFeature.TIMESERIES.value]), rank=3)
        # window_size = self.config.window_size


        processor.window_size = window_size[0]
        self.config.window_size = window_size
        self.model = RQKMeans(self.config)
        all_data = processor.prepare_dataset(all_data)

        inputs = processor(
            timeslide=all_data[DatasetFeature.TIMESLIDE.value],
            timestamp=all_data[DatasetFeature.TIMESTAMP.value]
        )
        timeslide = inputs.data[DatasetFeature.TIMESLIDE.value]
        # timeslide = self._window_zscore(timeslide)
        self.model.fit(timeslide)
        self.save_pretrained(train_args.output_dir)
        processor.save_pretrained(train_args.output_dir)
        return self


    def forward(self, timeslide, timestamp, **kwargs):
        # timeslide = self._window_zscore(timeslide)
        idx, dists, score = self.model(timeslide, return_dist=True)
        if score.dim() == 2 and score.shape[0] == 1:
            score = score.squeeze(0)
        return RQTADModelOutput(
            score=score,
            idx=idx
        )


