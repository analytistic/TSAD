"""
This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
Original source: [https://github.com/TimeEval/TimeEval-algorithms]
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
# from ..utils.utility import zscore
from ..base.modeling_base import BaseTASDModel
from .configuration_KMeansAD import KMeansADConfig
from torch import nn
import torch
from transformers.training_args import TrainingArguments
from .. import register_model
from ..base.processing_base import BaseProcessor



class KMeans(nn.Module):
    def __init__(self, config: KMeansADConfig):
        super().__init__()
        self.k = config.k
        self.window_size = config.window_size
        self.codebook = nn.Embedding(self.k, self.window_size)
        self.n_iter = config.n_iter
        self.tol = config.tol

        nn.init.zeros_(self.codebook.weight)
        self.codebook.weight.requires_grad = False

    def forward(self, X, return_dist=False):
        # X shape: (n_samples, window_size)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.codebook.weight.device).requires_grad_(False)
        distance =  torch.cdist(X, self.codebook.weight)
        idx = torch.argmin(distance, dim=-1)
        min_dists = distance.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        if return_dist:
            return idx, min_dists
        else:
            return idx, None
    
    def fit(self, X):
        # X shape: (n_samples, window_size)
        for _ in range(self.n_iter):
            idx, _ = self.forward(X, return_dist=True)
            new_centers = torch.zeros_like(self.codebook.weight, requires_grad=False, device=X.device)
            new_centers = new_centers.scatter_add_(0, idx.unsqueeze(-1).expand(-1, X.shape[-1]), X)

            counts = torch.bincount(idx, minlength=self.k).unsqueeze(-1) # k,1
            new_centers[counts==0] = self.codebook.weight[counts==0]
            new_centers[counts!=0] /= counts[counts!=0]

            shift = torch.norm(self.codebook.weight - new_centers, dim=1).max()
            self.codebook.weight.data.copy_(new_centers)
            if shift < self.tol:
                break
        return self
    




@register_model("KMeansAD")
class KMeansAD(BaseTASDModel):
    def __init__(self, config: KMeansADConfig):
        super().__init__(config)
        self.model = KMeans(config)
    
    def fit(self, X, train_config: TrainingArguments, processor: BaseProcessor, **kwargs):
        self.model.fit(X)
        self.save_pretrained(train_config.output_dir)
        processor.save_pretrained(train_config.output_dir)
        return self
    
    # def get_point_scores(self, window_scores, window_size, stride, padding_length):
    #     # compute begin and end indices of windows
    #     begins = np.array([i * stride for i in range(window_scores.shape[0])])
    #     ends = begins + window_size

    #     # prepare target array
    #     unwindowed_length = stride * (window_scores.shape[0] - 1) + window_size + padding_length
    #     mapped = np.full(unwindowed_length, fill_value=np.nan)

    #     # only iterate over window intersections
    #     indices = np.unique(np.r_[begins, ends])
    #     for i, j in zip(indices[:-1], indices[1:]):
    #         window_indices = np.flatnonzero((begins <= i) & (j-1 < ends))
    #         mapped[i:j] = np.nanmean(window_scores[window_indices])

    #     # replace untouched indices with 0 (especially for the padding at the end)
    #     np.nan_to_num(mapped, copy=False)
    #     return mapped
    
    def predict(self, timeslides, **kwargs):
        # X shape: (n_samples, window_size)
        _, dists = self.forward(timeslides)
        return dists
    
    def forward(self, timeslides, **kwargs):
        idx, dists = self.model(timeslides, return_dist=True)
        return dists






# class KMeansAD(BaseEstimator, OutlierMixin):
#     def __init__(self, k, window_size, stride, n_jobs=1, normalize=True):
#         self.k = k
#         self.window_size = window_size
#         self.stride = stride
#         self.model = KMeans(n_clusters=k)
#         self.padding_length = 0
#         self.normalize = normalize

#     def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
#         flat_shape = (X.shape[0] - (self.window_size - 1), -1)  # in case we have a multivariate TS
#         slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
#         self.padding_length = X.shape[0] - (slides.shape[0] * self.stride + self.window_size - self.stride)
#         print(f"Required padding_length={self.padding_length}")
#         # if self.normalize: slides = zscore(slides, axis=1, ddof=1)
#         return slides

#     def _custom_reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
#         print("Reversing window-based scores to point-based scores:")
#         print(f"Before reverse-windowing: scores.shape={scores.shape}")
#         # compute begin and end indices of windows
#         begins = np.array([i * self.stride for i in range(scores.shape[0])])
#         ends = begins + self.window_size

#         # prepare target array
#         unwindowed_length = self.stride * (scores.shape[0] - 1) + self.window_size + self.padding_length
#         mapped = np.full(unwindowed_length, fill_value=np.nan)

#         # only iterate over window intersections
#         indices = np.unique(np.r_[begins, ends])
#         for i, j in zip(indices[:-1], indices[1:]):
#             window_indices = np.flatnonzero((begins <= i) & (j-1 < ends))
#             # print(i, j, window_indices)
#             mapped[i:j] = np.nanmean(scores[window_indices])

#         # replace untouched indices with 0 (especially for the padding at the end)
#         np.nan_to_num(mapped, copy=False)
#         print(f"After reverse-windowing: scores.shape={mapped.shape}")
#         return mapped

#     def fit(self, X: np.ndarray, y=None, preprocess=True) -> 'KMeansAD':
#         if preprocess:
#             X = self._preprocess_data(X)
#         self.model.fit(X)
#         return self

#     def predict(self, X: np.ndarray, preprocess=True) -> np.ndarray:
#         if preprocess:
#             X = self._preprocess_data(X)
#         clusters = self.model.predict(X)
#         diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
#         return self._custom_reverse_windowing(diffs)

#     def fit_predict(self, X, y=None) -> np.ndarray:
#         X = self._preprocess_data(X)
#         self.fit(X, y, preprocess=False)
#         return self.predict(X, preprocess=False)
    

# if __name__ == "__main__":
#     import numpy as np

#     period = 6
#     length = 36 * 6
#     t = np.arange(length)
#     signal = np.sin(2 * np.pi * t / period) + 0.05 * np.random.randn(length)

#     model = KMeansAD(k=3, window_size=6, stride=1)
#     anomaly_scores = model.fit_predict(signal.reshape(-1, 1))

  