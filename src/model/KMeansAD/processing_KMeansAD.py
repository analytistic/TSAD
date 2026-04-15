from transformers import FeatureExtractionMixin
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from ..utils.scaler import ScalerType, StandaryScaler, MinMaxScaler, BaseScaler
from numpy.lib.stride_tricks import sliding_window_view
from .. import register_processor
from ..base.processing_base import BaseProcessor
from datasets import Dataset
from ...dataset import DatasetFeature

@register_processor("KMeansAD")
class KMeansADProcessor(BaseProcessor):
    def __init__(self,
                 window_size: int=24,
                 stride: int=1,
                 scale: bool=False,
                 scaler_type: Optional[ScalerType] = None,
                 **kwargs
                 ):
        super().__init__(window_size=window_size, stride=stride, scale=scale, scaler_type=scaler_type, **kwargs)
        self.window_size = window_size
        self.stride = stride
        self.scale = scale
        self.scaler = self._build_scaler(scaler_type, **kwargs) if scale else None

    
    def _build_scaler(self, scaler_type: ScalerType | None, **kwargs) -> BaseScaler:
        if scaler_type == ScalerType.STANDARY:
            return StandaryScaler(**kwargs)
        elif scaler_type == ScalerType.MINMAX:
            return MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
    def fit(self, inputs: np.ndarray):
        if self.scaler is not None:
            self.scaler.fit(inputs)
    
    def transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(inputs)
        else:
            return inputs
        
    def inverse_transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.inverse_transform(inputs)
        else:
            return inputs

    
    def __call__(self, 
                 timeseries: np.ndarray | List | None = None,
                 timeslide: np.ndarray | List | None = None, 
                 timestamp: np.ndarray | List | None = None,
                 ex_features: np.ndarray | None = None,
                 labels: np.ndarray | None = None,
                 scale: Optional[bool] = None, 
                 return_tensors: str = 'pt', 
                 **kwargs) -> BatchFeature:
        """
        if timeslide is provided, use timeslide directly and ignore timeseries and timestamp. Otherwise, compute timeslide from timeseries and timestamp.
        """
        if timeseries is None and timeslide is None:
            raise ValueError("At least one of timeseries or timeslide must be provided.")
        
        timeslide = np.array(timeslide) if timeslide is not None else self._slide_window(np.array(timeseries))
        if timeslide.shape[0] == self.window_size:
            timeslide = timeslide[None, ...]
        timeslide = self.transform(timeslide)

        outputs = {}
        outputs[DatasetFeature.TIMESLIDE.value] = timeslide

        return BatchFeature(
            data=outputs,
            tensor_type=return_tensors,
        )
    
    def to_dict(self):
        output = super().to_dict()
        scaler = output.pop("scaler", None)
        if scaler is not None:
            for key, value in scaler.__dict__.items():
                if isinstance(value, np.ndarray):
                    output[key] = value.tolist()
                elif isinstance(value, (list, int, float, str, bool, type(None))):
                    output[key] = value
        return output
    
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
    
    def get_point_scores(self, window_scores, window_size, stride, padding_length):
        num_windows = len(window_scores)
        begins = np.arange(num_windows) * stride
        ends = begins + window_size
        unwindowed_length = stride * (num_windows - 1) + window_size + padding_length

        cum = np.zeros(unwindowed_length)
        cnt = np.zeros(unwindowed_length)
        for b, e, s in zip(begins, ends, window_scores):
            cum[b:e] += s
            cnt[b:e] += 1

        cnt[cnt == 0] = 1
        mapped = cum / cnt
        return mapped
    
    def decode(self, window_scores, padding_length=None):
        padding_length = padding_length if padding_length is not None else self.padding_length
        point_scores = self.get_point_scores(window_scores, self.window_size, self.stride, padding_length)
        return point_scores
    
    def _slide_window(self, series: np.ndarray):

        flat_shape = (series.shape[0] - (self.window_size - 1), -1)  # in case we have a multivariate TS
        slides = sliding_window_view(series, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        self.padding_length = series.shape[0] - (slides.shape[0] * self.stride + self.window_size - self.stride)

        return slides

    
    def prepare_dataset(self, data: Dataset, **kwargs) -> Dataset:

        def slide_window_for_batch(batch):
            timeseries = np.array(batch[DatasetFeature.TIMESERIES.value])[:, None]
            timestamp = np.array(batch[DatasetFeature.TIMESTAMP.value])[:, None]
            timeslides = self._slide_window(timeseries)
            timestamp = self._slide_window(timestamp)
            return {
                DatasetFeature.TIMESTAMP.value: timestamp.tolist(),
                DatasetFeature.TIMESLIDE.value: timeslides.tolist(),
            }

            
        data = data.map(
            slide_window_for_batch,
            batched=True,
            batch_size=None,
            load_from_cache_file=False,
            remove_columns=data.column_names,
        )
        return data

