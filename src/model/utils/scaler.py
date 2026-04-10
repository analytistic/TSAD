from enum import Enum
from transformers.utils import logging
from typing import List
import numpy as np

logger = logging.get_logger(__name__)

class ScalerType(str, Enum):
    STANDARY = 'standary'
    MINMAX = 'minmax'



class BaseScaler:
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def transform(self, inputs):
        raise NotImplementedError
    
    def fit(self, inputs):
        raise NotImplementedError

    def inverse_transform(self, inputs):
        raise NotImplementedError
    
class StandaryScaler(BaseScaler):
    def __init__(self, mean: List | np.ndarray | None = None, std: List | np.ndarray | None = None):
        if mean is not None and std is not None:
            super().__init__(mean=np.ndarray(mean), std=np.ndarray(std))
        else:
            super().__init__()

    def transform(self, inputs: np.ndarray):
        assert inputs.ndim == 2
        return (inputs - self.mean) / self.std
    
    def fit(self, inputs: np.ndarray):
        assert inputs.ndim == 2
        self.mean = inputs.mean(axis=0)
        self.std = inputs.std(axis=0)

    def inverse_transform(self, inputs: np.ndarray):
        assert inputs.ndim == 2
        return inputs * self.std + self.mean
    
class MinMaxScaler(BaseScaler):
    def __init__(self, max: List | np.ndarray | None = None, min: List | np.ndarray | None = None, feature_range=(-1,1)):
        if max is not None and min is not None:
            super().__init__(max=np.ndarray(max), min=np.ndarray(min), feature_range=feature_range)
        else:
            super().__init__(feature_range=feature_range)

    def transform(self, inputs: np.ndarray):
        assert inputs.ndim == 2
        return (self.feature_range[1] - self.feature_range[0]) * ((inputs - self.min) / np.abs((self.max - self.min) + (np.finfo(float).eps if np.any(self.max - self.min == 0) else 0))) + self.feature_range[0]
    
    def fit(self, inputs: np.ndarray):
        assert inputs.ndim == 2
        self._reset()
        self.min = inputs.min(axis=0)
        self.max = inputs.max(axis=0)

    def _reset(self):
        if hasattr(self, "min"):
            del self.min
        if hasattr(self, "max"):
            del self.max

    def inverse_transform(self, inputs: np.ndarray):
        assert inputs.ndim == 2
        inputs = (inputs - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return inputs * np.abs((self.max - self.min) + (np.finfo(float).eps if np.any(self.max - self.min == 0) else 0)) + self.min
    

        

         
    