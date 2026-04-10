from transformers import FeatureExtractionMixin
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from ..utils.scaler import ScalerType, StandaryScaler, MinMaxScaler, BaseScaler



class RQTADProcessor(FeatureExtractionMixin):
    def __init__(self, 
                 scale: bool,
                 scaler_type: Optional[ScalerType] = None,
                 **kwargs
                 ):
        super().__init__(scale=scale, scaler_type=scaler_type, **kwargs)
        
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
        else:
            raise ValueError("No scaler defined to fit the data.")
    
    def transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(inputs)
        else:
            raise ValueError("No scaler defined to transform the data.")
        
    def inverse_transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.inverse_transform(inputs)
        else:
            raise ValueError("No scaler defined to inverse transform the data.")

    
    def __call__(self, 
                 timeseries: np.ndarray | List, 
                 timestamp: Optional[np.ndarray] = None,
                 ex_features: np.ndarray | List | None = None,
                 labels: np.ndarray | List | None = None,
                 scale: Optional[bool] = None, 
                 return_tensors: str = 'pt', 
                 **kwargs) -> BatchFeature:
        outputs = {}
        scale = scale if scale is not None else self.scale

        if timestamp is not None:
            timestamp = np.array(timestamp.astype('int64'), dtype=np.int64)
        else:
            timeseries = np.arange(len(timeseries)) if isinstance(timeseries, list) else np.arange(timeseries.shape[0])


        labels = np.array(labels) if labels is not None else None
        if self.scaler is not None and self.scale:
            timeseries = self.scaler.transform(timeseries) if scale else timeseries

        outputs["timeseries"] = timeseries
        outputs["timestamp"] = timestamp
        outputs["labels"] = labels

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