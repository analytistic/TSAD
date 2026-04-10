from abc import ABC, abstractmethod
from transformers import FeatureExtractionMixin

class BaseProcessor(FeatureExtractionMixin):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError("The transform method must be implemented by the processor.")
    
    @abstractmethod
    def inverse_transform(self, *args, **kwargs):
        raise NotImplementedError("The inverse_transform method must be implemented by the processor.")