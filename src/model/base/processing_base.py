from abc import ABC, abstractmethod
from transformers import FeatureExtractionMixin

class BaseProcessor(FeatureExtractionMixin, ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("The fit method must be implemented by the processor.")
    
    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError("The transform method must be implemented by the processor.")
    
    @abstractmethod
    def inverse_transform(self, *args, **kwargs):
        raise NotImplementedError("The inverse_transform method must be implemented by the processor.")
    
    @abstractmethod
    def prepare_dataset(self, *args, **kwargs):
        """
        The raw dataset is like 'date, value' in 'n_samples, -1'
        This method should handle it to the dataset can be feed to model like 'n_sampel, window_size, -1'
        """
        raise NotImplementedError("The prepare_dataset method must be implemented by the processor.")
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        return the model inputs
        """
        raise NotImplementedError("The __call__ method must be implemented by the processor.")
    
    @abstractmethod
    def decode(self, *args, **kwargs):
        """
        Decode the model outputs to scores and labels
        """
        raise NotImplementedError("The decode method must be implemented by the processor.")
    