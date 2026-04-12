from transformers import PreTrainedModel
from abc import abstractmethod, ABC

class BaseTASDModel(PreTrainedModel, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def partial_fit(self, *args, **kwargs):
        raise NotImplementedError
    

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError