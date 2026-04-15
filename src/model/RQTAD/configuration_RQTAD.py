from transformers import PreTrainedConfig
from .. import register_config

@register_config("RQTAD")
class RQTADConfig(PreTrainedConfig):
    def __init__(self, 
                 k=8, 
                 window_size=10, 
                 stride=1, 
                 n_jobs=1, 
                 normalize=True, 
                 n_iter=300,
                 tol=1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.n_jobs = n_jobs
        self.normalize = normalize
        self.n_iter = n_iter
        self.tol = tol