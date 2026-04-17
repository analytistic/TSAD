from transformers import PreTrainedConfig
from .. import register_config

@register_config("RQTAD")
class RQTADConfig(PreTrainedConfig):
    def __init__(self, 
                 k_list=[40, 20, 10], 
                 window_size=10, 
                 stride=1, 
                 n_jobs=1, 
                 normalize=True, 
                 n_iter=300,
                 tol=1e-4,
                 codebook_num=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.k_list = k_list
        self.window_size = window_size
        self.stride = stride
        self.n_jobs = n_jobs
        self.normalize = normalize
        self.n_iter = n_iter
        self.tol = tol
        self.codebook_num = codebook_num