from transformers import PreTrainedConfig

class RQTADConfig(PreTrainedConfig):
    model_type = "rqtad"

    def __init__(
        self,
        window_size=100,
        stride=50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride