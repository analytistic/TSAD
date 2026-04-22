

MODEL_REGISTRY = {}
CONFIG_REGISTRY = {}
PROCESSOR_REGISTRY = {}

def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

def register_config(name):
    def register_config_cls(cls):
        if name in CONFIG_REGISTRY:
            raise ValueError(f"Cannot register duplicate config ({name})")
        CONFIG_REGISTRY[name] = cls
        return cls
    return register_config_cls

def register_processor(name):
    def register_processor_cls(cls):
        if name in PROCESSOR_REGISTRY:
            raise ValueError(f"Cannot register duplicate processor ({name})")
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return register_processor_cls

from .KMeansAD.modeling_KMeansAD import KMeansAD
from .KMeansAD.processing_KMeansAD import KMeansADProcessor
from .KMeansAD.configuration_KMeansAD import KMeansADConfig
from .RQTAD.configuration_RQTAD import RQTADConfig
from .RQTAD.modeling_RQTAD import RQTAD
from .RQTAD.processing_RQTAD import RQTADProcessor


__all__ = [
           "KMeansAD", 
           "KMeansADProcessor", 
           "KMeansADConfig", 
           "RQTAD", 
           "RQTADProcessor", 
           "RQTADConfig", 
           "MODEL_REGISTRY", 
           "CONFIG_REGISTRY"
        ]