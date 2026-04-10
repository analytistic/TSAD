

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import transformers
import toml


@dataclass
class DataArguments:
    datasets: str = field(
        default="",
        metadata={
            "help": "data name "
        },
    )
    data_path: str = field(
        default="",
        metadata={
            "help": "data path"
        },
    )
    data_dir: str = field(
        default="",
        metadata={
            "help": "data dir"
        },
    )

    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                toml_args = data.get('data_args', {})
            args = cls()
            for arg in toml_args.keys():
                setattr(args, arg, toml_args[arg])
            return args
        else:
            return cls()
    

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        default="",
        metadata={
            "help": "model name or path"
        },
    )
    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether to use cache"
        },
    )
    config: Dict = field(
        default_factory=dict,
        metadata={
            "help": "Additional configuration for the model."
        },
    )
    processor_config: Dict = field(
        default_factory=dict,
        metadata={
            "help": "Additional configuration for the processor."
        },
    )

    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                toml_args = data.get('model_args', {})
            args = cls()
            for arg in toml_args.keys():
                setattr(args, arg, toml_args[arg])
            return args
        else:
            return cls()
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=False, metadata={"help": "Whether to use mixed precision (bf16) training."})
    fp16: bool = field(default=False, metadata={"help": "Whether to use mixed precision (fp16) training."})
    fp32: bool = field(default=False, metadata={"help": "Whether to use mixed precision (fp32) training."})

    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                toml_args = data.get('training_args', {})
            return cls(**toml_args)
        else:
            return cls()
    
