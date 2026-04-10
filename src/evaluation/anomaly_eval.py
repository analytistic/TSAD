from ..model import MODEL_REGISTRY, CONFIG_REGISTRY, PROCESSOR_REGISTRY
from transformers import PreTrainedModel, PreTrainedConfig
from transformers import set_seed
from pathlib import Path
from datasets import load_dataset
import datetime
import numpy as np
from ..model.base.processing_base import BaseProcessor
from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments
import torch

class AnomalyEvaluation:
    def __init__(
            self,
            data_args: DataArguments,
            model_args: ModelArguments,
            train_args: TrainingArguments,
    ):
        pretrained_model_name_or_path = model_args.pretrained_model_name_or_path

        if Path(pretrained_model_name_or_path).exists():
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        else:
            model_name = pretrained_model_name_or_path
            config_cls: PreTrainedConfig = CONFIG_REGISTRY[model_name]
            model_cls: PreTrainedModel = MODEL_REGISTRY[model_name]
            processor_cls: BaseProcessor = PROCESSOR_REGISTRY[model_name]
            config = config_cls.from_dict(model_args.config)
            self.model = model_cls(config)
            self.processor = processor_cls.from_dict(model_args.processor_config)

        self.random_seed = train_args.seed
        set_seed(self.random_seed)

        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args

    def prepare_data(self, path):

        path = Path(path)
        datas = load_dataset(f'{path.suffix[1:]}', data_files=str(path), split="train")
        features = datas.features

        data_column = None
        label_column = None
        timestamp_column = None

        if len(datas) == 0:
            print(f"Empty dataset for file: {path}")
            return [], [], [], []

        for feat, val in features.items():
            if val.dtype in ['float32', 'float64']:
                data_column = feat
            elif val.dtype in ['int32', 'int64']:
                label_column = feat
            elif val.dtype == 'string':
                timestamp_column = feat


        cols = list(features.keys())
        if data_column is None and len(cols) > 0:
            data_column = cols[0]
        if label_column is None and len(cols) > 1:
            label_column = cols[1]

        if data_column is None or label_column is None:
            print(f"Could not determine data/label columns for file: {path}")
            return [], [], [], []

        if timestamp_column is None:
            col_name = 'date'
            datas = datas.add_column(col_name, list(range(len(datas))))
            timestamp_column = col_name

        if len(datas) == 0:
            print(f"Empty dataset for file: {path}")
            return [], [], [], []

        file_name_parts = path.name.split('_')
        train_end_part = [file_name_parts[i + 1] for i, part in enumerate(file_name_parts) if part == 'tr']        

        if train_end_part and len(train_end_part[0]) > 1:
            train_end = int(train_end_part[0])
            train_dataset = datas.select(range(0, train_end))
            test_dataset = datas.select(range(train_end, len(datas)))
            train_data = train_dataset.select_columns([timestamp_column, data_column])
            train_labels = train_dataset.select_columns([timestamp_column, label_column])
            test_data = test_dataset.select_columns([timestamp_column, data_column])
            test_labels = test_dataset.select_columns([timestamp_column, label_column])

        else:
            print(f"Invalid file format or missing 'tr_' in: {path.name}")
            return [], [], [], []
        
        return train_data, train_labels, test_data, test_labels
    
    def evaluate(self, train_data, train_labels, test_data, test_labels):
        # self.processor.fit(
        #     np.array(train_data[train_data.column_names[1]])
        # )
        # train_transformed_data = self.processor.transform(
        #     np.array(train_data[train_data.column_names[1]])
        # )

        # test_transformed_data = self.processor.transform(
        #     np.array(test_data[test_data.column_names[1]])
        # )
        # train_col = train_data.column_names[1]
        # test_col = test_data.column_names[1]

        # train_data = train_data.remove_columns(train_col)
        # test_data = test_data.remove_columns(test_col)
        # train_data = train_data.add_column(train_col, train_transformed_data)
        # test_data = test_data.add_column(test_col, test_transformed_data)

        """
            unsupervised fit at both train and test data
        """

        self.model.fit(train_data, test_data, self.train_args, self.processor)
        self._in_loop(test_data, test_labels)

    def _in_loop(self, test_data, test_labels):
        # in loop evaluation 
        with torch.no_grad():
            
      
        pass




            



    def evaluate_loop(self):
        # Implement your evaluation logic here
        # For example, you can calculate precision, recall, F1-score, etc.
        pass

