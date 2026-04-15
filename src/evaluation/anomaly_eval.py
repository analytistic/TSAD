from ..model import MODEL_REGISTRY, CONFIG_REGISTRY, PROCESSOR_REGISTRY
from transformers import PreTrainedModel, PreTrainedConfig
from transformers import set_seed
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import datetime
import numpy as np
from ..model.base.processing_base import BaseProcessor
from ..model.base.output_base import BaseTASDModelOutput
from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from .metrics import get_metrics, ResultMerger
from ..dataset import DatasetFeature
import json
from typing import Dict, Any
from scipy.signal import argrelextrema


class AnomalyEvaluation:
    def __init__(
            self,
            data_args: DataArguments,
            model_args: ModelArguments,
            train_args: TrainingArguments,
    ):

        self.result_merger = ResultMerger()
        self.random_seed = train_args.seed
        set_seed(self.random_seed)

        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args

    def _detect_period(self, data, rank=1)-> int:
        """
        calculate the period window of the data by acf
        """
        assert data.ndim == 1, "Only support univariate data for period detection"
        data = data[:min(2000, len(data))]

        base = 3
        auto_corr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[len(data)-1:]
        auto_corr /= auto_corr[0]
        auto_corr = auto_corr[base:]

        local_max = argrelextrema(auto_corr, np.greater)[0]

        try:
            sorted_local_max = np.argsort(auto_corr[local_max])[::-1]    # Ascending order
            max_local_max = sorted_local_max[0]     # Default
            if rank == 1: max_local_max = sorted_local_max[0]
            if rank == 2: 
                for i in sorted_local_max[1:]: 
                    if i > sorted_local_max[0]: 
                        max_local_max = i 
                        break
            if rank == 3:
                for i in sorted_local_max[1:]: 
                    if i > sorted_local_max[0]: 
                        id_tmp = i
                        break
                for i in sorted_local_max[id_tmp:]:
                    if i > sorted_local_max[id_tmp]: 
                        max_local_max = i           
                        break

            final_period = local_max[max_local_max]+base
            if final_period>300:
                return 125
            return int(final_period)
        except:
            return 125
            

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
            datas = datas.rename_columns(
                {
                    timestamp_column: DatasetFeature.TIMESTAMP.value,
                    data_column: DatasetFeature.TIMESERIES.value,
                    label_column: DatasetFeature.LABELS.value
                }
            )
            train_end = int(train_end_part[0])
            train_dataset = datas.select(range(0, train_end))
            test_dataset = datas.select(range(train_end, len(datas)))
            train_data = train_dataset.select_columns([DatasetFeature.TIMESTAMP.value, DatasetFeature.TIMESERIES.value])
            train_labels = train_dataset.select_columns([DatasetFeature.TIMESTAMP.value, DatasetFeature.LABELS.value])
            test_data = test_dataset.select_columns([DatasetFeature.TIMESTAMP.value, DatasetFeature.TIMESERIES.value])
            test_labels = test_dataset.select_columns([DatasetFeature.TIMESTAMP.value, DatasetFeature.LABELS.value])

        else:
            print(f"Invalid file format or missing 'tr_' in: {path.name}")
            return [], [], [], []
        
        return train_data, train_labels, test_data, test_labels
    
    def evaluate(self, data_path)-> Dict:
        """
            train:
                unsupervised fit at both train and test data
                semisupervised fit at train data
            eval:
                unsupervised and semisupervised eval at both train and test data
        """
        train_data, train_labels, test_data, test_labels = self.prepare_data(data_path)
        
        pretrained_model_name_or_path = self.model_args.pretrained_model_name_or_path
        # TODO: support use auto to load from both local and config
        if Path(pretrained_model_name_or_path).exists():
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
            self.processor: BaseProcessor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        else:
            model_name = pretrained_model_name_or_path
            config_cls: PreTrainedConfig = CONFIG_REGISTRY[model_name]
            model_cls: PreTrainedModel = MODEL_REGISTRY[model_name]
            processor_cls: BaseProcessor = PROCESSOR_REGISTRY[model_name]
            config = config_cls.from_dict(self.model_args.config)
            self.model = model_cls(config)
            self.processor = processor_cls.from_dict(self.model_args.processor_config) # type: ignore

        self.model.fit(train_data, test_data, self.train_args, self.processor)
        all_data = concatenate_datasets([train_data, test_data])
        period = self._detect_period(np.array(all_data[DatasetFeature.TIMESERIES.value]))
        all_labels = concatenate_datasets([train_labels, test_labels])
        sorce = self._in_loop(all_data)
        evaluation_result: Dict = get_metrics(sorce, np.array(all_labels[DatasetFeature.LABELS.value]), period)
        return evaluation_result

    def _in_loop(self, all_data):
        # in loop evaluation 
        with torch.no_grad():
            self.model.eval()
            all_data = self.processor.prepare_dataset(all_data)
            results = []
            for _, inputs in enumerate(all_data):
                outputs: BaseTASDModelOutput = self.model(**self.processor(**inputs))
                results.append(outputs.sorce.cpu().numpy())
        sorce = self.processor.decode(np.array(results))
        return sorce

    def evaluate_loop(self, save_dir):
        # check file and dir
        if self.data_args.data_dir is None and self.data_args.data_file is None:
            raise ValueError("Need either a data_dir or a data_file.")
        data_path_list = sorted([
            p for p in Path(self.data_args.data_dir).rglob("*") if p.is_file()
        ]) if self.data_args.data_dir is not None else [Path(self.data_args.data_file)]

        save_dir = Path(save_dir) if save_dir is not None else Path(self.train_args.output_dir if self.train_args.output_dir is not None else "./results")
        save_file = save_dir / f"evaluation_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        save_dir.mkdir(parents=True, exist_ok=True)
        results_tosave = []
        for idx, data_path in enumerate(data_path_list):
            print(f"{'='*5} Evaluating file {idx+1}/{len(data_path_list)}: {data_path.name} {'='*5}")
            evaluation_result = self.evaluate(data_path)
            evaluation_result['file_name'] = data_path.name
            print(json.dumps(evaluation_result, indent=4, ensure_ascii=False))
            results_tosave = self.result_merger(evaluation_result)

            with open(save_file, 'w') as f:
                json.dump(sum(results_tosave, []) , f, indent=4)

        return sum(results_tosave, [])



    def __call__(self, save_path=None):
        results = self.evaluate_loop(save_path)
        return results

