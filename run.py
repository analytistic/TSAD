from src.evaluation.anomaly_eval import AnomalyEvaluation
import argparse
if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="config/RQTAD.toml", help="Path to the configuration file")
    args = parser.parse_args()

    from src.utils.arguments import DataArguments, ModelArguments, TrainingArguments
    evaler = AnomalyEvaluation(
        data_args=DataArguments.from_toml(args.config_path),
        model_args=ModelArguments.from_toml(args.config_path),
        train_args=TrainingArguments.from_toml(args.config_path)
    )
    evaler()
    