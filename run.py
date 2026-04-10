from src.evaluation.anomaly_eval import AnomalyEvaluation
if __name__ == "__main__":
    # Example usage
    config_path = "src/config/KMeansAD.toml"
    from src.utils.arguments import DataArguments, ModelArguments, TrainingArguments
    evaluation = AnomalyEvaluation(
        data_args=DataArguments.from_toml(config_path),
        model_args=ModelArguments.from_toml(config_path),
        train_args=TrainingArguments.from_toml(config_path)
    )
    train_data, train_labels, test_data, test_labels = evaluation.prepare_data(evaluation.data_args.data_path)
    evaluation.evaluate(train_data, train_labels, test_data, test_labels)