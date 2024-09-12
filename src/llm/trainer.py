import argparse
from datetime import datetime
from typing import Tuple, Dict
from loguru import logger

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed, EvalPrediction
)

def load_and_validate_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the train and test datasets."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    required_columns = {'smiles', 'label'}
    for df, name in [(train_data, 'train'), (test_data, 'test')]:
        if not required_columns.issubset(df.columns):
            raise ValueError(f"The {name} data does not have the required columns: {required_columns}")

    return train_data, test_data

def prepare_datasets(train_data: pd.DataFrame, test_data: pd.DataFrame, train_val_split: float, seed: int) -> DatasetDict:
    """Prepare and split the datasets."""
    train_val_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_data, preserve_index=False)

    split_train_data = train_val_dataset.train_test_split(test_size=train_val_split, seed=seed)

    return DatasetDict({
        'train': split_train_data['train'],
        'valid': split_train_data['test'],
        'test': test_dataset
    })

def tokenize_function(examples: Dict, tokenizer) -> Dict:
    """Tokenize the SMILES strings."""
    return tokenizer(examples["smiles"], padding=True, truncation=True, max_length=None)

def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """Compute evaluation metrics."""
    preds, labels = eval_pred
    return {
        "mse": mean_squared_error(labels, preds),
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds)
    }

def main(args: argparse.Namespace):

    # Set reproducibility
    logger.info("Setting random seed for reproducibility...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    # Load and prepare data
    logger.info("Loading and validating data...")
    train_data, test_data = load_and_validate_data(
                train_path = f"{args.main_directory}/data/{args.train_data}", 
                test_path = f"{args.main_directory}/data/{args.test_data}")
    datasets = prepare_datasets(train_data=train_data, test_data=test_data, train_val_split=args.train_val_split, seed=args.seed)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Tokenize datasets
    tokenized_datasets = datasets.map(lambda examples: tokenize_function(examples, tokenizer), batched=True).remove_columns(["smiles"])
    train_dataset, valid_dataset, test_dataset = tokenized_datasets["train"], tokenized_datasets["valid"], tokenized_datasets['test']
    # save tokenized_datasets
    train_dataset.save_to_disk(f"{args.main_directory}/{args.result_dir}/llm_{args.identifier}/train_dataset")
    valid_dataset.save_to_disk(f"{args.main_directory}/{args.result_dir}/llm_{args.identifier}/valid_dataset")
    test_dataset.save_to_disk(f"{args.main_directory}/{args.result_dir}/llm_{args.identifier}/test_dataset")

    # Prepare training arguments
    logger.info("Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=f"{args.main_directory}/{args.result_dir}/llm_{args.identifier}",
        num_train_epochs=args.steps,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        save_strategy="no",
        seed=args.seed,
        do_eval=True,
        do_train=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        optim='adamw_torch'
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    
    # Train model
    logger.info("Training model...")
    trainer.train()

    # Save the model
    trainer.save_model(f"{args.main_directory}/{args.model_dir}/llm_{args.identifier}")
    logger.success(f"Model saved to {args.main_directory}/{args.model_dir}/llm_{args.identifier}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on SMILES data")
    parser.add_argument("--train_data", type=str, default='train_data_cleaned.csv', help="Path to the train data CSV")
    parser.add_argument("--test_data", type=str, default='test_data_cleaned.csv', help="Path to the test data CSV")
    parser.add_argument("--train_val_split", type=float, default=0.2, help="Fraction of training data to use for validation")
    parser.add_argument("--steps", type=int, default=10, help="number of steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_name", type=str, default="seyonec/PubChem10M_SMILES_BPE_120k", help="Name of the pre-trained model to use")
    parser.add_argument("--main_directory", type=str, default='/Users/gbemidebe/Documents/GitHub/BatteryInformatics', help="Main directory")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="Directory to save the trained model")
    parser.add_argument("--identifier", type=str, default='20240828_21', help="Identifier for the trained model")

    args = parser.parse_args()
    main(args)