import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datasets import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from trainer import compute_metrics as transformer_compute_metrics

def compute_metrics(actual: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    return {
        "mse": mean_squared_error(actual, preds),
        "mae": mean_absolute_error(actual, preds),
        "r2": r2_score(actual, preds)
    }

def load_trained_model(model_path: str) -> AutoModelForSequenceClassification:
    """Load the trained model."""
    return AutoModelForSequenceClassification.from_pretrained(model_path)

def load_dataset(path: str) -> Dataset:
    """Load a dataset from disk."""
    return Dataset.load_from_disk(path)

def plot_results(datasets, output_path: str):
    """Plot the results for train, validation, and test datasets."""
    fig, ax = plt.subplots()

    for data, label, color in datasets:
        r2 = r2_score(data.label_ids, data.predictions)
        mse = mean_squared_error(data.label_ids, data.predictions)
        ax.scatter(data.predictions, data.label_ids, 
                   label=f'{label}', #: $R^2 = {r2:.2f}, MSE = {mse:.2f}$',
                   alpha=0.6, color=color)

    # use the min and max values of the data to set the axis limits
    ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c='black', alpha=0.5)
    ax.legend(loc='upper left')
    ax.set_xlabel('ML output (mV)')
    ax.set_ylabel('DFT output (mV)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):

    # Set paths
    model_path = Path(args.main_directory) / args.model_dir / f"llm_{args.identifier}"
    results_path = Path(args.main_directory) / args.result_dir / f"llm_{args.identifier}"

    # Load the trained model
    model = load_trained_model(str(model_path))
    # Load the tokenize dataset
    train_data = load_dataset(str(results_path / "train_dataset"))
    valid_data = load_dataset(str(results_path / "valid_dataset"))
    test_data = load_dataset(str(results_path / "test_dataset"))
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=str(results_path),
        num_train_epochs=15,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        save_strategy="no",
        seed=args.seed,
        do_eval=True,
        do_train=True,
        load_best_model_at_end=False,
        save_total_limit=2,
        optim='adamw_torch'
    )
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=transformer_compute_metrics,
        train_dataset=train_data,
        eval_dataset=valid_data
    )
    # Predict on train, validation, and test datasets
    predictions = {
        'train': trainer.predict(train_data),
        'valid': trainer.predict(valid_data),
        'test': trainer.predict(test_data)
    }
    # Compute metrics
    results = {split: compute_metrics(pred.label_ids, pred.predictions) 
               for split, pred in predictions.items()}
    # Log and save results
    for split, metrics in results.items():
        logger.info(f"{split.capitalize()}: {metrics}")

    pd.DataFrame(results).T.to_csv(results_path / "results_metrics.csv")

    # Plot results
    plot_results([
        (predictions['train'], 'Train', 'blue'),
        (predictions['valid'], 'Validation', 'green'),
        (predictions['test'], 'Test', 'red')
    ], str(results_path / "results_plot.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on SMILES data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--main_directory", type=str, default='/Users/gbemidebe/Documents/GitHub/BatteryInformatics', help="Main directory")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="Directory to save the trained model")
    parser.add_argument("--identifier", type=str, default='20240828_21', help="Identifier for the trained model")

    args = parser.parse_args()
    main(args)