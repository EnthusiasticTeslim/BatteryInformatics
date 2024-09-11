
import os, yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from molecules import canonical_smiles

def read_and_process_data(file_path, canonicalize=True, columns = ['smiles', 'redox_potential']):
    """Read and process the original data file."""
    with open(file_path, 'r') as f:
        next(f)  # Skip header
        data = [line.strip().split(",") for line in f]
    
    df = pd.DataFrame(data, columns=columns)
    df[columns[1]] = df[columns[1]].astype(float)
    
    if canonicalize:
        df[columns[0]] = df[columns[0]].apply(canonical_smiles)
    
    df.columns = ['smiles', 'label']
    return df

def save_data(df, path, filename):
    """Save DataFrame to CSV."""
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved data to {full_path}")



def scale(X_train_orig, X_test_orig):
    """Split into training and test sets and scale."""

    scaler = StandardScaler()

    scaler.fit(X_train_orig)

    X_train = scaler.transform(X_train_orig)
    X_test = scaler.transform(X_test_orig)

    return X_train, X_test



def hyperparameter_optimization(
                                model, 
                                param_space, 
                                X_train, y_train, 
                                cv=5, n_iter=50, random_state=42
                                ):
    """Perform hyperparameter optimization using Bayesian optimization."""

    # Create the BayesSearchCV object
    opt = BayesSearchCV(
        estimator=model, # The model
        search_spaces=param_space, # The hyperparameter space
        n_iter=n_iter,  # Number of parameter settings that are sampled
        cv=cv, # Number of folds in cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=0, # Controls the verbosity: the higher, the more messages
        random_state=random_state # Random seed
    )
        
    # Perform optimization
    opt.fit(X_train, y_train)

    return opt


def load_hyperparameter_space(yaml_file):
    """Load hyperparameter space from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    space = {}
    for model, params in config.items():
        space[model] = {}
        for param, settings in params.items():
            if settings['type'] == 'Real':
                space[model][param] = Real(settings['low'], settings['high'], prior=settings['prior'])
            elif settings['type'] == 'Integer':
                space[model][param] = Integer(settings['low'], settings['high'])
            elif settings['type'] == 'Categorical':
                space[model][param] = Categorical(settings['choices'])
    
    return space


def plot_prediction_scatter(
        y_true_train, y_pred_train, y_true_test, y_pred_test,                     
        figsize=(4, 3), xlabel=r"$\rm Target$", ylabel=r"$\rm Prediction$", title=None, save_path=None):
    """
    Create a scatter plot of predicted vs true values for train and test sets.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot actual vs predicted
    ax.scatter(y_true_train, y_pred_train, label=r"$\rm Train$", c="red") # Train
    ax.scatter(y_true_test, y_pred_test, label=r"$\rm Test$", c="blue") # Test
    # plot best fit line
    all_values = np.concatenate([y_true_train, y_true_test, y_pred_train, y_pred_test])
    min_val, max_val = np.min(all_values), np.max(all_values)
    ax.plot([min_val-1, max_val+1], [min_val-1, max_val+1], "--", color="gray") # low the bounds
    # set ticks
    ax.set_xticks(np.arange(min_val, max_val, 1))
    ax.set_yticks(np.arange(min_val, max_val, 1))
    # set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    # save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return None

