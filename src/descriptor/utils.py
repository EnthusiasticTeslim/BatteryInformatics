import os, yaml
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdCoordGen, Descriptors

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

def setup_environment(args):
    """Setup the environment"""
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set the result directory
    if args.docker:
       dir = f"{args.parent_directory}"
    else:
        dir = f"{args.parent_directory}/{args.result_directory}"
        
    if not args.morgan_fingerprint:
        result_dir = f"{dir}/{args.model}/RDKit"
    else:
        result_dir = f"{dir}/{args.model}/MFF"
    
    # create directory
    if os.path.exists(result_dir):
        if input(f"Directory {result_dir} exists. Overwrite? (y/n)") == 'y':
            os.system(f"rm -r {result_dir}")
        else:
            return
        os.makedirs(result_dir)
    else:
        os.makedirs(result_dir)
    return result_dir

def molecules_from_smiles(smiles):
    '''
    Generate RDKit molecules from SMILES strings
    '''
    molecules = []
    for smilei in tqdm(smiles):
        mol = Chem.MolFromSmiles(smilei)
        rdCoordGen.AddCoords(mol)
        molecules.append(mol)
    return molecules


def canonical_smiles(smiles):
    '''
    Generate canonical SMILES strings
    '''
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))



def compute_rdkit_descriptors(molecules, drop_na=False):
    """
    Compute RDKit descriptors for a list of molecules, dropping NaNs by default and returning a DataFrame.
    """
    # Get the list of descriptor names and functions
    desc_list = Descriptors.descList
    
    # Initialize dictionary to store results
    results = {name: [] for name, _ in desc_list}
    
    # Iterate over molecules with a progress bar
    for mol in tqdm(molecules, desc="Computing descriptors"):
        for name, func in desc_list:
            try:
                value = func(mol)
            except:
                value = np.nan
            results[name].append(value)
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Drop NaNs if requested
    if drop_na:
        df = df.dropna(axis=1)
    
    return df


def compute_MFF_descriptors(molecules, nbits=128, radius=2, useChirality=True,):
    """
    Compute Morgan fingerprints for a list of molecules and return a DataFrame.
    """
    # Initialize dictionary to store results
    results = {f"mfp{i}": [] for i in range(nbits)}
    
    # Iterate over molecules with a progress bar
    for mol in tqdm(molecules, desc="Computing MFF descriptors"):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, useChirality=useChirality, nBits=nbits)
        for i in range(nbits):
            results[f"mfp{i}"].append(fp[i])
        
    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    return df

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

def evaluate_model(result_dir, results, identifier='noCV'):

    save = []
    for dataset, y_true, y_pred in results:
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        save.append((dataset, y_true, y_pred, rmse, mae, r2))
    
    # should be form of [('Train', (rmse, r2), ('Test', (rmse, r2)), ('Val', (rmse, r2))]
    if len(save) == 2:
        tqdm.write(f"{identifier} ---> Train RMSE: {save[0][3]:.2f}, Test RMSE: {save[1][3]:.2f}, Train R2: {save[0][5]:.2f}, Test R2: {save[1][5]:.2f}")
    else:
        tqdm.write(f"{identifier} ---> Train RMSE: {save[0][3]:.2f}, Val RMSE: {save[1][3]:.2f}, Test RMSE: {save[2][3]:.2f}, Train R2: {save[0][5]:.2f}, Val R2: {save[1][5]:.2f}, Test R2: {save[2][5]:.2f}")

    # metrics to file
    save_metrics=[]
    with open(f"{result_dir}/performance_metrics_{identifier}.txt", 'w') as f:
        for dataset, y_true, y_pred in results:
            rmse = root_mean_squared_error(y_true, y_pred) 
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            save_metrics.append((dataset, rmse, mae, r2))
        if len(save) == 2:
            f.write(f"{identifier} ---> Train RMSE: {save[0][3]:.2f}, Test RMSE: {save[1][3]:.2f}, Train R2: {save[0][5]:.2f}, Test R2: {save[1][5]:.2f}")
        else:
            f.write(f"{identifier} ---> Train RMSE: {save[0][3]:.2f}, Val RMSE: {save[1][3]:.2f}, Test RMSE: {save[2][3]:.2f}, Train R2: {save[0][5]:.2f}, Val R2: {save[1][5]:.2f}, Test R2: {save[2][5]:.2f}")

    
    
