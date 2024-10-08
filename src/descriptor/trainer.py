# import libraries
import warnings
import pickle
import pandas as pd
import numpy as np
import argparse, os, sys
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings('ignore', category=UserWarning, append=True)
from utils import setup_environment, molecules_from_smiles, compute_rdkit_descriptors, compute_MFF_descriptors, scale, hyperparameter_optimization, load_hyperparameter_space, evaluate_model

def main(args):

    result_dir = setup_environment(args)
    # read data
    if args.docker:
        train_data = pd.read_csv(f'{args.parent_directory}/{args.train_data}')
        test_data = pd.read_csv(f'{args.parent_directory}/{args.test_data}')
    else:
        train_data = pd.read_csv(f'{args.parent_directory}/{args.data_directory}/train_data_cleaned.csv')
        test_data = pd.read_csv(f'{args.parent_directory}/{args.data_directory}/test_data_cleaned.csv')

    tqdm.write(f"Data loaded: {np.ceil(100 * len(train_data) / (len(train_data) + len(test_data)))} % train set, "
               f"{np.floor(100 * len(test_data) / (len(train_data) + len(test_data)))} % test set")

    # check if columns smiles and label are in the data
    if 'smiles' not in train_data.columns or 'label' not in train_data.columns or \
       'smiles' not in test_data.columns or 'label' not in test_data.columns:
        raise ValueError('Columns smiles and label are required in both train and test data')
    
    # make molecules
    train_molecules = molecules_from_smiles(train_data.smiles)
    test_molecules = molecules_from_smiles(test_data.smiles)

    # descriptors
    if not args.morgan_fingerprint:
        train_descriptors = compute_rdkit_descriptors(train_molecules)
        test_descriptors = compute_rdkit_descriptors(test_molecules)
    else:
        train_descriptors = compute_MFF_descriptors(train_molecules, nbits=args.nbits, radius=args.radius, useChirality=True)
        test_descriptors = compute_MFF_descriptors(test_molecules, nbits=args.nbits, radius=args.radius, useChirality=True)


    # target variable
    y_train, y_test = train_data.label, test_data.label

    # scale data
    if args.scale and not args.morgan_fingerprint:
        X_train, X_test = scale(train_descriptors, test_descriptors)
    else:
        X_train, X_test = train_descriptors.values, test_descriptors.values

    # optimize hyperparameters
    tqdm.write(f'Optimizing hyperparameters for model {args.model}')
    model = eval(f"{args.model}()")
    
    if args.docker:
        hyperparameter_file = f"{args.parent_directory}/{args.hyperparameter}"
    else:
        hyperparameter_file = f"{args.parent_directory}/regenerate/{args.hyperparameter}"
    param_space = load_hyperparameter_space(hyperparameter_file)[args.model]

    optimized_result = hyperparameter_optimization(model, param_space, X_train, y_train, cv=args.cv, n_iter=args.iterations, random_state=int(args.seed))
    # write it to a file
    with open(f"{result_dir}/best_hyperparameters.txt", 'w') as f:
        f.write("Best parameters:\n")
        for param, value in optimized_result.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Best score: {optimized_result.best_score_:.2f}\n")

    # 
    if args.skip_cv:
        tqdm.write('Skipping cross-validation')
        fname = 'noCV'
        model = eval(f"{args.model}()")
        model = model.set_params(**optimized_result.best_params_)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # evaluate model   
        evaluate_model(result_dir=result_dir, results= [('Train', y_train, y_pred_train), ('Test', y_test, y_pred_test)], identifier=fname)
        # save predictions
        pd.DataFrame({'smiles': train_data.smiles, 'target': y_train, 'prediction': y_pred_train}).to_csv(f"{result_dir}/train_predictions_{fname}.csv", index=False)
        pd.DataFrame({'smiles': test_data.smiles, 'target': y_test, 'prediction': y_pred_test}).to_csv(f"{result_dir}/test_predictions_{fname}.csv", index=False)
        # save model
        with open(f"{result_dir}/model_{fname}.pkl", 'wb') as f:
            pickle.dump(model, f)

    else:
        kf = KFold(n_splits=5, random_state=40, shuffle=True)

        for cv_index, (train_indices, valid_indices) in enumerate(kf.split(range(len(train_data)))):
            fname = f"cvid{cv_index}"
            # create model with best hyperparameters
            model = eval(f"{args.model}()")
            model = model.set_params(**optimized_result.best_params_)
            # split data
            X_train_cv = X_train[train_indices]
            y_train_cv = y_train[train_indices]
            X_valid_cv = X_train[valid_indices]
            y_valid_cv = y_train[valid_indices]
            # fit model
            model.fit(X_train_cv, y_train_cv)
            # predict
            y_pred_train = model.predict(X_train_cv)
            y_pred_valid = model.predict(X_valid_cv)
            y_pred_test = model.predict(X_test)
            # save predictions
            evaluate_model(result_dir=result_dir, results= [('Train', y_train_cv, y_pred_train), ('Val', y_valid_cv, y_pred_valid), ('Test', y_test, y_pred_test)], identifier=fname)
            # save predictions
            pd.DataFrame({'smiles': train_data.smiles[train_indices], 'target': y_train[train_indices], 'prediction': y_pred_train}).to_csv(f"{result_dir}/train_predictions_{fname}.csv", index=False)
            pd.DataFrame({'smiles': train_data.smiles[valid_indices], 'target': y_train[valid_indices], 'prediction': y_pred_valid}).to_csv(f"{result_dir}/valid_predictions_{fname}.csv", index=False)
            pd.DataFrame({'smiles': test_data.smiles, 'target': y_test, 'prediction': y_pred_test}).to_csv(f"{result_dir}/test_predictions_{fname}.csv", index=False)
            # save model
            with open(f"{result_dir}/model_{fname}.pkl", 'wb') as f:
                pickle.dump(model, f)

    # write arguments in parser to a file
    with open(f"{result_dir}/args.txt", 'w') as f:
        f.write(str(args))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance ML with descriptor')
    parser.add_argument('--parent_directory', type=str, default='/Users/gbemidebe/Documents/GitHub/BatteryInformatics', help='Path to main directory')
    parser.add_argument('--data_directory', default='data', type=str, help='where the data is stored in parent directory')
    parser.add_argument('--result_directory', default='results', type=str, help='Path to result directory')
    parser.add_argument('--src', type=str, default='src', help='function source directory')
    parser.add_argument('--train_data', type=str, default='train_data_cleaned.csv', help='Path to train data')
    parser.add_argument('--test_data', type=str, default='test_data_cleaned.csv', help='Path to test data')
    parser.add_argument('--scale', action='store_true', help='Scale data')
    parser.add_argument('--hyperparameter', type=str, default='hp_descriptor.yaml', help='Hyperparameter space')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for hyperparameter optimization')
    parser.add_argument('--skip_cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--model', type=str, default='RandomForestRegressor', help='Model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--morgan_fingerprint', action='store_true', help='Use Morgan Fingerprint (MFF) instead of RDKit descriptors')
    parser.add_argument('--nbits', type=int, default=256, help='Number of bits for MFF')
    parser.add_argument('--radius', type=int, default=2, help='Radius for MFF')
    parser.add_argument('--docker', action='store_true', help='Run in docker')

    args = parser.parse_args()
    main(args)