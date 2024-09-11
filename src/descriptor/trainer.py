# import libraries
import warnings
import pickle
import pandas as pd
import argparse, os, sys
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings('ignore', category=UserWarning, append=True)

def main(args):

    # create save directory
    if args.save_result and not args.docker:
        result_dir = f"{args.parent_directory}/results/{args.model}"
        if os.path.exists(result_dir):
            if input(f"Directory {result_dir} exists. Overwrite? (y/n)") == 'y':
                os.system(f"rm -r {result_dir}")
            else:
                return
        os.makedirs(result_dir)

    result_path = f"{args.parent_directory}/results/{args.model}/{args.model}" if not args.docker else f"{args.parent_directory}/{args.model}"
    
    # add ../ to the path
    if not args.docker:
        sys.path.append('../')

    from molecules import molecules_from_smiles, compute_rdkit_descriptors
    from utils import scale, hyperparameter_optimization, load_hyperparameter_space, plot_prediction_scatter

    # read data
    tqdm.write('Data loaded')
    data_dir = f'{args.parent_directory}/data/' if not args.docker else f'{args.parent_directory}/'
    train_data = pd.read_csv(f'{data_dir}{args.train_data}')
    test_data = pd.read_csv(f'{data_dir}{args.test_data}')
    # print % of data
    tqdm.write(f'% Train data: {100*len(train_data)/(len(train_data)+len(test_data)):.2f}')
    tqdm.write(f'% Test data: {100*len(test_data)/(len(train_data)+len(test_data)):.2f}')

    # check if columns smiles and label are in the data
    if 'smiles' not in train_data.columns or 'label' not in train_data.columns or \
       'smiles' not in test_data.columns or 'label' not in test_data.columns:
        raise ValueError('Columns smiles and label are required in both train and test data')
    
    # make molecules
    train_molecules = molecules_from_smiles(train_data.smiles)
    test_molecules = molecules_from_smiles(test_data.smiles)

    # descriptors
    train_rdkit_descriptors = compute_rdkit_descriptors(train_molecules)
    test_rdkit_descriptors = compute_rdkit_descriptors(test_molecules)

    # target variable
    y_train, y_test = train_data.label, test_data.label

    # scale data
    if args.scale:
        X_train, X_test = scale(train_rdkit_descriptors, test_rdkit_descriptors)

    # optimize hyperparameters
    tqdm.write(f'Optimizing hyperparameters for model {args.model}')
    model = eval(f"{args.model}()")
    
    hyperparameter_file = f"{args.parent_directory}/src/descriptor/{args.hyperparameter}" if not args.docker else f"{args.parent_directory}/{args.hyperparameter}"
    param_space = load_hyperparameter_space(hyperparameter_file)[args.model]

    optimized_result = hyperparameter_optimization(model, param_space, X_train, y_train, cv=args.cv, n_iter=args.iterations, random_state=int(args.seed))
    # write it to a file
    with open(f"{result_path}_best_hyperparameters.txt", 'w') as f:
        f.write("Best parameters:\n")
        for param, value in optimized_result.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Best score: {optimized_result.best_score_:.2f}\n")

    # evaluate model
    best_model = optimized_result.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    tqdm.write(f"Model performance on test set: {best_model.score(X_test, y_test):.2f}")
    
    for dataset, y_true, y_pred in [('Train', y_train, y_pred_train), ('Test', y_test, y_pred_test)]:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        tqdm.write(f'{dataset} set')
        tqdm.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    if args.save_result:
        # metrics to file
        with open(f"{result_path}_performance_metrics.txt", 'w') as f:
            f.write(f"Model performance on test set: {best_model.score(X_test, y_test):.2f}\n")
            for dataset, y_true, y_pred in [('Train', y_train, y_pred_train), ('Test', y_test, y_pred_test)]:
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                f.write(f'{dataset} set\n')
                f.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}\n")
        # save plot
        plot_prediction_scatter(y_true_train=y_train, y_pred_train=y_pred_train, 
                                y_true_test=y_test, y_pred_test=y_pred_test, 
                                save_path=f"{result_path}_prediction_scatter.png")
        # save predictions
        pd.DataFrame({'smiles': train_data.smiles, 'target': y_train, 'prediction': y_pred_train}).to_csv(f"{result_path}_train_predictions.csv", index=False)
        pd.DataFrame({'smiles': test_data.smiles, 'target': y_test, 'prediction': y_pred_test}).to_csv(f"{result_path}_test_predictions.csv", index=False)

        # save model
        with open(f"{result_path}_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)

        # write arguments in parser to a file
        with open(f"{result_path}_args.txt", 'w') as f:
            f.write(str(args))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance ML with descriptor')
    parser.add_argument('--parent_directory', type=str, default='/Users/gbemidebe/Documents/GitHub/BatteryInformatics', help='Path to data')
    parser.add_argument('--train_data', type=str, default='train_data_cleaned.csv', help='Path to train data')
    parser.add_argument('--test_data', type=str, default='test_data_cleaned.csv', help='Path to test data')
    parser.add_argument('--scale', action='store_true', help='Scale data')
    parser.add_argument('--hyperparameter', type=str, default='hyperparameters.yaml', help='Hyperparameter space')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for hyperparameter optimization')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--model', type=str, default='RandomForestRegressor', help='Model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_result', action='store_true', help='Save results')
    parser.add_argument('--docker', action='store_true', help='Docker environment')

    args = parser.parse_args()
    main(args)