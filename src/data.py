import pandas as pd
from sklearn.model_selection import train_test_split
import argparse, os
from utils import read_and_process_data, save_data


def main(args):
    if args.read_original_data:
        file_path = os.path.join(args.path, args.data)
        data = read_and_process_data(file_path, args.canonicalize)
        cleaned_filename = f"{os.path.splitext(args.data)[0]}_cleaned.csv"
        save_data(data, args.path, cleaned_filename)
    else:
        file_path = os.path.join(args.path, args.data)
        data = pd.read_csv(file_path)

    # Split data
    train, test = train_test_split(data, test_size=args.split_ratio, random_state=args.seed)

    # Save train and test data
    name_tag = os.path.splitext(args.data)[0].split('_cleaned')[0]
    save_data(train, args.path, f"train_{name_tag}_cleaned.csv")
    save_data(test, args.path, f"test_{name_tag}_cleaned.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--path', default='../data', help='Path to the data directory')
    parser.add_argument('--data', default="data.csv", help='Name of the data file')
    parser.add_argument('--read_original_data', action='store_true', help='Read original data')
    parser.add_argument('--canonicalize', action='store_true', help='Convert SMILES to canonical form')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Train-test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)

# Run the script: python src/data.py --path 'data' --read_original_data