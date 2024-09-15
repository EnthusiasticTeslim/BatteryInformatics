#!/bin/bash

# Define variables
MAIN_DIRECTORY='.'
PYTHON_SCRIPT="trainer.py"
MODEL_NAME="SVR"
SEED=42
ITERATIONS=100
CV=10
train_data="train_data_cleaned.csv"
test_data="test_data_cleaned.csv"
parameters="hp_descriptor.yaml"
# if docker, add --docker flag

python "$PYTHON_SCRIPT" \
    --parent_directory "$MAIN_DIRECTORY" --result_directory "$MAIN_DIRECTORY" \
    --train_data "$train_data" --test_data "$test_data" --scale --model "$MODEL_NAME" \
    --seed $SEED --iterations $ITERATIONS --hyperparameter "$parameters" --cv $CV \
    --docker