#!/bin/bash

# Define variables
MAIN_DIRECTORY="/Users/gbemidebe/Documents/GitHub/BatteryInformatics"
RESULT_DIRECTORY="results/GNN"
DATA_DIRECTORY="data"
PYTHON_SCRIPT="$MAIN_DIRECTORY/src/graph/trainer.py"
seed=42
iterations=1000
cv=10
train_data="train_data_cleaned.csv"
test_data="test_data_cleaned.csv"


# Run the Python script with arguments
python "$PYTHON_SCRIPT" \
    --parent_directory "$MAIN_DIRECTORY" \
    --result_directory "$RESULT_DIRECTORY" \
    --data_directory "$DATA_DIRECTORY" \
    --train_data "$train_data" \
    --test_data "$test_data" \
    --seed $seed \
    --epoch $iterations \
    --cv $cv \
    --train \
    #--skip_cv \