#!/bin/bash

# Define variables
data_SCRIPT="data.py"
train_SCRIPT="trainer.py"
MODEL="SVR()"
MODEL_NAME="SVR"
seed=42
iterations=100
cv=10
path="."
data="data_cleaned.csv"
# if docker, add --docker

# Run the Python script with arguments
python "$data_SCRIPT" \
    --path "$path" \
    --data "$data" \
    --seed $seed

python "$train_SCRIPT" \
    --parent_directory "$path" \
    --scale \
    --save_result \
    --model "$MODEL" \
    --seed $seed \
    --iterations $iterations \
    --cv $cv \
    --docker