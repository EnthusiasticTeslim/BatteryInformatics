#!/bin/bash

# Define variables
train_SCRIPT="trainer.py"
bash_SCRIPT="docker.sh"
seed=42
iterations=100
cv=10
train_data="train_data_cleaned.csv"
test_data="test_data_cleaned.csv"
directory="." # use the current directory i.e. /app
# if docker, add --docker

# Run the Python script with arguments
python "$train_SCRIPT" \
    --parent_directory "$directory" --result_directory "$directory" \
    --train_data "$train_data" --test_data "$test_data" \
    --train --skip_cv --seed $seed --epoch $iterations --cv $cv --print_result \
    --docker