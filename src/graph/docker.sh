#!/bin/bash

# Define variables
train_SCRIPT="trainer.py"
bash_SCRIPT="docker.sh"
seed=42
iterations=100
cv=10
train_data="train_data_cleaned.csv"
test_data="test_data_cleaned.csv"
# if docker, add --docker

# Run the Python script with arguments

python "$train_SCRIPT" \
    --parent_directory "." \
    --result_directory "." \
    --data_directory "." \
    --train_data "$train_data" \
    --test_data "$test_data" \
    --skip_cv \
    --seed $seed \
    --epoch $iterations \
    --cv $cv 