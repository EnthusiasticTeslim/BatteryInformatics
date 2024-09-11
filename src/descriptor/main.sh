#!/bin/bash

# Define variables
PYTHON_SCRIPT="trainer.py"
MODEL="GradientBoostingRegressor" # SVR, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
seed=104
iterations=100
cv=10
# if docker, add --docker

# Run the Python script with arguments
python "$PYTHON_SCRIPT" \
    --scale \
    --save_result \
    --model "$MODEL" \
    --seed $seed \
    --iterations $iterations \
    --cv $cv


# current model: see the folders in src/descriptor/ for the results
# SVR: seed=104, iterations=20, cv=10
# RandomForestRegressor: seed=42, iterations=30, cv=10
# AdaBoostRegressor: seed=104, iterations=100, cv=10
# GradientBoostingRegressor: seed=104, iterations=100, cv=10