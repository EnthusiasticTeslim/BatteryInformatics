#!/bin/bash

# Set default values
MAIN_DIRECTORY="/Users/gbemidebe/Documents/GitHub/BatteryInformatics"
RESULT_DIRECTORY="results"
DATA_DIRECTORY="data"
PYTHON_SCRIPT="${MAIN_DIRECTORY}/src/descriptor/trainer.py"
train_data="train_data_cleaned.csv"
test_data="test_data_cleaned.csv"
MODELS=("SVR") # ("GradientBoostingRegressor" "SVR" "RandomForestRegressor" "AdaBoostRegressor")
SEED=104
ITERATIONS=1
CV=5

# Function to run the Python script
run_model() {
    local model=$1
    echo "Running model: $model"
    python "$PYTHON_SCRIPT" \
        --parent_directory "$MAIN_DIRECTORY" \
        --result_directory "$RESULT_DIRECTORY" \
        --data_directory "$DATA_DIRECTORY" \
        --train_data "$train_data" \
        --test_data "$test_data" \
        --scale \
        --model "$model" \
        --seed $SEED \
        --iterations $ITERATIONS \
        --hyperparameter "hp_descriptor.yaml" \
        --cv $CV
}

# Main execution
echo "Starting model execution..."

for model in "${MODELS[@]}"; do
    run_model "$model"
done

echo "All models completed."

# Model-specific configurations (for reference)
# SVR: seed=104, iterations=20, cv=10
# RandomForestRegressor: seed=42, iterations=30, cv=10
# AdaBoostRegressor: seed=104, iterations=100, cv=10
# GradientBoostingRegressor: seed=104, iterations=100, cv=10