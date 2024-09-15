#!/bin/bash

# Activate the virtual environment 'separationPINN'
# conda activate separationPINN

# Set variables
SEED=4059
IDENTIFIER="20240828_21"
MAIN_DIR="/Users/gbemidebe/Documents/GitHub/BatteryInformatics"
CODE_DIR="src/llm"
# check if the results folder {MAIN_DIR}/results/llm_{IDENTIFIER} & {MAIN_DIR}/saved_models/llm_{IDENTIFIER} exists and delete them
if [ -d "${MAIN_DIR}/results/llm_${IDENTIFIER}" ]; then
    echo "Deleting existing results folder"
    rm -r "${MAIN_DIR}/results/llm_${IDENTIFIER}"
fi
if [ -d "${MAIN_DIR}/saved_models/llm_${IDENTIFIER}" ]; then
    echo "Deleting existing saved_models folder"
    rm -r "${MAIN_DIR}/saved_models/llm_${IDENTIFIER}"
fi

# Run the training script
python ${CODE_DIR}/trainer.py \
    --train_data 'train_data_cleaned.csv' \
    --test_data 'test_data_cleaned.csv' \
    --train_val_split 0.2 \
    --seed ${SEED} \
    --model_name "seyonec/PubChem10M_SMILES_BPE_120k" \
    --main_directory "${MAIN_DIR}" \
    --result_dir "results" \
    --model_dir "saved_models" \
    --identifier "${IDENTIFIER}" \
    --steps 32 \

# Test the model
python ${CODE_DIR}/evaluate.py \
    --seed ${SEED} \
    --main_directory "${MAIN_DIR}" \
    --result_dir "results" \
    --model_dir "saved_models" \
    --identifier "${IDENTIFIER}"