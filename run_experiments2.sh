#!/bin/bash
# run_experiments.sh
# This script runs the training script for multiple combinations of Smart Grid settings.

# Common parameters
TRAIN_STEPS=200   # Number of training steps
LR=0.0008
MAX_STEPS=24
BS=12
UPDATE=1
DELTAR=1
ENV_NAME="Smart_Grids"
SAVE_MODEL="./saved_models_experiments_2/"
SAVE_FILE_NAME="test_run_"
SEED=42  # Seed value included
#FIXED_START="27.11.2016"   # Fixed start date
#FIXED_START=None

echo "Starting experiments for Smart Grids two houses..."

#########################################
# Two-house experiments (broken == 1)
#########################################
for break_src in 0 1; do
  for noise in 0.0 1.0; do
    echo "-----------------------------------------------------"
    echo "Running two-house: broken=1, break_src=${break_src}, noise=${noise}"
    python3 train_darc_SmartGrids.py \
      --env-name ${ENV_NAME} \
      --broken 1 \
      --break_src ${break_src} \
      --noise ${noise} \
      --lr ${LR} \
      --train-steps ${TRAIN_STEPS} \
      --max-steps ${MAX_STEPS} \
      --bs ${BS} \
      --update ${UPDATE} \
      --deltar ${DELTAR} \
      --save-model ${SAVE_MODEL} \
      --save_file_name ${SAVE_FILE_NAME} \
      --seed ${SEED}
  done
done

echo "All experiments finished :)"
echo "---------------------------------------"
