#!/bin/bash
# run_experiments.sh
# This script runs the training script for multiple combinations of Smart Grid settings.
# Ensure this script has executable permissions (chmod +x run_experiments.sh).

# Common parameters
TRAIN_STEPS=4000
LR=0.0001
MAX_STEPS=24
BS=12
UPDATE=1
DELTAR=1
NOISE=0.2
ENV_NAME="Smart_Grids"
SAVE_MODEL="./saved_models/"
SAVE_FILE_NAME="test_run_"

echo "Starting experiments for Smart Grids..."

# One-house experiments: broken==0 with variety modifications.
# We loop over lin_src, variety names, and degree values.
for lin_src in 0 1; do
  for variety in s c d v lc lp; do
    for degree in 0.1 0.5 0.8; do
      echo "-----------------------------------------------------"
      echo "Running one-house: broken=0, lin_src=${lin_src}, variety=${variety}, degree=${degree}"
      python3 train_darc_clean.py \
        --env-name ${ENV_NAME} \
        --broken 0 \
        --lin_src ${lin_src} \
        --variety-name ${variety} \
        --degree ${degree} \
        --noise ${NOISE} \
        --lr ${LR} \
        --train-steps ${TRAIN_STEPS} \
        --max-steps ${MAX_STEPS} \
        --bs ${BS} \
        --update ${UPDATE} \
        --deltar ${DELTAR} \
        --save-model ${SAVE_MODEL} \
        --save_file_name ${SAVE_FILE_NAME}
    done
  done
done

# Two-house experiments: broken==1 (two-house setting).
# We loop over the two possible break_src values.
for break_src in 0 1; do
  echo "-----------------------------------------------------"
  echo "Running two-house: broken=1, break_src=${break_src}"
  python3 train_darc_clean.py \
    --env-name ${ENV_NAME} \
    --broken 1 \
    --break_src ${break_src} \
    --lr ${LR} \
    --train-steps ${TRAIN_STEPS} \
    --max-steps ${MAX_STEPS} \
    --bs ${BS} \
    --update ${UPDATE} \
    --deltar ${DELTAR} \
    --save-model ${SAVE_MODEL} \
    --save_file_name ${SAVE_FILE_NAME}
done

echo "All experiments finished."
echo "---------------------------------------"

# PENDING EDIT THE BUFFER TO PICK ONLY THE SOC AND NOT THE WHOLE STATE !!!!!!!
# COULD BE NOT THAT DIRECT SINCE IT IS DEPENDING ON WHETHER THERE ARE TWO HOUSES OR NOT