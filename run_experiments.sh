#!/bin/bash
# run_experiments.sh
# This script runs the training script for multiple combinations of Smart Grid settings.
# Ensure this script has executable permissions (chmod +x run_experiments.sh).

# Common parameters
TRAIN_STEPS=120   # For now !!!!!!!!!!!!11
LR=0.0008
MAX_STEPS=24
BS=12
UPDATE=1
DELTAR=1
NOISE=0.0    # Noise of the clasifier for now 0 !!!!!!
ENV_NAME="Smart_Grids"
SAVE_MODEL="./saved_models/"
SAVE_FILE_NAME="test_run_"

echo "Starting experiments for Smart Grids..."

#########################################
# One-house experiments (broken == 0)
#########################################

# (i) Degree-based experiments: for varieties that use a degree (e.g., s, c, d, v)
for lin_src in 0 1; do
  for variety in s c d v lc lp; do
    for degree in 0.1 0.5 0.8; do
      echo "-----------------------------------------------------"
      echo "Running one-house (degree): broken=0, lin_src=${lin_src}, variety=${variety}, degree=${degree}"
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

# (ii) Capacity-based experiments: for variety "lc"
for lin_src in 0 1; do
  for capacity in 2 4; do
    echo "-----------------------------------------------------"
    echo "Running one-house (capacity): broken=0, lin_src=${lin_src}, variety=lc, capacity=${capacity}"
    python3 train_darc_clean.py \
      --env-name ${ENV_NAME} \
      --broken 0 \
      --lin_src ${lin_src} \
      --variety-name lc \
      --capacity ${capacity} \
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

# (iii) p_lim-based experiments: for variety "lp"
for lin_src in 0 1; do
  for p_lim in 1.0 2.0; do
    echo "-----------------------------------------------------"
    echo "Running one-house (p_lim): broken=0, lin_src=${lin_src}, variety=lp, p_lim=${p_lim}"
    python3 train_darc_clean.py \
      --env-name ${ENV_NAME} \
      --broken 0 \
      --lin_src ${lin_src} \
      --variety-name lp \
      --p_lim ${p_lim} \
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

#########################################
# Two-house experiments (broken == 1)
#########################################
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

echo "All experiments finished (46 combinations)."
echo "---------------------------------------"
