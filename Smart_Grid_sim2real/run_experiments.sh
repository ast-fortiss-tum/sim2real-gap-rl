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
SAVE_MODEL="./saved_models_experiments_5/"
SAVE_FILE_NAME="test_run_"
SEED=98  # Seed value included
#FIXED_START="27.11.2016"   # Fixed start date
#FIXED_START=None

echo "Starting experiments for Smart Grids..."

#########################################
# One-house experiments (broken == 0)
#########################################

# (i) Degree-based experiments: for varieties that use a degree (e.g., s, c, d, v)
for lin_src in 0 1; do
  for variety in v; do
    for degree in 0.2 0.5 0.8; do
      for noise in 0.0 1.0; do
        echo "-----------------------------------------------------"
        echo "Running one-house (degree): broken=0, lin_src=${lin_src}, variety=${variety}, degree=${degree}, noise=${noise}"
        python3 train_darc_SmartGrids.py \
          --env-name ${ENV_NAME} \
          --broken 0 \
          --lin_src ${lin_src} \
          --variety-name ${variety} \
          --degree ${degree} \
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
  done
done

# (ii) Capacity-based experiments: for variety "lc"
for lin_src in 0 1; do
  for capacity in 2.0 4.0 5.0; do
    for noise in 0.0 1.0; do
      echo "-----------------------------------------------------"
      echo "Running one-house (capacity): broken=0, lin_src=${lin_src}, variety=lc, capacity=${capacity}, noise=${noise}"
      python3 train_darc_SmartGrids.py \
        --env-name ${ENV_NAME} \
        --broken 0 \
        --lin_src ${lin_src} \
        --variety-name lc \
        --capacity ${capacity} \
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
done

# (iii) p_lim-based experiments: for variety "lp"
for lin_src in 0 1; do
  for p_lim in 0.5 1.0 2.0; do
    for noise in 0.0 1.0; do
      echo "-----------------------------------------------------"
      echo "Running one-house (p_lim): broken=0, lin_src=${lin_src}, variety=lp, p_lim=${p_lim}, noise=${noise}"
      python3 train_darc_SmartGrids.py \
        --env-name ${ENV_NAME} \
        --broken 0 \
        --lin_src ${lin_src} \
        --variety-name lp \
        --p_lim ${p_lim} \
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
done

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
