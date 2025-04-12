#!/bin/bash
# run_experiments.sh
# This script runs the training script for multiple combinations of Smart Grid settings.

# Noise set (1): noise on linear environment (only obs. shift) GAUSSIAN mean:0.2 bias:0.5
# Assume we have acces to a perfect linear real data (offline) but with measurement noise (systematic and random).

#Noise set (2): noise on "damaged" (eff:0.5) environment (obs. and dyn. shift) GAUSSIAN mean:0.2 bias:0.5
# Assume we have acces to a realistic real data (offline) but with measurement noise (systematic and random).

#Noise set (3): noise on "damaged" (eff:0.5) or perfect linear environment (obs. and (dyn.) shift) Another random distribution

# Common parameters
TRAIN_STEPS=200   # Number of training steps
LR=0.0008
MAX_STEPS=24
BS=12
UPDATE=1
DELTAR=1
ENV_NAME="Smart_Grids"
SAVE_MODEL="./saved_models_experiments_5/"
SAVE_FILE_NAME="test_run_NOISE_SET(1)" 
SEED=20  # Seed value included
#FIXED_START="27.11.2016"   # Fixed start date
#FIXED_START=None

echo "Starting experiments for Smart Grids..."

#########################################
# One-house experiments (broken == 0)
#########################################

# (i) Degree-based experiments: for varieties that use a degree (e.g., s, c, d, v)
for lin_src in 1; do
  for variety in v; do
    for degree in 0.2 0.5 0.8 1.0; do
      for noise in 0.0 0.2; do
        for bias in 0.5; do
          for noise_cfrs in 0.0 0.2; do
            for use_denoiser in 0 1; do
              echo "-----------------------------------------------------"
              echo "Running one-house (degree): broken=0, lin_src=${lin_src}, variety=${variety}, degree=${degree}, noise=${noise}"
              python3 train_darc_SmartGrids.py \
                --save_file_name ${SAVE_FILE_NAME} \
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
                --seed ${SEED} \
                --bias ${bias} \
                --noise_cfrs ${noise_cfrs} \
                --use_denoiser ${use_denoiser}
            done
          done
        done
      done
    done
  done
done


echo "All experiments finished :)"
echo "---------------------------------------"
