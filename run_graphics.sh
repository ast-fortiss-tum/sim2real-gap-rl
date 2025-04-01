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
SAVE_MODEL="."
SAVE_FILE_NAME="."
SEED=42  # Seed value included
#FIXED_START="27.11.2016"   # Fixed start date
#FIXED_START=None
NUM_GAMES=25
EVAL_SEED=126


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
        python3 evaluate_all2.py \
          --env-name ${ENV_NAME} \
          --broken 0 \
          --lin_src ${lin_src} \
          --variety-name ${variety} \
          --degree ${degree} \
          --noise ${noise} \
          --lr ${LR} \
          --seed ${SEED} \
          --num-games ${NUM_GAMES} \
          --run-mpc \
          --eval-seed ${EVAL_SEED}
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
      python3 evaluate_all2.py \
        --env-name ${ENV_NAME} \
        --broken 0 \
        --lin_src ${lin_src} \
        --variety-name lc \
        --capacity ${capacity} \
        --noise ${noise} \
        --lr ${LR} \
        --seed ${SEED} \
        --num-games ${NUM_GAMES} \
        --run-mpc \
        --eval-seed ${EVAL_SEED}
    done
  done
done

# (iii) p_lim-based experiments: for variety "lp"
for lin_src in 0 1; do
  for p_lim in 0.5 1.0 2.0; do
    for noise in 0.0 1.0; do
      echo "-----------------------------------------------------"
      echo "Running one-house (p_lim): broken=0, lin_src=${lin_src}, variety=lp, p_lim=${p_lim}, noise=${noise}"
      python3 evaluate_all2.py  \
        --env-name ${ENV_NAME} \
        --broken 0 \
        --lin_src ${lin_src} \
        --variety-name lp \
        --p_lim ${p_lim} \
        --noise ${noise} \
        --lr ${LR} \
        --seed ${SEED} \
        --num-games ${NUM_GAMES} \
        --run-mpc \
        --eval-seed ${EVAL_SEED}
    done
  done
done