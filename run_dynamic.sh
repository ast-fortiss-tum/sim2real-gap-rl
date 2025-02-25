#!/bin/bash
# run_experiments.sh
# This script runs experiments for Smart Grid and Gymnasium environments.
#
# For Smart Grid experiments, we use the hyper parameters specified in your file:
#   --train-steps 4000, --max-steps 1000, --lr 1e-4, --bs 256, --update 1, --deltar 1,
#   --normalize 1, --broken-p 1.0, --variety-name g, --degree 0.5, --noise 0.2, --policynet 256,
#   --classifier 32, --warmup 50.
#
# For Smart Grid experiments, we only test the broken case (i.e. --broken 1). Then:
#   - When --break_src 1, the source environment will be the linear version and the target the nonlinear.
#   - When --break_src 0, the roles are reversed.
#
# For Gymnasium experiments (e.g., HalfCheetah-v4, Reacher-v2), all combinations of --broken and --break_src are explored.
#

BASE_LOG_DIR="logs"
DATE=$(date +%Y%m%d_%H%M%S)

#####################################
# 1. Smart Grid Experiments
#####################################
echo "Running Smart Grid experiments..."
# Environment flag used by your code for Smart Grid logic.
ENV="Smart_Grids"

# Hyper parameters (as in your file)
TRAIN_STEPS=4000
MAX_STEPS=1000
LR=1e-4
BS=256
UPDATE=1
DELTAR=1
NORMALIZE=1
BROKEN_P=1.0
VARIETY="g"
DEGREE=0.5
NOISE=0.2
POLICENET=256
CLASSIFIER=32
WARMUP=50

# For Smart Grid experiments, we fix broken=1 and break_joint=0.
BROKEN=1
BREAK_JOINT=0

# Loop over break_src values (0 and 1)
for BREAK_SRC in 0 1; do
    SAVE_MODEL_PATH="${BASE_LOG_DIR}/${ENV}/${ENV}_broken${BROKEN}_breaksrc${BREAK_SRC}_${DATE}"
    mkdir -p "$SAVE_MODEL_PATH"
    
    echo "Running Smart Grid experiment with break_src=${BREAK_SRC}"
    
    python3 train_darc.py \
      --env-name "${ENV}" \
      --save-model "${SAVE_MODEL_PATH}" \
      --save_file_name "${ENV}_broken${BROKEN}_breaksrc${BREAK_SRC}_" \
      --train-steps "${TRAIN_STEPS}" \
      --max-steps "${MAX_STEPS}" \
      --lr "${LR}" \
      --bs "${BS}" \
      --update "${UPDATE}" \
      --deltar "${DELTAR}" \
      --broken "${BROKEN}" \
      --break_src "${BREAK_SRC}" \
      --break_joint "${BREAK_JOINT}" \
      --normalize "${NORMALIZE}" \
      --broken-p "${BROKEN_P}" \
      --variety-name "${VARIETY}" \
      --degree "${DEGREE}" \
      --noise "${NOISE}" \
      --policynet "${POLICENET}" \
      --classifier "${CLASSIFIER}" \
      --warmup "${WARMUP}" \
      &> "${SAVE_MODEL_PATH}/training.log"
      
    echo "Completed Smart Grid experiment with break_src=${BREAK_SRC}"
done

echo "Smart Grid experiments completed."
echo "---------------------------------------"

#####################################
# 2. Gymnasium Experiments
# (e.g., HalfCheetah-v4, Reacher-v2)
#####################################
echo "Running Gymnasium experiments..."
# List the Gymnasium environments you want to test.
gym_envs=("HalfCheetah-v4" "Reacher-v2")

# Loop over all combinations of broken and break_src for these environments.
for ENV in "${gym_envs[@]}"; do
  for BROKEN in 0 1; do
    for BREAK_SRC in 0 1; do
      SAVE_MODEL_PATH="${BASE_LOG_DIR}/${ENV}/${ENV}_broken${BROKEN}_breaksrc${BREAK_SRC}_${DATE}"
      mkdir -p "$SAVE_MODEL_PATH"
      
      echo "Running Gym experiment for ${ENV} with broken=${BROKEN} and break_src=${BREAK_SRC}"
      
      python3 train_darc2.py \
        --env-name "${ENV}" \
        --save-model "${SAVE_MODEL_PATH}" \
        --save_file_name "${ENV}_broken${BROKEN}_breaksrc${BREAK_SRC}_" \
        --train-steps "${TRAIN_STEPS}" \
        --max-steps "${MAX_STEPS}" \
        --lr "${LR}" \
        --bs "${BS}" \
        --update "${UPDATE}" \
        --deltar "${DELTAR}" \
        --broken "${BROKEN}" \
        --break_src "${BREAK_SRC}" \
        --break_joint "${BREAK_JOINT}" \
        --normalize "${NORMALIZE}" \
        --broken-p "${BROKEN_P}" \
        --variety-name "${VARIETY}" \
        --degree "${DEGREE}" \
        --noise "${NOISE}" \
        --policynet "${POLICENET}" \
        --classifier "${CLASSIFIER}" \
        --warmup "${WARMUP}" \
        &> "${SAVE_MODEL_PATH}/training.log"
      
      echo "Completed Gym experiment for ${ENV} with broken=${BROKEN} and break_src=${BREAK_SRC}"
      echo "---------------------------------------"
    done
  done
done

echo "All experiments completed."
echo "---------------------------------------"