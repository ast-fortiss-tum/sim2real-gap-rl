#!/bin/bash
# run_experiments.sh
# This script runs the training script for multiple combinations of Smart Grid settings.

# Noise set (1): noise on linear environment (only obs. shift) GAUSSIAN mean:0.2 bias:0.5
# Assume we have acces to a perfect linear real data (offline) but with measurement noise (systematic and random).

#Noise set (2): noise on "damaged" (eff:0.5) environment (obs. and dyn. shift) GAUSSIAN mean:0.2 bias:0.5
# Assume we have acces to a realistic real data (offline) but with measurement noise (systematic and random).

#Noise set (3): noise on "damaged" (eff:0.5) or perfect linear environment (obs. and (dyn.) shift) Another random distribution

# Common parameters

SEED=42  # Seed value included

echo "Starting experiments for Smart Grids..."

#########################################
# One-house experiments (broken == 0)
#########################################

# (i) Degree-based experiments: for varieties that use a degree (e.g., s, c, d, v)
for bias in 0.2; do
  for noise in 0.2; do
    for degree in 0.5 0.65 0.8; do
      echo "-----------------------------------------------------"
      echo "Preparing Datasets: bias=${bias}, degree=${degree}, noise=${noise}"
      python3 extract_dataset.py \
        --seed ${SEED} \
        --bias ${bias} \
        --degree ${degree} \
        --noise ${noise}
      sleep 3
      echo "-----------------------------------------------------"
      echo "Generating DAE: bias=${bias}, degree=${degree}, noise=${noise}"
      python3 online_denoising_AE.py \
        --seed ${SEED} \
        --bias ${bias} \
        --degree ${degree} \
        --noise ${noise}
      echo "-----------------------------------------------------"
      sleep 2
    done
  done
done

echo "All dataset produced finished :)"
echo "---------------------------------------"
