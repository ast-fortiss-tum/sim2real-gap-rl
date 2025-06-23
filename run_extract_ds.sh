#!/bin/bash

# Common parameters

SEED=42  # Seed value included

echo "Starting experiments for Smart Grids..."

#########################################
# One-house experiments (broken == 0)
#########################################

# (i) Degree-based experiments: for varieties that use a degree (e.g., s, c, d, v)
for bias in 0.5; do
  for noise in 0.2; do
    for degree in 0.5 0.65 0.8; do
      echo "-----------------------------------------------------"
      echo "Preparing Datasets: bias=${bias}, degree=${degree}, noise=${noise}"
      python3 extract_dataset_DAE.py \
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
