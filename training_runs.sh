#!/bin/bash
# run_trainings.sh
# This script trains an algorithm using different flag combinations, saves the results/model,
# and logs the outputs and timings for each run.

LOGFILE="training_runs.log"

# Start logging
echo "Starting training runs at $(date)" > "$LOGFILE"
echo "----------------------------------------" >> "$LOGFILE"

# Define arrays for environments and flags.
envs=("HalfCheetah" "Hopper" "Walker2d")
broken_flags=(0 1)
break_src_flags=(0 1)

# Loop over each combination.
for env in "${envs[@]}"; do
  for broken in "${broken_flags[@]}"; do
    for break_src in "${break_src_flags[@]}"; do
      # Create a unique save file name based on the current parameters.
      save_file_name="${env}_b${broken}_bs${break_src}"
      
      # Log the start of the training run.
      echo "Starting training with env: ${env}, broken: ${broken}, break_src: ${break_src} at $(date)" >> "$LOGFILE"
      echo "Results and model will be saved as: ${save_file_name}" >> "$LOGFILE"
      
      # Execute the training command and log its output.
      python3 train_darc.py --env "${env}" --save_file_name "${save_file_name}" --broken "${broken}" --break_src "${break_src}" &>> "$LOGFILE"
      
      # Log the completion of the training run.
      echo "Finished training with env: ${env}, broken: ${broken}, break_src: ${break_src} at $(date)" >> "$LOGFILE"
      echo "----------------------------------------" >> "$LOGFILE"
    done
  done
done

echo "All training runs completed at $(date)" >> "$LOGFILE"
echo "Training runs completed."
