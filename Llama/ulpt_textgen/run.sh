#!/bin/bash

# Define the log file
LOG_FILE="gsm8k_log/3b_r_experiment.log"
echo "Starting experiments" > "$LOG_FILE"

rank=(2 64 256)

for r in "${rank[@]}"
do
  echo "---------------------------------" >> "$LOG_FILE"
  echo "Running experiment with rank: $r" >> "$LOG_FILE"
  # Run the Python script with the current learning rate; adjust dataset or other parameters as needed
  CUDA_VISIBLE_DEVICES=0 python main.py --lr "1e-3" --dataset "gsm8k" --model_name "3b" --m 10 --bs 2 --acc_step 2 --epochs 3 --r "$r" 2>&1 | tee >(grep "Final Answer Accuracy" >> "$LOG_FILE")
  echo "Completed experiment with rank: $r" >> "$LOG_FILE"
done

echo "All experiments completed" >> "$LOG_FILE"



