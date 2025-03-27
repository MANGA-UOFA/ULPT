#!/bin/bash

# Define the log file
LOG_FILE="mbpp_log/3b_fourier_r_experiment.log"
echo "Starting experiments" > "$LOG_FILE"

rank=(128 512 1024)

for r in "${rank[@]}"
do
  echo "---------------------------------" >> "$LOG_FILE"
  echo "Running experiment with rank: $r" >> "$LOG_FILE"
  # Run the Python script with the current learning rate; adjust dataset or other parameters as needed
  CUDA_VISIBLE_DEVICES=0 python main.py --r "$r" --dataset "mbpp" --model_name "3b" --bs 2 --acc_step 2 --epochs 10 --lora_method "fourier" --lr "1e-2" 2>&1 | tee >(grep "Final Answer Accuracy" >> "$LOG_FILE")
  echo "Completed experiment with learning rate: $r" >> "$LOG_FILE"
done

echo "All experiments completed" >> "$LOG_FILE"