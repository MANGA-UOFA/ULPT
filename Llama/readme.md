# Llama Experiment

## Overview

This repository contains code for experimenting with various fine-tuning and prompting strategies on LLaMA 3.2 models, with a focus on the ULPT method. It includes implementations and evaluation scripts for:
1. **Few-Shot Prompting** – Standard in-context learning with handcrafted prompts.
2. **LoRA-based Finetuning** – Includes vanilla LoRA, VeRA, and Fourier FT.
3. **Prompt Tuning** – Optimization of soft prompts for downstream tasks.
4. **ULPT** – Our method

Supported tasks include mathematical reasoning (GSM8K) and code generation (MBPP).

## Directory Structure
```
├── fewshot_textgen/           # Few-shot prompting code
│   ├── few_shot_gsm8k.py      # GSM8K few-shot evaluation
│   └── few_shot_mbpp.py       # MBPP few-shot evaluation
├── lora_textgen/              # LoRA, VeRA, and Fourier FT training/evaluation
│   ├──code_evaluator.py       # Code evaluation utility
│   ├──main.py                 # Entry point for training and evaluation
│   ├──run.sh                  # Training script
│   ├──util_mbpp.py            # MBPP evaluation utility
│   └──util_gsm8k.py           # GSM8K evaluation utility
├── ulpt_textgen/              # ULPT training and evaluation       
│   ├──code_evaluator.py       # Code evaluation utility
│   ├──main.py                 # Entry point for training and evaluation
│   ├──run.sh                  # Training script
│   ├──util_mbpp.py            # MBPP evaluation utility
│   └──util_gsm8k.py           # GSM8K evaluation utility  
└── readme.md                  # This file
```

## Requirements
In order to support the Llama 3.2 models, we need to update the following dependencies:
```bash
pip install transformers==4.48.2 peft==0.15.0 datasets==3.2.0 accelerate==1.3.0
```

## Inference only
```bash
cd fewhot_textgen
python few_shot_gsm8k.py # for math reasoning
python few_shot_mbpp.py # for code generation
```

## Training and Inference
We provide a script to train and evaluate the models on the GSM8K and MBPP tasks. The script trains the models for 3 epochs on GSM8K and 10 epochs on MBPP. The models are then evaluated using greedy decoding to ensure consistent answers. The learning rate is set to 1e-3 for ULPT, LoRA and VeRA and 1e-2 for PT and Fourier FT.

```bash
cd lora_textgen
sh run.sh # for LoRA, VeRA, and Fourier FT
```

```bash
cd ulpt_textgen
sh run.sh # for ULPT
```




