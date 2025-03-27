import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          get_linear_schedule_with_warmup, set_seed)
from datasets import load_dataset
import evaluate
import math
from tqdm import tqdm
import os
import re

set_seed(42)

# -----------------------------
# 0. Utility: Extract Final Answer
# -----------------------------
def extract_final_answer(text):
    """
    Extracts the final answer from generated text for GSM8K.
    Handles cases where the answer follows '####', including prefixed text.
    If no '####' is found, it returns the last number in the text.
    """
    # Remove commas that appear between digits (e.g., "1,234" -> "1234").
    clean_text = re.sub(r"(?<=\d),(?=\d)", "", text)
    
    # Look for "####" and extract the first number that follows, even with text in between.
    match = re.search(r"####.*?([-+]?\d*\.\d+|\d+)", clean_text)
    if match:
        return match.group(1).strip()
    
    # Fallback: find the last number in the text.
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
    return numbers[-1].strip() if numbers else clean_text.strip()

# -----------------------------
# 6. Few-Shot Prompting Baseline Evaluation
# -----------------------------
def build_few_shot_prompt(demos, test_question, prompt):
    """
    Constructs a prompt containing:
      - A system-level prompt.
      - A set of few-shot demonstration examples.
      - The test question with an answer cue.
    """
    system_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"
    demo_texts = ""
    for demo in demos:
        demo_texts += (
            f"<|start_header_id|>user<|end_header_id|>\n\nquestion: {demo['question']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nanswer: {demo['answer']}<|eot_id|>"
        )
    test_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\nquestion: {test_question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return system_text + demo_texts + test_text

def evaluate_few_shot(model, tokenizer, dataset, demos, prompt, max_length, device="cpu"):
    """
    Evaluate a baseline that uses few-shot prompting.
    `dataset` is expected to be a Hugging Face Dataset (with fields "question" and "answer").
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions = []
    ground_truths = []
    for sample in tqdm(dataset, desc="Evaluating Few-Shot Baseline"):
        test_question = sample["question"]
        gt = extract_final_answer(sample["answer"])
        few_shot_prompt = build_few_shot_prompt(demos, test_question, prompt)

        inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0][prompt_length:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(generated_text)

        pred = extract_final_answer(generated_text)

        predictions.append(pred)
        ground_truths.append(gt)
        if pred.strip() == gt.strip():
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions, ground_truths

# -----------------------------
# 7. Main: Train Soft Prompt and Evaluate Both Methods
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token


    max_length = 512
    prompt = "You are a helpful assistant."

    # Load GSM8K dataset splits
    gsm8k_train = load_dataset("gsm8k", "main", split="train")
    gsm8k_split = gsm8k_train.train_test_split(test_size=0.1, seed=42)
    gsm8k_train = gsm8k_split["train"]
    gsm8k_val = gsm8k_split["test"]
    gsm8k_test = load_dataset("gsm8k", "main", split="test")
  

    # -----------------------------
    # Few-Shot Prompting Baseline
    # -----------------------------
    print("Evaluating Few-Shot Prompting Baseline...")
    # For the baseline, load the base LM without any soft prompt tuning.
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=AutoConfig.from_pretrained(model_name, use_flash_attention=True)
    )
    baseline_model.to(device)
    baseline_model.eval()

    # Select a fixed set of demonstration examples from the training set.
    few_shot_count = 4
    demos = []
    for sample in gsm8k_train.select(range(few_shot_count)):
        demos.append({
            "question": sample["question"],
            "answer": sample["answer"]  # Use the full original chain-of-thought
        })

    # Evaluate on the GSM8K test split (using the raw dataset)
    few_shot_accuracy, few_shot_preds, few_shot_gts = evaluate_few_shot(
        baseline_model, tokenizer, gsm8k_test, demos, prompt, max_length, device=device
    )
    print(f"Few-Shot Prompting Baseline Accuracy: {few_shot_accuracy*100:.2f}%")