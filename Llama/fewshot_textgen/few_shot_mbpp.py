import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed)
from datasets import load_dataset
from tqdm import tqdm
import random
import json
from util_mbpp import evaluate_accuracy_mbpp_from_file

set_seed(42)

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
            f"<|start_header_id|>user<|end_header_id|>\n\nproblem: {demo['text']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{demo['code']}<|eot_id|>"
        )
    test_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\nproblem: {test_question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return system_text + demo_texts + test_text

def evaluate_few_shot(model, tokenizer, dataset, demos, prompt, max_length, device="cpu", output_file="output/few_shot_baseline.json"):
    """
    Evaluate a baseline that uses few-shot prompting.
    `dataset` is expected to be a Hugging Face Dataset (with fields "question" and "answer").
    """
    model.to(device)
    model.eval()
    predictions = []

    for sample in tqdm(dataset, desc="Evaluating Few-Shot Baseline"):
        test_question = sample["text"]
        gt = sample['test_list']
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
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # print(test_question)
        # print(gen_text)
        # print(gt)
        # exit()

        predictions.append({"gen_text": gen_text, "ground_truth": gt})

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return evaluate_accuracy_mbpp_from_file(output_file)

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
    gsm8k_train = load_dataset("mbpp", split="train")
    gsm8k_test = load_dataset("mbpp", split="test")
  
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
    random_indices = random.sample(range(len(gsm8k_train)), few_shot_count)
    for i in random_indices:
        sample = gsm8k_train[i]
        demos.append({
            "text": sample["text"],
            "code": sample["code"]
        })

    # Evaluate on the GSM8K test split (using the raw dataset)
    results = evaluate_few_shot(
        baseline_model, tokenizer, gsm8k_test, demos, prompt, max_length, device=device
    )
    print(f"Few-Shot Prompting Baseline Accuracy: {results*100:.2f}%")