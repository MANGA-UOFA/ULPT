import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

# ------------------------------------------------------------------------------
# 0. Extraction Tool: Extract Final Answer from Generated Text
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 1. GSM8K Dataset (with an option for inference)
# ------------------------------------------------------------------------------
class GSM8KDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, prompt="", max_length=256, inference=False):
        """
        hf_dataset: Hugging Face GSM8K dataset split.
        tokenizer: Tokenizer compatible with your LM.
        prompt: Optional text prompt inserted at the beginning.
        max_length: Maximum length for tokenization.
        inference: If True, only the question (plus an "answer:" cue) is used as input.
                   When False (training), the full text ("question: ...\nanswer: <answer>") is used.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.inference = inference

        self.prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
            self.prompt +
            "<|eot_id|>"
        )
        # self.prompt = "<|begin_of_text|>"

        # Pre-tokenize the fixed text prompt (to be prepended).
        self.prompt_ids = tokenizer(self.prompt, add_special_tokens=False)["input_ids"]
        self.prompt_length = len(self.prompt_ids)
        print("Text prompt tokens:", self.prompt_ids)
        print("Text prompt length:", self.prompt_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        prefix_template = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "question: {question}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        prefix_text = prefix_template.format(question=sample["question"])

        # prefix_text = f"question: {sample['question']}\nanswer:"

        if self.inference:
            text = prefix_text
            # text = f"question: {sample['question']}\nanswer:"
            answer_offset = None
            padding_option = False
        else:
            answer_template = (
                "answer: {answer}<|eot_id|>"
            )
            text = prefix_text + answer_template.format(answer=sample["answer"])

            # text = f"question: {sample['question']}\nanswer: {sample['answer']}<|end_of_text|>"

            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            answer_offset = len(prefix_ids)
            padding_option = "max_length"

        # Tokenize the full text.
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=padding_option,  # Use "max_length" for training, no padding for inference.
            add_special_tokens=False,
            return_tensors="pt"
        )
        text_ids = text_encoding["input_ids"].squeeze(0)      # shape: (L,) where L may be < max_length in inference.
        text_mask = text_encoding["attention_mask"].squeeze(0)
        text_length = int(text_mask.sum().item())  # Number of non-padded tokens.

        non_padded_ids = text_ids[text_mask.bool()].tolist()
        truncated_text = self.tokenizer.decode(non_padded_ids, skip_special_tokens=True)
        
        # Construct the final input by concatenating the fixed text prompt with the tokenized text.
        final_input_ids = self.prompt_ids + text_ids.tolist()
        final_attention_mask = [1] * self.prompt_length + text_mask.tolist()

        ret = {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(final_attention_mask, dtype=torch.long),
            "text": truncated_text,
            "prompt_length": self.prompt_length,
            "ground_truth": extract_final_answer(sample["answer"])
        }

        # For training, include answer_offset and text_length to later mask out loss.
        if not self.inference:
            ret["answer_offset"] = torch.tensor(answer_offset, dtype=torch.long)
            ret["text_length"] = torch.tensor(text_length, dtype=torch.long)

        return ret


# ------------------------------------------------------------------------------
# 2. Evaluation Function: Evaluate Accuracy of GSM8K Model
# ------------------------------------------------------------------------------
def evaluate_accuracy_gsm8k(model, dataloader, tokenizer, max_length, device="cpu"):
    """
    Generates outputs using input = soft prompt + text prompt + question,
    then extracts the final answer from the generated text and computes accuracy
    against the ground truth final answer.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    generated_answers_all = []
    ground_truths_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Accuracy"):
            input_ids = batch["input_ids"].to(device)

            prefix_embeds = model.base_token_embeddings(input_ids)
            prefix_attention_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device)

            generated_ids = model.gpt2.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_attention_mask,
                max_length=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # print(generated_texts)
            # print(batch['text'])
            # print(batch['ground_truth'])

            for gen_text, gt in zip(generated_texts, batch["ground_truth"]):
                pred_final = extract_final_answer(gen_text)
                generated_answers_all.append(pred_final)
                ground_truths_all.append(gt)
                if pred_final.strip() == gt.strip():
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, generated_answers_all, ground_truths_all