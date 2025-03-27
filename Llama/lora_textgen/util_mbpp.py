import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from code_evaluator import CodeEvaluator
import json


class MBPPDataset(Dataset):
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
                "problem: {text}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        prefix_text = prefix_template.format(text=sample["text"])

        # prefix_text = f"question: {sample['question']}\nanswer:"

        if self.inference:
            text = prefix_text
            # text = f"question: {sample['question']}\nanswer:"
            answer_offset = None
            padding_option = False
        else:
            answer_template = (
                "{code}<|eot_id|>"
            )
            text = prefix_text + answer_template.format(code=sample["code"])

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
            "test_list": '\n'.join(sample['test_list']),
        }

        # For training, include answer_offset and text_length to later mask out loss.
        if not self.inference:
            ret["answer_offset"] = torch.tensor(answer_offset, dtype=torch.long)
            ret["text_length"] = torch.tensor(text_length, dtype=torch.long)

        return ret
    

def evaluate_accuracy_mbpp(model, dataloader, tokenizer, max_length, device="cpu", output_file="code_predictions.json"):
    model.to(device)
    model.eval()
    predictions = []

    # Generate and collect predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating code..."):
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

            # Collect predictions with Python code
            for gen_text, gt in zip(generated_texts, batch["test_list"]):
                predictions.append({"gen_text": gen_text, "ground_truth": gt.split("\n")})

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return evaluate_accuracy_mbpp_from_file(output_file)


def evaluate_accuracy_mbpp_from_file(output_file="code_predictions.json"):
    # Evaluate from saved file
    evaluator = CodeEvaluator(timeout=2.0, num_workers=4)
    correct = 0
    total = 0

    with open(output_file, "r", encoding="utf-8") as f:
        loaded_predictions = json.load(f)
        total = len(loaded_predictions)
        for pred in tqdm(loaded_predictions, desc="Runing code..."):
            gen_text = pred["gen_text"]  # This is the generated Python code
            gt = pred["ground_truth"]  # Convert ground truth back to tuple
            try:
                pass_at_1 = evaluator.evaluate(gen_text, gt)['pass@1']
                # print(f"Pass at 1: {pass_at_1}")
                correct += pass_at_1
            except Exception as e:
                print(f"Error evaluating code: {e}")
                print(f"Generated code:\n{gen_text}")
                print(f"Ground truth: {gt}")
                pass_at_1 = 0  # Count as incorrect if evaluation fails

    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    print(evaluate_accuracy_mbpp_from_file("output/1b_mbpp_lr0.3_pt100_epoch10_r164.json"))


