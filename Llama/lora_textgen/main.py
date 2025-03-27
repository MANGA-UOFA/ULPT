import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup, set_seed
from peft import get_peft_model, LoraConfig, VeraConfig, FourierFTConfig, PeftModel
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from util_gsm8k import GSM8KDataset, evaluate_accuracy_gsm8k
from util_mbpp import MBPPDataset, evaluate_accuracy_mbpp

class GPT2LoRA(nn.Module):
    def __init__(self, lora_config, model_name="gpt2", checkpoint_path=None):
        """
        model_name: Name of the pretrained LM.
        m: Number of learned soft prompt tokens (used only in prompt tuning).
        tuning_mode: "prompt" for soft prompt tuning or "lora" for LoRA adaptation.
        lora_r: LoRA low-rank dimension (only used when tuning_mode=="lora").
        """
        super().__init__()

        # Load the base GPT-2 model.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            config=AutoConfig.from_pretrained(model_name, use_flash_attention=True)
        )

        self.base_token_embeddings = self.base_model.get_input_embeddings()

        if checkpoint_path is not None:
            # trainable parameters are zero, only for inference
            self.gpt2 = PeftModel.from_pretrained(self.base_model, checkpoint_path)
        else:
            self.gpt2 = get_peft_model(self.base_model, lora_config)

    def forward(self, input_ids, attention_mask, prompt_length, answer_offset=None, text_length=None):
        """
        input_ids: (batch_size, seq_len) - tokenized text.
        attention_mask: (batch_size, seq_len)
        prompt_length: Length of the fixed text prompt (an integer).
        answer_offset: (optional) Either a scalar or list/tensor of length batch_size; the number of tokens
                       (after the prompt) before the answer begins.
        text_length: (optional) Either a scalar or list/tensor of length batch_size; the total number of
                     non-padded tokens in the tokenized text.
        When answer_offset and text_length are provided, loss is computed only on the answer part.
        """

        batch_size, seq_len = input_ids.shape
        # For LoRA tuning, no soft prompt is injected.
        # Build labels in a similar way to compute loss only on the answer tokens.
        if answer_offset is not None and text_length is not None:
            labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=input_ids.device)
            if isinstance(answer_offset, torch.Tensor):
                answer_offset = answer_offset.tolist()
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.tolist()
            for i in range(batch_size):
                start_idx = prompt_length + answer_offset[i]
                end_idx = prompt_length + text_length[i]
                labels[i, start_idx:end_idx] = input_ids[i, prompt_length + answer_offset[i]: prompt_length + text_length[i]]
        else:
            labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=input_ids.device)
            labels[:, prompt_length:] = input_ids[:, prompt_length:]
            labels[attention_mask == 0] = -100

        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels)

        return outputs
    
def train(model, train_dataloader, val_dataloader, epochs=5, lr=3e-3,
            eval_interval=100, gradient_accumulation_steps=1,
            checkpoint_path="test", device="cpu"):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    total_steps = (epochs * len(train_dataloader)) // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(total_steps*0.05),
                                                num_training_steps=total_steps)
    start_epoch = 0
    resume_batch_idx = 0
    update_step = 0
    best_perplexity = float("inf")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}", total=len(train_dataloader))
        for step, batch in pbar:
            # If resuming from a checkpoint, skip batches as needed.
            if epoch == start_epoch and step < resume_batch_idx:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_length = batch["prompt_length"]
            prompt_length = prompt_length[0] if isinstance(prompt_length, (list, torch.Tensor)) else prompt_length

            # Retrieve answer_offset and text_length (they exist only during training).
            answer_offset = batch["answer_offset"].to(device)
            text_length = batch["text_length"].to(device)
            # answer_offset, text_length = None, None

            outputs = model(input_ids, attention_mask, prompt_length=prompt_length,
                            answer_offset=answer_offset, text_length=text_length)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                update_step += 1

                pbar.set_postfix({
                    "loss": loss.item() * gradient_accumulation_steps,
                    "update_step": update_step,
                    "lr": scheduler.get_last_lr()[0],
                })

                if update_step % eval_interval == 0:
                    model.eval()
                    perplexity_val = evaluate_validation_loss(model, val_dataloader, device=device)
                    print(f"Update step {update_step}: Validation loss: {perplexity_val:.4f}")

                    if perplexity_val < best_perplexity:
                        best_perplexity = perplexity_val
                        model.gpt2.save_pretrained(checkpoint_path)
                        print(f"Saving checkpoint to {checkpoint_path}")

                    model.train()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        resume_batch_idx = 0

    print("Best Validation Perplexity:", best_perplexity)
    

def evaluate_validation_loss(model, dataloader, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_length = batch["prompt_length"]
            prompt_length = prompt_length[0] if isinstance(prompt_length, (list, torch.Tensor)) else prompt_length

            answer_offset = batch["answer_offset"].to(device)
            text_length = batch["text_length"].to(device)

            outputs = model(input_ids, attention_mask, prompt_length=prompt_length,
                            answer_offset=answer_offset, text_length=text_length)
            total_loss += outputs.loss.item()
            total_batches += 1
    avg_loss = total_loss / total_batches if total_batches > 0 else float("inf")
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train soft prompt with configurable learning rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="mbpp", help="Dataset to use: 'mbpp' or 'gsm8k'")
    parser.add_argument("--model_name", type=str, default="1b", help="model to use '1b' or '3b'")
    parser.add_argument("--r", type=int, default=8, help="r value")
    parser.add_argument("--bs", type=int, default=4, help="batch step")
    parser.add_argument("--acc_step", type=int, default=1, help="acc_steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lora_method", type=str, default="lora", help="LoRA method to use: 'lora', 'vera', 'fourier'")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class TrainingConfig:
        # Hyperparameters
        epochs = args.epochs
        batch_size = args.bs
        max_length = 1024
        r = args.r
        lr = args.lr
        eval_interval = 100
        acc_steps = args.acc_step
        prompt = "You are a helpful assistant."
        model_name = args.model_name
        dataset = args.dataset
        seed=42
        lora_method = args.lora_method

    config = TrainingConfig()

    if config.lora_method == 'lora':
        lora_config = LoraConfig(
            r=args.r,
            target_modules=['q_proj', 'v_proj'],
        )
    elif config.lora_method == 'vera':
        lora_config = VeraConfig(
            r=args.r,
            target_modules=['q_proj', 'v_proj'],
        )
    elif config.lora_method == 'fourier':
        lora_config = FourierFTConfig(
            n_frequency=args.r,
            target_modules=['q_proj', 'v_proj'],
        )
    else:
        raise ValueError(f"Unknown LoRA method: {config.lora_method}")
    
    set_seed(config.seed)

    if config.model_name == "1b":
        model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    elif config.model_name == "3b":
        model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    else:
        raise ValueError("Invalid model name.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LoRA(lora_config, model_name=model_name)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")


    if config.dataset == "gsm8k":
        checkpoint_path = f"checkpoint/{config.model_name}_{config.lora_method}_gsm8k_lr{config.lr}_epoch{config.epochs}_r{config.r}"

        gsm8k_train = load_dataset("gsm8k", "main", split="train")
        gsm8k_split = gsm8k_train.train_test_split(test_size=0.1, seed=42)
        gsm8k_train = gsm8k_split["train"]
        gsm8k_val = gsm8k_split["test"]

        gsm8k_test = load_dataset("gsm8k", "main", split="test")

        train_dataset = GSM8KDataset(gsm8k_train, tokenizer, prompt=config.prompt, max_length=config.max_length, inference=False)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
       
        val_dataset = GSM8KDataset(gsm8k_val, tokenizer, prompt=config.prompt, max_length=config.max_length, inference=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        test_dataset = GSM8KDataset(gsm8k_test, tokenizer, prompt=config.prompt, max_length=config.max_length, inference=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    elif config.dataset == "mbpp":
        checkpoint_path = f"checkpoint/{config.model_name}_{config.lora_method}_mbpp_lr{config.lr}_epoch{config.epochs}_r{config.r}"
        output_file = f"{config.model_name}_{config.lora_method}_mbpp_lr{config.lr}_epoch{config.epochs}_r{config.r}.json"
    
        mbpp_train= load_dataset("mbpp", split="train")
        train_dataset = MBPPDataset(mbpp_train, tokenizer, prompt=config.prompt, max_length=config.max_length, inference=False)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        mbpp_val = load_dataset("mbpp", split="validation")
        val_dataset = MBPPDataset(mbpp_val, tokenizer, prompt=config.prompt, max_length=config.max_length, inference=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        mbpp_test = load_dataset("mbpp", split="test")
        test_dataset = MBPPDataset(mbpp_test, tokenizer, prompt=config.prompt, max_length=config.max_length, inference=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    else:
        raise ValueError("Invalid dataset name.")
    

    train(
        model, 
        train_dataloader, 
        val_dataloader, 
        epochs=config.epochs, 
        lr=config.lr,
        eval_interval=config.eval_interval, 
        gradient_accumulation_steps=config.acc_steps,
        checkpoint_path=checkpoint_path, 
        device=device
        )

    # Load the best checkpoint for generation evaluation.
    if os.path.isdir(checkpoint_path):
        print("Loading best checkpoint for generation evaluation.")
        model = GPT2LoRA(lora_config, model_name=model_name, checkpoint_path=checkpoint_path)
    else:
        print("Best checkpoint not found. Using current model weights.")


    if config.dataset == "gsm8k":
        # Evaluate final answer accuracy.
        accuracy, preds, gts = evaluate_accuracy_gsm8k(model, test_dataloader, tokenizer, config.max_length, device=device)
        print(f"Final Answer Accuracy: {accuracy*100:.2f}%")

    elif config.dataset == "mbpp":
        accuracy = evaluate_accuracy_mbpp(model, test_dataloader, tokenizer, config.max_length, device=device, output_file=output_file)
        print(f"Final Answer Accuracy: {accuracy*100:.2f}%")
