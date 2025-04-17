import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup, set_seed
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse

from util_gsm8k import GSM8KDataset, evaluate_accuracy_gsm8k
from util_mbpp import MBPPDataset, evaluate_accuracy_mbpp

class CustomNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        # self.variance_epsilon = eps

    def forward(self, hidden_states):
        return self.weight * hidden_states + self.bias
    

def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, fresh_start):
    if os.path.isfile(checkpoint_path) and not fresh_start:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        if model.r is not None:
            model.linear.load_state_dict(checkpoint["linear_state"])
            model.norm.load_state_dict(checkpoint["norm_state"])
        model.soft_prompt.data.copy_(checkpoint["soft_prompt"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        update_step = checkpoint["update_step"]
        resume_batch_idx = checkpoint.get("batch_idx", 0)
        best_perplexity = checkpoint["best_perplexity"]
        print(f"Resuming from epoch {start_epoch}, batch {resume_batch_idx}, update step {update_step}")
        return start_epoch, update_step, resume_batch_idx, best_perplexity
    else:
        print("Starting from scratch.")
        return 0, 0, 0, float("inf")


# ------------------------------------------------------------------------------
# 3. GPT-2 Wrapper with a Shared Soft Prompt
# ------------------------------------------------------------------------------
class GPT2WithSoftToken(nn.Module):
    def __init__(self, model_name="gpt2", m=10, r=None):
        """
        model_name: Name of the pretrained LM.
        m: Number of learned soft prompt tokens.
        """
        super().__init__()
        self.gpt2 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            config=AutoConfig.from_pretrained(model_name, use_flash_attention=True)
        )
        # Freeze GPT-2 parameters.
        for param in self.gpt2.parameters():
            param.requires_grad = False

        hidden_size = self.gpt2.config.hidden_size
        
        self.r = r
        if self.r is None:
            self.soft_prompt = nn.Parameter(torch.randn(m, hidden_size, dtype=torch.bfloat16))
        else:
            self.soft_prompt = nn.Parameter(torch.randn(m, r, dtype=torch.bfloat16))
            self.linear = nn.Linear(r, hidden_size, bias=False, dtype=torch.bfloat16)
            self.linear.weight.requires_grad = False
            self.norm = CustomNorm(hidden_size)

        self.base_token_embeddings = self.gpt2.get_input_embeddings()
        self.m = m

    def forward(self, input_ids, attention_mask, prompt_length, answer_offset=None, text_length=None):
        """
        input_ids: (batch_size, seq_len) --> final input = text_prompt + tokenized text.
        attention_mask: (batch_size, seq_len)
        prompt_length: Length of the fixed text prompt (integer).
        answer_offset: (optional) Either a scalar or list/tensor of length batch_size; the number of text tokens
                       (in the tokenized text, not counting the fixed prompt) before the answer begins.
        text_length: (optional) Either a scalar or list/tensor of length batch_size; the number of non-padded
                     tokens in the tokenized text.
        When answer_offset and text_length are provided (training/evaluation), loss is computed only on the answer part.
        """
        batch_size, seq_len = input_ids.shape
        base_embeds = self.base_token_embeddings(input_ids)  # (B, seq_len, hidden_size)

        # Expand the shared soft prompt.
        if self.r is None:
            soft_tokens = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            soft_tokens = self.norm(self.linear(self.soft_prompt))
            soft_tokens = soft_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, m, hidden_size)

        soft_token_length = soft_tokens.shape[1]
        # Concatenate soft prompt with base embeddings.
        final_embeds = torch.cat([soft_tokens, base_embeds], dim=1)
        # Adjust attention mask: prepend ones for soft prompt tokens.
        new_attention_mask = torch.cat(
            [torch.ones(batch_size, soft_token_length, dtype=attention_mask.dtype, device=attention_mask.device),
             attention_mask],
            dim=1
        )

        # If answer_offset and text_length are provided, apply loss only on the answer tokens.
        if answer_offset is not None and text_length is not None:
            final_seq_length = seq_len + soft_token_length
            labels = torch.full((batch_size, final_seq_length), -100, dtype=torch.long, device=input_ids.device)
            # Ensure answer_offset and text_length are lists.
            if isinstance(answer_offset, torch.Tensor):
                answer_offset = answer_offset.tolist()
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.tolist()
            for i in range(batch_size):
                start_idx = soft_token_length + prompt_length + answer_offset[i]
                end_idx = soft_token_length + prompt_length + text_length[i]
                labels[i, start_idx:end_idx] = input_ids[i, prompt_length + answer_offset[i] : prompt_length + text_length[i]]
        else:
            # Fallback: supervise all tokens after the fixed text prompt.
            labels = torch.full((batch_size, seq_len + soft_token_length), -100, dtype=torch.long, device=input_ids.device)
            labels[:, soft_token_length + prompt_length:] = input_ids[:, prompt_length:]
            labels[new_attention_mask == 0] = -100

        outputs = self.gpt2(inputs_embeds=final_embeds, attention_mask=new_attention_mask, labels=labels, prompt_length=self.m)
        return outputs
    

# ------------------------------------------------------------------------------
# 4. Training, Perplexity Evaluation, and Generation/Accuracy Evaluation
# ------------------------------------------------------------------------------
def train_soft_token_mapping(model, train_dataloader, val_dataloader, epochs=5, lr=1e-3,
                             eval_interval=100, gradient_accumulation_steps=1,
                             checkpoint_path="checkpoint.pt", fresh_start=False, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    total_steps = (epochs * len(train_dataloader)) // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(total_steps*0.05),
                                                num_training_steps=total_steps)

    start_epoch, update_step, resume_batch_idx, best_perplexity = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, fresh_start,
    )

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
                        state = {
                            "epoch": epoch,
                            "batch_idx": step + 1,
                            "update_step": update_step,
                            "soft_prompt": model.soft_prompt.data,
                            "linear_state": model.linear.state_dict() if model.r is not None else None, 
                            "norm_state": model.norm.state_dict() if model.r is not None else None, 
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_perplexity": best_perplexity,
                        }
                        save_checkpoint(state, checkpoint_path)

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train soft prompt with configurable learning rate.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="mbpp", help="Dataset to use: 'mbpp' or 'gsm8k'")
    parser.add_argument("--model_name", type=str, default="1b", help="model to use '1b' or '3b'")
    parser.add_argument("--r", type=int, default=None, help="r value")
    parser.add_argument("--m", type=int, default=100, help="m value")
    parser.add_argument("--bs", type=int, default=4, help="batch step")
    parser.add_argument("--acc_step", type=int, default=1, help="acc_steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class TrainingConfig:
        # Hyperparameters
        epochs = args.epochs
        batch_size = args.bs
        max_length = 1024
        m = args.m
        r = args.r
        lr = args.lr
        eval_interval = 100
        acc_steps = args.acc_step
        prompt = "You are a helpful assistant."
        fresh_start = True
        model_name = args.model_name
        dataset = args.dataset
        seed=42

    config = TrainingConfig()

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
    

    if config.dataset == "gsm8k":
        checkpoint_path = f"checkpoint/{config.model_name}_gsm8k_lr{config.lr}_pt{config.m}_epoch{config.epochs}_r{config.r}.pt"

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
        checkpoint_path = f"checkpoint/{config.model_name}_mbpp_lr{config.lr}_pt{config.m}_epoch{config.epochs}_r{config.r}.pt"
        output_file = f"{config.model_name}_mbpp_lr{config.lr}_pt{config.m}_epoch{config.epochs}_r{config.r}.json"
    
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


    # Instantiate the model.
    model = GPT2WithSoftToken(model_name=model_name, m=config.m, r=config.r)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # Train the model.
    train_soft_token_mapping(
        model,
        train_dataloader,
        val_dataloader,
        epochs=config.epochs,
        lr=config.lr,
        eval_interval=int(config.eval_interval / config.acc_steps),
        gradient_accumulation_steps=config.acc_steps,
        checkpoint_path=checkpoint_path,
        fresh_start=config.fresh_start,
        device=device
    )

    # Load the best checkpoint for generation evaluation.
    if os.path.isfile(checkpoint_path):
        print("Loading best checkpoint for generation evaluation.")
        checkpoint = torch.load(checkpoint_path)
        if config.r is not None:
            model.linear.load_state_dict(checkpoint["linear_state"])
            model.norm.load_state_dict(checkpoint["norm_state"])   
            
        model.soft_prompt.data.copy_(checkpoint["soft_prompt"])
    else:
        print("Best checkpoint not found. Using current model weights.")


    if config.dataset == "gsm8k":
        # Evaluate final answer accuracy.
        accuracy, preds, gts = evaluate_accuracy_gsm8k(model, test_dataloader, tokenizer, config.max_length, device=device)
        print(f"Final Answer Accuracy: {accuracy*100:.2f}%")

    elif config.dataset == "mbpp":
        accuracy = evaluate_accuracy_mbpp(model, test_dataloader, tokenizer, config.max_length, device=device, output_file=output_file)
        print(f"Final Answer Accuracy: {accuracy*100:.2f}%")
