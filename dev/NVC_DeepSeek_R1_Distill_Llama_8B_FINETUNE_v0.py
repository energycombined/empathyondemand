# NVC_DeepSeek_R1_Distill_Llama_8B_FINETUNE.py
# -*- coding: utf-8 -*-
"""
This script fine-tunes the DeepSeek model for chain-of-thought (CoT) reasoning
using a Nonviolent Communication (NVC) dataset.
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import pandas as pd
from datasets import Dataset

# --- 1. Prepare CoT Dataset ---
csv_file = "nvc_cot_dataset.csv"  # Path to your generated CoT dataset
cot_dataset_df = pd.read_csv(csv_file)

# Rename columns to match expected format and drop rows with NaN values
cot_dataset_df = cot_dataset_df.rename(columns={"instruction": "Question", "chain_of_thought": "Chain of Thought"})
cot_dataset_df = cot_dataset_df.dropna()


# Convert pandas DataFrame to Hugging Face Dataset format
dataset = Dataset.from_pandas(cot_dataset_df)


# --- 2. Load Model with Memory Optimization and Configure for LoRA Training ---
max_seq_length = 2048
dtype = None  # Auto-detect dtype
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# --- 3. Dataset Formatting ---
def format_cot(example):
    """Formats the dataset example into the desired CoT format."""
    return f"""### Question:
{example['Question']}

### Chain of Thought:
{example['Chain of Thought']}
""" # Removed Answer from format as it is not generated in data creation step.

# --- 4. Training Configuration ---
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=1000, # Reduced for example, adjust for full training
    learning_rate=2e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=25,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    output_dir="deepseek-nvc-cot-finetuned", # Changed output dir to be NVC specific
    save_strategy="steps",
    save_steps=500, # Save every 500 steps
    logging_dir="logs",        # Directory for storing logs
    logging_strategy="steps", # Log every logging_steps
    logging_first_step=True,
    report_to="tensorboard" # or "wandb" if you have it setup
)

# --- 5. Create Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="formatted",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
    formatting_func=format_cot,
)

# --- 6. Fine-tuning Execution ---
print("--- Start Fine-tuning ---")
trainer.train()
print("--- Fine-tuning Completed ---")

# --- 7. Save Model ---
print("--- Save Fine-tuned Model ---")
model.save_pretrained("deepseek-nvc-cot-finetuned") # Changed save dir to be NVC specific
tokenizer.save_pretrained("deepseek-nvc-cot-finetuned")
print("--- Model Saved ---")

# --- 8. Generation Example ---
print("--- Generation Example ---")
inputs = tokenizer(
    """### Question:
My partner never listens to me, what should I do?

### Chain of Thought:""",
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
print("--- Generation Example Completed ---")

print("NVC DeepSeek CoT Fine-tuning Script Completed.")
