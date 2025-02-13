# method_1_nvc_chatbot_sft_improved.py
# -*- coding: utf-8 -*-
"""
Improved SFT Fine-tuning for NVC Chatbot using Synthetic Data

This script fine-tunes an LLM for creating an NVC chatbot using Supervised
Fine-Tuning (SFT). It is designed to work with synthetic NVC data generated
by `data_generation.py` (specifically, `synthetic_nvc_data_detailed_prompt.csv`).

**IMPORTANT:** This script is improved but still relies on SYNTHETIC DATA.
For a truly effective NVC chatbot, consider:
1. Replacing synthetic data with a HIGH-QUALITY, HUMAN-CURATED dataset.
2. Exploring Gradient Ratio Policy Optimization (GRPO) for better rule enforcement.

This version includes:
- Correct data loading from `synthetic_nvc_data_detailed_prompt.csv`.
- Adjusted training hyperparameters for potentially better learning with
  the synthetic dataset.
- Increased logging frequency for monitoring training progress.
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# --- 1. Model and Tokenizer Loading ---
max_seq_length = 2048  # Adjust if needed
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  # Or choose a different base model
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# --- 2. LoRA Configuration ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Adjust LoRA rank as needed (experiment with 8, 16, 32)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", ],  # Adjust if needed
    lora_alpha=16,
    lora_dropout=0.05, # Added dropout for potential regularization
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# --- 3. Load Synthetic NVC Dataset from CSV ---
dataset = load_dataset('csv', data_files='synthetic_nvc_data_detailed_prompt.csv', split='train')

# --- 4. Data Formatting (Using Alpaca Prompt - Optional, but kept for consistency) ---
alpaca_prompt = """Below is a user input for an NVC chatbot, paired with an ideal NVC chatbot response. Your task is to learn to generate similar responses following nonviolent communication principles.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True,)


# --- 5. SFT Trainer Setup ---
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Adjust based on your GPU memory (experiment with 2, 4)
    gradient_accumulation_steps=4, # Keep or adjust (2, 4)
    warmup_steps=20,  # Increased warmup steps (experiment with 20, 50)
    max_steps=150,  # Increased max steps for more training (experiment with 150, 300, 500 or more)
    learning_rate=1e-4,  # Reduced learning rate slightly (experiment with 2e-4, 1e-4, 5e-5)
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=5, # Increased logging frequency to every 5 steps
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine", # Changed scheduler to cosine for potential better convergence
    seed=3407,
    output_dir="sft_nvc_chatbot_outputs_improved",  # New output directory
    report_to="none",
    save_strategy="steps", # Save checkpoints periodically
    save_steps=50, # Save checkpoint every 50 steps
    evaluation_strategy="no" # No evaluation during training for simplicity in this SFT example
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# --- 6. Train the Model ---
trainer.train()

# --- 7. Inference Example ---
alpaca_prompt_inference = """Below is a user input for an NVC chatbot. Generate an NVC-based response.

### Instruction:
{}

### Response:
"""

inference_prompt = alpaca_prompt_inference.format("I'm feeling ignored by my friends, they never text me back.")

inputs = tokenizer(
    [inference_prompt], return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print("\n--- Inference Output ---")
print(f"Prompt:\n{inference_prompt}")
print(f"Generated NVC Chatbot Response:\n{generated_text}")


# --- 8. Saving the Trained LoRA Adapters ---
model.save_pretrained("lora_nvc_chatbot_model_improved")
tokenizer.save_pretrained("lora_nvc_chatbot_model_improved")

print("\n--- Training and Saving Complete (Improved SFT) ---")
print("LoRA adapters saved to 'lora_nvc_chatbot_model_improved' directory.")
print("**Remember**: This script uses SYNTHETIC data. For a functional NVC chatbot,")
print("you'll likely need a HUMAN-CURATED dataset and potentially GRPO for better results.")
