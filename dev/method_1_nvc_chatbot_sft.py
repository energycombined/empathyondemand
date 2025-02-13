# method_1_nvc_chatbot_sft.py
# -*- coding: utf-8 -*-
"""
SFT Fine-tuning for NVC Chatbot (Illustrative Example - Requires High-Quality NVC Dataset)

This script demonstrates Supervised Fine-Tuning (SFT) for creating an NVC chatbot
based on the prompt system described previously.

**IMPORTANT:** This code uses a placeholder dataset for demonstration purposes only.
For SFT to be effective for this complex task, you MUST replace the placeholder
dataset with a carefully curated, high-quality dataset of NVC conversations
that strictly adhere to ALL the rules and guidelines of your prompt system.

SFT may be less ideal than GRPO for this task due to the need to enforce
complex rules and behaviors, but this script shows how you would approach SFT.
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
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
    r=16,  # Adjust LoRA rank as needed
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", ],  # Adjust if needed
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# --- 3. Placeholder NVC Dataset (REPLACE THIS WITH YOUR REAL DATASET!) ---
# **WARNING:** This is a VERY basic placeholder.  It's insufficient for a real NVC chatbot.
# You need to create a dataset of conversations that PERFECTLY follow your NVC prompt rules.

placeholder_data = [
    {
        "instruction": "I'm feeling really down today.",
        "output": "Would you like to share more about what's happening so I can try to understand you better? It sounds like you are feeling down. Are you perhaps longing for some comfort or support?"
    },
    {
        "instruction": "My boss is always criticizing me!",
        "output": "It sounds like you are experiencing criticism from your boss.  If you're willing to share, I'm here to listen.  Could it be that you are feeling hurt because you have a need for respect or appreciation in your work environment?"
    },
    {
        "instruction": "My friend didn't invite me to their party and I'm angry.",
        "output": "If you'd like to tell me more, I'm here to listen. It sounds like you're feeling anger right now because your friend didn't invite you. Is it possible you are needing inclusion or connection with your friend group?"
    },
    {
        "instruction": "I feel misunderstood.",
        "output": "When you say you feel misunderstood, do you feel frustration or sadness because you want to be heard and understood?"
    },
    {
        "instruction": "I feel rejected.",
        "output": "When you think you are being rejected, do you feel sadness or loneliness?"
    }
]

dataset = Dataset.from_list(placeholder_data)

# --- 4. Data Formatting (Using Alpaca Prompt - Optional) ---
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
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4, # Adjust
    warmup_steps=10,  # Adjust
    max_steps=50,  # Reduced steps for demonstration - increase for real training (e.g., 1000 or more)
    learning_rate=2e-4,  # Adjust learning rate
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="sft_nvc_chatbot_outputs",  # Output directory for trained model
    report_to="none",
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

inference_prompt = alpaca_prompt_inference.format("I'm feeling so overwhelmed with work and family obligations.")

inputs = tokenizer(
    [inference_prompt], return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print("\n--- Inference Output ---")
print(f"Prompt:\n{inference_prompt}")
print(f"Generated NVC Chatbot Response:\n{generated_text}")


# --- 8. Saving the Trained LoRA Adapters ---
model.save_pretrained("lora_nvc_chatbot_model")
tokenizer.save_pretrained("lora_nvc_chatbot_model")

print("\n--- Training and Saving Complete ---")
print("LoRA adapters saved to 'lora_nvc_chatbot_model' directory.")
print("**Remember**: This script uses a PLACEHOLDER dataset. For a functional NVC chatbot,")
print("you MUST replace 'placeholder_data' with a high-quality, rule-adhering NVC dataset.")
print("SFT might have limitations for this complex task, consider GRPO for better rule enforcement.")
