# NVC_DeepSeek_R1_Distill_Llama_8B_FINETUNE.py
##DeepSeek R1 version
# -*- coding: utf-8 -*-
"""
Enhanced fine-tuning script for DeepSeek-R1 with improved CoT handling
and API response structure alignment
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset, Dataset
import pandas as pd
import torch

# --- 1. Enhanced Model Loading with Architecture Alignment ---
max_seq_length = 4096  # Increased to match DeepSeek's 64K context handling
lora_rank = 64  # Increased rank for better CoT capture

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.95,
)

# Expanded target modules for better reasoning capture
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=lora_rank*2,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# --- 2. Enhanced Dataset Processing with API Structure Alignment ---
def format_api_response(example):
    """Formats examples to match DeepSeek's API response structure"""
    return tokenizer.apply_chat_template([
        {"role": "user", "content": example["instruction"]},
        {
            "role": "assistant",
            "content": example["final_answer"],
            "reasoning_content": example["chain_of_thought"]
        }
    ], tokenize=False)

class CoTLogitsProcessor(LogitsProcessor):
    """Encourages CoT-style reasoning through logit biasing"""
    def __call__(self, input_ids, scores):
        # Boost probability of reasoning-related tokens
        reasoning_tokens = [
            tokenizer.encode(" step", add_special_tokens=False)[-1],
            tokenizer.encode(" reason", add_special_tokens=False)[-1],
            tokenizer.encode(" because", add_special_tokens=False)[-1],
            tokenizer.encode(" therefore", add_special_tokens=False)[-1],
        ]
        scores[:, reasoning_tokens] += 2.0  # Adjust boost strength as needed
        return scores

# --- 3. Optimized Training Configuration ---
training_args = TrainingArguments(
    output_dir="./nvc_deepseek_cot_finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_torch_fused",
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    num_train_epochs=5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    report_to="wandb",
    remove_unused_columns=False,
    max_grad_norm=0.5,
)

# --- 4. Enhanced SFTTrainer Configuration ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=nvc_cot_dataset,
    formatting_func=format_api_response,
    max_seq_length=max_seq_length,
    args=training_args,
    logits_processor=LogitsProcessorList([CoTLogitsProcessor()]),
    packing=True,
    dataset_num_proc=2,
)

# --- 5. Improved Training Execution ---
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    trainer.train()

# --- 6. Enhanced Model Saving ---
model.save_pretrained_merged(
    "nvc_deepseek_cot_merged",
    tokenizer,
    save_method="merged_16bit",
    push_to_hub=True,
    hf_model_name="your_hf_username/nvc_deepseek_cot",
)

# --- 7. API-Aligned Inference ---
def generate_cot_response(prompt, max_cot_tokens=512, max_answer_tokens=128):
    """Mimics the DeepSeek API response structure"""
    chat_prompt = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "", "reasoning_content": ""}
    ]
    
    # First generate CoT reasoning
    cot_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    ) + "\n### Chain of Thought:"
    
    cot_inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)
    cot_outputs = model.generate(
        **cot_inputs,
        max_new_tokens=max_cot_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    cot_content = tokenizer.decode(cot_outputs[0][cot_inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Then generate final answer
    answer_prompt = f"{cot_prompt}{cot_content}\n### Final Answer:"
    answer_inputs = tokenizer(answer_prompt, return_tensors="pt").to(model.device)
    answer_outputs = model.generate(
        **answer_inputs,
        max_new_tokens=max_answer_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    final_answer = tokenizer.decode(answer_outputs[0][answer_inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "content": final_answer.strip(),
        "reasoning_content": cot_content.strip()
    }

# Example usage
if __name__ == "__main__":
    test_prompt = "A colleague took credit for my work in a team meeting. How should I handle this using NVC?"
    response = generate_cot_response(test_prompt)
    
    print("### User Input:")
    print(test_prompt)
    print("\n### Chain of Thought:")
    print(response["reasoning_content"])
    print("\n### Final Answer:")
    print(response["content"])
