# method_2_nvc_chatbot_grpo.py
# -*- coding: utf-8 -*-
"""
GRPO Fine-tuning for NVC Chatbot (Method 2) using Synthetic Data

This script fine-tunes an LLM for creating an NVC chatbot using Gradient
Ratio Policy Optimization (GRPO). It is adapted from the Unsloth GRPO notebook
for Llama 3.1, and designed to incorporate NVC principles.

**IMPORTANT:** This script is a METHOD 2 EXPERIMENT using GRPO with
SYNTHETIC DATA.

**Limitations and Caveats:**
- GRPO is generally more effective with comparative data or human preference data,
  not directly with synthetic SFT-style data.
- The reward functions defined in this script are rudimentary and serve as a
  starting point. They might not perfectly capture all nuances of NVC.
- For a truly effective NVC chatbot using GRPO, you would ideally need:
    1. A different data generation strategy focused on comparative examples or
       human preference data related to NVC conversations.
    2. More sophisticated and nuanced NVC-specific reward functions, possibly
       involving NLP techniques to assess feeling/need identification,
       conversation flow, and adherence to all NVC principles.
- This script is intended for EXPERIMENTAL purposes to explore GRPO's
  potential in the NVC chatbot context using available synthetic data.
- For a production-ready NVC chatbot, consider human-curated datasets and
  more refined GRPO reward mechanisms.
"""

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
max_seq_length = 512  # Adjust if needed
lora_rank = 32       # Adjust LoRA rank as needed

# --- 1. Model and Tokenizer Loading ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct", # Or choose a different base model
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

from datasets import load_dataset, Dataset
import re

# --- 2. Data Preparation ---
# Load synthetic NVC data (replace with your actual data path if different)
dataset = load_dataset('csv', data_files='synthetic_nvc_data_detailed_prompt.csv', split='train')

SYSTEM_PROMPT = """
Respond based on Nonviolent Communication principles following all instructions.
"""

def prepare_grpo_prompts(examples):
    prompts = []
    answers = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': instruction}
        ]
        prompts.append(prompt)
        answers.append(output) # Using the synthetic SFT output as "answer" for GRPO (simplification)
    return {'prompt': prompts, 'answer': answers}

dataset = dataset.map(prepare_grpo_prompts, batched=True)


# --- 3. NVC Reward Functions (Rudimentary - to be improved) ---
def avoid_advice_reward_func(completions, **kwargs) -> list[float]:
    """Penalize responses that give advice (basic keyword check)."""
    advice_keywords = ["you should", "you could", "maybe try", "try to", "advise", "suggestion"]
    rewards = []
    for completion in completions:
        response_text = completion[0]['content'].lower()
        reward = 1.0  # Default reward
        for keyword in advice_keywords:
            if keyword in response_text:
                reward -= 0.5  # Penalize for advice keywords
        rewards.append(max(0.0, reward)) # Reward should not be negative
    return rewards

def nvc_question_reward_func(completions, **kwargs) -> list[float]:
    """Reward responses using NVC-style questions, penalize forbidden questions."""
    nvc_question_starters = ["would you like", "do you want", "do you need", "is important to you", "do you find important"]
    forbidden_question_starters = ["do you feel", "do you have the feeling that"]
    rewards = []
    for completion in completions:
        response_text = completion[0]['content'].lower()
        reward = 0.0
        for starter in nvc_question_starters:
            if starter in response_text:
                reward += 0.25 # Reward for NVC question starter
                break # Only reward once per response even if multiple starters used (adjust as needed)
        for forbidden_starter in forbidden_question_starters:
            if forbidden_starter in response_text:
                reward -= 0.5 # Penalize forbidden question starter
        rewards.append(max(0.0, reward))
    return rewards

def length_penalty_reward_func(completions, **kwargs) -> list[float]:
    """Penalize responses exceeding 100 words."""
    max_words = 100
    rewards = []
    for completion in completions:
        response_text = completion[0]['content']
        word_count = len(response_text.split())
        if word_count > max_words:
            penalty = (word_count - max_words) * 0.01 # Penalty per extra word
            rewards.append(max(0.0, 1.0 - penalty)) # Reward decreases with length over limit
        else:
            rewards.append(1.0)
    return rewards

def combined_nvc_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Combines multiple NVC reward functions."""
    rewards = [0.0] * len(completions) # Initialize rewards to 0

    # Apply individual reward functions and add to combined reward
    advice_rewards = avoid_advice_reward_func(completions, **kwargs)
    question_rewards = nvc_question_reward_func(completions, **kwargs)
    length_rewards = length_penalty_reward_func(completions, **kwargs)

    for i in range(len(completions)):
        rewards[i] += advice_rewards[i] * 0.5  # Weight advice avoidance
        rewards[i] += question_rewards[i] * 0.3 # Weight NVC questions
        rewards[i] += length_rewards[i] * 0.2   # Weight length penalty

    return rewards


# --- 4. GRPO Trainer Configuration ---
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,      # Adjust learning rate as needed (experiment)
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1, # Keep batch size low for GRPO
    gradient_accumulation_steps = 4, # Adjust gradient accumulation (experiment)
    num_generations = 4,         # Number of generations per prompt (adjust memory)
    max_prompt_length = 256,
    max_completion_length = 100, # Match max response length from instructions
    max_steps = 300,             # Adjust max steps for training duration (experiment)
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "grpo_nvc_chatbot_outputs", # New output directory for GRPO
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [combined_nvc_reward_func], # Use the combined reward function
    args = training_args,
    train_dataset = dataset,
)

# --- 5. Train the Model with GRPO ---
trainer.train()

# --- 6. Inference with GRPO Fine-tuned Model ---
inference_prompt_text = """Hello, I'm feeling really down about my work situation."""

inference_prompt = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : inference_prompt_text},
], tokenize = False, add_generation_prompt = True)


from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.7,
    top_p = 0.95,
    max_tokens = 100, # Limit response length for inference as well
)

output = model.fast_generate(
    inference_prompt,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_nvc_chatbot_outputs/checkpoint-250/lora"), # Load the saved LoRA
)[0].outputs[0].text

print("\n--- GRPO Inference Output ---")
print(f"Prompt:\n{inference_prompt_text}")
print(f"Generated NVC Chatbot Response (GRPO Fine-tuned):\n{output}")


# --- 7. Saving the GRPO Fine-tuned LoRA Adapters ---
model.save_lora("lora_nvc_chatbot_model_grpo")
tokenizer.save_pretrained("lora_nvc_chatbot_model_grpo")

print("\n--- GRPO Fine-tuning and Saving Complete ---")
print("GRPO LoRA adapters saved to 'lora_nvc_chatbot_model_grpo' directory.")
print("**Remember**: This is an EXPERIMENTAL GRPO fine-tuning using SYNTHETIC data.")
print("For a functional NVC chatbot, consider human-curated data and refined reward functions.")
